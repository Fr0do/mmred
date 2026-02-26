import math
from typing import Any, Dict, List, Optional

import torch
from torch.nn import CrossEntropyLoss, Parameter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM 
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

class MemoryCell(torch.nn.Module):
    def __init__(self, embeddings, num_mem_tokens):
        super().__init__()
        self._create_memory(embeddings, num_mem_tokens)
    
    def _create_memory(self, embeddings, num_mem_tokens: int) -> None:
        self.num_mem_tokens = num_mem_tokens
        if num_mem_tokens in {0, None}:
            if hasattr(self, "memory"):
                del self.memory
            self.register_parameter("memory", None)
        self.memory = Parameter(torch.empty(num_mem_tokens, embeddings.embedding_dim, dtype=embeddings.weight.dtype), requires_grad=False)
        with torch.no_grad():
            self.memory.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input):
        memory = self.memory.repeat(input.size(0), 1, 1)
        return memory.to(input.device, input.dtype)

class RMTQwen3Config(Qwen3Config):
    model_type = "rmt-qwen3"

    def __init__(
        self,
        segment_size: Optional[int] = None,
        max_n_segments: Optional[int] = None,
        num_mem_tokens: int = 0,
        segment_alignment: Optional[str] = None,
        k2: int = -1,
        sliding_window: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.segment_size = segment_size
        self.max_n_segments = max_n_segments
        self.num_mem_tokens = num_mem_tokens
        self.segment_alignment = segment_alignment
        self.k2 = k2
        self.sliding_window = sliding_window


class RMTQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = RMTQwen3Config

    def __init__(self, config: RMTQwen3Config):
        super().__init__(config)
        self.memory_cell = MemoryCell(self.get_input_embeddings(), config.num_mem_tokens)
        self.num_mem_tokens = config.num_mem_tokens

    def _pad_attention_mask(self, attention_mask: torch.Tensor, shape: torch.Size, write_mem: bool = True) -> torch.Tensor:
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            last_ids = -self.num_mem_tokens if write_mem else None
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens:last_ids] = attention_mask
            return mask
    
    def _prepare_segment_inputs(
        self,
        segment_kwargs: Dict[str, torch.Tensor],
        memory_state: Optional[torch.Tensor],
        write_mem: bool,
        output_attentions: Optional[bool],
    ) -> Dict[str, Any]:
        seg_kwargs: Dict[str, Any] = {k: v for k, v in segment_kwargs.items() if v is not None}
        input_ids = seg_kwargs.pop("input_ids", None)
        inputs_embeds = seg_kwargs.get("inputs_embeds")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided for each segment.")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if self.num_mem_tokens not in {0, None}:
            if memory_state is None:
                # memory_state = self.memory_cell(inputs_embeds)
                memory_state = inputs_embeds.mean(1, keepdim=True).repeat(1, self.memory_cell.memory.size(0), 1)
            if write_mem:
                inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
        seg_kwargs["inputs_embeds"] = inputs_embeds
        seg_kwargs["input_ids"] = None

        # Generate position_ids for this segment (each segment starts at position 0
        # because memory tokens are prepended and RoPE should be relative to segment start)
        batch_size, seq_len, _ = inputs_embeds.shape
        seg_kwargs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

        attention_mask = seg_kwargs.get("attention_mask")
        if attention_mask is not None:
            padded_mask = self._pad_attention_mask(attention_mask, inputs_embeds.shape, write_mem)
            prev_attn_mask = seg_kwargs.pop("prev_attn_mask", None)
            if prev_attn_mask is not None:
                padded_mask = torch.cat([prev_attn_mask, padded_mask], dim=-1)
            seg_kwargs["attention_mask"] = padded_mask

        seg_kwargs["output_hidden_states"] = True
        if output_attentions is not None:
            seg_kwargs["output_attentions"] = output_attentions
        seg_kwargs["return_dict"] = True

        return seg_kwargs

    def _process_segment_output(
        self,
        model_outputs: CausalLMOutputWithPast,
        write_mem: bool,
        output_hidden_states: bool,
    ) -> tuple[CausalLMOutputWithPast, Optional[torch.Tensor]]:
        if self.num_mem_tokens in {0, None}:
            return model_outputs, None

        out = CausalLMOutputWithPast()
        memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]

        logit_start = self.num_mem_tokens
        logit_end = -self.num_mem_tokens if write_mem else None
        out["logits"] = model_outputs.logits[:, logit_start:logit_end]
        if output_hidden_states:
            out["hidden_states"] = [hs[:, logit_start:logit_end] for hs in model_outputs.hidden_states]
        if model_outputs.attentions is not None:
            out["attentions"] = model_outputs.attentions
        if hasattr(model_outputs, "past_key_values"):
            out["past_key_values"] = model_outputs.past_key_values

        return out, memory_state

    def _forward_segment(
        self,
        segment: Dict[str, torch.Tensor],
        memory_state: Optional[torch.Tensor],
        write_mem: bool = True,
        output_hidden_states: bool = True,
        **kwargs: Any,
    ) -> tuple[CausalLMOutputWithPast, Optional[torch.Tensor]]:
        seg_kwargs = self._prepare_segment_inputs(
            segment,
            memory_state=memory_state,
            write_mem=write_mem,
            output_attentions=kwargs.get("output_attentions"),
        )
        filtered_kwargs = {k: v for k, v in seg_kwargs.items() if k not in seg_kwargs}
        base_outputs = super().forward(labels=None, use_cache=False, **seg_kwargs, **filtered_kwargs)
        return self._process_segment_output(base_outputs, write_mem=write_mem, output_hidden_states=output_hidden_states)

    def segment(self, **kwargs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        segment_size = self.config.segment_size
        if segment_size in {None, 0}:
            return [{k: v for k, v in kwargs.items()}]

        segments: List[Dict[str, torch.Tensor]] = []
        for key, tensor in kwargs.items():
            if tensor is None or tensor.ndim < 2:
                continue
            key_segments = self._split_tensor(tensor, segment_size)
            for index, key_segment in enumerate(key_segments):
                if index < len(segments):
                    segments[index][key] = key_segment
                else:
                    segments.append({key: key_segment})
        return segments

    def _split_tensor(self, tensor: torch.Tensor, segment_size: int) -> List[torch.Tensor]:
        align = getattr(self.config, "segment_alignment", None)
        if align in {"left", None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            return [tensor[:, start:end] for start, end in zip(split_inds, split_inds[1:])]
        if align == "right":
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            return [tensor[:, start:end] for start, end in zip(split_inds, split_inds[1:])]
        if align == "center":
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            return list(torch.chunk(tensor, n_seg, dim=1))
        raise NotImplementedError(f"Unsupported segment alignment: {align}")

    def _process_outputs(
        self,
        cell_outputs: List[CausalLMOutputWithPast],
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        out = CausalLMOutputWithPast()
        logits = torch.cat([output.logits for output in cell_outputs], dim=1)
        out["logits"] = logits
        if labels is not None:
            out["loss"] = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
        else:
            out["loss"] = None

        if output_hidden_states:
            hidden_sequences = [o.hidden_states for o in cell_outputs if o.hidden_states is not None]
            if hidden_sequences:
                out["hidden_states"] = tuple(
                    [torch.cat(layer_hidden, dim=1) for layer_hidden in zip(*hidden_sequences)]
                )
        if output_attentions:
            attention_sequences = [o.attentions for o in cell_outputs if o.attentions is not None]
            if attention_sequences:
                out["attentions"] = tuple(attention_sequences)

        return out

    def _manage_gradients(self, memory_state: Optional[torch.Tensor], seg_num: int) -> Optional[torch.Tensor]:
        k2 = getattr(self.config, "k2", None)
        max_n_segments = getattr(self.config, "max_n_segments", None)
        if memory_state is None or k2 in {-1, None} or max_n_segments is None:
            return memory_state
        if seg_num == 0 or seg_num + k2 > max_n_segments:
            return memory_state
        return memory_state.detach()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        segment_size = getattr(self.config, "segment_size", None)
        if segment_size in {None, 0} or getattr(self, '_use_original_forward', False):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if return_dict is None:
            return_dict = self.config.use_return_dict

        segmented = self.segment(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        memory_state = None
        cell_outputs: List[CausalLMOutputWithPast] = []
        output_attentions = kwargs.pop("output_attentions", False)
        output_hidden_states = kwargs.pop("output_hidden_states", False)

        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self._forward_segment(
                {k: segment.get(k) for k in ("input_ids", "inputs_embeds", "attention_mask", "position_ids")},
                memory_state=memory_state,
                write_mem=True,# write_mem=seg_num < len(segmented) - 1,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            cell_outputs.append(cell_out)
            memory_state = self._manage_gradients(memory_state, seg_num)
        out = self._process_outputs(
            cell_outputs,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

        if not return_dict:
            return tuple(v for v in out.to_tuple() if v is not None)
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs: Any,
    ):
        segment_size = getattr(self.config, "segment_size", None)
        if segment_size in {None, 0}:
            return super().generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)

        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)
        memory_state = None
        for seg_num, segment in enumerate(segmented[:-1]):
            _, memory_state = self._forward_segment(
                segment,
                memory_state=memory_state,
                write_mem=True,
                output_hidden_states=False,
                output_attentions=False,
            )

        final_segment = segmented[-1]
        seg_kwargs = self._prepare_segment_inputs(
            final_segment,
            memory_state=memory_state,
            write_mem=False,
            output_attentions=False,
        )
        self._use_original_forward = True
        generation_output = super().generate(
            inputs_embeds=seg_kwargs["inputs_embeds"],
            attention_mask=seg_kwargs.get("attention_mask"),
            **generate_kwargs,
        )
        self._use_original_forward = False
        return generation_output

try:
    AutoConfig.register(RMTQwen3Config.model_type, RMTQwen3Config)
    AutoModelForCausalLM.register(RMTQwen3Config, RMTQwen3ForCausalLM)
except ValueError:
    # Config or model may already be registered in some environments.
    pass