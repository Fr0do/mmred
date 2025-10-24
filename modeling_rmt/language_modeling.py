import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import PreTrainedModel


class MemoryCell(torch.nn.Module):
    """Wraps a transformer model with learnable memory tokens."""

    def __init__(self, base_model, num_mem_tokens: int):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens: int):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim), dtype=embeddings.weight.dtype)
        memory_weights *= embeddings.weight.data.std()
        self.register_parameter("memory", torch.nn.Parameter(memory_weights, requires_grad=True))
        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        return self.memory.repeat(input_shape[0], 1, 1)

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        write_mem = kwargs.get("write_mem", True)
        seg_kwargs = self.process_input(input_ids, memory_state, write_mem=write_mem, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, write_mem=write_mem, **kwargs)
        return out, new_memory_state

    def generate(self, input_ids, memory_state=None, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)
        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask, write_mem=False)
        return self.model.generate(
            inputs_embeds=seg_kwargs["inputs_embeds"], attention_mask=seg_kwargs["attention_mask"], **generate_kwargs
        )

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if self.num_mem_tokens > 0:
            if write_mem:
                inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
        seg_kwargs["input_ids"] = None
        seg_kwargs["inputs_embeds"] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
            if kwargs.get('prev_attn_mask') is not None:
                seg_kwargs['attention_mask'] = torch.cat([kwargs['prev_attn_mask'], seg_kwargs['attention_mask']], dim=-1)
            if 'prev_attn_mask' in seg_kwargs:
                seg_kwargs.pop('prev_attn_mask')
        seg_kwargs["output_hidden_states"] = True
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        mask = torch.ones(*shape[:2], dtype=torch.int64, device=attention_mask.device)
        mask[:, self.num_mem_tokens : self.num_mem_tokens + attention_mask.shape[1]] = attention_mask
        return mask

    def process_output(self, model_outputs, write_mem=True, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]

            logit_start = self.num_mem_tokens
            logit_end = -self.num_mem_tokens if write_mem else None

            out["logits"] = model_outputs.logits[:, logit_start:logit_end]
            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [lh[:, logit_start:logit_end] for lh in model_outputs.hidden_states]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs.attentions
            if hasattr(model_outputs, "past_key_values"):
                out["past_key_values"] = model_outputs.past_key_values
        else:
            memory_state = None
            out = model_outputs
        return out, memory_state


class RecurrentWrapper(PreTrainedModel):
    """Segments long sequences and processes them recurrently using MemoryCell."""
    # TODO: necessary?
    base_model_prefix = "memory_cell.model"  # Changed from "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    
    def __init__(self, memory_cell: MemoryCell, **rmt_kwargs):
        self.config = memory_cell.model.base_model.config
        super().__init__(self.config)
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
    
    
    def forward(
        self, input_ids, labels=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None, **kwargs
    ):
        sliding_window = self.rmt_config['sliding_window'] if 'sliding_window' in self.rmt_config else False
        memory_state = None
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)
        out = self.process_outputs(
            cell_outputs, labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)
        for seg_num, segment in enumerate(segmented[:-1]):
            _, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
        final_segment = segmented[-1]
        return self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is None:
                continue
            k_segments = self.split_tensor(tensor)
            for s, k_seg in enumerate(k_segments):
                if s < len(segments):
                    segments[s][k] = k_seg
                else:
                    segments.append({k: k_seg})
        return segments

    def split_tensor(self, tensor):
        align = self.rmt_config.get("segment_alignment")
        segment_size = self.rmt_config.get("segment_size")
        if align in {"left", None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for start, end in zip(split_inds, split_inds[1:])]
        elif align in {"right"}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for start, end in zip(split_inds, split_inds[1:])]
        elif align == "center":
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
        labels = kwargs.get("labels")
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss_fct = CrossEntropyLoss()
            out["loss"] = loss_fct(flat_logits, flat_labels)
        else:
            out["loss"] = 0
        out["logits"] = full_logits
        if kwargs.get("output_hidden_states"):
            out["hidden_states"] = full_hidden_states
        return out

    def manage_gradients(self, memory_state, seg_num):
        k2 = self.rmt_config.get("k2")
        max_n_segments = self.rmt_config.get("max_n_segments")
        if seg_num == 0 or k2 in {-1, None} or seg_num + k2 > max_n_segments:
            return memory_state
        return memory_state.detach()
