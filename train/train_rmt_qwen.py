import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


import torch
from datasets import load_dataset
from modeling_rmt import RMTQwen3Config, RMTQwen3ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetArgs:
    dataset_name: str
    subset: str = field(default="default", metadata={"help": "Dataset subset."})
    split: str = field(default="train", metadata={"help": "Dataset split."})
    val_split: str = field(default="val", metadata={"help": "Dataset validation split."})
    system_prompt: Optional[str] = "You are a helpful AI assistant."
    task_prompt: Optional[str] = "Answer with a single word or number."
    subsample_train: float = field(default=1.00, metadata={"help": "Subsample train set."})


@dataclass
class RMTArgs:
    segment_size: int = field(metadata={"help": "Segment size for RMT."})
    num_mem_tokens: int = field(default=16, metadata={"help": "Number of memory tokens."})
    max_n_segments: Optional[int] = field(default=None, metadata={"help": "Maximum number of segments to train on."})
    k2: int = field(default=-1, metadata={"help": "BPTT unroll steps for RMT."})
    segment_alignment: Optional[str] = field(default=None, metadata={"help": "Segment alignment strategy."})
    sliding_window: bool = field(default=False, metadata={"help": "Whether to enable sliding window alignment."})


@dataclass
class CurriculumArgs:
    curriculum_stages: List[Dict[str, int]] = field(default_factory=list, metadata={"help": "Curriculum stages."})
    early_stopping_patience: int = field(
        default=1,
        metadata={"help": "Number of evaluation rounds to wait for improvement before early stopping."},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum improvement in eval loss to reset patience."},
    )


def preprocess_function(example, ds_args: "DatasetArgs", tokenizer, segment_size: int) -> Dict[str, Any]:
    """
    Segments:
        [S0 (padded to segment_size), S1 (padded), ..., Sk (padded), S_final (assistant_prefix + completion, padded)]

    S0..Sk together contain:
        system + user (chat template, pre-assistant area) + steps (whole list broken into pieces)

    S_final contains:
        assistant_prefix (chat template) + completion (json answer + eos)
    """
    # 1) Build user message text (task prompt + question)
    user_content_parts: List[str] = []
    if getattr(ds_args, "task_prompt", None):
        user_content_parts.append(ds_args.task_prompt)
    user_content_parts.append(example["question"])
    user_content = "\n".join(user_content_parts)

    # 2) Messages up to assistant (no assistant content yet)
    messages = []
    if getattr(ds_args, "system_prompt", None):
        messages.append({"role": "system", "content": ds_args.system_prompt})
    messages.append({"role": "user", "content": user_content})

    # 3) We need to split the chat-template tokens into:
    #    - pre_assistant_tokens: everything BEFORE the assistant header/prefix
    #    - assistant_prefix_tokens: the assistant header/prefix itself (no assistant content)
    #
    #    Robust approach:
    #      A) Template with add_generation_prompt=False  → ends right after the user content (no assistant header)
    #      B) Template with add_generation_prompt=True + dummy assistant content → assistant header + dummy content
    #      C) Find where dummy content begins; tokens before it contain both pre_assistant + assistant_prefix.
    #         Then subtract (A) from the front to isolate the assistant prefix.
    #
    #   NOTE: This avoids relying on EOS positioning or tokenizer-specific special IDs.
    baseline = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_dict=True
    )
    with_gen_and_dummy = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": "DUMMY"}],
        add_generation_prompt=False, tokenize=True, return_dict=True, enable_thinking=False,
    )

    base_ids = list(baseline.input_ids)  # up to end of user
    gen_dummy_ids = list(with_gen_and_dummy.input_ids)  # includes assistant prefix + tokens for "DUMMY"

    # Tokenize the dummy string alone to locate its start reliably.
    dummy_only = tokenizer("DUMMY", add_special_tokens=False).input_ids

    # Find the dummy content start inside gen_dummy_ids
    # We search for the first exact match of the dummy_only sequence.
    def find_subseq(haystack: List[int], needle: List[int]) -> int:
        if not needle:
            return -1
        L, N = len(haystack), len(needle)
        for i in range(max(0, len(base_ids)), L - N + 1):  # start search at least after baseline
            if haystack[i:i+N] == needle:
                return i
        return -1

    dummy_start = find_subseq(gen_dummy_ids, dummy_only)
    if dummy_start == -1:
        # Fallback: if we can't find it, we treat the entire difference as assistant prefix.
        # This is rare, but keeps behavior defined.
        pre_assistant_tokens = base_ids
        assistant_prefix_tokens = gen_dummy_ids[len(base_ids):]
    else:
        # Tokens before dummy_start contain: base_ids + assistant_prefix_tokens
        assert len(gen_dummy_ids) >= dummy_start
        pre_plus_prefix = gen_dummy_ids[:dummy_start]
        # Split into (pre-assistant, assistant_prefix) using base_ids length.
        assert len(pre_plus_prefix) >= len(base_ids)
        pre_assistant_tokens = pre_plus_prefix[:len(base_ids)]
        assistant_prefix_tokens = pre_plus_prefix[len(base_ids):]

    # Sanity: pre_assistant_tokens should equal base_ids (they should!),
    # but keep the variable for clarity.
    # Now, we will construct:
    #   prompt_tokens_before_assistant = pre_assistant_tokens + steps_tokens
    #   completion_segment = assistant_prefix_tokens + completion_tokens (+ eos)
    prompt_tokens_before_assistant = list(pre_assistant_tokens)

    # 4) Steps
    steps = example["sequence_json"].replace("'", '"')
    if isinstance(steps, str):
        steps = json.loads(steps)
    # Ensure steps are strings with a trailing newline (as the original code)
    step_texts = [str(s) + "\n" for s in steps]

    # Tokenize steps one-by-one, then append into the prompt area (before assistant)
    tokenized_steps = tokenizer(step_texts, add_special_tokens=False)
    for step_ids in tokenized_steps.input_ids:
        prompt_tokens_before_assistant.extend(step_ids)

    # 5) Pack pre-assistant prompt content into fixed-size segments, padding each to segment_size
    input_ids: List[int] = []
    num_segments = 0

    def flush_segment(seg: List[int], pad_to: int, pad_id: int):
        nonlocal input_ids, num_segments
        if not seg:
            return
        if len(seg) < pad_to:
            seg = seg + [pad_id] * (pad_to - len(seg))
        input_ids.extend(seg)
        num_segments += 1

    # Greedy packer for the prompt content (pre-assistant + steps)
    current_seg: List[int] = []
    for tid in prompt_tokens_before_assistant:
        if len(current_seg) == segment_size:
            flush_segment(current_seg, segment_size, tokenizer.pad_token_id)
            current_seg = []
        current_seg.append(tid)
    # If there's remaining content in the current segment, pad and flush
    if current_seg:
        flush_segment(current_seg, segment_size, tokenizer.pad_token_id)

    # 6) Build the completion segment (NOT padded):
    #    assistant_prefix + JSON completion + eos
    completion_text = json.dumps({"answer": example["answer"]})
    completion_ids = tokenizer(completion_text, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None; please set eos token.")
    completion_segment = list(assistant_prefix_tokens) + completion_ids + [eos_id]
    num_completion_paddings = len(completion_segment) - segment_size
    # num_completion_paddings = 0
    completion_segment = [tokenizer.pad_token_id] * num_completion_paddings + completion_segment

    # Append completion segment WITHOUT padding and count exactly one segment
    input_ids.extend(completion_segment)
    num_segments += 1  # final segment

    # 7) Build completion mask:
    #    - mask 0 over all padded prompt segments + assistant_prefix
    #    - mask 1 over the completion part of answer segment
    total_prompt_len = len(input_ids) - len(completion_segment) + num_completion_paddings + len(assistant_prefix_tokens)
    completion_mask = [0] * total_prompt_len + [1] * (len(input_ids) - total_prompt_len)

    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "num_segments": num_segments,
        "length": len(input_ids),
    }


def preprocess_for_train(example, ds_args: "DatasetArgs", tokenizer, segment_size: int) -> Dict[str, Any]:
    """
    Constructs the prompt, appends the answer in the assistant role, 
    and masks everything except the assistant's answer content.
    """
    # --- A. Construct Inputs ---
    user_content_parts: List[str] = []
    if ds_args.task_prompt:
        user_content_parts.append(ds_args.task_prompt)
    user_content_parts.append(example["question"])
    steps = example["sequence_json"].replace("'", '"')
    if isinstance(steps, str):
        steps = json.loads(steps)
    user_content_parts += [str(s) for s in steps]
    user_content = "\n".join(user_content_parts)
    
    # Create the answer JSON string
    answer_content = json.dumps({"answer": example["answer"]})

    messages = [
        {"role": "system", "content": ds_args.system_prompt or ""},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_content}
    ]
    
    # --- B. Tokenize Full Sequence ---
    full_enc = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        return_dict=True,
        add_generation_prompt=False, 
    )
    input_ids = full_enc['input_ids']
    attention_mask = full_enc['attention_mask']

    # --- C. Find Mask Boundary ---
    # To strictly mask the user prompt, we re-tokenize just the prompt part.
    prompt_enc = tokenizer.apply_chat_template(
        messages[:2], 
        tokenize=True, 
        return_dict=True,
        add_generation_prompt=True 
    )
    prompt_len = len(prompt_enc['input_ids'])

    # --- D. Create Labels ---
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "length": len(input_ids),
        "num_segments": len(input_ids) // segment_size + 1,
    }


def filter_by_max_segments(dataset, max_segments: Optional[int]):
    if max_segments is None:
        return dataset
    return dataset.filter(lambda x: x["num_segments"] <= max_segments)


if __name__ == "__main__":
    parser = TrlParser((SFTConfig, ModelConfig, DatasetArgs, RMTArgs, CurriculumArgs))
    training_args, model_args, ds_args, rmt_args, curriculum_args = parser.parse_args_and_config()
    if training_args.run_name:
        os.environ["CLEARML_TASK"] = training_args.run_name

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    curriculum_stages = curriculum_args.curriculum_stages or [
        {"max_n_segments": rmt_args.max_n_segments, "num_train_epochs": training_args.num_train_epochs}
    ]
    curriculum_max_segments = max(
        (stage.get("max_n_segments") for stage in curriculum_stages if stage.get("max_n_segments") is not None),
        default=None,
    )

    config = RMTQwen3Config.from_pretrained(
        model_args.model_name_or_path,
        segment_size=rmt_args.segment_size,
        max_n_segments=curriculum_max_segments or rmt_args.max_n_segments,
        num_mem_tokens=rmt_args.num_mem_tokens,
        segment_alignment=rmt_args.segment_alignment,
        k2=rmt_args.k2,
        sliding_window=rmt_args.sliding_window,
        local_files_only=True,
    )
    config.save_pretrained(training_args.output_dir)

    raw_ds = load_dataset(ds_args.dataset_name, ds_args.subset)
    raw_train_ds = raw_ds[ds_args.split]
    if ds_args.subsample_train < 1.0:
        raw_train_ds = raw_train_ds.shuffle(seed=42).select(range(int(len(raw_train_ds) * ds_args.subsample_train)))
    raw_val_ds = raw_ds[ds_args.val_split]

    processed_train_ds = raw_train_ds.map(
        lambda ex: preprocess_for_train(ex, ds_args, tokenizer, rmt_args.segment_size),
        remove_columns=raw_train_ds.column_names,
        num_proc=8,
    )
    processed_val_ds = raw_val_ds.map(
        lambda ex: preprocess_for_train(ex, ds_args, tokenizer, rmt_args.segment_size),
        remove_columns=raw_val_ds.column_names,
        num_proc=8,
    )

    model = RMTQwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=getattr(torch, model_args.dtype),
        attn_implementation=model_args.attn_implementation,
        local_files_only=True,
    )

    if model_args.use_peft:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            modules_to_save=model_args.lora_modules_to_save,
            target_modules=model_args.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.base_model.memory_cell.requires_grad_(False)
        model.print_trainable_parameters()

    training_args.remove_unused_columns = False
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better = False
    training_args.load_best_model_at_end = True
    
    data_collator = DataCollatorForLanguageModeling(tokenizer.pad_token_id, padding_free=True)

    logger.info("Running curriculum training across %d stages.", len(curriculum_stages))
    
    # Create trainer once - will be reused across stages
    trainer = None
    
    for stage_idx, stage in enumerate(curriculum_stages):
        stage_max_segments = stage.get("max_n_segments")
        stage_epochs = stage.get("num_train_epochs")
        if stage_epochs is None:
            raise ValueError("Each curriculum stage must define 'num_train_epochs'.")

        stage_train_ds = filter_by_max_segments(processed_train_ds, stage_max_segments)

        if len(stage_train_ds) == 0:
            raise ValueError(f"Stage {stage_idx} has no training samples after filtering with max_n_segments={stage_max_segments}.")

        logger.info(
            "Starting curriculum stage %d with max_n_segments=%s for %s epochs (%d training samples).",
            stage_idx,
            str(stage_max_segments),
            str(stage_epochs),
            len(stage_train_ds),
        )

        # First stage: create trainer
        stage_training_args = copy.deepcopy(training_args)
        stage_training_args.num_train_epochs = stage_epochs
        stage_training_args.output_dir = f"{training_args.output_dir}/stage_{stage_idx}"
        if training_args.run_name:
            os.environ["CLEARML_TASK"] = training_args.run_name + f"_stage_{stage_idx}"
        
        trainer = SFTTrainer(
            model=model,
            args=stage_training_args,
            train_dataset=stage_train_ds,
            eval_dataset=processed_val_ds,
            data_collator=data_collator,
            processing_class=tokenizer,
        )
        
        # trainer.add_callback(
        #     EarlyStoppingCallback(
        #         early_stopping_patience=curriculum_args.early_stopping_patience,
        #         early_stopping_threshold=curriculum_args.early_stopping_threshold,
        #     )
        # )
        
        if stage_idx:
            trainer.args.warmup_steps = 0
            trainer.args.warmup_ratio = 0.0
        trainer.train()
        trainer.save_model()
