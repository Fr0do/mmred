import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from modeling_rmt import RMTQwen3Config, RMTQwen3ForCausalLM, Qwen3ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
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
    subsample_train: float = field(default=1.0, metadata={"help": "Subsample train set."})


@dataclass
class RMTArgs:
    segment_size: int = field(metadata={"help": "Segment size for RMT."})
    num_mem_tokens: int = field(default=16, metadata={"help": "Number of memory tokens."})
    max_n_segments: Optional[int] = field(default=None, metadata={"help": "Maximum number of segments to train on."})
    k2: int = field(default=-1, metadata={"help": "BPTT unroll steps for RMT."})
    segment_alignment: Optional[str] = field(default=None, metadata={"help": "Segment alignment strategy."})
    sliding_window: bool = field(default=False, metadata={"help": "Whether to enable sliding window alignment."})


def preprocess_function(example, ds_args: DatasetArgs, tokenizer, segment_size: int):
    user_content_parts = []
    if ds_args.task_prompt:
        user_content_parts.append(ds_args.task_prompt)
    user_content_parts.append(example["question"])
    user_content = "\n".join(user_content_parts)

    prompt = []
    if ds_args.system_prompt:
        prompt.append({"role": "system", "content": ds_args.system_prompt})
    prompt.append({"role": "user", "content": user_content})

    tokenized_prompt = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        thinking_mode=False,
        tokenize=True,
        return_dict=True,
    )

    steps = example["sequence_json"].replace("\'", "\"")
    if isinstance(steps, str):
        steps = json.loads(steps)
    steps = [str(s) + "\n" for s in steps]

    tokenized_steps = tokenizer([str(s) for s in steps])

    user_end_idx = len(tokenized_prompt.input_ids) - 1 - next(
        i
        for i, ids in enumerate(tokenized_prompt.input_ids[::-1])
        if ids == tokenizer.eos_token_id
    )

    prompt_input_ids = tokenized_prompt.input_ids[:user_end_idx]
    prompt_input_ids += [tokenizer.pad_token_id] * (segment_size - len(prompt_input_ids))

    tokenized_steps.input_ids[-1] += tokenized_prompt.input_ids[user_end_idx:]
    current_step_segment_ids = []
    num_segments = 1
    for step_ids in tokenized_steps.input_ids:
        if len(current_step_segment_ids) + len(step_ids) < segment_size:
            current_step_segment_ids += step_ids
        else:
            current_step_segment_ids += [tokenizer.pad_token_id] * (segment_size - len(current_step_segment_ids))
            prompt_input_ids += current_step_segment_ids
            current_step_segment_ids = step_ids
            num_segments += 1

    if current_step_segment_ids:
        prompt_input_ids += current_step_segment_ids
        num_segments += 1

    completion_input_ids = tokenizer(json.dumps({"answer": example["answer"]})).input_ids + [tokenizer.eos_token_id]
    num_segments += 1

    return {
        "input_ids": prompt_input_ids + completion_input_ids,
        "completion_mask": [0] * len(prompt_input_ids) + [1] * len(completion_input_ids),
        "num_segments": num_segments,
        "length": len(prompt_input_ids) + len(completion_input_ids),
    }


if __name__ == "__main__":
    parser = TrlParser((SFTConfig, ModelConfig, DatasetArgs, RMTArgs))
    training_args, model_args, ds_args, rmt_args = parser.parse_args_and_config()

    if training_args.run_name:
        os.environ["CLEARML_TASK"] = training_args.run_name

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = RMTQwen3Config.from_pretrained(
        model_args.model_name_or_path,
        segment_size=rmt_args.segment_size,
        max_n_segments=rmt_args.max_n_segments,
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
        raw_train_ds = raw_train_ds.shuffle(seed=42).select(
            range(int(len(raw_train_ds) * ds_args.subsample_train))
        )
    raw_val_ds = raw_ds[ds_args.val_split]

    processed_train_ds = raw_train_ds.map(
        lambda ex: preprocess_function(ex, ds_args, tokenizer, rmt_args.segment_size),
        remove_columns=raw_train_ds.column_names,
    )
    processed_val_ds = raw_val_ds.map(
        lambda ex: preprocess_function(ex, ds_args, tokenizer, rmt_args.segment_size),
        remove_columns=raw_val_ds.column_names,
    )

    if rmt_args.max_n_segments is not None:
        processed_train_ds = processed_train_ds.filter(lambda x: x["num_segments"] <= rmt_args.max_n_segments)
        processed_val_ds = processed_val_ds.filter(lambda x: x["num_segments"] <= rmt_args.max_n_segments)
    
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
        model.print_trainable_parameters()

    training_args.remove_unused_columns = False

    data_collator = DataCollatorForLanguageModeling(tokenizer.pad_token_id)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_ds,
        eval_dataset=processed_val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model()
    logger.info("Training complete. Running evaluation.")
    metrics = trainer.evaluate()
    logger.info(metrics)
