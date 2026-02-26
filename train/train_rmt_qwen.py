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

# torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

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
        print("patching [PAD] token with ", tokenizer.eos_token)
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
        attn_implementation="flash_attention_2",
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
    
    data_collator = DataCollatorForLanguageModeling(tokenizer.pad_token_id, padding_free=False, pad_to_multiple_of=rmt_args.max_n_segments * rmt_args.segment_size)

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

