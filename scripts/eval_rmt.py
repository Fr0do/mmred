import logging
import json
import os
from dataclasses import dataclass, field
from functools import partial

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import PeftModel
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper
from trl import (
    TrlParser,
    ModelConfig,
)
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetArgs:
    dataset_name: str
    subset: str = field(default="default", metadata={"help": "Dataset subset."})
    split: str = field(default="val", metadata={"help": "Dataset split."})
    system_prompt: str | None = "You are a helpful AI assistant."
    task_prompt: str | None = "Answer with a single word or number."

@dataclass
class EvalArgs:
    output_dir: str = field(default="results", metadata={"help": "Output directory for results."})
    exp_name: str = field(default="rmt_eval", metadata={"help": "Experiment name."})
    batch_size: int = field(default=1, metadata={"help": "Evaluation batch size."})
    max_new_tokens: int = field(default=50, metadata={"help": "Max new tokens for generation."})

@dataclass
class RMTArgs:
    segment_size: int = field(metadata={"help": "Segment size for RMT."})
    num_mem_tokens: int = field(default=16, metadata={"help": "Number of memory tokens."})
    max_n_segments: int = field(default=10, metadata={"help": "Max number of segments for RMT."})
    k2: int = field(default=-1, metadata={"help": "BPTT unroll steps of RMT."})
    segment_alignment: str | None = field(default=None, metadata={"help": "Segment alignment of RMT."})
    sliding_window: bool = field(default=False, metadata={"help": "Sliding alignment of RMT."})


def get_prompt(example, ds_args: DatasetArgs):
    user_content_parts = []
    if ds_args.task_prompt:
        user_content_parts.append(ds_args.task_prompt)

    steps = example["sequence_json"].replace("'", '"')
    if isinstance(steps, str):
        steps = json.loads(steps)
    
    context = "".join([f"{s}\n" for s in steps])
    user_content_parts.append(context)
    user_content_parts.append(example["question"])
    user_content = "\n".join(user_content_parts)

    prompt = []
    if ds_args.system_prompt:
        prompt.append({"role": "system", "content": ds_args.system_prompt})
    prompt.append({"role": "user", "content": user_content})
    
    return prompt


def main():
    parser = TrlParser((ModelConfig, DatasetArgs, RMTArgs, EvalArgs))
    model_args, ds_args, rmt_args, eval_args = parser.parse_args_and_config()

    # --- Load Model and Tokenizer ---
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.dtype) if hasattr(model_args, 'dtype') and model_args.dtype else torch.bfloat16,
        attn_implementation=model_args.attn_implementation if hasattr(model_args, 'attn_implementation') else None,
        trust_remote_code=True,
        device_map="auto"
    )

    if model_args.use_peft:
        logger.info("Applying PEFT adapter.")
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=False)

    # --- Wrap model with RMT ---
    logger.info("Wrapping model with RecurrentWrapper for RMT.")
    cell = MemoryCell(model, num_mem_tokens=rmt_args.num_mem_tokens)
    rmt_kwargs = dict(
        segment_size=rmt_args.segment_size,
        max_n_segments=rmt_args.max_n_segments,
        segment_alignment=rmt_args.segment_alignment,
        k2=rmt_args.k2,
    )
    rmt_model = RecurrentWrapper(cell, **rmt_kwargs)
    rmt_model.eval()

    # --- Load Dataset ---
    logger.info(f"Loading dataset {ds_args.dataset_name} split {ds_args.split}")
    raw_ds = load_dataset(ds_args.dataset_name, ds_args.subset)
    eval_ds = raw_ds[ds_args.split]

    # --- Prepare for output ---
    model_format = model_args.model_name_or_path.strip("/").split("/")[-1]
    output_dir = os.path.join(eval_args.output_dir, eval_args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"qa_pairs_answers_{model_format}.csv")
    
    # --- Preprocess Dataset ---
    logger.info("Preprocessing dataset.")
    def preprocess(example):
        prompt = get_prompt(example, ds_args)
        prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(prompt_str, truncation=False)
        example['input_ids'] = tokenized.input_ids
        example['attention_mask'] = tokenized.attention_mask
        example['input_len'] = len(tokenized.input_ids)
        return example

    processed_eval_ds = eval_ds.map(preprocess)

    # --- Generation ---
    logger.info("Starting generation.")
    rmt_model.generation_config = GenerationConfig(
        max_new_tokens=eval_args.max_new_tokens,
        do_sample=False, # Greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=eval_args.batch_size,
        dataloader_drop_last=False,
        logging_steps=10,
        predict_with_generate=True,
        report_to=[],
    )

    trainer = Trainer(
        model=rmt_model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    predictions = trainer.predict(processed_eval_ds)

    # --- Process results ---
    results = []
    input_lengths = processed_eval_ds["input_len"]
    qids = processed_eval_ds["qid"]
    questions = processed_eval_ds["question"]
    answers = processed_eval_ds["answer"]

    for i, pred_ids in enumerate(predictions.predictions):
        in_len = input_lengths[i]
        generated_part = pred_ids[in_len:]
        predicted_answer = tokenizer.decode(generated_part, skip_special_tokens=True)

        results.append({
            "qid": qids[i],
            "question": questions[i],
            "answer": answers[i],
            "Predicted_Answer": predicted_answer.strip(),
        })

    # --- Save results ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()