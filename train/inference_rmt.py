"""Run greedy inference with an RMT Qwen3 model.

This script mirrors the preprocessing logic used during training in
``train/train_rmt_qwen.py`` so that the test split is tokenised into
segments that are compatible with the recurrent memory transformer
architecture.  The resulting predictions are written to a CSV file that
matches the format produced by ``scripts/openai_server_inference.py`` –
all original dataset columns are preserved and an additional
``Predicted_Answer`` column is appended.

The script is designed to be launched with ``accelerate`` so that tensor
parallelism can be used when multiple devices are available.  Generation
is performed greedily (``do_sample=False``) as required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from modeling_rmt import RMTQwen3Config, RMTQwen3ForCausalLM
from pprint import pprint
from transformers import DataCollatorWithPadding

@dataclass
class DatasetArgs:
    dataset_name: str = "dataset/hf_main_1mv_train_full"
    subset: str = "default"
    split: str = "test"
    system_prompt: str | None = "You are a helpful AI Assistant."
    task_prompt: str | None = (
        'Format your final answer with a {"answer": <value>}, where <value> is:\n'
        "  - A **single room name** (e.g., 'Kitchen') for location answers.\n"
        "  - A **number** (e.g., '3') for counting answers.\n"
        "  - A **single person name** (e.g., 'Michael') for people answers or 'Nobody' "
        "if no person satisfies given conditions."
    )


@dataclass
class ModelArgs:
    model_name_or_path: str | None = None
    attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"
    local_files_only: bool = True


@dataclass
class InferenceArgs:
    config: str = "train/config_rmt.yaml"
    output_csv: str = "inference_outputs/rmt_qwen_predictions.csv"
    batch_size: int = 1
    max_new_tokens: int = 32
    max_samples: int | None = None


def _load_yaml_defaults(config_path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    dataset_keys = {field.name for field in dataclass_fields(DatasetArgs)}
    model_keys = {field.name for field in dataclass_fields(ModelArgs)}

    dataset_cfg = {k: raw_cfg[k] for k in dataset_keys & raw_cfg.keys()}
    model_cfg = {k: raw_cfg[k] for k in model_keys & raw_cfg.keys()}
    return {"dataset": dataset_cfg, "model": model_cfg}


def dataclass_fields(cls):
    return getattr(cls, "__dataclass_fields__").values()


def load_arguments() -> tuple[InferenceArgs, DatasetArgs, ModelArgs]:
    parser = argparse.ArgumentParser(description="Run greedy inference with an RMT Qwen3 model.")
    parser.add_argument("--config", default="train/config_rmt.yaml", help="Path to a YAML config file.")
    parser.add_argument("--output_csv", default="data/main_1mv/rmt.csv", help="Where to save the CSV with predictions.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", default=None, help="Path to the fine-tuned checkpoint.")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--task_prompt", default=None)

    cli_args = parser.parse_args()

    defaults = _load_yaml_defaults(cli_args.config)

    inference_args = InferenceArgs(
        config=cli_args.config,
        output_csv=cli_args.output_csv,
        batch_size=cli_args.batch_size,
        max_new_tokens=cli_args.max_new_tokens,
        max_samples=cli_args.max_samples,
    )

    dataset_args = DatasetArgs(**defaults["dataset"])
    if cli_args.dataset_name is not None:
        dataset_args.dataset_name = cli_args.dataset_name
    if cli_args.subset is not None:
        dataset_args.subset = cli_args.subset
    if cli_args.split is not None:
        dataset_args.split = cli_args.split
    if cli_args.system_prompt is not None:
        dataset_args.system_prompt = cli_args.system_prompt
    if cli_args.task_prompt is not None:
        dataset_args.task_prompt = cli_args.task_prompt

    model_args = ModelArgs(**defaults["model"])
    if cli_args.model_name_or_path is not None:
        model_args.model_name_or_path = cli_args.model_name_or_path
    if cli_args.attn_implementation is not None:
        model_args.attn_implementation = cli_args.attn_implementation
    if cli_args.dtype is not None:
        model_args.dtype = cli_args.dtype
    if cli_args.local_files_only:
        model_args.local_files_only = True

    if model_args.model_name_or_path is None:
        raise ValueError("--model_name_or_path must be provided or specified in the config file.")

    return inference_args, dataset_args, model_args


from typing import Dict, Any, List
import json

def preprocess_function_for_generation(
    example,
    ds_args: "DatasetArgs",
    tokenizer,
    segment_size: int,
) -> Dict[str, Any]:
    """
    Generation version (for benchmarking):

    Segments:
        [S0 (padded to segment_size), S1 (padded), ..., Sk (padded), S_final (assistant_prefix, NOT padded)]

    S0..Sk together contain:
        system + user (chat template, pre-assistant area) + steps (whole list broken into pieces)

    S_final contains:
        assistant_prefix (chat template), WITHOUT the answer JSON and WITHOUT eos.
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

    # 3) Locate assistant_prefix_tokens using dummy content trick
    baseline = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_dict=True
    )
    with_gen_and_dummy = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": "DUMMY"}],
        add_generation_prompt=True, tokenize=True, return_dict=True
    )

    base_ids = list(baseline.input_ids)          # up to end of user
    gen_dummy_ids = list(with_gen_and_dummy.input_ids)  # includes assistant prefix + "DUMMY"

    dummy_only = tokenizer("DUMMY", add_special_tokens=False).input_ids

    def find_subseq(haystack: List[int], needle: List[int]) -> int:
        if not needle:
            return -1
        L, N = len(haystack), len(needle)
        # start search at/after baseline length
        for i in range(max(0, len(base_ids)), L - N + 1):
            if haystack[i:i+N] == needle:
                return i
        return -1

    dummy_start = find_subseq(gen_dummy_ids, dummy_only)
    if dummy_start == -1:
        # Fallback: treat entire suffix as assistant prefix if dummy not found
        pre_assistant_tokens = base_ids
        assistant_prefix_tokens = gen_dummy_ids[len(base_ids):]
    else:
        pre_plus_prefix = gen_dummy_ids[:dummy_start]
        assert len(pre_plus_prefix) >= len(base_ids)
        pre_assistant_tokens = pre_plus_prefix[:len(base_ids)]
        assistant_prefix_tokens = pre_plus_prefix[len(base_ids):]

    # 4) Steps (same as in training version)
    steps = example["sequence_json"].replace("'", '"')
    if isinstance(steps, str):
        steps = json.loads(steps)
    step_texts = [str(s) + "\n" for s in steps]

    tokenized_steps = tokenizer(step_texts, add_special_tokens=False)

    # prompt_tokens_before_assistant = pre_assistant + steps
    prompt_tokens_before_assistant: List[int] = list(pre_assistant_tokens)
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

    current_seg: List[int] = []
    for tid in prompt_tokens_before_assistant:
        if len(current_seg) == segment_size:
            flush_segment(current_seg, segment_size, tokenizer.pad_token_id)
            current_seg = []
        current_seg.append(tid)

    if current_seg:
        flush_segment(current_seg, segment_size, tokenizer.pad_token_id)

    # 6) Final (un-padded) segment: assistant prefix ONLY (no answer, no eos)
    # completion_segment = [tokenizer.pad_token_id] * (segment_size - len(list(assistant_prefix_tokens))) + list(assistant_prefix_tokens)
    completion_segment = [tokenizer.pad_token_id] * (0) + list(assistant_prefix_tokens)
    input_ids.extend(completion_segment)
    num_segments += 1

    return {
        "input_ids": input_ids,
        "num_segments": num_segments,
        "prompt_length": len(input_ids),
        "attention_mask": [1 if i != tokenizer.pad_token_id or 1 else 0 for i in input_ids],
    }

def collate_fn(features, processor, metadata_columns):
    columns = list(features[0].keys())
    batch_features = [{col: f[col] for col in columns if col not in metadata_columns} for f in features]
    batch = processor(batch_features)
    metadata = {}
    for col in metadata_columns:
        metadata[col] = [f[col] for f in features] 
    return batch, metadata

def main():
    inference_args, dataset_args, model_args = load_arguments()

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=model_args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = RMTQwen3Config.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=model_args.local_files_only,
    )

    with accelerator.main_process_first():
        raw_dataset: Dataset = load_dataset(dataset_args.dataset_name, dataset_args.subset)[dataset_args.split]
        if inference_args.max_samples:
            raw_dataset = raw_dataset.select(range(min(inference_args.max_samples, len(raw_dataset))))

        processed_dataset = raw_dataset.map(
            lambda ex: preprocess_function_for_generation(ex, dataset_args, tokenizer, config.segment_size),
            desc="Tokenising dataset",
        )
    #    processed_dataset = processed_dataset.filter(lambda x: x["num_segments"] <= config.max_n_segments)

    metadata_columns = raw_dataset.column_names + ["num_segments", "prompt_length"]
    data_collator = DataCollatorWithPadding(tokenizer)

    dataloader = DataLoader(
        processed_dataset,
        batch_size=inference_args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, data_collator, metadata_columns),
    )

    dtype = getattr(torch, model_args.dtype) if isinstance(model_args.dtype, str) else model_args.dtype

    model = RMTQwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=dtype,
        attn_implementation=model_args.attn_implementation,
        local_files_only=model_args.local_files_only,
    )
    model.generation_config.max_new_tokens = inference_args.max_new_tokens
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model = accelerator.prepare(model)
    model.eval()

    csv_writer = None
    csv_file = None
    if accelerator.is_main_process:
        output_dir = os.path.dirname(inference_args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        csv_file = open(inference_args.output_csv, "w", encoding="utf-8", newline="")
        fieldnames = metadata_columns + ["Predicted_Answer"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    unwrapped = accelerator.unwrap_model(model)

    for batch, metadata in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        pprint(tokenizer.batch_decode(batch["input_ids"]))
        pprint(batch["input_ids"])
        model_inputs = {k: v.to(accelerator.device) for k, v in batch.items() if k != "prompt_length"}
        with torch.no_grad():
            generated = unwrapped.generate(
                **model_inputs,
                max_new_tokens=inference_args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        pprint(tokenizer.batch_decode(generated))
        pprint(generated)
        break

        prompt_lengths = prompt_lengths.tolist()
        records: List[Dict[str, object]] = []
        for idx in range(generated.size(0)):
            prompt_len = prompt_lengths[idx]
            output_tokens = generated[idx, prompt_len:]
            text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            row = {col: metadata[col][idx] for col in metadata_columns}
            row["Predicted_Answer"] = text
            records.append(row)

        try:
            gathered = accelerator.gather_object(records)
        except AttributeError:
            gathered = [records]
        if accelerator.is_main_process and gathered:
            for group in gathered:
                for record in group:
                    csv_writer.writerow(record)

    accelerator.wait_for_everyone()

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
