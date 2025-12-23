import argparse
import torch
import json
import pandas as pd
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from accelerate import Accelerator

from modeling_rmt import RMTQwen3Config, RMTQwen3ForCausalLM

# --- 1. Data Classes & Arguments ---

@dataclass
class InferenceArgs:
    output_csv: str
    batch_size: int
    max_samples: Optional[int]

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
    model_name_or_path: str
    attn_implementation: str = "flash_attention_2"

def load_arguments() -> Tuple[InferenceArgs, DatasetArgs, ModelArgs]:
    # Initialize defaults from the Dataclass to keep Single Source of Truth
    defaults = DatasetArgs()

    parser = argparse.ArgumentParser(description="Run greedy exact-match benchmark with online CSV writing.")
    parser.add_argument("--output_csv", default="data/main_1mv/rmt_forward.csv", help="Path to save CSV.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", required=True, help="Model checkpoint.")
    
    # Dataset Args (Now optional with new defaults)
    parser.add_argument("--dataset_name", default=defaults.dataset_name)
    parser.add_argument("--subset", default=defaults.subset)
    parser.add_argument("--split", default=defaults.split)
    parser.add_argument("--system_prompt", default=defaults.system_prompt)
    parser.add_argument("--task_prompt", default=defaults.task_prompt)
    
    args = parser.parse_args()
    
    return (
        InferenceArgs(args.output_csv, args.batch_size, args.max_samples),
        DatasetArgs(args.dataset_name, args.subset, args.split, args.system_prompt, args.task_prompt),
        ModelArgs(args.model_name_or_path)
    )

# --- 2. CSV Helper ---

def write_csv_batch(file_path, data_list, mode='a'):
    """
    Writes a list of dictionaries to CSV batch by batch.
    """
    df = pd.DataFrame(data_list)
    # If file doesn't exist, write header. If it does, skip header.
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode=mode, header=header, index=False)

# --- 3. Preprocessing Logic ---

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
        "row_idx": example["row_idx"] # Pass index for lookup
    }


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_value=0, label_pad_value=-100):
    """
    Args:
        batch: List of dictionaries provided by the dataset.
        pad_value: Padding value for input_ids (usually tokenizer.pad_token_id).
        label_pad_value: Padding value for labels (usually -100 for CrossEntropyLoss).
    """
    
    # 1. Extract and convert to tensors (if they aren't already)
    # We use clone().detach() if they are tensors, or torch.tensor() if they are lists
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    attention_mask = [torch.tensor(b['attention_mask']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    row_idx = [b['row_idx'] for b in batch]

    # 2. Pad sequences
    # batch_first=True results in shape (Batch_Size, Max_Length)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_value)
    
    # Attention masks are usually padded with 0 (indicating ignored positions)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    # Labels are often padded with -100 so PyTorch's CrossEntropyLoss ignores them
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=label_pad_value)

    # 3. Stack simple fields
    row_idx_tensor = torch.tensor(row_idx)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "row_idx": row_idx_tensor
    }

# --- 4. Main Logic ---

def main():
    inf_args, ds_args, model_args = load_arguments()
    accelerator = Accelerator()
    
    # Clear existing CSV if starting fresh (Optional, currently appends)
    if accelerator.is_main_process and not os.path.exists(os.path.dirname(inf_args.output_csv)):
        os.makedirs(os.path.dirname(inf_args.output_csv), exist_ok=True)

    # --- Load Model ---
    if accelerator.is_main_process:
        print(f"Loading model: {model_args.model_name_or_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = RMTQwen3Config.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
    )
    
    if accelerator.num_processes > 1:
        device = torch.device(f"cuda:{accelerator.local_process_index}")
        device_map = f"cuda:{accelerator.local_process_index}"
    else:
        device = torch.device("cuda:0")
        device_map = device
    
    model = RMTQwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=model_args.attn_implementation,
        local_files_only=True,
    )
    model.eval()
    

    # --- Load & Prep Dataset ---
    if accelerator.is_main_process:
        print(f"Loading dataset: {ds_args.dataset_name} ({ds_args.split})...")
        
    raw_dataset = load_dataset(ds_args.dataset_name, ds_args.subset)[ds_args.split]
    
    if inf_args.max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), inf_args.max_samples)))
    
    # Add Index
    raw_dataset = raw_dataset.add_column("row_idx", range(len(raw_dataset)))

    with accelerator.main_process_first():
        processed_dataset = raw_dataset.map(
            lambda ex: preprocess_for_train(ex, ds_args, tokenizer, config.segment_size),
            desc="Tokenizing",
            num_proc=8,
            remove_columns=[c for c in raw_dataset.column_names if c != "row_idx"]
        )

    dataloader = DataLoader(
        processed_dataset, 
        batch_size=inf_args.batch_size,
        collate_fn=lambda x: collate_fn(x, pad_value=tokenizer.pad_token_id),
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4,
    )
    
    model, dataloader = accelerator.prepare(model, dataloader)

    # --- Inference Loop ---
    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    
    # For quick metadata lookup on main process
    meta_lookup = None
    if accelerator.is_main_process:
        # Convert to pandas for easy indexing
        meta_lookup = raw_dataset.to_pandas().set_index("row_idx")

    for batch in dataloader:
        with torch.inference_mode():
            outputs = model(**batch)
        logits = outputs.logits
        
        # Shift Logic
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        
        # Greedy Argmax
        pred_tokens_ids = torch.argmax(shift_logits, dim=-1)
        
        # pred_tokens_ids = accelerator.pad_across_processes(
        #     pred_tokens_ids, 
        #     dim=1, 
        #     pad_index=tokenizer.pad_token_id
        # )
        
        del logits, shift_logits, outputs
        
        # Mask (ignore -100)
        mask = shift_labels != -100
        
        # Calculate Hits
        correct_tokens = (pred_tokens_ids == shift_labels) & mask
        has_valid_tokens = mask.sum(dim=1) > 0
        seq_correct = (correct_tokens == mask).all(dim=1) & has_valid_tokens
        seq_lens = mask.sum(dim=1)

        # --- Gather Basic Metrics ---
        gathered_hits = accelerator.gather_for_metrics(seq_correct)
        gathered_lens = accelerator.gather_for_metrics(seq_lens)
        gathered_idxs = accelerator.gather_for_metrics(batch["row_idx"])
        
        # --- Decode Predicted Strings ---
        # Mask out prompt predictions with padding for cleaner decoding
        safe_preds = pred_tokens_ids.clone()
        safe_preds[~mask] = tokenizer.pad_token_id 
        
        gathered_preds = accelerator.gather_for_metrics(safe_preds)
        
        if accelerator.is_main_process:
            # Decode the batch
            decoded_preds = tokenizer.batch_decode(gathered_preds, skip_special_tokens=True)
            
            hits = gathered_hits.cpu().tolist()
            lens = gathered_lens.cpu().tolist()
            idxs = gathered_idxs.cpu().tolist()
            
            batch_results = []
            
            for h, l, idx, pred_str in zip(hits, lens, idxs, decoded_preds):
                # Fetch metadata
                meta_row = meta_lookup.loc[idx]
                
                row = {
                    "row_idx": idx,
                    "qid": meta_row.get("qid", None),
                    "qtype": meta_row.get("qtype", None),
                    "atype": meta_row.get("atype", None),
                    "question": meta_row.get("question", ""),
                    "answer_gt": meta_row.get("answer", meta_row.get("sequence_json", "")),
                    "Predicted_Answer": pred_str.strip(),
                    "seq_len": int(l),
                    "hit": 1 if h else 0,
                }
                batch_results.append(row)
            
            # Write to CSV immediately
            write_csv_batch(inf_args.output_csv, batch_results)
        
        progress_bar.update(1)
        
        del pred_tokens_ids, mask, correct_tokens, shift_labels, safe_preds
        # Only needed if you are strictly hitting OOM limits; otherwise slows down slightly
        # torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print(f"\nDone. Results saved to {inf_args.output_csv}")

if __name__ == "__main__":
    main()