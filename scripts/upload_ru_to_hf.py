#!/usr/bin/env python3
"""Upload Russian MMReD dataset to HuggingFace Hub."""

import json
import os
from pathlib import Path

import datasets
from tqdm import tqdm

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "dondosss/mmred_mera"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mera_mmred_ru"
META_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mmred"

TASK_TYPES = ["dc_sa_c", "dc_sr_i", "dc_cc_i", "dc_ws_r", "dc_whs_c"]
SEQ_LENS = [32, 64, 128]
SUBSETS = [f"mmred_{task}_{sl}" for task in TASK_TYPES for sl in SEQ_LENS]

# Load prompts from meta
meta_path = META_DIR / "raw_dataset_meta.json"
if meta_path.exists():
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    prompts = meta.get("prompts", [])
else:
    prompts = []

# Features matching the current HF schema
features = datasets.Features({
    "instruction": datasets.Value("string"),
    "inputs": {
        "context": datasets.Value("string"),
        "question": datasets.Value("string"),
    },
    "outputs": datasets.Value("string"),
    "meta": {
        "id": datasets.Value("int32"),
        "categories": {
            "task_type": datasets.Value("string"),
            "seq_len": datasets.Value("int32"),
            "atype": datasets.Value("string"),
        },
    },
})


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def convert_card(card):
    """Convert from local format to HF format."""
    # Handle instruction — if it's an index, replace with prompt
    instruction = card.get("instruction", "")
    if isinstance(instruction, int) and instruction < len(prompts):
        instruction = prompts[instruction]

    # Build meta with categories sub-dict
    local_meta = card.get("meta", {})
    categories = local_meta.get("categories", {})
    if not categories:
        # Convert flat meta to categories format
        categories = {
            "task_type": local_meta.get("task", "unknown"),
            "seq_len": local_meta.get("seq_len", 0),
            "atype": local_meta.get("atype", "person"),
        }

    return {
        "instruction": instruction,
        "inputs": card.get("inputs", {}),
        "outputs": card.get("outputs", ""),
        "meta": {
            "id": local_meta.get("id", 0),
            "categories": categories,
        },
    }


all_dicts = {}
for subset in tqdm(SUBSETS, desc="Processing subsets"):
    subset_dir = DATA_DIR / subset
    test_path = subset_dir / "test.json"
    shots_path = subset_dir / "shots.json"

    if not test_path.exists():
        print(f"WARNING: {test_path} not found — skipping {subset}")
        continue

    test_raw = load_json(test_path)["data"]
    test_cards = [convert_card(c) for c in test_raw]

    shots_cards = []
    if shots_path.exists():
        shots_raw = load_json(shots_path)["data"]
        shots_cards = [convert_card(c) for c in shots_raw]

    # Build datasets
    test_ds = datasets.Dataset.from_list(test_cards, features=features)
    splits = {"test": test_ds}
    if shots_cards:
        shots_ds = datasets.Dataset.from_list(shots_cards, features=features)
        splits["shots"] = shots_ds

    all_dicts[subset] = datasets.DatasetDict(splits)
    print(f"  {subset}: test={len(test_ds)}, shots={len(shots_cards)}")

print(f"\nReady to upload {len(all_dicts)} subsets to {REPO_ID}")

for subset, dd in tqdm(all_dicts.items(), desc="Uploading"):
    dd.push_to_hub(REPO_ID, config_name=subset, token=HF_TOKEN)
    print(f"  Uploaded: {subset}")

print("\nDone! All subsets uploaded.")
