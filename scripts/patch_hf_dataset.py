#!/usr/bin/env python
"""Patch the published dondosss/mmred_mera dataset in place.

Two fixes, preserving all sequences and gold answers:
1. DC-WHS-C (who_spend): remove "в одиночестве" from question texts.
   Golds were computed as *total* time spent in the room, so the old
   wording ("time alone in the room") contradicted them. The corrected
   wording matches the golds; no gold changes are needed.
2. All configs: replace the legacy instruction with the 10 SAP-formatted
   prompts (semantic blocks per MERA docs/dataset_formatting.md),
   assigned cyclically per sample.

Usage:
    python scripts/patch_hf_dataset.py --save-dir /tmp/mmred_mera_patched
    python scripts/patch_hf_dataset.py --save-dir /tmp/mmred_mera_patched --push
"""

import argparse
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.upload_hf_dataset import PROMPTS

REPO_ID = "dondosss/mmred_mera"

TASK_TYPES = ["dc_sa_c", "dc_sr_i", "dc_cc_i", "dc_ws_r", "dc_whs_c"]
SEQ_LENS = [32, 64, 128]
CONFIGS = [f"mmred_{task}_{sl}" for task in TASK_TYPES for sl in SEQ_LENS]

ALONE_FRAGMENT = " в одиночестве"


def patch_split(split: datasets.Dataset, is_whs: bool) -> datasets.Dataset:
    """Return a copy of the split with patched instructions and questions."""

    def _patch(sample: dict, idx: int) -> dict:
        sample["instruction"] = PROMPTS[idx % len(PROMPTS)]
        if is_whs:
            sample["inputs"]["question"] = sample["inputs"]["question"].replace(
                ALONE_FRAGMENT, ""
            )
        return sample

    return split.map(_patch, with_indices=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", required=True, help="Local dir for patched configs.")
    parser.add_argument("--push", action="store_true", help="Push patched configs to the Hub.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        is_whs = "dc_whs_c" in config
        dd = datasets.load_dataset(REPO_ID, config)
        patched = datasets.DatasetDict(
            {name: patch_split(split, is_whs) for name, split in dd.items()}
        )
        patched.save_to_disk(str(save_dir / config))
        print(f"  {config}: patched (whs_fix={is_whs}), splits={list(patched.keys())}")

        if args.push:
            patched.push_to_hub(REPO_ID, config_name=config, private=True)
            print(f"  {config}: pushed to {REPO_ID}")


if __name__ == "__main__":
    main()
