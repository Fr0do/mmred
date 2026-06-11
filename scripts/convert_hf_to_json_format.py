#!/usr/bin/env python
"""Migrate the published dondosss/mmred_mera dataset to the JSONL context format.

1. test split: deterministically convert each context from the legacy
   pseudo-text form ("Шаг 1: Кухня: [Сандра]; ...") to JSONL — one
   {"step_id": N, "rooms": {room: [chars]}} object per line, matching the
   library's serialize_sequence schema (the paper's original format).
   Questions, golds, and meta are unchanged; instructions are re-assigned
   cyclically from the JSON-aware SAP PROMPTS.
2. shots split: regenerated from scratch at seq_len 8/16 (the published
   shots used 1-2 steps, which makes aggregation questions degenerate),
   with the corrected question wording and JSONL contexts.

Usage:
    python scripts/convert_hf_to_json_format.py --save-dir /tmp/mmred_json
    python scripts/convert_hf_to_json_format.py --save-dir /tmp/mmred_json --push
"""

import argparse
import json
import re
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.upload_hf_dataset import PROMPTS, SHOTS_ID_OFFSET
from scripts.generate_mera_dataset import generate_mera_dataset, QTYPE_TO_TASK

REPO_ID = "dondosss/mmred_mera"
TASK_TYPES = ["dc_sa_c", "dc_sr_i", "dc_cc_i", "dc_ws_r", "dc_whs_c"]
SEQ_LENS = [32, 64, 128]
CONFIGS = [f"mmred_{task}_{sl}" for task in TASK_TYPES for sl in SEQ_LENS]

SHOT_LENS = [8, 16]
N_SHOTS = 5  # 3 from len-8 + 2 from len-16


def text_context_to_jsonl(ctx: str) -> str:
    """Parse the legacy text context and re-emit it as JSONL steps."""
    lines = []
    for line in ctx.strip().split("\n"):
        m = re.match(r"(?:Step|Шаг) (\d+): (.*)", line)
        if not m:
            raise ValueError(f"unparseable context line: {line!r}")
        rooms = {}
        for part in m.group(2).split("; "):
            room, occ = part.split(": ", 1)
            occ = occ.strip()
            if not (occ.startswith("[") and occ.endswith("]")):
                raise ValueError(f"unparseable occupants: {part!r}")
            inner = occ[1:-1].strip()
            rooms[room] = [o.strip() for o in inner.split(",")] if inner else []
        lines.append(
            json.dumps(
                {"step_id": int(m.group(1)), "rooms": rooms}, ensure_ascii=False
            )
        )
    return "\n".join(lines)


def jsonl_roundtrip_equal(text_ctx: str, jsonl_ctx: str) -> bool:
    """Verify the JSONL context encodes exactly the same states as the text."""
    steps = [json.loads(l) for l in jsonl_ctx.split("\n")]
    rebuilt = text_context_to_jsonl(text_ctx)
    return rebuilt == jsonl_ctx and all(
        s["step_id"] == i + 1 for i, s in enumerate(steps)
    )


def convert_test_split(split: datasets.Dataset) -> datasets.Dataset:
    def _convert(sample: dict, idx: int) -> dict:
        jsonl = text_context_to_jsonl(sample["inputs"]["context"])
        assert jsonl_roundtrip_equal(sample["inputs"]["context"], jsonl)
        sample["inputs"]["context"] = jsonl
        sample["instruction"] = PROMPTS[idx % len(PROMPTS)]
        return sample

    return split.map(_convert, with_indices=True)


def build_new_shots(work_dir: Path) -> dict[str, list[dict]]:
    """Generate fresh shots at SHOT_LENS; returns task_code -> 5 shot records."""
    task_datasets = generate_mera_dataset(
        output_dir=work_dir,
        mode="text",
        language="ru",
        seq_lengths=SHOT_LENS,
        n_questions=3,
        seed=0xBADFACE + 777,  # disjoint from the test-set seed
        n_few_shot=0,
        render_images=False,
    )

    shots_by_code: dict[str, list[dict]] = {}
    for name, samples in task_datasets.items():
        # name like mmred_dc_sa_c_8 -> code DC-SA-C, length 8
        slug, sl = name.removeprefix("mmred_").rsplit("_", 1)
        code = slug.upper().replace("_", "-")
        per_len = 3 if int(sl) == SHOT_LENS[0] else 2
        shots_by_code.setdefault(code, [])
        for s in samples[:per_len]:
            i = len(shots_by_code[code])
            shots_by_code[code].append(
                {
                    "instruction": PROMPTS[i % len(PROMPTS)],
                    "inputs": s["inputs"],
                    "outputs": s["outputs"],
                    "meta": {
                        "id": SHOTS_ID_OFFSET + i + 1,
                        "categories": {
                            "task_type": code,
                            "seq_len": int(sl),
                            "atype": s["meta"]["atype"],
                        },
                    },
                }
            )
    for code, shots in shots_by_code.items():
        assert len(shots) == N_SHOTS, f"{code}: {len(shots)} shots"
    return shots_by_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--token-file", default="/home/jovyan/kurkin/hf_write.txt")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    token = Path(args.token_file).read_text().strip() if args.push else None

    print("Generating new shots (seq_len 8/16)...")
    shots_by_code = build_new_shots(save_dir / "_shots_work")

    features = None
    for config in CONFIGS:
        slug = config.removeprefix("mmred_").rsplit("_", 1)[0]
        code = slug.upper().replace("_", "-")
        dd = datasets.load_dataset(REPO_ID, config)
        features = dd["test"].features

        new_test = convert_test_split(dd["test"])
        new_shots = datasets.Dataset.from_list(shots_by_code[code], features=features)
        out = datasets.DatasetDict({"shots": new_shots, "test": new_test})
        out.save_to_disk(str(save_dir / config))
        print(f"  {config}: test={len(new_test)} converted, shots={len(new_shots)} regenerated")

        if args.push:
            out.push_to_hub(REPO_ID, config_name=config, token=token)
            print(f"  {config}: pushed")


if __name__ == "__main__":
    main()
