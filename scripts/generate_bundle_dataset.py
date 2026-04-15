#!/usr/bin/env python
"""Generate episode-bundle JSON for sequence-bundle benchmarking.

Each episode uses one shared sequence of length ``seq_len``. Every episode has
``bundle_size`` questions (default ``bundle_size == seq_len``): ``k_target``
questions of type ``spend_alone_at_step`` at steps 1..k_target, plus filler
questions of other types on the same sequence.

Filler types must have fixed-sequence generators in ``mmred.qgen.bundles``:
``crowded_room``, ``room_empty``.

Example::

    python scripts/generate_bundle_dataset.py \\
        --output_path data/bundles/seq16_k8.json \\
        --seq_len 16 \\
        --k_target 8 \\
        --n_episodes 1200 \\
        --target_question_type spend_alone_at_step \\
        --question_types spend_alone_at_step crowded_room \\
        --seed 12345
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmred.qgen.bundles import generate_bundle_dataset
from mmred.qgen.questions import QUESTIONS


def main() -> None:
    p = argparse.ArgumentParser(description="Generate MMReD episode-bundle JSON.")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--k_target", type=int, required=True, help="Target questions at steps 1..k_target.")
    p.add_argument("--n_episodes", type=int, default=1200)
    p.add_argument("--target_question_type", type=str, default="spend_alone_at_step")
    p.add_argument(
        "--question_types",
        type=str,
        nargs="+",
        required=True,
        help="Must include target and at least one filler (e.g. crowded_room).",
    )
    p.add_argument(
        "--bundle_size",
        type=int,
        default=None,
        help=(
            "Total questions per episode. Default: same as --seq_len (fixed-L bundles). "
            "If set, must be >= k_target; k_target must be <= seq_len for spend_alone_at_step."
        ),
    )
    p.add_argument("--seed", type=int, default=0xBADFACE)
    args = p.parse_args()

    unknown = set(args.question_types) - set(QUESTIONS.keys())
    if unknown:
        print(f"Error: unknown question types: {unknown}", file=sys.stderr)
        sys.exit(1)
    if args.target_question_type not in args.question_types:
        print("Error: target_question_type must appear in --question_types", file=sys.stderr)
        sys.exit(1)
    if len(set(args.question_types)) < 2:
        print("Error: need at least one filler question type besides target", file=sys.stderr)
        sys.exit(1)

    bundle_size = args.bundle_size if args.bundle_size is not None else args.seq_len
    if bundle_size < args.k_target:
        print("Error: bundle_size must be >= k_target", file=sys.stderr)
        sys.exit(1)

    samples = generate_bundle_dataset(
        n_episodes=args.n_episodes,
        seq_len=args.seq_len,
        k_target=args.k_target,
        bundle_size=bundle_size,
        target_question_type=args.target_question_type,
        question_types=list(args.question_types),
        seed=args.seed,
    )

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Wrote {len(samples)} rows ({args.n_episodes} episodes) -> {out}")


if __name__ == "__main__":
    main()
