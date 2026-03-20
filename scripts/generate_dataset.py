#!/usr/bin/env python
"""Generate MMReD benchmark dataset.

This script generates the MMReD benchmark dataset with configurable parameters
for sequence lengths, question types, and output format.

Example usage:
    # Generate with default settings
    python scripts/generate_dataset.py --output_path data/dataset.json

    # Generate with custom settings
    python scripts/generate_dataset.py \\
        --output_path data/custom_dataset.json \\
        --seq_lengths 16 32 64 128 \\
        --n_questions 100 \\
        --question_types spend_alone where_spend \\
        --seed 12345

    Currently used for my new question:
    python generate_dataset.py --seq_lengths 16 --n_questions 1 --question_types spend_alone_at_step --seed 12345 --output_path /workspace-SR004.nfs2/acherepanov/mmred_project/mmred/data/dataset.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmred.config import GenerationConfig, DEFAULT_SEQ_LENGTHS
from mmred.qgen.qgen import generate_questions, generate_questions_sequential, save_dataset
from mmred.qgen.questions import QUESTIONS


def _distribute_remainder_across_types(remainder: int, other_types: list[str]) -> dict[str, int]:
    """Split `remainder` evenly; first (remainder % m) types get one extra question."""
    m = len(other_types)
    if m == 0:
        if remainder != 0:
            raise ValueError("No non-target question types to distribute remainder into")
        return {}
    base = remainder // m
    extra = remainder % m
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(other_types)}


def _split_output_path(base_path: Path, k: int, split_total: int) -> Path:
    width = max(2, len(str(split_total)))
    suffix = base_path.suffix if base_path.suffix else ".json"
    return base_path.parent / f"{base_path.stem}_split_{k:0{width}d}{suffix}"


def _build_n_questions_per_type_for_split(
    question_types: list[str],
    target: str,
    k: int,
    split_total: int,
) -> dict[str, int]:
    others = [t for t in question_types if t != target]
    per_type: dict[str, int] = {target: k}
    per_type.update(_distribute_remainder_across_types(split_total - k, others))
    return {qt: per_type[qt] for qt in question_types}


def main():
    parser = argparse.ArgumentParser(
        description="Generate MMReD benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available question types:
  NIAH (Needle-in-a-Haystack):
    first_app, final_app, char_on_char_first_app, char_on_char_final_app,
    char_at_frame, first_at_room, last_at_room, room_on_char_first_app,
    room_on_char_final_app, room_at_frame, char_on_char_at_frame,
    n_room_on_char_first_app, n_room_on_char_final_app, n_char_at_frame, n_empty

  MMLong (Aggregation over range):
    room_empty, where_spend, crowded_room, who_spend, spend_alone,
    spend_together, steps_in_room, rooms_visited, crowd_count
""",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the JSON dataset file.",
    )
    
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=None,
        help=f"Sequence lengths to generate. Default: {DEFAULT_SEQ_LENGTHS}",
    )
    
    parser.add_argument(
        "--n_questions",
        type=int,
        default=50,
        help="Number of questions per question type per sequence length. Default: 50",
    )
    
    parser.add_argument(
        "--question_types",
        type=str,
        nargs="+",
        default=None,
        help="Question types to include. Default: all",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0xBADFACE,
        help="Random seed for reproducibility. Default: 0xBADFACE",
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential generation (no parallelization). Useful for debugging.",
    )
    
    parser.add_argument(
        "--list_question_types",
        action="store_true",
        help="List all available question types and exit.",
    )

    parser.add_argument(
        "--question_split",
        action="store_true",
        help=(
            "Emit one dataset per k in 0..split_total: k questions of --target_question_type "
            "and (split_total - k) spread across other --question_types."
        ),
    )
    parser.add_argument(
        "--target_question_type",
        type=str,
        default=None,
        help="Target question type for --question_split (must be in --question_types).",
    )
    parser.add_argument(
        "--split_total",
        type=int,
        default=None,
        help=(
            "Total questions per split output file. Default: the single --seq_lengths value "
            "when exactly one length is given; otherwise required."
        ),
    )

    args = parser.parse_args()
    
    # Handle --list_question_types
    if args.list_question_types:
        print("Available question types:")
        for qtype in sorted(QUESTIONS.keys()):
            print(f"  - {qtype}")
        return
    
    # Validate question types if provided
    if args.question_types:
        unknown = set(args.question_types) - set(QUESTIONS.keys())
        if unknown:
            print(f"Error: Unknown question types: {unknown}")
            print(f"Use --list_question_types to see available types.")
            sys.exit(1)

    if args.question_split:
        if not args.target_question_type:
            print("Error: --question_split requires --target_question_type.", file=sys.stderr)
            sys.exit(1)
        if not args.question_types:
            print(
                "Error: --question_split requires explicit --question_types "
                "(target + at least one other type).",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.target_question_type not in args.question_types:
            print(
                f"Error: --target_question_type {args.target_question_type!r} "
                "must appear in --question_types.",
                file=sys.stderr,
            )
            sys.exit(1)
        others = [t for t in args.question_types if t != args.target_question_type]
        if not others:
            print(
                "Error: --question_split needs at least one non-target type in --question_types.",
                file=sys.stderr,
            )
            sys.exit(1)
        seq_lens = args.seq_lengths or DEFAULT_SEQ_LENGTHS
        if len(seq_lens) != 1:
            print(
                "Error: --question_split requires exactly one value in --seq_lengths "
                f"(got {seq_lens}).",
                file=sys.stderr,
            )
            sys.exit(1)
        split_total = args.split_total if args.split_total is not None else seq_lens[0]
        if split_total < 0:
            print("Error: --split_total must be >= 0.", file=sys.stderr)
            sys.exit(1)

        base_out = Path(args.output_path)
        base_out.parent.mkdir(parents=True, exist_ok=True)

        print("Question-split mode:")
        print(f"  Seed: {args.seed}")
        print(f"  Sequence lengths: {seq_lens}")
        print(f"  split_total: {split_total}")
        print(f"  Target type: {args.target_question_type}")
        print(f"  All types: {args.question_types}")
        print(f"  Output pattern: {_split_output_path(base_out, 0, split_total)} ... "
              f"{_split_output_path(base_out, split_total, split_total)}")
        print()

        for k in range(split_total + 1):
            n_per_type = _build_n_questions_per_type_for_split(
                list(args.question_types),
                args.target_question_type,
                k,
                split_total,
            )
            config = GenerationConfig(
                seed=args.seed,
                seq_lengths=seq_lens,
                n_questions=1,
                question_types=list(args.question_types),
                n_questions_per_type=n_per_type,
            )
            split_path = _split_output_path(base_out, k, split_total)
            print(f"--- split k={k} -> {split_path.name} counts={n_per_type}")
            if args.sequential:
                samples = generate_questions_sequential(config)
            else:
                samples = generate_questions(config)
            save_dataset(samples, split_path)
            print(f"    Total samples: {len(samples)}")

        print()
        print("Question-split complete.")
        return

    # Create configuration
    config = GenerationConfig(
        seed=args.seed,
        seq_lengths=args.seq_lengths or DEFAULT_SEQ_LENGTHS,
        n_questions=args.n_questions,
        question_types=args.question_types,
    )

    print(f"Generating dataset with configuration:")
    print(f"  Seed: {config.seed}")
    print(f"  Sequence lengths: {config.seq_lengths}")
    print(f"  Questions per type per length: {config.n_questions}")
    print(f"  Question types: {config.question_types or 'all'}")
    print()

    # Generate dataset
    if args.sequential:
        print("Using sequential generation...")
        samples = generate_questions_sequential(config)
    else:
        print("Using parallel generation...")
        samples = generate_questions(config)

    # Save dataset
    save_dataset(samples, args.output_path)

    # Print summary
    print()
    print("Summary:")
    print(f"  Total samples: {len(samples)}")

    # Count by seq_len
    by_seq_len = {}
    for s in samples:
        sl = s["seq_len"]
        by_seq_len[sl] = by_seq_len.get(sl, 0) + 1
    for sl in sorted(by_seq_len.keys()):
        print(f"  seq_len={sl}: {by_seq_len[sl]} samples")


if __name__ == "__main__":
    main()
