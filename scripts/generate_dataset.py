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
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmred.config import GenerationConfig, DEFAULT_SEQ_LENGTHS
from mmred.qgen.qgen import generate_questions, generate_questions_sequential, save_dataset
from mmred.qgen.questions import QUESTIONS


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
