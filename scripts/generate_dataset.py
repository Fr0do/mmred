import argparse
from pathlib import Path

from mmred.const import SEQ_LENGTHS
from mmred.in_context import generate_in_context_examples
from mmred.qgen.qgen import generate_questions
from mmred.vgen.vgen import generate_videos


def create_exp_structure(base_path, exp_name):
    # Create experiment folder
    exp_path = Path(base_path) / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    print(f"Created experiment folder: {exp_path}")

    # Create sub-folders for sequence lengths
    for length in SEQ_LENGTHS:
        length_folder = exp_path / f"len_{length}"
        (length_folder / "sequences").mkdir(parents=True, exist_ok=True)
        (length_folder / "videos").mkdir(parents=True, exist_ok=True)
        print(f"Created folders for len_{length}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment folder structure."
    )
    parser.add_argument(
        "--base_path", type=str, required=True, help="Base path for the experiment."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False,
        default="main",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--with_in_context",
        action="store_true",
        help="Generate a compact in-context dataset for few-shot prompts.",
    )
    parser.add_argument(
        "--in_context_examples",
        type=int,
        default=5,
        help="Number of examples per task for in-context dataset.",
    )
    args = parser.parse_args()
    
    
    if args.with_in_context:
        output_path = generate_in_context_examples(
            args.base_path, args.exp_name, n_examples_per_task=args.in_context_examples
        )
        print(f"Saved in-context dataset to {output_path}")
    else:
        create_exp_structure(args.base_path, args.exp_name)
        generate_questions(args.base_path, args.exp_name)
        generate_videos(args.base_path, args.exp_name)



if __name__ == "__main__":
    main()
