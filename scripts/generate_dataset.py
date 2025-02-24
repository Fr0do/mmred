import argparse
from pathlib import Path

from qgen.const import SEQ_LENGTHS
from qgen.qgen import generate_questions, generate_videos


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
    args = parser.parse_args()

    create_exp_structure(args.base_path, args.exp_name)
    generate_questions(args.base_path, args.exp_name)
    generate_videos(args.base_path, args.exp_name)


if __name__ == "__main__":
    main()
