#!/usr/bin/env python
"""Generate MMReD benchmark dataset in MERA-compatible format.

This script generates a dataset compatible with the MERA evaluation harness,
supporting both text (JSON) and image-based context representations.

Example usage:
    # Generate DC tasks for MERA leaderboard (lengths 32, 64, 128)
    python scripts/generate_mera_dataset.py \\
        --output_dir data/mera_mmred \\
        --mode image \\
        --language en

    # Generate Russian version
    python scripts/generate_mera_dataset.py \\
        --output_dir data/mera_mmred_ru \\
        --mode image \\
        --language ru
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmred.config import GenerationConfig
from mmred.qgen.qgen import generate_questions_sequential
from mmred.localization import Language
from mmred.in_context import generate_in_context_examples


# Task mapping from question types to MERA task codes
QTYPE_TO_TASK = {
    'first_app': 'FA-FA-R',
    'char_on_char_first_app': 'FA-CCFA-R',
    'first_at_room': 'FA-FR-C',
    'room_on_char_first_app': 'FA-RCFA-C',
    'n_room_on_char_first_app': 'FA-NRFA-I',
    'final_app': 'FI-FA-R',
    'char_on_char_final_app': 'FI-CCFA-R',
    'last_at_room': 'FI-LR-C',
    'room_on_char_final_app': 'FI-RCFA-C',
    'n_room_on_char_final_app': 'FI-NRFA-I',
    'char_at_frame': 'FX-CF-R',
    'room_at_frame': 'FX-RF-C',
    'char_on_char_at_frame': 'FX-CCF-C',
    'n_char_at_frame': 'FX-NCF-I',
    'n_empty': 'FX-NE-I',
    'room_empty': 'DC-RE-R',
    'where_spend': 'DC-WS-R',
    'crowded_room': 'DC-CR-R',
    'who_spend': 'DC-WHS-C',
    'spend_alone': 'DC-SA-C',
    'spend_together': 'DC-ST-C',
    'steps_in_room': 'DC-SR-I',
    'rooms_visited': 'DC-RV-I',
    'crowd_count': 'DC-CC-I'
}

# DC (Dense Context) tasks to keep for MERA leaderboard
DC_TASKS = ['spend_alone', 'steps_in_room', 'crowd_count', 'where_spend', 'who_spend']

# MERA leaderboard sequence lengths
MERA_SEQ_LENGTHS = [32, 64, 128]

# Few-shot example lengths (shorter contexts for demonstration)
FEW_SHOT_SEQ_LENGTHS = [1, 2, 4, 8, 16]


def format_sequence_as_text(sequence: list[dict], language: str = "en") -> str:
    """Format sequence data as text context.
    
    Args:
        sequence: List of step dictionaries with rooms and occupants
        language: Language for formatting ("en" or "ru")
        
    Returns:
        Formatted text representation of the sequence
    """
    lines = []
    for step in sequence:
        step_id = step["step_id"]
        room_parts = []
        for room, occupants in step["rooms"].items():
            if occupants:
                occupants_str = ", ".join(occupants)
                room_parts.append(f"{room}: [{occupants_str}]")
            else:
                room_parts.append(f"{room}: []")
        
        if language == "ru":
            lines.append(f"Шаг {step_id}: " + "; ".join(room_parts))
        else:
            lines.append(f"Step {step_id}: " + "; ".join(room_parts))
    
    return "\n".join(lines)


def get_instruction_prompt(language: str = "en", mode: str = "text") -> str:
    """Get the instruction prompt for the task.
    
    Args:
        language: Language for the prompt ("en" or "ru")
        mode: Context mode ("text" or "image")
        
    Returns:
        Instruction prompt string with {context} and {question} placeholders
    """
    if language == "ru":
        if mode == "image":
            return (
                "Вы анализируете последовательность изображений, показывающих расположение персонажей "
                "в разных комнатах на каждом шаге. Изучите изображения и ответьте на вопрос.\n\n"
                "{context}\n\n"
                "Вопрос: {question}\n\n"
                "Ответьте одним словом или числом."
            )
        else:
            return (
                "Вы анализируете последовательность состояний комнат, где указано, "
                "какие персонажи находятся в каких комнатах на каждом шаге.\n\n"
                "{context}\n\n"
                "Вопрос: {question}\n\n"
                "Ответьте одним словом или числом."
            )
    else:
        if mode == "image":
            return (
                "You are analyzing a sequence of images showing character positions "
                "in different rooms at each step. Study the images and answer the question.\n\n"
                "{context}\n\n"
                "Question: {question}\n\n"
                "Answer with a single word or number."
            )
        else:
            return (
                "You are analyzing a sequence of room occupancy states showing "
                "which characters are in which rooms at each step.\n\n"
                "{context}\n\n"
                "Question: {question}\n\n"
                "Answer with a single word or number."
            )


def convert_to_mera_format(
    sample: dict,
    prompt_template: str,
    mode: Literal["text", "image"] = "text",
    image_dir: Path | None = None,
    language: str = "en",
) -> dict:
    """Convert MMReD sample to MERA dataset format.
    
    Args:
        sample: MMReD sample dictionary
        prompt_template: Instruction template with placeholders
        mode: "text" for JSON context, "image" for image references
        image_dir: Directory containing rendered images (for image mode)
        
    Returns:
        MERA-formatted sample dictionary
    """
    qid = sample["qid"]
    seq_len = sample["seq_len"]
    qtype = sample["qtype"]
    task_code = QTYPE_TO_TASK.get(qtype, qtype)
    
    # Format context based on mode
    if mode == "image":
        # Reference images - actual image paths will be set during HuggingFace upload
        context = f"[See {seq_len} attached images showing room states]"
        # Image paths for doc_to_image
        image_paths = [f"{qid}/step_{i+1:04d}.png" for i in range(seq_len)]
    else:
        context = format_sequence_as_text(sample["sequence"], language)
        image_paths = None
    
    # Build MERA sample
    mera_sample = {
        "instruction": prompt_template,
        "inputs": {
            "context": context,
            "question": sample["question"],
        },
        "outputs": str(sample["answer"]),
        "meta": {
            "id": int(qid),
            "task": task_code,
            "qtype": qtype,
            "seq_len": seq_len,
            "atype": sample["atype"],
        }
    }
    
    # Add image paths for multimodal setup
    if image_paths:
        mera_sample["meta"]["images"] = image_paths
    
    return mera_sample


def generate_mera_dataset(
    output_dir: Path,
    mode: Literal["text", "image"] = "image",
    language: str = "en",
    question_types: list[str] | None = None,
    seq_lengths: list[int] | None = None,
    n_questions: int = 50,
    seed: int = 0xBADFACE,
    render_images: bool = False,
    n_few_shot: int = 5,
    few_shot_max_len: int = 16,
) -> dict[str, list[dict]]:
    """Generate MERA-formatted dataset.
    
    Args:
        output_dir: Output directory for dataset files
        mode: Context representation mode ("text" or "image")
        language: Language for prompts and questions ("en" or "ru")
        question_types: Question types to include (default: DC tasks only)
        seq_lengths: Sequence lengths (default: [32, 64, 128])
        n_questions: Questions per type per length
        seed: Random seed
        render_images: Whether to render images (requires mmred.vgen)
        n_few_shot: Number of few-shot examples per question type
        few_shot_max_len: Maximum sequence length for few-shot examples
        
    Returns:
        Dictionary mapping dataset names to sample lists
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use DC tasks by default
    if question_types is None:
        question_types = DC_TASKS
    
    if seq_lengths is None:
        seq_lengths = MERA_SEQ_LENGTHS
    
    # Set up language-specific names
    lang = Language.from_code(language)

    # Generate base dataset
    config = GenerationConfig(
        seed=seed,
        seq_lengths=seq_lengths,
        n_questions=n_questions,
        question_types=question_types,
        rooms=lang.rooms,
        chars=lang.chars,
    )
    
    print(f"Generating MMReD dataset...")
    print(f"  Language: {language}")
    print(f"  Mode: {mode}")
    print(f"  Question types: {question_types}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Questions per type per length: {n_questions}")
    
    samples = generate_questions_sequential(config)
    print(f"  Generated {len(samples)} samples")

    # Translate question texts if needed
    if language == "ru":
        for sample in samples:
            sample["question"] = lang.translate_question(sample["question"])
            # Translate "Nobody" answer
            if sample["answer"] == "Nobody":
                sample["answer"] = lang.nobody
    
    # Get instruction prompt
    prompt_template = get_instruction_prompt(language, mode)
    
    # Group samples by task and length
    datasets = {}
    for sample in samples:
        task_code = QTYPE_TO_TASK.get(sample["qtype"], sample["qtype"])
        seq_len = sample["seq_len"]
        dataset_name = f"mmred_{task_code.lower().replace('-', '_')}_{seq_len}"
        
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        
        mera_sample = convert_to_mera_format(
            sample, prompt_template, mode,
            image_dir=output_dir / "images" if render_images else None,
            language=language,
        )
        datasets[dataset_name].append(mera_sample)
    
    # Optionally render images
    if render_images and mode == "image":
        print(f"\nRendering images...")
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        try:
            from mmred.vgen.visualization import render_sequence_from_json
            
            for sample in samples:
                qid = sample["qid"]
                sample_dir = images_dir / qid
                sample_dir.mkdir(exist_ok=True)
                render_sequence_from_json(
                    sample["sequence"],
                    output_path=str(sample_dir),
                    as_gif=False,
                )
        except ImportError:
            print("Warning: Could not import vgen module. Images not rendered.")
    
    # Save datasets
    print(f"\nSaving datasets to {output_dir}...")
    for dataset_name, dataset_samples in datasets.items():
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Save test.json (main evaluation set)
        test_data = {
            "access": "public",
            "data": dataset_samples
        }
        with open(dataset_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # Save shots.json (empty for 0-shot)
        shots_data = {"data": []}
        with open(dataset_dir / "shots.json", "w", encoding="utf-8") as f:
            json.dump(shots_data, f, ensure_ascii=False, indent=2)
        
        print(f"  {dataset_name}: {len(dataset_samples)} samples")
    
    # Generate few-shot examples using in_context module
    if n_few_shot > 0:
        print(f"\nGenerating few-shot examples...")
        few_shot_file = output_dir / "in_context_examples.json"
        
        # Generate examples at shorter lengths for each question type
        from mmred.config import DEFAULT_SEED
        generate_in_context_examples(
            output_path=few_shot_file,
            n_examples_per_task=n_few_shot,
            seq_lengths=FEW_SHOT_SEQ_LENGTHS[:few_shot_max_len] if few_shot_max_len < len(FEW_SHOT_SEQ_LENGTHS) else FEW_SHOT_SEQ_LENGTHS,
            question_types=question_types,
            seed=seed + 12345,  # Different seed to avoid overlap with test set
            overwrite=True,
            chars=lang.chars,
            rooms=lang.rooms,
        )

        # Translate few-shot questions if needed
        if language == "ru":
            with open(few_shot_file, "r") as f:
                raw_examples_pre = json.load(f)
            for ex in raw_examples_pre:
                ex["question"] = lang.translate_question(ex["question"])
                if ex["answer"] == "Nobody":
                    ex["answer"] = lang.nobody
            with open(few_shot_file, "w", encoding="utf-8") as f:
                json.dump(raw_examples_pre, f, ensure_ascii=False, indent=2)
        
        # Load and format for MERA shots.json
        with open(few_shot_file, "r") as f:
            raw_examples = json.load(f)
        
        # Group by question type
        examples_by_qtype = {}
        for ex in raw_examples:
            qtype = ex["qtype"]
            if qtype not in examples_by_qtype:
                examples_by_qtype[qtype] = []
            
            # Convert to MERA format
            mera_ex = {
                "instruction": prompt_template,
                "inputs": {
                    "context": format_sequence_as_text(ex["sequence"], language),
                    "question": ex["question"],
                },
                "outputs": str(ex["answer"]),
                "meta": {
                    "seq_len": ex["seq_len"],
                    "qtype": qtype,
                    "atype": ex["atype"],
                }
            }
            examples_by_qtype[qtype].append(mera_ex)
        
        # Save shots.json for each dataset
        for dataset_name in datasets.keys():
            dataset_dir = output_dir / dataset_name
            
            # Extract qtype from dataset name (e.g., mmred_dc_sa_c_64 -> spend_alone)
            task_code = dataset_name.replace('mmred_', '').rsplit('_', 1)[0].upper().replace('_', '-')
            qtype = next((q for q, t in QTYPE_TO_TASK.items() if t == task_code), None)
            
            if qtype and qtype in examples_by_qtype:
                shots_data = {"data": examples_by_qtype[qtype][:n_few_shot]}
            else:
                shots_data = {"data": []}
            
            with open(dataset_dir / "shots.json", "w", encoding="utf-8") as f:
                json.dump(shots_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Generated {len(raw_examples)} few-shot examples")
    
    # Save combined dataset info
    info = {
        "name": "MMReD",
        "version": "1.0.0",
        "language": language,
        "mode": mode,
        "tasks": list(datasets.keys()),
        "total_samples": sum(len(d) for d in datasets.values()),
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nDone! Total: {info['total_samples']} samples across {len(datasets)} datasets")
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate MMReD dataset in MERA-compatible format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for MERA dataset files.",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "image"],
        default="image",
        help="Context representation mode. Default: image",
    )
    
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "ru"],
        default="en",
        help="Language for prompts and questions. Default: en",
    )
    
    parser.add_argument(
        "--question_types",
        type=str,
        nargs="+",
        default=None,
        help=f"Question types to include. Default: DC tasks ({', '.join(DC_TASKS)})",
    )
    
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=None,
        help=f"Sequence lengths. Default: {MERA_SEQ_LENGTHS}",
    )
    
    parser.add_argument(
        "--n_questions",
        type=int,
        default=50,
        help="Questions per type per length. Default: 50",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0xBADFACE,
        help="Random seed. Default: 0xBADFACE",
    )
    
    parser.add_argument(
        "--render_images",
        action="store_true",
        help="Render images for each sample (requires vgen module).",
    )
    
    parser.add_argument(
        "--n_few_shot",
        type=int,
        default=5,
        help="Number of few-shot examples per question type. Default: 5",
    )
    
    parser.add_argument(
        "--few_shot_max_len",
        type=int,
        default=16,
        help="Maximum sequence length for few-shot examples. Default: 16",
    )
    
    args = parser.parse_args()
    
    generate_mera_dataset(
        output_dir=Path(args.output_dir),
        mode=args.mode,
        language=args.language,
        question_types=args.question_types,
        seq_lengths=args.seq_lengths,
        n_questions=args.n_questions,
        seed=args.seed,
        render_images=args.render_images,
        n_few_shot=args.n_few_shot,
        few_shot_max_len=args.few_shot_max_len,
    )


if __name__ == "__main__":
    main()
