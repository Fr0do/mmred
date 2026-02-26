#!/usr/bin/env python
"""Render images from a generated MMReD dataset.

This script takes a JSON dataset file as input and renders frame images
for each sequence. Images can be rendered as individual PNGs or combined GIFs.

Example usage:
    # Render all sequences as individual frames
    python scripts/render_images.py \\
        --input_path data/dataset.json \\
        --output_dir data/images/

    # Render as GIFs
    python scripts/render_images.py \\
        --input_path data/dataset.json \\
        --output_dir data/gifs/ \\
        --format gif

    # Limit number of sequences to render
    python scripts/render_images.py \\
        --input_path data/dataset.json \\
        --output_dir data/images/ \\
        --limit 10
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def render_single_sample(
    sample: dict[str, Any],
    output_dir: Path,
    format: str = "png",
) -> str:
    """Render images for a single sample.
    
    Args:
        sample: Sample dictionary with 'qid' and 'sequence' keys
        output_dir: Base output directory
        format: Output format ('png' or 'gif')
        
    Returns:
        Path to the rendered output
    """
    from mmred.vgen.visualization import render_sequence_from_json
    
    qid = sample["qid"]
    sequence = sample["sequence"]
    
    if format == "gif":
        output_path = output_dir / f"{qid}.gif"
        render_sequence_from_json(sequence, output_path, as_gif=True)
    else:
        output_path = output_dir / qid
        render_sequence_from_json(sequence, output_path, as_gif=False)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Render images from MMReD dataset.",
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input JSON dataset file.",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for rendered images.",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "gif"],
        default="png",
        help="Output format. 'png' creates folders with frame images, 'gif' creates animated GIFs. Default: png",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sequences to render (for testing).",
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential rendering (no parallelization).",
    )
    
    parser.add_argument(
        "--qids",
        type=str,
        nargs="+",
        default=None,
        help="Specific question IDs to render.",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    with open(input_path, "r") as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples from {input_path}")
    
    # Filter by qids if specified
    if args.qids:
        qid_set = set(args.qids)
        dataset = [s for s in dataset if s["qid"] in qid_set]
        print(f"Filtered to {len(dataset)} samples matching specified qids")
    
    # Apply limit if specified
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Limited to {len(dataset)} samples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering to {output_dir} as {args.format}...")
    
    # Render
    if args.sequential:
        for sample in dataset:
            output_path = render_single_sample(sample, output_dir, args.format)
            print(f"  Rendered {sample['qid']} -> {output_path}")
    else:
        render_fn = partial(render_single_sample, output_dir=output_dir, format=args.format)
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(render_fn, dataset))
        print(f"Rendered {len(results)} sequences")
    
    print("Done!")


if __name__ == "__main__":
    main()
