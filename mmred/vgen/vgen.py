"""Video generation from JSON dataset."""

from pathlib import Path
from typing import Any

from .visualization import render_sequence_from_json


def render_dataset(
    dataset: list[dict[str, Any]],
    output_dir: str | Path,
    as_gif: bool = False,
) -> None:
    """Render images for all samples in a dataset.
    
    Args:
        dataset: List of sample dictionaries with 'qid' and 'sequence' keys
        output_dir: Base output directory
        as_gif: If True, create GIFs; otherwise create PNG frame directories
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for sample in dataset:
        qid = sample["qid"]
        sequence = sample["sequence"]
        
        if as_gif:
            output_path = output_dir / f"{qid}.gif"
        else:
            output_path = output_dir / qid
        
        render_sequence_from_json(sequence, output_path, as_gif=as_gif)
