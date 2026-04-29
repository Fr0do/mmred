"""Generate in-context examples for few-shot prompting.

This module provides functionality to generate compact in-context examples
that can be used for few-shot prompts.
"""

import json
from pathlib import Path
from typing import Sequence

from .config import GenerationConfig, DEFAULT_SEED
from .data_model import serialize_sequence, MetadataStep
from .qgen.qgen import _generate_single_question
from .qgen.questions import QUESTIONS
from .qgen.utils import create_rng


def generate_in_context_examples(
    output_path: str | Path,
    n_examples_per_task: int = 5,
    seq_lengths: Sequence[int] = (1, 2, 4, 8, 16),
    question_types: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    overwrite: bool = True,
    chars: list[str] | None = None,
    rooms: list[str] | None = None,
) -> Path:
    """Generate a compact in-context dataset for few-shot prompts.

    The resulting file contains serialized sequences, questions, and
    answers across the provided ``seq_lengths``.
    
    Args:
        output_path: Path to output JSON file
        n_examples_per_task: Number of examples per question type per seq_len
        seq_lengths: Sequence lengths to include
        question_types: Question types to include (None = all)
        seed: Random seed for reproducibility
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to the generated file
    """
    from .config import DEFAULT_ROOMS, DEFAULT_CHARS

    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    question_types = question_types or list(QUESTIONS.keys())
    examples = []
    
    for seq_len in seq_lengths:
        for question_type in question_types:
            if question_type not in QUESTIONS:
                continue
                
            question_fn = QUESTIONS[question_type]
            
            # Special cases
            q_kwargs = {}
            if (question_type == "where_spend") and (seq_len <= 4):
                q_kwargs["is_more"] = True
            elif (question_type == "spend_alone") and (seq_len <= 2):
                q_kwargs["is_more"] = True
            
            # Create deterministic RNG for this combination
            batch_seed = seed + hash(f"incontext_{seq_len}_{question_type}") % (2**31)
            rng = create_rng(batch_seed)
            
            seen_hashes = []
            for ex_idx in range(n_examples_per_task):
                seq_df, question, answer, atype, relevant_map, seq_hash = _generate_single_question(
                    question_fn, seq_len, seen_hashes,
                    chars, rooms, rng, **q_kwargs
                )
                seen_hashes.append(seq_hash)
                
                # Serialize sequence
                sequence = serialize_sequence(seq_df, rooms)
                
                # Create metadata
                metadata = []
                for step_id in range(1, seq_len + 1):
                    step_rooms = relevant_map.get(step_id, [])
                    room_relevance = {room: room in step_rooms for room in rooms}
                    metadata.append(MetadataStep(step_id=step_id, rooms=room_relevance))
                
                examples.append({
                    "example_id": f"ctx_{seq_len}_{question_type}_{ex_idx}",
                    "seq_len": seq_len,
                    "qtype": question_type,
                    "atype": atype,
                    "question": question,
                    "answer": answer,
                    "sequence": [s.to_dict() for s in sequence],
                    "metadata": [m.to_dict() for m in metadata],
                })

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    return output_path
