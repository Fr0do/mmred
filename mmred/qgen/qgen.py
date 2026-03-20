"""Question generation orchestration with parallelization.

This module provides the main entry point for generating MMReD benchmark
questions with support for:
- Parallelization across sequence lengths and question types
- Reproducible seeding
- Configurable question type selection
- Inline JSON serialization
"""

import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import GenerationConfig
from ..data_model import (
    Sample,
    Step,
    MetadataStep,
    serialize_sequence,
    create_metadata_from_relevance,
    aggregate_metadata_step,
    aggregate_metadata_global,
)
from .questions import QUESTIONS
from .utils import hash_seq_df, create_rng


def _generate_single_question(
    q_fn,
    seq_len: int,
    q_hashes: list[str],
    chars: list[str],
    rooms: list[str],
    rng: random.Random,
    **kwargs,
) -> tuple[pd.DataFrame, str, Any, str, dict[int, list[str]], str]:
    """Generate a single question, ensuring unique sequence hash."""
    seq, q, a, atype, relevant_map = q_fn(
        seq_len, chars=chars, rooms=rooms, rng=rng, **kwargs
    )
    h = hash_seq_df(seq)
    attempts = 0
    max_attempts = 100
    while h in q_hashes and attempts < max_attempts:
        seq, q, a, atype, relevant_map = q_fn(
            seq_len, chars=chars, rooms=rooms, rng=rng, **kwargs
        )
        h = hash_seq_df(seq)
        attempts += 1
    return seq, q, a, atype, relevant_map, h


def _generate_batch(
    seq_len: int,
    question_type: str,
    n_questions: int,
    seed: int,
    chars: list[str],
    rooms: list[str],
) -> list[dict[str, Any]]:
    """Generate a batch of questions for a specific seq_len and question type.
    
    This function is designed to be called in a subprocess for parallelization.
    """
    # Create a deterministic seed based on seq_len and question_type
    batch_seed = seed + hash(f"{seq_len}_{question_type}") % (2**31)
    rng = create_rng(batch_seed)
    
    question_fn = QUESTIONS[question_type]
    q_hashes = []
    samples = []
    
    # Handle special cases for certain question types
    q_kwargs = {}
    if (question_type == "where_spend") and (seq_len <= 4):
        q_kwargs["is_more"] = True
    elif (question_type == "spend_alone") and (seq_len <= 2):
        q_kwargs["is_more"] = True
    
    for _ in range(n_questions):
        seq, q, a, atype, relevant_map, h = _generate_single_question(
            question_fn, seq_len, q_hashes, chars, rooms, rng, **q_kwargs
        )
        q_hashes.append(h)
        
        # Serialize sequence inline
        sequence = serialize_sequence(seq, rooms)
        
        # Create metadata from the relevant_map
        metadata = []
        for step_id in range(1, seq_len + 1):
            step_rooms = relevant_map.get(step_id, [])
            room_relevance = {room: room in step_rooms for room in rooms}
            metadata.append(MetadataStep(step_id=step_id, rooms=room_relevance))

        # Aggregate metadata fields
        n_per_step = aggregate_metadata_step(metadata)
        n_total = aggregate_metadata_global(metadata)

        samples.append({
            "seq_len": seq_len,
            "qtype": question_type,
            "atype": atype,
            "question": q,
            "answer": a,
            "sequence": [s.to_dict() for s in sequence],
            "metadata": [m.to_dict() for m in metadata],
            "n_relevant_rooms_per_step": n_per_step,
            "n_relevant_rooms": n_total,
        })
    
    return samples


def generate_questions(config: GenerationConfig) -> list[dict[str, Any]]:
    """Generate all questions according to the configuration.
    
    Args:
        config: Generation configuration
        
    Returns:
        List of sample dictionaries ready for JSON serialization
    """
    question_types = config.get_question_types(QUESTIONS.keys())
    all_samples = []
    
    # Generate tasks for parallel execution
    tasks = []
    for seq_len in config.seq_lengths:
        for qtype in question_types:
            n_q = config.n_for_question_type(qtype)
            if n_q > 0:
                tasks.append((seq_len, qtype, n_q))
    
    # Use ProcessPoolExecutor for parallel generation
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _generate_batch,
                seq_len,
                qtype,
                n_q,
                config.seed,
                config.chars,
                config.rooms,
            ): (seq_len, qtype)
            for seq_len, qtype, n_q in tasks
        }
        
        for future in as_completed(futures):
            seq_len, qtype = futures[future]
            try:
                batch_samples = future.result()
                all_samples.extend(batch_samples)
            except Exception as e:
                print(f"Error generating {qtype} for seq_len={seq_len}: {e}")
                raise
    
    # Sort by seq_len, then by question type for deterministic ordering
    all_samples.sort(key=lambda x: (x["seq_len"], x["qtype"]))
    
    # Assign qids after sorting
    for i, sample in enumerate(all_samples):
        sample["qid"] = f"{i:07d}"
    
    return all_samples


def generate_questions_sequential(config: GenerationConfig) -> list[dict[str, Any]]:
    """Generate questions sequentially (for debugging or small datasets).
    
    This is useful when parallel execution is not desired or for debugging.
    """
    question_types = config.get_question_types(QUESTIONS.keys())
    all_samples = []
    
    for seq_len in config.seq_lengths:
        for qtype in question_types:
            n_q = config.n_for_question_type(qtype)
            if n_q <= 0:
                continue
            batch_samples = _generate_batch(
                seq_len,
                qtype,
                n_q,
                config.seed,
                config.chars,
                config.rooms,
            )
            all_samples.extend(batch_samples)
            print(f"Generated {len(batch_samples)} samples for {qtype} @ len={seq_len}")
    
    # Sort and assign qids
    all_samples.sort(key=lambda x: (x["seq_len"], x["qtype"]))
    for i, sample in enumerate(all_samples):
        sample["qid"] = f"{i:07d}"
    
    return all_samples


def save_dataset(samples: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save the generated dataset to a JSON file.
    
    Args:
        samples: List of sample dictionaries
        output_path: Path to the output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_path}")
