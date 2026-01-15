"""Utility functions for question generation.

This module provides helper functions for sequence generation, hashing,
sampling, and reproducible random state management.
"""

import hashlib
import json
import random
from typing import Callable

import numpy as np
import pandas as pd

from ..config import DEFAULT_CHARS, DEFAULT_ROOMS


def inv1d_argmax(a):
    """Find the index of the last maximum value in a 1D array."""
    return len(a) - np.argmax(a[::-1]) - 1


def generate_sequence_df(
    seq_len: int,
    one_move: bool = True,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
) -> pd.DataFrame:
    """Generate a random sequence of character positions.
    
    Args:
        seq_len: Number of steps in the sequence
        one_move: If True, only one character moves per step
        chars: List of character names (default: DEFAULT_CHARS)
        rooms: List of room names (default: DEFAULT_ROOMS)
        rng: Random number generator instance for reproducibility
        
    Returns:
        DataFrame with characters as columns and rooms as values
    """
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    if one_move:
        seq = [{char: rng.choice(rooms) for char in chars}]
        for _ in range(1, seq_len):
            char_to_move = rng.choice(chars)
            room_to_move = rng.choice(sorted(set(rooms) - {seq[-1][char_to_move]}))
            seq.append(
                {
                    char: (room_to_move if (char == char_to_move) else seq[-1][char])
                    for char in chars
                }
            )
        return pd.DataFrame.from_records(seq)
    else:
        return pd.DataFrame(
            {char: [rng.choice(rooms) for _ in range(seq_len)] for char in chars}
        )


def generate_sequence_df_controlled(
    seq_len: int,
    target_keyframes: int,
    keyframe_checker: Callable[[pd.DataFrame, int], bool],
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    max_attempts: int = 1000,
) -> pd.DataFrame | None:
    """Generate a sequence with a specific number of 'keyframes' (relevant steps).
    
    This function generates sequences until one is found that has exactly
    the target number of keyframes, as determined by the keyframe_checker function.
    
    Args:
        seq_len: Number of steps in the sequence
        target_keyframes: Desired number of keyframes
        keyframe_checker: Function that takes (df, step_index) and returns True if step is a keyframe
        chars: List of character names
        rooms: List of room names
        rng: Random number generator instance
        max_attempts: Maximum generation attempts before giving up
        
    Returns:
        DataFrame with the target keyframe count, or None if not achievable
    """
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    for _ in range(max_attempts):
        df = generate_sequence_df(seq_len, one_move=True, chars=chars, rooms=rooms, rng=rng)
        keyframe_count = sum(1 for i in range(seq_len) if keyframe_checker(df, i))
        if keyframe_count == target_keyframes:
            return df
    
    return None


def hash_seq_df(df: pd.DataFrame) -> str:
    """Generate a unique hash for a sequence DataFrame."""
    return hashlib.sha256(
        json.dumps(tuple(tuple(row) for row in df.values.tolist())).encode("utf-8")
    ).hexdigest()


def sample_steps(
    seq_len: int,
    fraction: float,
    rng: random.Random = None,
) -> tuple[int, int, str]:
    """Sample a contiguous range of steps within a sequence.
    
    Args:
        seq_len: Total number of steps
        fraction: Fraction of the sequence to include (0 < fraction <= 1)
        rng: Random number generator instance
        
    Returns:
        Tuple of (start_index, end_index, question_suffix)
    """
    rng = rng or random
    n_frames = max(1, int(round(seq_len * fraction)))
    start = rng.randint(0, seq_len - n_frames)
    q_finish = (
        "?"
        if (fraction == 1)
        else f" between steps {start + 1} and {start + n_frames}?"
    )
    return start, start + n_frames - 1, q_finish


def sample_comparison(
    superlative: bool = True,
    is_more: bool = None,
    rng: random.Random = None,
) -> tuple[bool, str]:
    """Sample a comparison direction (more/fewer or most/least).
    
    Args:
        superlative: If True, use "most/least amount of", else use "more/fewer"
        is_more: Force a specific direction (True=more, False=fewer, None=random)
        rng: Random number generator instance
        
    Returns:
        Tuple of (is_more_direction, question_text)
    """
    rng = rng or random
    is_more = (rng.random() >= 0.5) if is_more is None else is_more
    q_starts = ["least amount of", "most"] if superlative else ["fewer", "more"]
    return is_more, q_starts[int(is_more)]


def get_random_situation(
    seq_len: int,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
) -> tuple[pd.DataFrame, str, str, str, str, int]:
    """Generate a random situation for question generation.
    
    Returns:
        Tuple of (df, char_0, char_1, room_0, room_1, frame_index)
    """
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
    char_0 = rng.choice(chars)
    char_1 = rng.choice(sorted(set(chars) - {char_0}))
    room_0 = rng.choice(rooms)
    room_1 = rng.choice(sorted(set(rooms) - {room_0}))
    frame = rng.randint(0, seq_len - 1)
    return df, char_0, char_1, room_0, room_1, frame


def get_random_mmlong(
    seq_len: int,
    fraction: float,
    superlative: bool = True,
    is_more: bool = None,
    rng: random.Random = None,
) -> tuple[bool, str, int, int, str]:
    """Generate random parameters for MMLong-style questions.
    
    Returns:
        Tuple of (is_more, q_start_text, frame_0, frame_1, q_end_text)
    """
    rng = rng or random
    is_more, q_start = sample_comparison(superlative, is_more, rng)
    frame_0, frame_1, q_end = sample_steps(seq_len, fraction, rng)
    return is_more, q_start, frame_0, frame_1, q_end


def fix_seed(seed: int):
    """Set the global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def create_rng(seed: int) -> random.Random:
    """Create a new random.Random instance with the given seed."""
    return random.Random(seed)
