import hashlib
import json
import random

import numpy as np
import pandas as pd

from qgen.const import CHARS, ROOMS


def inv1d_argmax(a):
    return len(a) - np.argmax(a[::-1]) - 1


def generate_sequence_df(seq_len):
    return pd.DataFrame({char: [random.choice(ROOMS) for _ in range(seq_len)] for char in CHARS})


def hash_seq_df(df):
    return hashlib.sha256(json.dumps(tuple(tuple(row) for row in df.values.tolist())).encode('utf-8')).hexdigest()


def sample_steps(seq_len, fraction: float):
    n_frames = max(1, int(round(seq_len * fraction)))
    start = random.randint(0, seq_len - n_frames)
    q_finish = '?' if (fraction == 1) else f' between steps {start + 1} and {start + n_frames}?'
    return start, start + n_frames - 1, q_finish


def sample_comparison(superlative: bool = True, is_more: bool = None):
    is_more = (random.random() >= 0.5) if is_more is None else is_more
    q_starts = ['least amount of', 'most'] if superlative else ['fewer', 'more']
    return is_more, q_starts[int(is_more)]


def get_random_situation(seq_len):
    df = generate_sequence_df(seq_len)
    char_0 = random.choice(CHARS)
    char_1 = random.choice(sorted(set(CHARS) - {char_0}))
    room_0 = random.choice(ROOMS)
    room_1 = random.choice(sorted(set(ROOMS) - {room_0}))
    frame = random.randint(0, seq_len - 1)
    return df, char_0, char_1, room_0, room_1, frame


def get_random_mmlong(seq_len: int, fraction: float, superlative: bool = True, is_more: bool = None):
    is_more, q_start = sample_comparison(superlative, is_more)
    frame_0, frame_1, q_end = sample_steps(seq_len, fraction)
    return is_more, q_start, frame_0, frame_1, q_end


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
