import random
import sys
from pathlib import Path

import pandas as pd

# Add project root for imports
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from mmred.config import DEFAULT_CHARS, DEFAULT_ROOMS
from mmred.const import NOBODY, AnswerTypePerson
from mmred.qgen.questions import q_spend_alone_at_time


def _alone_at_frame(df: pd.DataFrame, frame: int) -> set:
    """Return the set of person names who are alone (sole occupant of a room) at the given 0-based frame."""
    row = df.iloc[frame]
    room_counts = row.value_counts()
    alone = set()
    for room, count in room_counts.items():
        if count == 1:
            person = row[row == room].index[0]
            alone.add(person)
    return alone


def test_q_alone_question(seq_len: int, seed: int = None):
    rng = random.Random(seed) if seed is not None else None
    result = q_spend_alone_at_time(seq_len, rng=rng)
    assert len(result) == 5, "QuestionResult must have exactly 5 elements"
    df, q, a, atype, relevant_map = result

    # 1. Return structure and types
    assert isinstance(df, pd.DataFrame), "df must be a DataFrame"
    assert isinstance(q, str) and len(q) > 0, "q must be a non-empty string"
    assert isinstance(a, str), "a must be a string"
    assert a == NOBODY or a in DEFAULT_CHARS, f"a must be NOBODY or one of {DEFAULT_CHARS}, got {a!r}"
    assert atype == AnswerTypePerson, f"atype must be AnswerTypePerson, got {atype!r}"
    assert isinstance(relevant_map, dict), "relevant_map must be a dict"
    for k, v in relevant_map.items():
        assert isinstance(k, int), "relevant_map keys must be int"
        assert isinstance(v, list) and all(r is None or isinstance(r, str) for r in v), "relevant_map values must be list of str or None"

    # 2. DataFrame shape and columns
    assert df.shape[0] == seq_len, f"df must have seq_len={seq_len} rows, got {df.shape[0]}"
    assert set(df.columns) == set(DEFAULT_CHARS), f"df columns must match DEFAULT_CHARS, got {set(df.columns)}"

    # 3. relevant_map structure
    assert len(relevant_map) == 1, "relevant_map must have exactly one key"
    frame = next(iter(relevant_map))
    assert 0 <= frame < seq_len, f"frame must be in [0, seq_len-1], got {frame}"
    rooms_list = relevant_map[frame]
    assert len(rooms_list) == 1, "relevant_map[frame] must be a list of one room"
    assert rooms_list[0] is None or rooms_list[0] in DEFAULT_ROOMS, f"room must be None or in DEFAULT_ROOMS, got {rooms_list[0]!r}"

    # 4. Answer consistency with the sequence
    row = df.iloc[frame]
    alone_persons = _alone_at_frame(df, frame)
    if alone_persons:
        expected_answer = min(alone_persons)
        assert a == expected_answer, f"Expected answer {expected_answer!r} (min of alone {alone_persons}), got {a!r}"
        assert row[a] == relevant_map[frame][0], f"Answer's room must match relevant_map: {row[a]!r} vs {relevant_map[frame][0]!r}"
    else:
        assert a == NOBODY, f"Expected NOBODY when no one is alone, got {a!r}"
        if relevant_map[frame][0] is not None:
            assert relevant_map[frame][0] in DEFAULT_ROOMS, "relevant_map room must be in DEFAULT_ROOMS when set"

    # 5. Question string content
    q_lower = q.lower()
    assert "alone" in q_lower, f"Question must contain 'alone', got: {q!r}"
    assert "time" in q_lower, f"Question must contain 'time', got: {q!r}"


if __name__ == "__main__":
    test_q_alone_question(1)
    test_q_alone_question(2)
    test_q_alone_question(4)
    test_q_alone_question(8)
    test_q_alone_question(16)
    test_q_alone_question(32)
    test_q_alone_question(64)
    test_q_alone_question(128)
    # Reproducible run (optional)
    test_q_alone_question(4, seed=42)
    print("All assertions passed.")
