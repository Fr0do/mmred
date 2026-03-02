"""Question generation functions for MMReD benchmark.

Each question function now returns metadata indicating which rooms at which steps
are relevant for answering the question.

Function signature:
    q_xxx(seq_len, ..., rng=None) -> tuple[pd.DataFrame, str, Any, str, dict[int, list[str]]]
    
Returns:
    - df: The sequence DataFrame
    - question: The question text
    - answer: The correct answer
    - atype: Answer type (person, room, or number)
    - relevant_map: Dict mapping step_id (1-indexed) to list of relevant room names
"""

import random
from typing import Any

import numpy as np
import pandas as pd

from ..config import DEFAULT_CHARS, DEFAULT_ROOMS
from ..const import NOBODY, AnswerTypePerson, AnswerTypeRoom, AnswerTypeNumber
from .utils import (
    inv1d_argmax,
    generate_sequence_df,
    get_random_situation,
    get_random_mmlong,
)


# Type alias for question function return type
QuestionResult = tuple[pd.DataFrame, str, Any, str, dict[int, list[str]]]


# ### NIAH questions: ###


def q_first_app(
    seq_len: int,
    inv: bool = False,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """In which room did [Person] first appear? / In which room was [Person] at the final step?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    
    df, char, _, _, _, _ = get_random_situation(seq_len, chars, rooms, rng)
    step_idx = (seq_len - 1) if inv else 0
    a = df[char].iloc[step_idx]
    q = (
        f"In which room was {char} at the final step?"
        if inv
        else f"In which room did {char} first appear?"
    )
    
    # Metadata: only the queried step, only the room where the character was
    relevant_map = {step_idx + 1: [a]}
    
    return df, q, a, AnswerTypeRoom, relevant_map


def q_final_app(seq_len: int, **kwargs) -> QuestionResult:
    """In which room was [Person] at the final step?"""
    return q_first_app(seq_len, inv=True, **kwargs)


def q_char_on_char_first_app(
    seq_len: int,
    inv: bool = False,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """In which room was [Person] when [Person] first appeared in the [Room]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char_0, char_1, room, _, _ = get_random_situation(seq_len, chars, rooms, rng)

    apps = (df[char_1] == room).values
    while not apps.any():
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        apps = (df[char_1] == room).values

    step_idx = inv1d_argmax(apps) if inv else np.argmax(apps)
    a = df[char_0].iloc[step_idx]
    q = (
        f"In which room was {char_0} when {char_1} made their final appearance in the {room}?"
        if inv
        else f"In which room was {char_0} when {char_1} first appeared in the {room}?"
    )
    
    # Metadata: the step where char_1 appeared, rooms for both chars
    relevant_map = {step_idx + 1: [room, a] if a != room else [room]}
    
    return df, q, a, AnswerTypeRoom, relevant_map


def q_char_on_char_final_app(seq_len: int, **kwargs) -> QuestionResult:
    """In which room was [Person] when [Person] made their final appearance in the [Room]?"""
    return q_char_on_char_first_app(seq_len, inv=True, **kwargs)


def q_char_at_frame(
    seq_len: int,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """In which room was [Person] at step X?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    
    df, char, _, _, _, frame = get_random_situation(seq_len, chars, rooms, rng)
    a = df[char].iloc[frame]
    q = f"In which room was {char} at step {frame + 1}?"
    
    # Metadata: only the queried frame, only the answer room
    relevant_map = {frame + 1: [a]}
    
    return df, q, a, AnswerTypeRoom, relevant_map


def q_first_at_room(
    seq_len: int,
    inv: bool = False,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Who was the first to appear in the [Room]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, _, _, room, _, _ = get_random_situation(seq_len, chars, rooms, rng)
    argmax_fn = inv1d_argmax if inv else np.argmax

    def _check_df_return_answer(_df):
        x = np.sum(_df.values == room, axis=1)
        if np.sum(x) == 0:
            return NOBODY, None
        if argmax_fn(x > 0) == argmax_fn(x > 1):
            return None, None
        step_idx = argmax_fn(x > 0)
        return _df.columns[_df.iloc[step_idx].values == room].item(), step_idx

    a, step_idx = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a, step_idx = _check_df_return_answer(df)

    q = (
        f"Who was the last to appear in the {room}?"
        if inv
        else f"Who was the first to appear in the {room}?"
    )
    
    # Metadata: the step where the person first/last appeared alone, the queried room
    if a == NOBODY:
        relevant_map = {}  # No relevant steps if nobody
    else:
        relevant_map = {step_idx + 1: [room]}
    
    return df, q, a, AnswerTypePerson, relevant_map


def q_last_at_room(seq_len: int, **kwargs) -> QuestionResult:
    """Who was the last to appear in the [Room]?"""
    return q_first_at_room(seq_len, inv=True, **kwargs)


def q_room_on_char_first_app(
    seq_len: int,
    inv: bool = False,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Who was in the [Room] when [Person] first appeared in the [Room]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, room_0, room_1, _ = get_random_situation(seq_len, chars, rooms, rng)
    argmax_fn = inv1d_argmax if inv else np.argmax

    def _check_df_return_answer(_df):
        apps = (_df[char] == room_1).values
        if not apps.any():
            return None, None
        step_idx = argmax_fn(apps)
        row = _df.iloc[step_idx]
        if np.sum(row.values == room_0) > 1:
            return None, None
        if room_0 not in row.values:
            return NOBODY, step_idx
        return _df.columns[row.values == room_0].item(), step_idx

    a, step_idx = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a, step_idx = _check_df_return_answer(df)

    q = (
        f"Who was in the {room_0} when {char} made their final appearance in the {room_1}?"
        if inv
        else f"Who was in the {room_0} when {char} first appeared in the {room_1}?"
    )
    
    # Metadata: the step, both rooms (char's room and queried room)
    relevant_rooms = list(set([room_0, room_1]))
    relevant_map = {step_idx + 1: relevant_rooms}
    
    return df, q, a, AnswerTypePerson, relevant_map


def q_room_on_char_final_app(seq_len: int, **kwargs) -> QuestionResult:
    """Who was in the [Room] when [Person] made their final appearance in the [Room]?"""
    return q_room_on_char_first_app(seq_len, inv=True, **kwargs)


def q_room_at_frame(
    seq_len: int,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Who was in the [Room] at step X?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, _, _, room, _, frame = get_random_situation(seq_len, chars, rooms, rng)

    def _check_df_return_answer(_df):
        row = _df.iloc[frame]
        if np.sum(row.values == room) > 1:
            return None
        if room not in row.values:
            return NOBODY
        return _df.columns[row.values == room].item()

    a = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a = _check_df_return_answer(df)

    q = f"Who was in the {room} at step {frame + 1}?"
    
    # Metadata: the queried frame, the queried room
    relevant_map = {frame + 1: [room]}
    
    return df, q, a, AnswerTypePerson, relevant_map


def q_char_on_char_at_frame(
    seq_len: int,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Who was in the same room as [Person] at step X?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, _, _, frame = get_random_situation(seq_len, chars, rooms, rng)

    def _check_df_return_answer(_df):
        row = _df.iloc[frame]
        if np.sum(row.values == row[char]) > 2:
            return None, None
        if np.sum(row.values == row[char]) == 1:
            return NOBODY, row[char]
        return sorted(set(_df.columns[row.values == row[char]].tolist()) - {char})[0], row[char]

    a, room = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a, room = _check_df_return_answer(df)

    q = f"Who was in the same room as {char} at step {frame + 1}?"
    
    # Metadata: the queried frame, the room where char was
    relevant_map = {frame + 1: [room]}
    
    return df, q, a, AnswerTypePerson, relevant_map


def q_n_room_on_char_first_app(
    seq_len: int,
    inv: bool = False,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """How many characters were in the [Room] when [Person] first appeared in the [Room]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, room_0, room_1, _ = get_random_situation(seq_len, chars, rooms, rng)
    argmax_fn = inv1d_argmax if inv else np.argmax

    def _check_df_return_answer(_df):
        apps = (_df[char] == room_1).values
        if not apps.any():
            return None, None
        step_idx = argmax_fn(apps)
        return np.sum(_df.iloc[step_idx].values == room_0).item(), step_idx

    a, step_idx = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a, step_idx = _check_df_return_answer(df)

    q = (
        f"How many characters were in the {room_0} when {char} made their final appearance in the {room_1}?"
        if inv
        else f"How many characters were in the {room_0} when {char} first appeared in the {room_1}?"
    )
    
    # Metadata: the step, both rooms
    relevant_rooms = list(set([room_0, room_1]))
    relevant_map = {step_idx + 1: relevant_rooms}
    
    return df, q, a, AnswerTypeNumber, relevant_map


def q_n_room_on_char_final_app(seq_len: int, **kwargs) -> QuestionResult:
    """How many characters were in the [Room] when [Person] made their final appearance in the [Room]?"""
    return q_n_room_on_char_first_app(seq_len, inv=True, **kwargs)


def q_n_char_at_frame(
    seq_len: int,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """How many other characters were in the same room as [Person] at step X?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    
    df, char, _, _, _, frame = get_random_situation(seq_len, chars, rooms, rng)
    room = df[char].iloc[frame]
    a = np.sum(df.iloc[frame].values == room).item() - 1
    q = f"How many other characters were in the same room as {char} at step {frame + 1}?"
    
    # Metadata: the queried frame, the room where char was
    relevant_map = {frame + 1: [room]}
    
    return df, q, a, AnswerTypeNumber, relevant_map


def q_n_empty(
    seq_len: int,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """How many rooms were empty at step X?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    
    df, _, _, _, _, frame = get_random_situation(seq_len, chars, rooms, rng)
    occupied_rooms = df.iloc[frame].unique().tolist()
    empty_rooms = [r for r in rooms if r not in occupied_rooms]
    a = len(empty_rooms)
    q = f"How many rooms were empty at step {frame + 1}?"
    
    # Metadata: the queried frame, all empty rooms (or all rooms if we need to check all)
    # We mark empty rooms as relevant since that's what we're counting
    relevant_map = {frame + 1: empty_rooms if empty_rooms else rooms}
    
    return df, q, a, AnswerTypeNumber, relevant_map


def q_spend_alone_at_time(
    seq_len: int,
    fraction: float = 1,
    is_more: bool = None,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    "Who was alone in the rooms at step X?"
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, _, _, frame = get_random_situation(seq_len, chars, rooms, rng)

    def _check_df_return_answer(_df):
        # breakpoint()
        row = _df.iloc[frame]
        alone_chars = {c: 0 for c in chars}
        comp_fn = max if is_more else min
        ur, urc = np.unique(row, return_counts=True)
        for r in ur[urc == 1]:
            alone_chars[np.array(chars)[row == r].item()] += 1

        alone_values = np.array(list(alone_chars.values()))
        all_persons = np.array(list(alone_chars.keys()))

        alone_persons = all_persons[alone_values == 1]
        if alone_persons.size == 0:
            return NOBODY, None
        return comp_fn(alone_persons).item(), row[comp_fn(alone_persons).item()]

    a, room = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a, room = _check_df_return_answer(df)

    q = f"Who was alone at time {frame}?"
    
    # Metadata: the queried frame, the room where char was
    relevant_map = {frame: [room]}
    
    return df, q, a, AnswerTypePerson, relevant_map


# ### DC questions: ###


def q_room_empty(
    seq_len: int,
    fraction: float = 1,
    is_more: bool = None,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Which room was empty for {more/fewer} steps than the other rooms [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, superlative=False, is_more=is_more, rng=rng
    )

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        room_non_visits = {r: 0 for r in rooms}
        for _, row in _df.iterrows():
            empty_rooms = sorted(set(rooms) - set(row.unique().tolist()))
            for r in empty_rooms:
                room_non_visits[r] += 1

        empty_counts = np.array(list(room_non_visits.values()))
        if np.sum(empty_counts == comp_fn(empty_counts)) > 1:
            return None
        return np.array(rooms)[empty_counts == comp_fn(empty_counts)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Which room was empty for {q_start} steps than the other rooms{q_end}"
    
    # Metadata: all steps in range, the answer room (where we're counting emptiness)
    relevant_map = {step_id: [a] for step_id in range(frame_0 + 1, frame_1 + 2)}
    
    return df, q, a, AnswerTypeRoom, relevant_map


def q_where_spend(
    seq_len: int,
    fraction: float = 1,
    is_more: bool = None,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """In which room did [Person] spend the {most/least amount of} time [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, _, _, _ = get_random_situation(seq_len, chars, rooms, rng)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more, rng=rng
    )

    if (not is_more) and (len(rooms) - len(df.iloc[frame_0 : frame_1 + 1]) > 1):
        raise ValueError("It is impossible to choose the least visited room")

    def _check_df_return_answer(_df):
        visits = _df[char].value_counts()
        unvisited = sorted(set(rooms) - set(visits.index.tolist()))
        visits = pd.concat(
            (
                visits,
                pd.Series(data=[0] * len(unvisited), index=unvisited, dtype=np.int64),
            )
        )
        if (
            (visits.iloc[0] == visits.iloc[1])
            if is_more
            else (visits.iloc[-1] == visits.iloc[-2])
        ):
            return None
        return visits.index[0] if is_more else visits.index[-1]

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"In which room did {char} spend the {q_start} time{q_end}"
    
    # Metadata: all steps in range where the character was in the answer room
    relevant_map = {}
    for step_id in range(frame_0 + 1, frame_1 + 2):
        if df[char].iloc[step_id - 1] == a:
            relevant_map[step_id] = [a]
    
    return df, q, a, AnswerTypeRoom, relevant_map


def q_crowded_room(
    seq_len: int,
    fraction: float = 1,
    n_crowd: int = 3,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Which room was crowded ([three] or more people in one room) for the most steps [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction, rng=rng)

    def _check_df_return_answer(_df):
        room_crowds = {r: 0 for r in rooms}
        for _, row in _df.iterrows():
            ur, urc = np.unique(row, return_counts=True)
            for r in ur[urc >= n_crowd]:
                room_crowds[r] += 1

        crowds = np.array(list(room_crowds.values()))
        if np.sum(crowds == np.max(crowds)) > 1:
            return None
        return np.array(rooms)[crowds == np.max(crowds)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Which room was crowded ({n_crowd} or more people in one room) for the most steps{q_end}"
    
    # Metadata: steps where the answer room was crowded
    relevant_map = {}
    for step_id in range(frame_0 + 1, frame_1 + 2):
        row = df.iloc[step_id - 1]
        ur, urc = np.unique(row, return_counts=True)
        if a in ur[urc >= n_crowd]:
            relevant_map[step_id] = [a]
    
    return df, q, a, AnswerTypeRoom, relevant_map


def q_who_spend(
    seq_len: int,
    fraction: float = 1,
    is_more: bool = None,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Who spent the {most/least amount of} time in the [Room] [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, _, _, room, _, _ = get_random_situation(seq_len, chars, rooms, rng)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more, rng=rng
    )

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        visit_counts = (_df == room).sum().values
        if np.sum(visit_counts == comp_fn(visit_counts)) > 1:
            return None
        return np.array(chars)[visit_counts == comp_fn(visit_counts)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Who spent the {q_start} time alone in the {room}{q_end}"
    
    # Metadata: all steps in range, the queried room
    relevant_map = {step_id: [room] for step_id in range(frame_0 + 1, frame_1 + 2)}
    
    return df, q, a, AnswerTypePerson, relevant_map


def q_spend_alone(
    seq_len: int,
    fraction: float = 1,
    is_more: bool = None,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """Who spent the {most/least amount of} time alone in the rooms [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    breakpoint()

    df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more, rng=rng
    )

    if (not is_more) and (frame_1 - frame_0 < 2):
        raise ValueError("It is impossible to choose the loneliest char")

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        alone_chars = {c: 0 for c in chars}
        for _, row in _df.iterrows():
            ur, urc = np.unique(row, return_counts=True)
            for r in ur[urc == 1]:
                alone_chars[np.array(chars)[row == r].item()] += 1

        alone = np.array(list(alone_chars.values()))
        if np.sum(alone == comp_fn(alone)) > 1:
            return None
        return np.array(chars)[alone == comp_fn(alone)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Who spent the {q_start} time alone in the rooms{q_end}"
    
    # Metadata: steps where the answer character was alone, and the room they were in
    relevant_map = {}
    for step_id in range(frame_0 + 1, frame_1 + 2):
        row = df.iloc[step_id - 1]
        ur, urc = np.unique(row, return_counts=True)
        alone_rooms = ur[urc == 1]
        for r in alone_rooms:
            if row[a] == r:  # The answer character was alone in this room
                relevant_map[step_id] = [r]
                break
    
    return df, q, a, AnswerTypePerson, relevant_map


def q_spend_together(
    seq_len: int,
    fraction: float = 1,
    is_more: bool = None,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """With whom did [Person] spend the {most/least amount of} time together in the same room [...]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, _, _, _ = get_random_situation(seq_len, chars, rooms, rng)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more, rng=rng
    )

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        rest_chars = np.array(sorted(set(chars) - {char}))
        alone_chars = {c: 0 for c in rest_chars}
        for _, row in _df.iterrows():
            for c in row.index[row == row[char]]:
                if c != char:
                    alone_chars[c] += 1
        alone = np.array(list(alone_chars.values()))
        if np.sum(alone == comp_fn(alone)) > 1:
            return None
        return rest_chars[alone == comp_fn(alone)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"With whom did {char} spend the {q_start} time together in the same room{q_end}"
    
    # Metadata: steps where the queried char and answer char were together
    relevant_map = {}
    for step_id in range(frame_0 + 1, frame_1 + 2):
        row = df.iloc[step_id - 1]
        if row[char] == row[a]:  # They were in the same room
            relevant_map[step_id] = [row[char]]
    
    return df, q, a, AnswerTypePerson, relevant_map


def q_steps_in_room(
    seq_len: int,
    fraction: float = 1,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """How many steps did [Person] spend in the [Room] [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, room, _, _ = get_random_situation(seq_len, chars, rooms, rng)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction, rng=rng)
    a = np.sum(df.iloc[frame_0 : frame_1 + 1][char].values == room).item()
    q = f"How many steps did {char} spend in the {room}{q_end}"
    
    # Metadata: steps where the character was in the queried room
    relevant_map = {}
    for step_id in range(frame_0 + 1, frame_1 + 2):
        if df[char].iloc[step_id - 1] == room:
            relevant_map[step_id] = [room]
    
    return df, q, a, AnswerTypeNumber, relevant_map


def q_rooms_visited(
    seq_len: int,
    fraction: float = 1,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """How many different rooms did [Person] visit [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df, char, _, _, _, _ = get_random_situation(seq_len, chars, rooms, rng)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction, rng=rng)
    visited_rooms = df.iloc[frame_0 : frame_1 + 1][char].unique().tolist()
    a = len(visited_rooms)
    q = f"How many different rooms did {char} visit{q_end}"
    
    # Metadata: all steps in range, rooms where the character was (first visit to each)
    relevant_map = {}
    seen_rooms = set()
    for step_id in range(frame_0 + 1, frame_1 + 2):
        room = df[char].iloc[step_id - 1]
        if room not in seen_rooms:
            relevant_map[step_id] = [room]
            seen_rooms.add(room)
    
    return df, q, a, AnswerTypeNumber, relevant_map


def q_crowd_count(
    seq_len: int,
    fraction: float = 1,
    n_crowd: int = 3,
    chars: list[str] = None,
    rooms: list[str] = None,
    rng: random.Random = None,
    **kwargs,
) -> QuestionResult:
    """How many times did a crowd ([three] or more people in one room) appear [between frames X and Y]?"""
    chars = chars or DEFAULT_CHARS
    rooms = rooms or DEFAULT_ROOMS
    rng = rng or random
    
    df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction, rng=rng)
    
    # Count crowds and track where they occurred
    crowd_steps = {}
    for step_id in range(frame_0 + 1, frame_1 + 2):
        row = df.iloc[step_id - 1]
        ur, urc = np.unique(row, return_counts=True)
        crowded_rooms = ur[urc >= n_crowd].tolist()
        if crowded_rooms:
            crowd_steps[step_id] = crowded_rooms
    
    a = sum(len(r) for r in crowd_steps.values())  # Total crowd occurrences
    
    # Recompute using original method to match
    a = (
        df.iloc[frame_0 : frame_1 + 1]
        .apply(lambda x: np.sum(np.unique(x, return_counts=True)[1] >= n_crowd), axis=1)
        .sum()
        .item()
    )
    
    q = f"How many times did a crowd ({n_crowd} or more people in one room) appear{q_end}"
    
    # Metadata: steps with crowded rooms
    relevant_map = crowd_steps
    
    return df, q, a, AnswerTypeNumber, relevant_map


# ===================================================== QUESTIONS =====================================================


QUESTIONS = {
    "first_app": q_first_app,
    "final_app": q_final_app,
    "char_on_char_first_app": q_char_on_char_first_app,
    "char_on_char_final_app": q_char_on_char_final_app,
    "char_at_frame": q_char_at_frame,
    "first_at_room": q_first_at_room,
    "last_at_room": q_last_at_room,
    "room_on_char_first_app": q_room_on_char_first_app,
    "room_on_char_final_app": q_room_on_char_final_app,
    "room_at_frame": q_room_at_frame,
    "char_on_char_at_frame": q_char_on_char_at_frame,
    "n_room_on_char_first_app": q_n_room_on_char_first_app,
    "n_room_on_char_final_app": q_n_room_on_char_final_app,
    "n_char_at_frame": q_n_char_at_frame,
    "n_empty": q_n_empty,
    "room_empty": q_room_empty,
    "where_spend": q_where_spend,
    "crowded_room": q_crowded_room,
    "who_spend": q_who_spend,
    "spend_alone": q_spend_alone,
    "spend_together": q_spend_together,
    "steps_in_room": q_steps_in_room,
    "rooms_visited": q_rooms_visited,
    "crowd_count": q_crowd_count,
    "spend_alone_at_step": q_spend_alone_at_time 
}
