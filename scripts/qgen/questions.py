import random

import numpy as np
import pandas as pd

from qgen.const import (
    CHARS,
    NOBODY,
    ROOMS,
    AnswerTypePerson,
    AnswerTypeRoom,
    AnswerTypeNumber,
)
from qgen.utils import (
    inv1d_argmax,
    generate_sequence_df,
    sample_comparison,
    sample_steps,
    get_random_situation,
    get_random_mmlong,
)


# ### NIAH questions: ###


def q_first_app(seq_len, inv=False, **kwargs):
    """In which room did [Person] first appear?"""
    df, char, _, _, _, _ = get_random_situation(seq_len)
    a = df[char].iloc[-1] if inv else df[char].iloc[0]
    q = (
        f"In which room was {char} at the final step?"
        if inv
        else f"In which room did {char} first appear?"
    )
    return df, q, a, AnswerTypeRoom


def q_final_app(seq_len, **kwargs):
    """In which room was [Person] at the final step?"""
    return q_first_app(seq_len, inv=True, **kwargs)


def q_char_on_char_first_app(seq_len, inv=False, **kwargs):
    """In which room was [Person] when [Person] first appeared in the [Room]?"""
    df, char_0, char_1, room, _, _ = get_random_situation(seq_len)

    apps = (df[char_1] == room).values
    while not apps.any():
        df = generate_sequence_df(seq_len)
        apps = (df[char_1] == room).values

    a = df[char_0].iloc[inv1d_argmax(apps)] if inv else df[char_0].iloc[np.argmax(apps)]
    q = (
        f"In which room was {char_0} when {char_1} made their final appearance in the {room}?"
        if inv
        else f"In which room was {char_0} when {char_1} first appeared in the {room}?"
    )
    return df, q, a, AnswerTypeRoom


def q_char_on_char_final_app(seq_len, **kwargs):
    """In which room was [Person] when [Person] made their final appearance in the [Room]?"""
    return q_char_on_char_first_app(seq_len, inv=True, **kwargs)


def q_char_at_frame(seq_len, **kwargs):
    """In which room was [Person] at step X?"""
    df, char, _, _, _, frame = get_random_situation(seq_len)
    a = df[char].iloc[frame]
    q = f"In which room was {char} at step {frame + 1}?"
    return df, q, a, AnswerTypeRoom


def q_first_at_room(seq_len, inv=False, **kwargs):
    """Who was the first to appear in the [Room]?"""
    df, _, _, room, _, _ = get_random_situation(seq_len)
    argmax_fn = inv1d_argmax if inv else np.argmax

    def _check_df_return_answer(_df):
        x = np.sum(_df.values == room, axis=1)
        if np.sum(x) == 0:
            return NOBODY
        if argmax_fn(x > 0) == argmax_fn(x > 1):  # also passes if no such room
            return
        return _df.columns[_df.iloc[argmax_fn(x > 0)].values == room].item()

    a = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df)

    q = (
        f"Who was the last to appear in the {room}?"
        if inv
        else f"Who was the first to appear in the {room}?"
    )
    return df, q, a, AnswerTypePerson


def q_last_at_room(seq_len, **kwargs):
    """Who was the last to appear in the [Room]?"""
    return q_first_at_room(seq_len, inv=True, **kwargs)


def q_room_on_char_first_app(seq_len, inv=False, **kwargs):
    """Who was in the [Room] when [Person] first appeared in the [Room]?"""
    df, char, _, room_0, room_1, _ = get_random_situation(seq_len)
    argmax_fn = inv1d_argmax if inv else np.argmax

    def _check_df_return_answer(_df):
        apps = (_df[char] == room_1).values
        if not apps.any():
            return
        row = _df.iloc[argmax_fn(apps)]
        if np.sum(row.values == room_0) > 1:
            return
        if room_0 not in row.values:
            return NOBODY
        return _df.columns[row.values == room_0].item()

    a = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df)

    q = (
        f"Who was in the {room_0} when {char} made their final appearance in the {room_1}?"
        if inv
        else f"Who was in the {room_0} when {char} first appeared in the {room_1}?"
    )
    return df, q, a, AnswerTypePerson


def q_room_on_char_final_app(seq_len, **kwargs):
    """Who was in the [Room] when [Person] made their final appearance in the [Room]?"""
    return q_room_on_char_first_app(seq_len, inv=True, **kwargs)


def q_room_at_frame(seq_len, **kwargs):
    """Who was in the [Room] at step X?"""
    df, _, _, room, _, frame = get_random_situation(seq_len)

    def _check_df_return_answer(_df):
        row = _df.iloc[frame]
        if np.sum(row.values == room) > 1:
            return
        if room not in row.values:
            return NOBODY
        return _df.columns[row.values == room].item()

    a = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df)

    q = f"Who was in the {room} at step {frame + 1}?"
    return df, q, a, AnswerTypePerson


def q_char_on_char_at_frame(seq_len, **kwargs):
    """Who was in the same room as [Person] at step X?"""
    df, char, _, _, _, frame = get_random_situation(seq_len)

    def _check_df_return_answer(_df):
        row = _df.iloc[frame]
        if np.sum(row.values == row[char]) > 2:
            return
        if np.sum(row.values == row[char]) == 1:
            return NOBODY
        return sorted(set(_df.columns[row.values == row[char]].tolist()) - {char})[0]

    a = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df)

    q = f"Who was in the same room as {char} at step {frame + 1}?"
    return df, q, a, AnswerTypePerson


def q_n_room_on_char_first_app(seq_len, inv=False, **kwargs):
    """How many characters were in the [Room] when [Person] first appeared in the [Room]?"""
    df, char, _, room_0, room_1, _ = get_random_situation(seq_len)
    argmax_fn = inv1d_argmax if inv else np.argmax

    def _check_df_return_answer(_df):
        apps = (_df[char] == room_1).values
        return (
            np.sum(_df.iloc[argmax_fn(apps)].values == room_0).item()
            if apps.any()
            else None
        )

    a = _check_df_return_answer(df)
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df)

    q = (
        f"How many characters were in the {room_0} when {char} made their final appearance in the {room_1}?"
        if inv
        else f"How many characters were in the {room_0} when {char} first appeared in the {room_1}?"
    )
    return df, q, a, AnswerTypeNumber


def q_n_room_on_char_final_app(seq_len, **kwargs):
    """How many characters were in the [Person] when [Person] made their final appearance in the [Room]?"""
    return q_n_room_on_char_first_app(seq_len, inv=True, **kwargs)


def q_n_char_at_frame(seq_len, **kwargs):
    """How many other characters were in the same room as [Person] at step X?"""
    df, char, _, _, _, frame = get_random_situation(seq_len)
    a = np.sum(df.iloc[frame].values == df[char].iloc[frame]).item() - 1
    q = f"How many other characters were in the same room as {char} at step {frame + 1}?"
    return df, q, a, AnswerTypeNumber


def q_n_empty(seq_len, **kwargs):
    """How many rooms were empty at step X?"""
    df, _, _, _, _, frame = get_random_situation(seq_len)
    a = len(ROOMS) - len(df.iloc[frame].unique())
    q = f"How many rooms were empty at step {frame + 1}?"
    return df, q, a, AnswerTypeNumber


# ### MMLong questions: ###


def q_room_empty(seq_len, fraction: float = 1, is_more: bool = None, **kwargs):
    """Which room was empty for {more/fewer} steps than the other rooms [between frames X and Y]?"""
    df = generate_sequence_df(seq_len)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, superlative=False, is_more=is_more
    )

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        room_non_visits = {r: 0 for r in ROOMS}
        for _, row in _df.iterrows():
            empty_rooms = sorted(set(ROOMS) - set(row.unique().tolist()))
            for r in empty_rooms:
                room_non_visits[r] += 1

        empty_counts = np.array(list(room_non_visits.values()))
        if np.sum(empty_counts == comp_fn(empty_counts)) > 1:
            return None
        return np.array(ROOMS)[empty_counts == comp_fn(empty_counts)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Which room was empty for {q_start} steps than the other rooms{q_end}"
    return df, q, a, AnswerTypeRoom


def q_where_spend(seq_len, fraction: float = 1, is_more: bool = None, **kwargs):
    """In which room did [Person] spend the {most/least amount of} time [between frames X and Y]?"""
    df, char, _, _, _, _ = get_random_situation(seq_len)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more
    )

    if (not is_more) and (len(ROOMS) - len(df.iloc[frame_0 : frame_1 + 1]) > 1):
        raise ValueError("It is impossible to choose the least visited room")

    def _check_df_return_answer(_df):
        visits = _df[char].value_counts()
        unvisited = sorted(set(ROOMS) - set(visits.index.tolist()))
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
            return
        return visits.index[0] if is_more else visits.index[-1]

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"In which room did {char} spend the {q_start} time{q_end}"
    return df, q, a, AnswerTypeRoom


def q_crowded_room(seq_len, fraction: float = 1, n_crowd: int = 3, **kwargs):
    """Which room was crowded ([three] or more people in one room) for the most steps [between frames X and Y]?"""
    df = generate_sequence_df(seq_len)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction)

    def _check_df_return_answer(_df):
        room_crowds = {r: 0 for r in ROOMS}
        for _, row in _df.iterrows():
            ur, urc = np.unique(row, return_counts=True)
            for r in ur[urc >= n_crowd]:
                room_crowds[r] += 1

        crowds = np.array(list(room_crowds.values()))
        if np.sum(crowds == np.max(crowds)) > 1:
            return
        return np.array(ROOMS)[crowds == np.max(crowds)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Which room was crowded ({n_crowd} or more people in one room) for the most steps{q_end}"
    return df, q, a, AnswerTypeRoom


def q_who_spend(seq_len, fraction: float = 1, is_more: bool = None, **kwargs):
    """Who spent the {most/least amount of} time in the [Room] [between frames X and Y]?"""
    df, _, _, room, _, _ = get_random_situation(seq_len)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more
    )

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        visit_counts = (_df == room).sum().values
        if np.sum(visit_counts == comp_fn(visit_counts)) > 1:
            return None
        return np.array(CHARS)[visit_counts == comp_fn(visit_counts)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Who spent the {q_start} time alone in the {room}{q_end}"
    return df, q, a, AnswerTypePerson


def q_spend_alone(seq_len, fraction: float = 1, is_more: bool = None, **kwargs):
    """Who spent the {most/least amount of} time alone in the rooms [between frames X and Y]?"""
    df = generate_sequence_df(seq_len)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more
    )

    if (not is_more) and (frame_1 - frame_0 < 2):
        raise ValueError("It is impossible to choose the loneliest char")

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        alone_chars = {c: 0 for c in CHARS}
        for _, row in _df.iterrows():
            ur, urc = np.unique(row, return_counts=True)
            for r in ur[urc == 1]:
                alone_chars[np.array(CHARS)[row == r].item()] += 1

        alone = np.array(list(alone_chars.values()))
        if np.sum(alone == comp_fn(alone)) > 1:
            return None
        return np.array(CHARS)[alone == comp_fn(alone)].item()

    a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])
    while a is None:
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"Who spent the {q_start} time alone in the rooms{q_end}"
    return df, q, a, AnswerTypePerson


def q_spend_together(seq_len, fraction: float = 1, is_more: bool = None, **kwargs):
    """With whom did [Person] spend the {most/least amount of} time together in the same room [...]?"""
    df, char, _, _, _, _ = get_random_situation(seq_len)
    is_more, q_start, frame_0, frame_1, q_end = get_random_mmlong(
        seq_len, fraction, is_more=is_more
    )

    def _check_df_return_answer(_df):
        comp_fn = np.max if is_more else np.min
        rest_chars = np.array(sorted(set(CHARS) - {char}))
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
        df = generate_sequence_df(seq_len)
        a = _check_df_return_answer(df.iloc[frame_0 : frame_1 + 1])

    q = f"With whom did {char} spend the {q_start} time together in the same room{q_end}"
    return df, q, a, AnswerTypePerson


def q_steps_in_room(seq_len, fraction: float = 1, **kwargs):
    """How many steps did [Person] spend in the [Room] [between frames X and Y]?"""
    df, char, _, room, _, _ = get_random_situation(seq_len)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction)
    a = np.sum(df.iloc[frame_0 : frame_1 + 1][char].values == room).item()
    q = f"How many steps did {char} spend in the {room}{q_end}"
    return df, q, a, AnswerTypeNumber


def q_rooms_visited(seq_len, fraction: float = 1, **kwargs):
    """How many different rooms did [Person] visit [between frames X and Y]?"""
    df, char, _, _, _, _ = get_random_situation(seq_len)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction)
    a = len(df.iloc[frame_0 : frame_1 + 1][char].unique())
    q = f"How many different rooms did {char} visit{q_end}"
    return df, q, a, AnswerTypeNumber


def q_crowd_count(seq_len, fraction: float = 1, n_crowd: int = 3, **kwargs):
    """How many times did a crowd ([three] or more people in one room) appear [between frames X and Y]?"""
    df = generate_sequence_df(seq_len)
    _, _, frame_0, frame_1, q_end = get_random_mmlong(seq_len, fraction)
    a = (
        df.iloc[frame_0 : frame_1 + 1]
        .apply(lambda x: np.sum(np.unique(x, return_counts=True)[1] >= n_crowd), axis=1)
        .sum()
        .item()
    )
    q = f"How many times did a crowd ({n_crowd} or more people in one room) appear{q_end}"
    return df, q, a, AnswerTypeNumber


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
}
