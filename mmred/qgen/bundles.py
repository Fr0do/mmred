"""Episode bundle generation: one shared sequence per episode, k target + filler questions."""

from __future__ import annotations

import random
from typing import Any, Callable

import pandas as pd

from ..config import DEFAULT_CHARS, DEFAULT_ROOMS
from ..data_model import (
    aggregate_metadata_global,
    aggregate_metadata_step,
    create_metadata_from_relevance,
    serialize_sequence,
)
from .questions import crowded_room_from_df, room_empty_from_df, spend_alone_at_time_from_df
from .utils import generate_sequence_df

FillerGen = Callable[
    [pd.DataFrame, int, list[str], list[str], random.Random],
    tuple[str, Any, str, dict[int, list[str]]] | None,
]

FILLER_REGISTRY: dict[str, FillerGen] = {
    "crowded_room": crowded_room_from_df,
    "room_empty": room_empty_from_df,
}


def _distribute_remainder_across_types(remainder: int, other_types: list[str]) -> dict[str, int]:
    m = len(other_types)
    if m == 0:
        if remainder != 0:
            raise ValueError("No non-target question types to distribute remainder into")
        return {}
    base = remainder // m
    extra = remainder % m
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(other_types)}


def select_target_frames(
    seq_len: int,
    k_target: int,
    target_step_strategy: str,
    rng: random.Random,
) -> list[int]:
    """Select zero-indexed frames for target questions."""
    if k_target < 0 or k_target > seq_len:
        raise ValueError("k_target must be in [0, seq_len]")
    if target_step_strategy == "prefix":
        return list(range(k_target))
    if target_step_strategy == "random":
        return sorted(rng.sample(range(seq_len), k_target))
    raise ValueError(
        f"Unknown target_step_strategy {target_step_strategy!r}; expected 'prefix' or 'random'"
    )


def _sample_to_dict(
    *,
    seq_len: int,
    qtype: str,
    atype: str,
    question: str,
    answer: Any,
    seq_df: pd.DataFrame,
    relevant_map: dict[int, list[str]],
    rooms: list[str],
    episode_id: str,
    is_target: bool,
    slot_index: int,
) -> dict[str, Any]:
    sequence = serialize_sequence(seq_df, rooms)
    metadata_list = create_metadata_from_relevance(relevant_map, seq_len, rooms)
    metadata = [m.to_dict() for m in metadata_list]
    n_per_step = aggregate_metadata_step(metadata_list)
    n_total = aggregate_metadata_global(metadata_list)
    return {
        "seq_len": seq_len,
        "qtype": qtype,
        "atype": atype,
        "question": question,
        "answer": answer,
        "sequence": [s.to_dict() for s in sequence],
        "metadata": metadata,
        "n_relevant_rooms_per_step": n_per_step,
        "n_relevant_rooms": n_total,
        "episode_id": episode_id,
        "is_target": is_target,
        "slot_index": slot_index,
    }


def try_build_episode_bundle(
    seq_len: int,
    k_target: int,
    bundle_size: int,
    target_question_type: str,
    question_types: list[str],
    episode_id: str,
    rng: random.Random,
    chars: list[str] | None = None,
    rooms: list[str] | None = None,
    max_episode_tries: int = 200,
    target_step_strategy: str = "prefix",
    target_rng: random.Random | None = None,
) -> list[dict[str, Any]] | None:
    """Build one episode: *bundle_size* questions on the same sequence (k target + fillers)."""
    chars = chars or list(DEFAULT_CHARS)
    rooms = rooms or list(DEFAULT_ROOMS)
    if target_question_type not in question_types:
        raise ValueError("target_question_type must be in question_types")
    others = [t for t in question_types if t != target_question_type]
    if not others:
        raise ValueError("question_types must include at least one non-target type")
    if k_target < 0 or k_target > bundle_size:
        raise ValueError(f"k_target must be in [0, {bundle_size}]")
    if k_target > seq_len:
        raise ValueError("k_target cannot exceed seq_len for spend_alone_at_step (one question per step index)")
    if bundle_size < 1:
        raise ValueError("bundle_size must be >= 1")
    target_frames = select_target_frames(
        seq_len, k_target, target_step_strategy, target_rng or rng
    )

    n_filler = bundle_size - k_target
    per_filler = _distribute_remainder_across_types(n_filler, others) if n_filler > 0 else {}

    if target_question_type != "spend_alone_at_step":
        raise ValueError(
            f"Unsupported target_question_type {target_question_type!r} (only 'spend_alone_at_step')"
        )

    for _ in range(max_episode_tries):
        df = generate_sequence_df(seq_len, chars=chars, rooms=rooms, rng=rng)
        rows: list[dict[str, Any]] = []
        slot = 0
        ok = True

        for frame in target_frames:
            is_more = rng.choice([True, False])
            q, a, atype, rm = spend_alone_at_time_from_df(df, frame, is_more, chars, rooms)
            rows.append(
                _sample_to_dict(
                    seq_len=seq_len,
                    qtype="spend_alone_at_step",
                    atype=atype,
                    question=q,
                    answer=a,
                    seq_df=df,
                    relevant_map=rm,
                    rooms=rooms,
                    episode_id=episode_id,
                    is_target=True,
                    slot_index=slot,
                )
            )
            slot += 1

        for ft, cnt in per_filler.items():
            gen = FILLER_REGISTRY.get(ft)
            if gen is None:
                raise ValueError(
                    f"No fixed-sequence filler generator for {ft!r}. Supported: {sorted(FILLER_REGISTRY)}"
                )
            for _j in range(cnt):
                out = gen(df, seq_len, chars, rooms, rng)
                if out is None:
                    ok = False
                    break
                fq, fa, fatype, frm = out
                rows.append(
                    _sample_to_dict(
                        seq_len=seq_len,
                        qtype=ft,
                        atype=fatype,
                        question=fq,
                        answer=fa,
                        seq_df=df,
                        relevant_map=frm,
                        rooms=rooms,
                        episode_id=episode_id,
                        is_target=False,
                        slot_index=slot,
                    )
                )
                slot += 1
            if not ok:
                break

        if ok and len(rows) == bundle_size:
            return rows
    return None


def generate_bundle_dataset(
    *,
    n_episodes: int,
    seq_len: int,
    k_target: int,
    bundle_size: int | None,
    target_question_type: str,
    question_types: list[str],
    seed: int,
    chars: list[str] | None = None,
    rooms: list[str] | None = None,
    target_step_strategy: str = "prefix",
) -> list[dict[str, Any]]:
    """
    Generate *n_episodes* episodes. Each episode has ``bundle_size`` questions on one sequence.

    ``bundle_size`` defaults to ``seq_len`` (fixed-L bundle: k target steps plus fillers).
    """
    resolved_bundle = bundle_size if bundle_size is not None else seq_len
    all_rows: list[dict[str, Any]] = []
    for ep in range(n_episodes):
        rng = random.Random(seed + ep * 1_000_003 + k_target * 17)
        target_rng = random.Random(seed + ep * 1_000_003 + k_target * 17 + 97_531)
        eid = f"{ep:05d}"
        bundle = try_build_episode_bundle(
            seq_len=seq_len,
            k_target=k_target,
            bundle_size=resolved_bundle,
            target_question_type=target_question_type,
            question_types=question_types,
            episode_id=eid,
            rng=rng,
            chars=chars,
            rooms=rooms,
            target_step_strategy=target_step_strategy,
            target_rng=target_rng,
        )
        if bundle is None:
            raise RuntimeError(
                f"Failed to build episode after max_episode_tries (ep={ep}, k_target={k_target})"
            )
        all_rows.extend(bundle)
    for i, row in enumerate(all_rows):
        row["qid"] = f"{i:07d}"
    return all_rows
