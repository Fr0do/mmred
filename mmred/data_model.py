"""Data model for the MMReD benchmark.

This module defines the core data structures used to represent generated samples,
sequences, and metadata for the MMReD benchmark dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Step:
    """A single step in a room-occupancy sequence.

    Attributes:
        step_id: 1-indexed step number within the sequence.
        rooms: Mapping from room name to the list of characters present in that room.
    """

    step_id: int
    rooms: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "step_id": self.step_id,
            "rooms": {room: list(chars) for room, chars in self.rooms.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        """Deserialize from a dictionary."""
        return cls(step_id=d["step_id"], rooms={k: list(v) for k, v in d["rooms"].items()})


@dataclass
class MetadataStep:
    """Relevance annotation for a single step.

    Attributes:
        step_id: 1-indexed step number within the sequence.
        rooms: Mapping from room name to a boolean indicating whether that room
               at this step was relevant for computing the answer.
    """

    step_id: int
    rooms: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "step_id": self.step_id,
            "rooms": {room: bool(flag) for room, flag in self.rooms.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetadataStep":
        """Deserialize from a dictionary."""
        return cls(step_id=d["step_id"], rooms={k: bool(v) for k, v in d["rooms"].items()})


@dataclass
class Sample:
    """A fully-specified benchmark sample.

    Attributes:
        qid: Unique question ID (7-digit, zero-padded string).
        seq_len: Number of steps in the sequence.
        qtype: Question-type identifier (e.g. "spend_alone", "first_app").
        atype: Answer type – one of "person", "room", or "number".
        question: Question text shown to the model.
        answer: Ground-truth answer (str for person/room, int for number).
        sequence: Sequence of room-occupancy steps.
        metadata: Per-step room-relevance annotations.
    """

    qid: str
    seq_len: int
    qtype: str
    atype: str
    question: str
    answer: str | int
    sequence: list[Step] = field(default_factory=list)
    metadata: list[MetadataStep] = field(default_factory=list)
    #: Per-step sum of relevant rooms (same length as ``metadata``).
    n_relevant_rooms_per_step: list[int] = field(default_factory=list)
    #: Total number of relevant room-step pairs across the whole sequence.
    n_relevant_rooms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "qid": self.qid,
            "seq_len": self.seq_len,
            "qtype": self.qtype,
            "atype": self.atype,
            "question": self.question,
            "answer": self.answer,
            "sequence": [s.to_dict() for s in self.sequence],
            "metadata": [m.to_dict() for m in self.metadata],
            "n_relevant_rooms_per_step": list(self.n_relevant_rooms_per_step),
            "n_relevant_rooms": self.n_relevant_rooms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sample":
        """Deserialize from a dictionary."""
        metadata = [MetadataStep.from_dict(m) for m in d.get("metadata", [])]
        return cls(
            qid=d["qid"],
            seq_len=d["seq_len"],
            qtype=d["qtype"],
            atype=d["atype"],
            question=d["question"],
            answer=d["answer"],
            sequence=[Step.from_dict(s) for s in d.get("sequence", [])],
            metadata=metadata,
            n_relevant_rooms_per_step=(
                d["n_relevant_rooms_per_step"]
                if "n_relevant_rooms_per_step" in d
                else aggregate_metadata_step(metadata)
            ),
            n_relevant_rooms=(
                d["n_relevant_rooms"]
                if "n_relevant_rooms" in d
                else aggregate_metadata_global(metadata)
            ),
        )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def serialize_sequence(seq_df: pd.DataFrame, rooms: list[str]) -> list[Step]:
    """Convert a sequence DataFrame to a list of :class:`Step` objects.

    The DataFrame is expected to have:
    - Row index corresponding to step positions (0-indexed).
    - Column names that are character names.
    - Cell values that are room names (where each character is located).

    Args:
        seq_df: DataFrame with shape ``(seq_len, n_chars)`` where each cell
                contains the name of the room occupied by that character at
                that step.
        rooms: Ordered list of all room names used in the benchmark (used to
               ensure a consistent key order in the output dictionaries).

    Returns:
        A list of :class:`Step` objects, one per row of the DataFrame.
    """
    steps: list[Step] = []
    chars = list(seq_df.columns)

    for step_idx, row in enumerate(seq_df.itertuples(index=False)):
        room_occupants: dict[str, list[str]] = {room: [] for room in rooms}
        for char, room in zip(chars, row):
            if room in room_occupants:
                room_occupants[room].append(char)
        steps.append(Step(step_id=step_idx + 1, rooms=room_occupants))

    return steps


def create_metadata_from_relevance(
    relevant_map: dict[int, list[str]],
    seq_len: int,
    rooms: list[str],
) -> list[MetadataStep]:
    """Build a :class:`MetadataStep` list from a relevance map.

    Args:
        relevant_map: Mapping from 1-indexed step ID to the list of room names
                      that are relevant at that step.
        seq_len: Total number of steps in the sequence.
        rooms: Ordered list of all room names.

    Returns:
        A list of :class:`MetadataStep` objects, one per step.
    """
    metadata: list[MetadataStep] = []
    for step_id in range(1, seq_len + 1):
        relevant_rooms = relevant_map.get(step_id, [])
        room_relevance = {room: room in relevant_rooms for room in rooms}
        metadata.append(MetadataStep(step_id=step_id, rooms=room_relevance))
    return metadata


def aggregate_metadata_step(metadata: list[MetadataStep]) -> list[int]:
    """Compute per-step relevance counts from a metadata list.

    For each step, sums the number of rooms whose relevance flag is ``True``.

    Args:
        metadata: List of :class:`MetadataStep` objects, one per step.

    Returns:
        A list of integers of the same length as *metadata*; element *i* is
        the number of relevant rooms at step *i+1*.
    """
    return [sum(flag for flag in step.rooms.values()) for step in metadata]


def aggregate_metadata_global(metadata: list[MetadataStep]) -> int:
    """Compute the total number of relevant room-step pairs.

    Args:
        metadata: List of :class:`MetadataStep` objects.

    Returns:
        Scalar integer — the sum of all ``True`` flags across every step and
        every room in *metadata*.
    """
    return sum(aggregate_metadata_step(metadata))
