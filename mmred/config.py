"""Configuration for MMReD dataset generation.

This module provides a flexible configuration system for the MMReD benchmark
generator, replacing the static const.py values with a dynamic dataclass-based
configuration.
"""

from dataclasses import dataclass, field
from typing import Sequence


# Default values (backward compatible with old const.py)
DEFAULT_SEED = 0xBADFACE
DEFAULT_SEQ_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_N_QUESTIONS = 50
DEFAULT_ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
DEFAULT_CHARS = ["Sandra", "Mary", "John", "Daniel", "Michael"]


@dataclass
class GenerationConfig:
    """Configuration for MMReD dataset generation.
    
    Attributes:
        seed: Random seed for reproducibility
        seq_lengths: List of sequence lengths to generate
        n_questions: Number of questions per question type per sequence length
        question_types: List of question types to include (None = all)
        rooms: List of room names
        chars: List of character names
        target_keyframes: Control for relevant step count:
            - None: random (default behavior)
            - int: exact number of keyframes
            - tuple[int, int]: range of keyframes (min, max)
            - list[int]: specific keyframe counts to generate (one sample per count)
        n_questions_per_type: If set, number of questions per question type per
            seq_len (types with 0 are skipped). When None, use uniform n_questions.
    """
    
    seed: int = DEFAULT_SEED
    seq_lengths: list[int] = field(default_factory=lambda: DEFAULT_SEQ_LENGTHS.copy())
    n_questions: int = DEFAULT_N_QUESTIONS
    question_types: list[str] | None = None
    rooms: list[str] = field(default_factory=lambda: DEFAULT_ROOMS.copy())
    chars: list[str] = field(default_factory=lambda: DEFAULT_CHARS.copy())
    target_keyframes: int | tuple[int, int] | list[int] | None = None
    n_questions_per_type: dict[str, int] | None = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_questions_per_type is None:
            if self.n_questions < 1:
                raise ValueError("n_questions must be at least 1")
        else:
            if self.question_types is None:
                raise ValueError(
                    "n_questions_per_type requires an explicit question_types list"
                )
            qt_set = set(self.question_types)
            per_keys = set(self.n_questions_per_type.keys())
            if qt_set != per_keys:
                raise ValueError(
                    "n_questions_per_type keys must exactly match question_types: "
                    f"question_types={sorted(qt_set)}, dict keys={sorted(per_keys)}"
                )
            for q, n in self.n_questions_per_type.items():
                if n < 0:
                    raise ValueError(f"n_questions_per_type[{q!r}] must be >= 0")
            if sum(self.n_questions_per_type.values()) < 1:
                raise ValueError("n_questions_per_type must have at least one positive count")
        if not self.seq_lengths:
            raise ValueError("seq_lengths cannot be empty")
        if not self.rooms:
            raise ValueError("rooms cannot be empty")
        if not self.chars:
            raise ValueError("chars cannot be empty")
        if self.target_keyframes is not None:
            if isinstance(self.target_keyframes, int):
                if self.target_keyframes < 1:
                    raise ValueError("target_keyframes must be at least 1")
            elif isinstance(self.target_keyframes, tuple):
                if len(self.target_keyframes) != 2:
                    raise ValueError("target_keyframes tuple must have exactly 2 elements")
                if self.target_keyframes[0] > self.target_keyframes[1]:
                    raise ValueError("target_keyframes range must be (min, max)")
            elif isinstance(self.target_keyframes, list):
                if not all(k >= 1 for k in self.target_keyframes):
                    raise ValueError("All target_keyframes values must be at least 1")
    
    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        """Create a configuration from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            "seed": self.seed,
            "seq_lengths": self.seq_lengths,
            "n_questions": self.n_questions,
            "question_types": self.question_types,
            "rooms": self.rooms,
            "chars": self.chars,
            "target_keyframes": self.target_keyframes,
            "n_questions_per_type": self.n_questions_per_type,
        }
    
    def get_question_types(self, available_types: Sequence[str]) -> list[str]:
        """Get the list of question types to generate.
        
        Args:
            available_types: All available question types
            
        Returns:
            List of question types to generate
        """
        if self.question_types is None:
            return list(available_types)
        
        # Validate that requested types exist
        unknown = set(self.question_types) - set(available_types)
        if unknown:
            raise ValueError(f"Unknown question types: {unknown}")
        
        return self.question_types

    def n_for_question_type(self, qtype: str) -> int:
        """Number of questions to generate for this type at each seq_len."""
        if self.n_questions_per_type is not None:
            return self.n_questions_per_type[qtype]
        return self.n_questions
