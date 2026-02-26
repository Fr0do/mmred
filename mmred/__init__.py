"""MMReD - Multi-Modal Reasoning in Dense Context.

A library for generating and working with the MMReD benchmark for
dense context reasoning evaluation.
"""

from .config import GenerationConfig
from .data_model import (
    Sample,
    Step,
    MetadataStep,
    serialize_sequence,
    aggregate_metadata_step,
    aggregate_metadata_global,
)
from .qgen.qgen import generate_questions, generate_questions_sequential, save_dataset
from .qgen.questions import QUESTIONS

__all__ = [
    "GenerationConfig",
    "Sample",
    "Step",
    "MetadataStep",
    "serialize_sequence",
    "aggregate_metadata_step",
    "aggregate_metadata_global",
    "generate_questions",
    "generate_questions_sequential",
    "save_dataset",
    "QUESTIONS",
]

__version__ = "2.0.0"
