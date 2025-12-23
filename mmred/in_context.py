import json
from pathlib import Path
from typing import List, Sequence

from .const import SEED
from .qgen.qgen import _generate_question, QUESTIONS
from .qgen.utils import fix_seed


def _serialize_sequence(sequence_df) -> List[dict]:
    """Serialize a pandas DataFrame sequence into a JSON-friendly list of dicts."""
    from .const import ROOMS

    serialized_steps: List[dict] = []
    for step_idx, (_, frame) in enumerate(sequence_df.iterrows(), start=1):
        rooms = {room: [] for room in ROOMS}
        for character, location in frame.items():
            rooms[location].append(character)

        serialized_steps.append({"step_id": step_idx, "rooms": rooms})

    return serialized_steps

def generate_in_context_examples(
    base_path: str,
    exp_name: str,
    n_examples_per_task: int = 5,
    seq_lengths: Sequence[int] = (1, 2, 4, 8, 16),
    overwrite: bool = True,
) -> Path:
    """Generate a compact in-context dataset for few-shot prompts.

    The resulting file is stored as ``in_context_examples.json`` under the
    experiment directory and contains serialized sequences, questions, and
    answers across the provided ``seq_lengths``.
    """

    fix_seed(SEED)
    exp_path = Path(base_path) / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    output_path = exp_path / "in_context_examples.json"
    if output_path.exists() and not overwrite:
        return output_path

    examples: List[dict] = []

    for seq_len in seq_lengths:
        for question_type, question_fn in QUESTIONS.items():
            q_kwargs = dict()
            if (question_type == "where_spend") and (seq_len <= 4):
                q_kwargs["is_more"] = True
            elif (question_type == "spend_alone") and (seq_len <= 2):
                q_kwargs["is_more"] = True

            seen_hashes: List[str] = []
            for ex_idx in range(n_examples_per_task):
                sequence_df, question, answer, atype, seq_hash = _generate_question(
                    question_fn, seq_len, seen_hashes, **q_kwargs
                )
                seen_hashes = [*seen_hashes, seq_hash]
                examples.append(
                    {
                        "example_id": f"ctx_{seq_len}_{question_type}_{ex_idx}",
                        "seq_len": seq_len,
                        "qtype": question_type,
                        "atype": atype,
                        "question": question,
                        "answer": answer,
                        "sequence_json": _serialize_sequence(sequence_df),
                    }
                )

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    return output_path

