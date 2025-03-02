#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

# Constants
room_names = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
people_names = ["Nobody", "Daniel", "Mary", "Michael", "Sandra", "John"]


def serialize_sequence(sequence: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Gets input sequence as a list of dict-like mapping: {'character': 'location'}.

    Example:
    [
        {
            'Sandra': 'Kitchen',
            'Mary': 'Bathroom',
            'John': 'Office',
            'Daniel': 'Kitchen',
            'Michael': 'Bedroom'
        },
        ...
    ]

    Outputs a list of dicts like:
    {'frame_id': integer number of image, 'rooms': {'location': ['char1', 'char2' ...]}}

    Enumeration of steps starts from 1.
    Example:
    [
        {
            'frame_id': 1,
            'rooms': {
                "Kitchen": ["Sandra", "Daniel"],
                "Bathroom": ["Mary"],
                "Garden": [],
                "Office": ["John"],
                "Bedroom": ["Michael"],
                "Hallway": []
            }
        },
        ...
    ]
    """
    output_sequence = []

    def _parse_rooms_mapping(char2room: Dict[str, str]) -> Dict[str, List[str]]:
        room2chars = {room: [] for room in room_names}
        for char, room in char2room.items():
            room2chars[room].append(char)
        return room2chars

    for step_i, step_map in enumerate(sequence, start=1):
        output_sequence.append(
            {"step_id": step_i, "rooms": _parse_rooms_mapping(step_map)}
        )
    return output_sequence


def serialize_dataset(
    dataset_fn: str, data_path: str, output_fn: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Loads a dataset questions from `dataset_fn` and serialize to a sequence of mappings.
    Save to `output_fn` if provided.
    """
    with open(dataset_fn, "r") as f:
        dataset = json.load(f)

    result_dataset = []
    seqlen = dataset_fn.split("/")[-2]

    for sample in dataset:
        sequence_fn = os.path.join(data_path, seqlen, sample["sequence"])
        sample_sequences = pd.read_csv(sequence_fn).to_dict("records")
        serialized_sequence = serialize_sequence(sample_sequences)
        sample["sequence_json"] = serialized_sequence
        result_dataset.append(sample)

    if output_fn is None:
        output_fn = os.path.join(data_path, seqlen, "text_serialized_questions.json")

    with open(output_fn, "w") as f:
        json.dump(result_dataset, f)

    return result_dataset


def process_dataset_file(dataset_fn: Path, data_path: Path) -> List[Dict[str, Any]]:
    output_fn = data_path / (
        dataset_fn.parent.name + "_" + "text_serialized_questions.json"
    )
    print(f"Will be saved to:\n{output_fn}")
    text_dataset_part = serialize_dataset(
        str(dataset_fn), str(data_path), str(output_fn)
    )
    print(f"Done.\n")
    return text_dataset_part


def main(data_path: str):
    data_path = Path(data_path)
    dataset_files = list(data_path.glob("len_*/*.json"))
    print(f"Found {len(dataset_files)} dataset files.")

    resulting_serialized_dataset = []
    for dataset_fn in dataset_files:
        output_fn = data_path / (
            dataset_fn.parent.name + "_" + "text_serialized_questions.json"
        )
        print(f"Will be saved to:\n{output_fn}")

        text_dataset_part = serialize_dataset(
            str(dataset_fn), str(data_path), str(output_fn)
        )
        resulting_serialized_dataset.extend(text_dataset_part)

    output_fn = data_path / "all_text_serialized_questions.json"
    print(f"Will be saved to:\n{output_fn}")

    with open(output_fn, "w") as f:
        json.dump(resulting_serialized_dataset, f)

    # Check
    with open(output_fn, "r") as f:
        all_data = json.load(f)

    print(f"Total samples in the final dataset: {len(all_data)}")
    print("Sample data:")
    print(all_data[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serialize dataset to text format.")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset directory."
    )
    args = parser.parse_args()

    main(args.data_path)
