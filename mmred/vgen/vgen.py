import concurrent.futures
import json
import time
from functools import partial
from pathlib import Path

import pandas as pd

from .visualization import seq2video
from ..const import SEQ_LENGTHS


def _dataset_entry_to_video(entry, dataset_path):
    seq = pd.read_csv(dataset_path / entry["sequence"])
    video_path = dataset_path / entry["video"]
    seq2video(seq, video_path)


def generate_videos(base_path, exp_name):
    exp_path = Path(base_path) / exp_name

    for seq_len in SEQ_LENGTHS:
        dataset_path = exp_path / f"len_{seq_len}"

        with open(str(dataset_path / "questions.json"), "r") as file:
            q_dataset = json.load(file)

        st = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(
                partial(_dataset_entry_to_video, dataset_path=dataset_path), q_dataset
            )
        et = time.time()
        print(
            f"Finished videos generation for len_{seq_len}. Elapsed time: {et - st:.1f} s"
        )
