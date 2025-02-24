import concurrent.futures
import json
import time
from functools import partial
from pathlib import Path

import pandas as pd

from qgen.const import N_QUESTIONS, SEQ_LENGTHS, SEED
from qgen.questions import QUESTIONS
from qgen.visualization import seq2video
from qgen.utils import fix_seed, hash_seq_df


def _generate_question(q_fn, seq_len, q_hashes, **kwargs):
    seq, q, a, atype = q_fn(seq_len, **kwargs)
    h = hash_seq_df(seq)
    while h in q_hashes:
        seq, q, a, atype = q_fn(seq_len, **kwargs)
        h = hash_seq_df(seq)
    return seq, q, a, atype, h


def generate_questions(base_path, exp_name):
    fix_seed(SEED)

    if exp_name == "main":
        exp_path = Path(base_path) / exp_name
        q_id = 0

        for seq_len in SEQ_LENGTHS:
            dataset_path = exp_path / f"len_{seq_len}"
            q_dataset = []
            q_hashes = []

            for question_type, question_fn in QUESTIONS.items():

                q_kwargs = dict()
                if (question_type == "where_spend") and (seq_len <= 4):
                    q_kwargs["is_more"] = True
                elif (question_type == "spend_alone") and (seq_len <= 2):
                    q_kwargs["is_more"] = True

                for _ in range(N_QUESTIONS):
                    # TODO: add optional re-generation to stratify by answer
                    seq, q, a, atype, h = _generate_question(
                        question_fn, seq_len, q_hashes, **q_kwargs
                    )
                    q_hashes.append(h)
                    q_dataset.append(
                        {
                            "qid": f"{q_id:07d}",
                            "seq_len": seq_len,
                            "qtype": question_type,
                            "atype": atype,
                            "question": q,
                            "answer": a,
                            "sequence": f"sequences/seq_{q_id:07d}.csv",
                            "video": f"videos/vid_{q_id:07d}",
                        }
                    )
                    seq.to_csv(dataset_path / q_dataset[-1]["sequence"], index=False)
                    q_id += 1

            with open(str(dataset_path / "questions.json"), "w") as file:
                json.dump(q_dataset, file, indent=4)
            print(f"Finished question generation for len_{seq_len}")

    else:
        raise NotImplementedError('For now, only "main" exp is implemented')


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
