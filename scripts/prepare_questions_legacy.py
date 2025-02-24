import os
import glob
import random
import pandas as pd
from collections import defaultdict
import numpy as np


def load_sequence_data(folder="data"):
    """
    Loads all CSV files from 'folder' matching 'sequence_*.csv'
    and returns a dict of DataFrames keyed by filename.
    Each DataFrame is assumed to have columns: [Step, Character, Room].
    """
    data_files = sorted(glob.glob(os.path.join(folder, "sequence_*.csv")))
    all_sequences = {}

    for file_path in data_files:
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        expected_cols = ["Step", "Character", "Room"]
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"File {file_path} must have columns {expected_cols}.")

        df = df.sort_values(by="Step").reset_index(drop=True)
        all_sequences[file_name] = df

    return all_sequences


def generate_questions_and_answers(df):
    characters = df.columns.tolist()
    rooms = np.unique(np.concatenate(df.apply(pd.unique).values))

    def first_time():
        character = random.choice(characters)
        record = df[character].iloc[:1]
        step, room = record.index[0], record.values[0]
        return {
            "Type": "first_time",
            "Question": f"Which room was {character} in for the first time?",
            "Answer": room,
            "Supporting_Steps": [step],
        }

    def first_time_count_other_room():
        character = random.choice(characters)
        record = df[character].drop_duplicates(keep="first").sample(1)
        step, room_orig = record.index[0], record.values[0]
        room_other = random.choice(rooms)
        count_other = (df.iloc[step] == room_other).sum()
        return {
            "Type": "first_time_count_other_room",
            "Question": f"How many people were in {room_other} when {character} first entered {room_orig}?",
            "Answer": count_other,
            "Supporting_Steps": [step],
        }

    def first_time_other_char():
        character = random.choice(characters)
        record = df[character].drop_duplicates(keep="first").sample(1)
        step, room = record.index[0], record.values[0]
        character_other = random.choice(list(set(characters) - {character}))
        room_other = df[character_other].iloc[step]
        return {
            "Type": "first_time_other_char",
            "Question": f"What room was {character_other} in when {character} first entered {room}?",
            "Answer": room_other,
            "Supporting_Steps": [step],
        }

    def last_time():
        character = random.choice(characters)
        record = df.iloc[-1:][character]
        step, room = record.index[0], record.values[0]
        return {
            "Type": "last_time",
            "Question": f"Which room was {character} in last time?",
            "Answer": room,
            "Supporting_Steps": [step],
        }

    def last_time_count_other_room():
        character = random.choice(characters)
        record = df[character].drop_duplicates(keep="last").sample(1)
        step, room_orig = record.index[0], record.values[0]
        room_other = random.choice(rooms)
        count_other = (df.iloc[step] == room_other).sum()
        return {
            "Type": "last_time_count_other_room",
            "Question": f"How many people were in {room_other} when {character} last entered {room_orig}?",
            "Answer": count_other,
            "Supporting_Steps": [step],
        }

    def last_time_other_char():
        character = random.choice(characters)
        record = df[character].drop_duplicates(keep="last").sample(1)
        step, room = record.index[0], record.values[0]
        character_other = random.choice(list(set(characters) - {character}))
        room_other = df[character_other].iloc[step]
        return {
            "Type": "last_time_other_char",
            "Question": f"What room was {character_other} in when {character} first entered {room}?",
            "Answer": room_other,
            "Supporting_Steps": [step],
        }

    def count_visits():
        character = random.choice(characters)
        person = df[character]
        stats = person.loc[person.shift(-1) != person]
        random_stat = stats.value_counts().sample(1)
        room, count = random_stat.index[0], random_stat.values[0]
        return {
            "Type": "count_all",
            "Question": f"How many times did {character} visit {room}?",
            "Answer": count,
            "Supporting_Steps": stats[stats == room].index.astype(int).tolist(),
        }

    question_types = [
        first_time,
        first_time_count_other_room,
        first_time_other_char,
        last_time,
        last_time_count_other_room,
        last_time_other_char,
        count_visits,
    ]

    qa_records = []

    for make_func in question_types:
        qa_records.append(make_func())
    return qa_records


def generate_all_sequences():
    # 1. Load all sequences
    all_sequences = load_sequence_data(folder="data/length_128")

    # 2. Step sizes
    step_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    # We collect rows for output CSV
    output_rows = []

    # 3. For each file, for each step size, generate Q&A
    for file_name, df in all_sequences.items():
        seq_id = os.path.splitext(file_name)[0]  # e.g. "sequence_000"

        df_static = df.pivot(index="Step", columns="Character", values="Room").ffill()
        for sz in step_sizes:
            truncated_df = df_static.iloc[df_static.index < sz]
            qa_list = generate_questions_and_answers(truncated_df)

            for qa in qa_list:
                output_rows.append(
                    {
                        "Seq_id": seq_id,
                        "N_steps": sz,
                        **qa,
                    }
                )

    # 4. Save to test_data.csv
    df_out = pd.DataFrame(
        output_rows,
        columns=["Seq_id", "N_steps", "Type", "Question", "Answer", "Supporting_Steps"],
    )
    df_out.to_csv("data/new_test_data.csv", index=False)


if __name__ == "__main__":
    generate_all_sequences()
