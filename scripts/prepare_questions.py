import os
import glob
import random
import pandas as pd
from tqdm.auto import tqdm
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
        df['Room'] = df["Room"].str.capitalize()
        all_sequences[file_name] = df

    return all_sequences


def generate_questions_and_answers(df):
    """
    Generate a list of Q&A dictionaries for a given DataFrame `df`.
    The DataFrame is assumed to have columns of characters,
    and rows indexed by step. Each cell is the room that the character is in at that step.
    """

    characters = df.columns.tolist()
    rooms = np.unique(np.concatenate(df.apply(pd.unique).values))
    
    def first_time():
        """
        e.g. "In which room did X first appear?"
        Picks the earliest step for a random character and returns that room.
        """
        character = random.choice(characters)
        record = df[character].iloc[:1]  # first row for that character
        step, room = record.index[0], record.values[0]
        return {
            "Type": "first_appearance_room",
            "Question": f"In which room did {character} first appear?",
            "Answer": room,
            "Supporting_Steps": [int(step)],
        }

    def first_time_count_other_room():
        """
        e.g. "How many characters were in room R2 when X first appeared in room R1?"
        Picks the first time the character appears in a room, then checks how many others
        were in a different room at that exact step.
        """
        character = random.choice(characters)
        # pick the first time the character entered each distinct room, then sample one
        record = df[character].drop_duplicates(keep='first').sample(1)
        step, room_entered = record.index[0], record.values[0]
        room_other = random.choice(rooms)
        count_other = (df.loc[step] == room_other).sum()
        return {
            "Type": "first_appearance_room_count",
            "Question": f"How many characters were in {room_other} when {character} first appeared in {room_entered}?",
            "Answer": int(count_other),
            "Supporting_Steps": [int(step)],
        }

    def first_time_other_char():
        """
        e.g. "In which room was Y when X first appeared in room R?"
        """
        character = random.choice(characters)
        record = df[character].drop_duplicates(keep='first').sample(1)
        step, room = record.index[0], record.values[0]
        character_other = random.choice(list(set(characters) - {character}))
        room_other_char = df[character_other].loc[step]
        return {
            "Type": "first_appearance_other_char",
            "Question": f"In which room was {character_other} when {character} first appeared in {room}?",
            "Answer": room_other_char,
            "Supporting_Steps": [int(step)],
        }

    def last_time():
        """
        e.g. "In which room was X found at the final step?"
        Takes the last row for a random character.
        """
        character = random.choice(characters)
        record = df[character].iloc[-1:]
        step, room = record.index[0], record.values[0]
        return {
            "Type": "final_room",
            "Question": f"In which room was {character} found at the final step?",
            "Answer": room,
            "Supporting_Steps": [int(step)],
        }
    
    def last_time_count_other_room():
        """
        e.g. "How many characters were in room R2 when X made their final appearance in R1?"
        Picks the last time the character entered a specific room, then counts how many
        were in another room at that step.
        """
        character = random.choice(characters)
        record = df[character].drop_duplicates(keep='last').sample(1)
        step, room_entered = record.index[0], record.values[0]
        room_other = random.choice(rooms)
        count_other = (df.loc[step] == room_other).sum()
        return {
            "Type": "final_room_count",
            "Question": f"How many characters were in {room_other} when {character} made their final appearance in {room_entered}?",
            "Answer": int(count_other),
            "Supporting_Steps": [int(step)],
        }

    def last_time_other_char():
        """
        e.g. "Which room was Y in when X made their final appearance in room R?"
        """
        character = random.choice(characters)
        record = df[character].drop_duplicates(keep='last').sample(1)
        step, room = record.index[0], record.values[0]
        character_other = random.choice(list(set(characters) - {character}))
        room_other_char = df.loc[step][character_other]
        return {
            "Type": "final_appearance_other_char",
            "Question": f"In which room was {character_other} when {character} made their final appearance in {room}?",
            "Answer": room_other_char,
            "Supporting_Steps": [int(step)],
        }

    def count_visits():
        """
        e.g. "How many separate times did X enter room R?"
        Finds how many times the character transitions into that room.
        """
        character = random.choice(characters)
        person = df[character]
        # Steps where the character "arrives" in a new room (comparing to the next step)
        # Another approach is to compare person != person.shift(1), etc.
        stats = person.loc[person.shift(-1) != person]
        if len(stats) == 0:
            # If we can't find transitions, fallback
            return None
        random_stat = stats.value_counts().sample(1)
        room, count = random_stat.index[0], random_stat.values[0]
        support_steps = stats[stats == room].index.astype(int).tolist()
        return {
            "Type": "visit_count",
            "Question": f"How many separate times did {character} enter room {room}?",
            "Answer": int(count),
            "Supporting_Steps": support_steps,
        }

    def changed_rooms_between_two_steps():
        """
        e.g. "Between step S and step S+1, which characters moved to a different room?"
        """
        # Pick a random step that has a "next" step
        if len(df.index) == 1:
            return {
                "Type": "compare_two_steps",
                "Question": f"Which characters moved to a different room?",
                "Answer": "Nobody",
                "Supporting_Steps": [1],
            }
        step, next_step = sorted(random.sample(df.index.tolist(), 2))

        row_current = df.loc[step]
        row_next = df.loc[next_step]

        changed_chars = []
        for char in characters:
            if row_current[char] != row_next[char]:
                changed_chars.append(char)

        answer = changed_chars if changed_chars else "Nobody"
        return {
            "Type": "compare_two_steps",
            "Question": f"Comparing step {step} and step {next_step}, which characters moved to a different room?  List characters or answer 'Nobody'.",
            "Answer": answer,
            "Supporting_Steps": [int(step), int(next_step)],
        }

    def chars_in_room_at_step():
        """
        e.g. "Who was in room R at step S?"
        """
        if len(df.index) == 0:
            return None
        step = random.choice(df.index)
        room_choice = random.choice(rooms)

        row = df.loc[step]
        # find which characters are in that room
        chars_in_that_room = row[row == room_choice].index.tolist()

        answer = chars_in_that_room if chars_in_that_room else "Nobody"
        return {
            "Type": "list_chars_in_room_at_step",
            "Question": f"Who was in room {room_choice} at step {step}? List characters or answer 'Nobody'.",
            "Answer": answer,
            "Supporting_Steps": [int(step)],
        }

    def how_many_with_char_at_step():
        """
        e.g. "How many characters were in the same room as X at step S?"
        """
        if len(df.index) == 0:
            return None
        step = random.choice(df.index)
        char = random.choice(characters)
        row = df.loc[step]
        room_of_char = row[char]
        # how many characters in the same room as 'char'
        same_room_count = (row == room_of_char).sum()
        return {
            "Type": "count_chars_with_char_at_step",
            "Question": f"How many characters were in the same room as {char} at step {step}?",
            "Answer": int(same_room_count),
            "Supporting_Steps": [int(step)],
        }

    def one_char_with_char_at_step():
        """
        e.g. "Name any other character who was also in the same room as X at step S."
        """
        if len(df.index) == 0:
            return None
        step = random.choice(df.index)
        char = random.choice(characters)
        row = df.loc[step]
        room_of_char = row[char]

        # find other characters in the same room
        others = [c for c in characters if c != char and row[c] == room_of_char]
        if not others:
            answer = "Nobody"
        else:
            answer = random.choice(others)
        return {
            "Type": "name_char_with_char_at_step",
            "Question": f"List other characters who were in the same room as {char} at step {step} or answer 'Nobody'.",
            "Answer": answer,
            "Supporting_Steps": [int(step)],
        }

    def most_time_in_room():
        """
        e.g. "Which character spent the most time overall in room R?"
        """
        if len(rooms) == 0:
            return None
        room_choice = random.choice(rooms)

        # Count the number of steps each character spent in that room
        time_in_room = {}
        for char in characters:
            time_in_room[char] = (df[char] == room_choice).sum()
        
        # find the character(s) with the maximum time
        max_time = max(time_in_room.values()) if time_in_room else 0
        # all chars with that count
        top_chars = [char for char, cnt in time_in_room.items() if cnt == max_time]

        if not top_chars:
            return {
                "Type": "most_time_in_room",
                "Question": f"Which character spent the most time (seen in the most steps) overall in room {room_choice}?",
                "Answer": "Nobody",
                "Supporting_Steps": [-1],
            }

        answer_char = random.choice(top_chars)
        return {
            "Type": "most_time_in_room",
            "Question": f"Which character spent the most time (seen in the most steps) overall in room {room_choice}?",
            "Answer": answer_char,
            "Supporting_Steps": [-1],  # Could include all steps but might be large.
        }

    question_generators = [
        first_time,
        first_time_count_other_room,
        first_time_other_char,
        last_time,
        last_time_count_other_room,
        last_time_other_char,
        count_visits,
        changed_rooms_between_two_steps,
        chars_in_room_at_step,
        how_many_with_char_at_step,
        one_char_with_char_at_step,
        most_time_in_room,
    ]

    qa_records = []
    for func in question_generators:
        result = func()
        # Some question generators might return None if they cannot produce a valid Q&A
        if result is not None:
            qa_records.append(result)

    return qa_records


def generate_all_sequences():
    # 1. Load all sequences
    all_sequences = load_sequence_data(folder="data/length_128")
    
    # 2. Step sizes
    step_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    # We collect rows for output CSV
    output_rows = []

    # 3. For each file, for each step size, generate Q&A
    for file_name, df in tqdm(all_sequences.items()):
        seq_id = os.path.splitext(file_name)[0]  # e.g. "sequence_000"

        df_static = df.pivot(index="Step", columns="Character", values="Room").ffill()
        for sz in step_sizes:
            truncated_df = df_static[df_static.index <= sz]
            qa_list = generate_questions_and_answers(truncated_df)
            
            for qa in qa_list:
                output_rows.append({
                    "Seq_id": seq_id,
                    "N_steps": sz,
                    **qa,
                })

    # 4. Save to test_data.csv
    df_out = pd.DataFrame(output_rows, columns=["Seq_id","N_steps","Type","Question","Answer", "Supporting_Steps"])
    df_out.to_csv("data/new_test_data.csv", index=False)


if __name__ == "__main__":
    generate_all_sequences()
