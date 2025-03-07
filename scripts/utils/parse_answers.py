import pandas as pd
import json
import os
import re
import glob
from ast import literal_eval
from pydantic import BaseModel, ValidationError
from typing import Optional, Set, Literal, Iterable
from json_repair import repair_json
import argparse
import multiprocessing as mp
from functools import partial

# Define the Pydantic models
room_names = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
people_names = ["Nobody", "Daniel", "Mary", "Michael", "Sandra", "John"]

# Define the regex pattern
all_names = "|".join(room_names + people_names + ["\\d"])
answer_pattern = rf"({all_names})"

# Add a pattern for numeric answers with garbage
numeric_pattern = r"[<>'\"`,.]*(\d+)"


class RoomAnswer(BaseModel):
    reasoning: Optional[str]
    answer: Literal[*room_names]


class NumberAnswer(BaseModel):
    reasoning: Optional[str]
    answer: int


class PersonAnswer(BaseModel):
    reasoning: Optional[str]
    answer: Set[Literal[*people_names]]


# Function to safely evaluate a string into a Python object
def evaluate_string(s):
    try:
        return literal_eval(s)
    except (ValueError, SyntaxError):
        # Check if it's a numeric answer with garbage
        if isinstance(s, str):
            numeric_match = re.search(numeric_pattern, s)
            if numeric_match:
                return int(numeric_match.group(1))
        return s  # Return the string as-is if evaluation fails


# Function to infer the correct Pydantic model based on the ground truth 'Answer'
def infer_model(answer):
    if isinstance(answer, int):
        return NumberAnswer
    elif isinstance(answer, str) and answer in room_names:
        return RoomAnswer
    elif isinstance(answer, (list, set)) and all(
        person in people_names for person in answer
    ):
        return PersonAnswer
    elif isinstance(answer, str) and answer in people_names:
        return PersonAnswer
    else:
        raise ValueError(f"Unable to infer the correct model for the answer: {answer}")


# Function to normalize the answer (convert lists or single items to sets for PersonAnswer)
def normalize_answer(answer, model):
    if model == PersonAnswer:
        if isinstance(answer, (str, int)):
            return {answer}  # Convert single item to a set
        elif isinstance(answer, list):
            return set(answer)  # Convert list to a set
        elif isinstance(answer, set):
            return answer  # Already a set
    if model == NumberAnswer:
        if isinstance(answer, (str, Iterable)) and "Nobody" in answer:
            answer = 0
        # Handle the case where answer is a string with a number
        elif isinstance(answer, str):
            numeric_match = re.search(numeric_pattern, answer)
            if numeric_match:
                answer = int(numeric_match.group(1))
    return answer  # No normalization needed for other models


# Function to validate and compare answers
def validate_and_compare(row):
    try:
        # Evaluate the ground truth 'Answer' string into a Python object
        evaluated_answer = evaluate_string(row["answer"])

        # Infer the model based on the evaluated ground truth 'Answer'
        model = infer_model(evaluated_answer)

        # Normalize the evaluated ground truth 'Answer'
        normalized_answer = normalize_answer(evaluated_answer, model)

        # Validate the ground truth 'Answer'
        validated_answer = model(reasoning=None, answer=normalized_answer)

        # Evaluate the 'Predicted_Answer' string into a Python object
        evaluated_prediction = evaluate_string(row["Final_Answer"])

        # Normalize the evaluated 'Predicted_Answer'
        normalized_prediction = normalize_answer(evaluated_prediction, model)

        # If this is a numeric answer, try to extract a number if it failed earlier
        if model == NumberAnswer and not isinstance(normalized_prediction, int):
            if isinstance(normalized_prediction, str):
                numeric_match = re.search(r"\d+", normalized_prediction)
                if numeric_match:
                    normalized_prediction = int(numeric_match.group(0))

        # Validate the 'Predicted_Answer'
        validated_prediction = model(reasoning=None, answer=normalized_prediction)

        # Compare the validated answers
        return validated_answer.answer == validated_prediction.answer
    except ValidationError as e:
        return False
    except ValueError as e:
        return False
    except Exception as e:
        return False


# Function to parse and clean the predicted answer
def parse_predicted_answer(predicted_answer):
    try:
        parsed_answer = json.loads(repair_json(predicted_answer))
        assert isinstance(parsed_answer, dict)
        parsed_answer = parsed_answer.get("answer", "None")
        if "no" in str(parsed_answer).lower():
            parsed_answer = {"Nobody"}
        if isinstance(parsed_answer, list):
            if len(parsed_answer) == 1:
                parsed_answer = {parsed_answer[0]}
            else:
                parsed_answer = {"Nobody"}
        return parsed_answer
    except Exception:
        # Try to extract a number if it looks like a numeric answer
        if isinstance(predicted_answer, str):
            numeric_match = re.search(numeric_pattern, predicted_answer)
            if numeric_match:
                return int(numeric_match.group(1))

        # Otherwise try the original pattern
        match = re.search(answer_pattern, predicted_answer, flags=re.IGNORECASE)
        return match[0] if match else "None"


def strip_until_first_brace(string):
    # Find the position of "</think>"
    think_end = string.find("</think>")

    # If "</think>" is not found, return the original string
    if think_end == -1:
        return string
    else:
        string = string[think_end:]
    string = string.replace("</answer>", "").replace("<answer>", "").strip()
    # Find the position of the first "{" after "</think>"
    brace_start = string.find("{", think_end)

    # If "{" is not found after "</think>", return the original string
    if brace_start == -1:
        return string

    # Return the substring starting from the first "{"
    return string[brace_start:]


# Function to process a single model's data
def process_model_data(path, debug=False):
    if not os.path.exists(path):
        return None

    if debug:
        print(f"Processing {path} in process {os.getpid()}")

    try:
        # Read CSV
        df_answers = pd.read_csv(path, sep=",", on_bad_lines="warn")
        if "Answer" in df_answers.columns:
            df_answers = df_answers.rename(
                columns={
                    "Answer": "answer",
                    "N_steps": "seq_len",
                    "Type": "qtype",
                }
            )

        # Filter out rows with errors in the predicted answer
        df_answers = df_answers[
            ~df_answers["Predicted_Answer"].str.lower().str.contains("error", na=True)
        ]

        if debug:
            print(f"Using {df_answers.shape[0]} answers for {path}")
        df_answers["Predicted_Answer"] = df_answers["Predicted_Answer"].apply(
            lambda x: strip_until_first_brace(x)
        )

        # Parse predicted answers
        parsed_answers = [
            parse_predicted_answer(row["Predicted_Answer"])
            for _, row in df_answers.iterrows()
        ]
        df_answers["Final_Answer"] = parsed_answers

        # Validate and compare answers
        df_answers["hit"] = df_answers.apply(validate_and_compare, axis=1).astype(int)

        # Extract model name from the file path
        model_name = "/".join(path.split("answers_")[-1].split(".csv")[0].split("_", 1))
        df_answers["model"] = model_name

        # Calculate hit rate
        hit_rate = (
            df_answers.groupby(["seq_len", "qtype"])["hit"]
            .mean()
            .sort_index()
            .to_frame("hit")
        )
        hit_rate["model"] = model_name

        return hit_rate.set_index(["model"], append=True)
    except Exception as e:
        if debug:
            print(f"Error processing {path}: {e}")
        return None


# Main function
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and analyze QA pairs.")
    parser.add_argument(
        "--exp_name", type=str, default="main", help="Name of the experiment"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory containing input CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output CSV files",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes to use (default: number of CPU cores)",
    )

    args = parser.parse_args()

    # Paths to answer files
    answers = glob.glob(f"{args.input_dir}/{args.exp_name}/qa_pairs_answers_*.csv")

    if args.debug:
        print(f"Found {len(answers)} files to process")

    # Determine number of processes
    num_processes = args.num_processes or min(len(answers), mp.cpu_count() - 1)

    if args.debug:
        print(f"Using {num_processes} processes")

    # Create a pool of worker processes
    with mp.Pool(processes=num_processes) as pool:
        # Create a partial function with the debug argument
        process_func = partial(process_model_data, debug=args.debug)

        # Process all files in parallel
        results = pool.map(process_func, answers)

    # Filter out None results
    heatmap_data = [result for result in results if result is not None]

    # Combine all results
    if heatmap_data:
        heatmap_data = pd.concat(heatmap_data)
        heatmap_data["hit"] = (heatmap_data["hit"].fillna(0) * 100).round(3)

        # Ensure the output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Save the results
        output_file = f"{args.output_dir}/{args.exp_name}_newest_results.csv"
        heatmap_data.to_csv(output_file)

        if args.debug:
            print(f"Results saved to {output_file}")
    else:
        print("No valid results to save")


if __name__ == "__main__":
    main()
