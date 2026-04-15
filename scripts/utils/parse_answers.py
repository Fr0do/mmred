import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from ast import literal_eval
from functools import partial
from typing import Optional, Set, Literal, Iterable

import pandas as pd
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from mmred.const import ROOMS, CHARS, NOBODY

# Define the Pydantic models
room_names = ROOMS
people_names = [NOBODY, *CHARS]

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


def _prepare_hits_dataframe(df_answers: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Normalize columns, parse predictions, add ``hit`` column (0/1 per row)."""
    if "Answer" in df_answers.columns:
        df_answers = df_answers.rename(
            columns={
                "Answer": "answer",
                "N_steps": "seq_len",
                "Type": "qtype",
            }
        )
    df_answers = df_answers[
        ~df_answers["Predicted_Answer"].str.lower().str.contains("error", na=True)
    ]
    if debug:
        print(f"Using {df_answers.shape[0]} rows after filtering errors")
    df_answers = df_answers.copy()
    df_answers["Predicted_Answer"] = df_answers["Predicted_Answer"].apply(
        strip_until_first_brace
    )
    df_answers["Final_Answer"] = [
        parse_predicted_answer(x) for x in df_answers["Predicted_Answer"]
    ]
    df_answers["hit"] = df_answers.apply(validate_and_compare, axis=1).astype(int)
    return df_answers


def evaluate_bundle_inference_csv(
    path: str,
    *,
    episode_column: str = "episode_id",
    scoring: str = "strict",
    min_correct: Optional[int] = None,
    seq_len: Optional[int] = None,
    k_target: Optional[int] = None,
    debug: bool = False,
) -> dict:
    """
    Sequence-level accuracy: fraction in [0, 1], mean over episodes of 1[episode passes threshold].

    strict: episode passes iff all rows in the episode are correct.
    at_least: episode passes iff sum(hits) >= min_correct.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df_answers = pd.read_csv(path, sep=",", on_bad_lines="warn")
    if episode_column not in df_answers.columns:
        raise ValueError(
            f"Column {episode_column!r} missing from {path}; need episode-bundle inference CSV."
        )
    df_answers = _prepare_hits_dataframe(df_answers, debug=debug)
    if df_answers.empty:
        return {
            "seq_len": seq_len,
            "k_target": k_target,
            "accuracy": 0.0,
            "n_episodes": 0,
            "scoring": scoring,
            "min_correct": min_correct if scoring == "at_least" else "",
        }

    episode_scores: list[float] = []
    for _, g in df_answers.groupby(episode_column, sort=True):
        n = len(g)
        c = int(g["hit"].sum())
        if scoring == "strict":
            episode_scores.append(1.0 if c == n else 0.0)
        else:
            if min_correct is None:
                raise ValueError("min_correct is required when scoring='at_least'")
            episode_scores.append(1.0 if c >= min_correct else 0.0)

    acc = (sum(episode_scores) / len(episode_scores)) if episode_scores else 0.0
    return {
        "seq_len": seq_len,
        "k_target": k_target,
        "accuracy": round(acc, 6),
        "n_episodes": len(episode_scores),
        "scoring": scoring,
        "min_correct": min_correct if scoring == "at_least" else "",
    }


# Function to process a single model's data
def process_model_data(path, debug=False):
    if not os.path.exists(path):
        return None

    if debug:
        print(f"Processing {path} in process {os.getpid()}")

    try:
        df_answers = pd.read_csv(path, sep=",", on_bad_lines="warn")
        df_answers = _prepare_hits_dataframe(df_answers, debug=debug)

        if debug:
            print(f"Using {df_answers.shape[0]} answers for {path}")

        # Extract model name from the file path
        model_name = "/".join(path.split("answers_")[-1].split(".csv")[0].split("_", 1))
        df_answers["model"] = model_name

        # Calculate hit rate (adaptive: use actual count per group instead of hardcoded 50)
        grouped = df_answers.groupby(["seq_len", "qtype"])["hit"].agg(["sum", "count"])
        grouped["hit"] = grouped["sum"] / grouped["count"]
        hit_rate = grouped[["hit"]].sort_index()
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
    parser.add_argument(
        "--bundle_scoring",
        choices=["off", "strict", "at_least"],
        default="off",
        help="If not off, compute sequence-level accuracy grouped by episode_id.",
    )
    parser.add_argument(
        "--bundle_min_correct",
        type=int,
        default=None,
        help="With --bundle_scoring at_least, minimum correct answers per episode to pass.",
    )
    parser.add_argument(
        "--bundle_episode_column",
        type=str,
        default="episode_id",
        help="CSV column for grouping rows into one sequence score.",
    )
    parser.add_argument(
        "--bundle_seq_len",
        type=int,
        default=None,
        help="Recorded in bundle metrics output (metadata only).",
    )
    parser.add_argument(
        "--bundle_k_target",
        type=int,
        default=None,
        help="Recorded in bundle metrics output (metadata only).",
    )

    args = parser.parse_args()

    if args.bundle_scoring != "off":
        if args.bundle_seq_len is None or args.bundle_k_target is None:
            print(
                "Error: --bundle_seq_len and --bundle_k_target are required when using bundle scoring.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.bundle_scoring == "at_least" and args.bundle_min_correct is None:
            print(
                "Error: --bundle_min_correct is required when --bundle_scoring at_least.",
                file=sys.stderr,
            )
            sys.exit(1)

        answers = glob.glob(f"{args.input_dir}/{args.exp_name}/qa_pairs_answers_*.csv")
        if len(answers) != 1:
            print(
                f"Error: bundle scoring expects exactly one qa_pairs_answers_*.csv, found {len(answers)}",
                file=sys.stderr,
            )
            sys.exit(1)
        metrics = evaluate_bundle_inference_csv(
            answers[0],
            episode_column=args.bundle_episode_column,
            scoring=args.bundle_scoring,
            min_correct=args.bundle_min_correct,
            seq_len=args.bundle_seq_len,
            k_target=args.bundle_k_target,
            debug=args.debug,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        row = {**metrics, "exp_name": args.exp_name}
        out_path = os.path.join(args.output_dir, f"{args.exp_name}_bundle_metrics.csv")
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"Bundle metrics -> {out_path}")
        print(row)
        return

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