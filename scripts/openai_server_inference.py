#!/usr/bin/env python
# coding: utf-8

import argparse
import asyncio
import csv
import glob
import os
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from pydantic import BaseModel, NonNegativeInt
from typing import Literal, Optional
from tqdm.asyncio import tqdm_asyncio
import pandas as pd

# Define the Pydantic models
room_names = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
people_names = ["No one", "Daniel", "Mary", "Michael", "Sandra", "John"]
prefix = ""

class RoomAnswer(BaseModel):
    reasoning: Optional[str]
    answer: Literal[*room_names]

class NumberAnswer(BaseModel):
    reasoning: Optional[str]
    answer: NonNegativeInt

class PersonAnswer(BaseModel):
    reasoning: Optional[str]
    answer: List[Literal[*people_names]]

schemas = {"room": RoomAnswer.model_json_schema(), "number": NumberAnswer.model_json_schema(), "person": PersonAnswer.model_json_schema()}

person_types = {"most_time_in_room", "compare_adjacent_steps", "list_chars_in_room_at_step", "name_char_with_char_at_step"}

async def process_qa_pair(qa: Dict, current_dir: str, client: AsyncOpenAI, model_name: str, semaphore: asyncio.Semaphore) -> Tuple[Dict, str]:
    image_urls = get_image_urls(qa, current_dir)
    if not image_urls:
        return (qa, "No images found for the given sequence.")
    
    user_content = [{'type': 'text', 'text': qa['Question']}]
    user_content += [url | {"type": "image_url"} for url in image_urls]

    messages = [
        {"role": "system", "content": "You are an assistant that analyzes image sequences. Reason and answer with one word or number (digit 0-9)."},
        {"role": "user", "content": user_content}
    ]

    async with semaphore:
        try:
            if qa['Type'] not in person_types:
                answer_type = "room" if "count" not in qa['Type'] else "number"
            else:
                answer_type = "person"
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=250,
                extra_body={"guided_json": schemas[answer_type], "guided_decoding_backend": "outlines"},
            )
            answer = response.choices[0].message.content
            return (qa, answer)
        except Exception as e:
            return (qa, f"Error: {e}")

def get_image_urls(qa: Dict, current_dir: str) -> List[Dict]:
    DATA_DIR = 'data/length_128'
    sequence_dir = os.path.join(DATA_DIR, qa['Seq_id'])
    image_pattern = os.path.join(sequence_dir, 'step_*.png')

    image_paths = sorted(glob.glob(image_pattern))

    if len(image_paths):
        image_paths = image_paths[:int(qa['N_steps'])]

    image_urls = []
    for path in image_paths:
        relative_path = os.path.relpath(path, current_dir)
        file_url = relative_path.replace(os.sep, "/")
        file_url =  prefix + os.path.abspath(relative_path)

        image_urls.append({'image_url': {'url': file_url}})
    return image_urls

def read_csv(file_path: str) -> List[Dict]:
    df = pd.read_csv(file_path).sort_values(["N_steps", "Seq_id"])
    return df.to_dict("records")

def get_completed_seq_ids(output_csv: str) -> List[str]:
    completed_seq_ids = []
    if not os.path.exists(output_csv):
        return completed_seq_ids
    with open(output_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'error' not in row['Predicted_Answer'].lower():
                completed_seq_ids.append(f"{row['Seq_id']}/{row['N_steps']}/{row['Type']}")
    return completed_seq_ids

async def process_all_qa_pairs(qa_pairs: List[Dict], current_dir: str, output_csv: str, client: AsyncOpenAI, model_name: str, semaphore_limit: int = 10):
    semaphore = asyncio.Semaphore(semaphore_limit)

    with open(output_csv, mode='a+', encoding='utf-8', newline='') as f_out:
        fieldnames = list(qa_pairs[0].keys()) + ['Predicted_Answer']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if not os.path.getsize(output_csv):
            writer.writeheader()

    tasks = []
    for qa_pair in qa_pairs:
        task = asyncio.create_task(process_qa_pair(qa_pair, current_dir, client, model_name, semaphore))
        tasks.append(task)

    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing QA Pairs"):
        qa_data, predicted_answer = await coro
        with open(output_csv, mode='a+', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writerow(qa_data | {'Predicted_Answer': predicted_answer})


async def main_async(args):
    CURRENT_DIR = os.getcwd()
    if args.vllm:
        global prefix
        prefix = "file://"

    if "base_url" not in args:
        args.base_url = f"http://{args.host}:{args.port}{args.endpoint}"

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    model_names = await client.models.list()
    global model_name
    model_name = model_names.data[0].id
    
    if "output_csv" not in args:
        model_format = "_".join(model_name.split("/"))
        args.output_csv = os.path.join('data', f'qa_pairs_answers_{model_format}.csv')
        os.environ['OUTLINES_CACHE_DIR'] = f"/tmp/outlines_{model_format}"
    # Step 1: Read all QA pairs synchronously
    qa_pairs = read_csv(args.qa_pairs_csv)
    print(f"Total QA pairs to process: {len(qa_pairs)}")

    # Step 2: Check for already completed QA pairs and filter them out
    completed_seq_ids = get_completed_seq_ids(args.output_csv)
    print(f"Already processed {len(completed_seq_ids)} QA pairs.")

    qa_pairs_remaining = [pair for pair in qa_pairs if f"{pair['Seq_id']}/{pair['N_steps']}/{pair['Type']}" not in completed_seq_ids]
    print(f"QA pairs remaining to process: {len(qa_pairs_remaining)}")

    # Step 3: Process remaining QA pairs asynchronously with intermediate storage
    if qa_pairs_remaining:
        print(f"Writing output to {args.output_csv}")
        await process_all_qa_pairs(qa_pairs_remaining, CURRENT_DIR, args.output_csv, client, model_name, args.semaphore_limit)
        print(f"Processing complete. Answers have been written to {args.output_csv}")
    else:
        print("No QA pairs remaining to process.")


def main():
    parser = argparse.ArgumentParser(description="Process QA pairs with image sequences.")
    parser.add_argument('--api_key', type=str, default="DUMMY", help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=argparse.SUPPRESS, help='Base URL for the API')
    parser.add_argument('--port', type=str, default='8000', help='Port for the API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Port for the API')
    parser.add_argument('--endpoint', type=str, default='/v1', help='Port for the API')
    parser.add_argument('--qa_pairs_csv', type=str, default='data/new_test_data.csv', help='Path to the QA pairs CSV file')
    parser.add_argument('--output_csv', type=str, default=argparse.SUPPRESS, help='Path to the output CSV file')
    parser.add_argument('--vllm', type=bool, action=argparse.BooleanOptionalAction, default=True, help='If server is vLLM')
    parser.add_argument('--semaphore_limit', type=int, default=16, help='Limit for concurrent requests')
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()