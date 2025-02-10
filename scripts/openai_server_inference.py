#!/usr/bin/env python
# coding: utf-8

import argparse
import asyncio
import csv
import glob
import os
from typing import List, Dict, Tuple, Union
from openai import AsyncOpenAI
from pydantic import BaseModel, NonNegativeInt, constr
from typing import Literal, Optional, Set
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
import requests

# Define the Pydantic models
room_names = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
people_names = ["Nobody", "Daniel", "Mary", "Michael", "Sandra", "John"]

class RoomAnswer(BaseModel):
    reasoning: Optional[str] = None
    answer: Literal[*room_names]

class NumberAnswer(BaseModel):
    reasoning: Optional[str] = None
    answer: int = 0

class PersonAnswer(BaseModel):
    reasoning: Optional[str] = None
    answer: Set[Literal[*people_names]] = {"Nobody"}

schemas = {"room": RoomAnswer.model_json_schema(), "number": NumberAnswer.model_json_schema(), "person": PersonAnswer.model_json_schema()}

person_types = {"most_time_in_room", "compare_two_steps", "list_chars_in_room_at_step", "name_char_with_char_at_step"}

SYSTEM_PROMPT = """
You are an assistant that analyzes image sequences.
Format your response as follows:
{
    "reasoning": Optional[str], # Your short explanation of answer, skip it if the answer is trivial
    "answer": <value>,
}

Where <value> is:
- A **single room name** (e.g., "Kitchen") for location answers.
- A **number** (e.g., "3") for count answers.
- A **set of names** (e.g., ["Michael"], ["Daniel", "Sandra"] or or ["Nobody"] if set should be empty) for people answers.
"""


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """encode raw date to base64 format."""
    buffered = BytesIO()
    FETCH_TIMEOUT = 10
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        if isinstance(image, str):
            url_or_path = image
            if url_or_path.startswith('http'):
                response = requests.get(url_or_path, headers=headers, timeout=FETCH_TIMEOUT)
                response.raise_for_status()
                buffered.write(response.content)
            elif os.path.exists(url_or_path):
                with open(url_or_path, 'rb') as image_file:
                    buffered.write(image_file.read())
        elif isinstance(image, Image.Image):
            image.save(buffered, format='PNG')
    except Exception as error:
        if isinstance(image, str) and len(image) > 100:
            image = image[:100] + ' ...'
        print(f'{error}, image={image}, using dummy image')
        # use dummy image
        image = Image.new('RGB', (32, 32))
        image.save(buffered, format='PNG')
    res = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return res


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
        file_url =  os.path.abspath(relative_path)
        image_urls.append({'image_url': {'url': f"data:image/png;base64,{encode_image_base64(file_url)}"}})
    return image_urls


async def process_qa_pair(qa: Dict, current_dir: str, client: AsyncOpenAI, model_name: str, semaphore: asyncio.Semaphore) -> Tuple[Dict, str]:
    image_urls = get_image_urls(qa, current_dir)
    if not image_urls:
        return (qa, "No images found for the given sequence.")
    user_content = [url | {"type": "image_url"} for url in image_urls] + [{'type': 'text', 'text': qa['Question']}]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    async with semaphore:
        try:
            if qa['Type'] in person_types:
                answer_type = "person"
            else:
                answer_type = "number" if "count" in qa['Type'] else "room"
            extra_body = {
                "repetition_penalty": 1.15, "guided_json": schemas[answer_type], 
                "guided_decoding_backend": "outlines"
            }
            if "Aria" in model_name: 
                extra_body |= {"stop_token_ids": [1970, 93653]}
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=512,
                extra_body=extra_body,
            )
            answer = response.choices[0].message.content
            return (qa, answer)
        except Exception as e:
            return (qa, f"Error: {e}")


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

    for coro in tqdm_asyncio.as_completed(tasks, position=hash(str(client.base_url)) % 10, total=len(tasks), desc="Processing QA Pairs"):
        qa_data, predicted_answer = await coro
        with open(output_csv, mode='a+', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writerow(qa_data | {'Predicted_Answer': predicted_answer})


async def main_async(args):
    CURRENT_DIR = os.getcwd()

    if "base_url" not in args:
        args.base_url = f"http://{args.host}:{args.port}{args.endpoint}"

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    try:
        model_names = await client.models.list()
        global model_name
        model_name = model_names.data[0].id
    except Exception as e:
        print(e, "No models available")
        return

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