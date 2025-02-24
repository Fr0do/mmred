import argparse
import asyncio
import base64
import csv
import json
import os
from pathlib import Path

import requests
from io import BytesIO
from typing import List, Dict, Tuple, Union, Literal, Optional

import pandas as pd
from openai import AsyncOpenAI
from PIL import Image
import base64
import requests
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from qgen.const import (
    ROOMS,
    CHARS,
    NOBODY,
    SEQ_LENGTHS,
    AnswerTypePerson,
    AnswerTypeRoom,
    AnswerTypeNumber,
)


class RoomAnswer(BaseModel):
    reasoning: Optional[str] = None
    answer: Literal[*ROOMS]


class NumberAnswer(BaseModel):
    reasoning: Optional[str] = None
    answer: int = 0


class PersonAnswer(BaseModel):
    reasoning: Optional[str] = None
    answer: Literal[*CHARS, NOBODY] = NOBODY


schemas = {
    AnswerTypeRoom: RoomAnswer.model_json_schema(),
    AnswerTypeNumber: NumberAnswer.model_json_schema(),
    AnswerTypePerson: PersonAnswer.model_json_schema(),
}

SYSTEM_PROMPT = """You are an assistant that analyzes sequences of human agents moving in an environment.
Format your response as a following json:
{ "answer": <value> }

Where <value> is:
- A **single room name** (e.g., "Kitchen") for location answers.
- A **number** (e.g., "3") for counting answers.
- A **single person name** (e.g., "Michael") for people answers or "Nobody" if no person satisfies given conditions."""


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """encode raw date to base64 format."""
    buffered = BytesIO()
    fetch_timeout = 10
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        if isinstance(image, str):
            url_or_path = image
            if url_or_path.startswith("http"):
                response = requests.get(
                    url_or_path, headers=headers, timeout=fetch_timeout
                )
                response.raise_for_status()
                buffered.write(response.content)
            elif os.path.exists(url_or_path):
                with open(url_or_path, "rb") as image_file:
                    buffered.write(image_file.read())
        elif isinstance(image, Image.Image):
            image.save(buffered, format="PNG")
    except Exception as error:
        if isinstance(image, str) and len(image) > 100:
            image = image[:100] + " ..."
        print(f"{error}, image={image}, using dummy image")
        # use dummy image
        image = Image.new("RGB", (32, 32))
        image.save(buffered, format="PNG")
    res = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return res


def get_image_urls(row: pd.Series, data_path: Path) -> List[Dict]:
    video_path = data_path / f"len_{row['seq_len']}" / "videos" / f"vid_{row['qid']}"
    image_paths = sorted(video_path.glob("frame_*.png"))
    return [
        {"image_url": {"url": f"data:image/png;base64,{encode_image_base64(str(p))}"}}
        for p in image_paths
    ]


def get_text_sequence(row: pd.Series) -> List[Dict]:
    return (
        "\n".join(str(frame) for frame in row["sequence_json"]) + "\n" + row["question"]
    )


async def process_row(
    row: pd.Series,
    data_path: Path,
    client: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    use_text_input: bool,
) -> Tuple[Dict, str]:
    if use_text_input:
        user_content = [{"type": "text", "text": get_text_sequence(row)}]
        row = row.drop("sequence_json")
    else:
        image_urls = get_image_urls(row, data_path)
        if not image_urls:
            return row.to_dict(), "Error: No images found for the given sequence"
        sequence = [url | {"type": "image_url"} for url in image_urls]
        user_content = sequence + [{"type": "text", "text": row["question"]}]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async with semaphore:
        try:
            extra_body = {
                "repetition_penalty": 1.1,
                "guided_json": schemas[row["atype"]],
                "guided_decoding_backend": "outlines",
            }
            if "Aria" in model_name:
                extra_body |= {
                    "stop_token_ids": [93532, 93653, 944, 93421, 1019, 93653, 93519]
                }

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=25,
                extra_body=extra_body,
            )
            answer = response.choices[0].message.content
            return row.to_dict(), answer

        except Exception as e:
            return row.to_dict(), f"Error: {e}"


async def process_dataset(
    dataset: pd.DataFrame,
    data_path: Path,
    output_csv: str,
    client: AsyncOpenAI,
    model_name: str,
    semaphore_limit: int = 10,
    use_text_input: bool = False,
):
    semaphore = asyncio.Semaphore(semaphore_limit)

    with open(output_csv, mode="a+", encoding="utf-8", newline="") as f_out:
        fieldnames = dataset.columns.tolist() + ["Predicted_Answer"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if not os.path.getsize(output_csv):
            writer.writeheader()

    tasks = []
    for _, row in dataset.iterrows():
        task = asyncio.create_task(
            process_row(row, data_path, client, model_name, semaphore, use_text_input)
        )
        tasks.append(task)

    for coro in tqdm_asyncio.as_completed(
        tasks,
        position=hash(str(client.base_url)) % 10,
        total=len(tasks),
        desc="Processing QA pairs",
    ):
        qa_data, predicted_answer = await coro
        with open(output_csv, mode="a+", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writerow(qa_data | {"Predicted_Answer": predicted_answer})


def _load_dataset(data_path: Path, use_text_input: bool) -> pd.DataFrame:
    dataset = []
    if use_text_input:
        with open(data_path, "r") as file:
            dataset = json.load(file)
    else:
        for seq_len in SEQ_LENGTHS:
            with open(
                str(data_path / f"len_{seq_len}" / "questions.json"), "r"
            ) as file:
                dataset_l = json.load(file)
            dataset += dataset_l
    return pd.DataFrame.from_records(dataset, index="qid")


def _get_completed_qids(output_csv: str) -> List[str]:
    completed_seq_ids = []
    if not os.path.exists(output_csv):
        return completed_seq_ids
    with open(output_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "error" not in row["Predicted_Answer"].lower():
                completed_seq_ids.append(row["qid"])
    return completed_seq_ids


async def main_async(args):
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
        args.output_csv = os.path.join("data", f"qa_pairs_answers_{model_format}.csv")

    # Step 1: Read all QA pairs synchronously
    use_text_input = args.text_json_path is not None
    data_path = (
        args.text_json_path if use_text_input else Path(args.data_path) / args.exp_name
    )

    full_dataset = _load_dataset(data_path, use_text_input)
    print(f"Total QA pairs to process: {len(full_dataset)}")

    # Step 2: Check for already completed QA pairs and filter them out
    completed_qids = _get_completed_qids(args.output_csv)
    print(f"Already processed {len(completed_qids)} QA pairs.")

    remaining_qids = sorted(set(full_dataset.index.tolist()) - set(completed_qids))
    print(f"QA pairs remaining to process: {len(remaining_qids)}")
    remaining_dataset = full_dataset.iloc[remaining_qids].reset_index()

    # Step 3: Process remaining QA pairs asynchronously with intermediate storage
    if remaining_qids or False:
        print(f"Writing output to {args.output_csv}")
        await process_dataset(
            remaining_dataset,
            data_path,
            args.output_csv,
            client,
            model_name,
            args.semaphore_limit,
            use_text_input,
        )
        print(f"Processing complete. Answers have been written to {args.output_csv}")
    else:
        print("No QA pairs remaining to process.")


def main():
    parser = argparse.ArgumentParser(
        description="Process QA pairs with image sequences or text JSON."
    )
    parser.add_argument("--api_key", type=str, default="DUMMY", help="OpenAI API key")
    parser.add_argument(
        "--base_url", type=str, default=argparse.SUPPRESS, help="Base URL for the API"
    )
    parser.add_argument("--port", type=str, default="8000", help="Port for the API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Port for the API")
    parser.add_argument("--endpoint", type=str, default="/v1", help="Port for the API")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/jovyan/shares/SR004.nfs2/data/long_vqa_synth",
        help="Path to synthetic dataset",
    )
    parser.add_argument(
        "--text_json_path", type=str, default=None, help="Path to the text JSON file"
    )
    parser.add_argument("--exp_name", type=str, default="main", help="Experiment name")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--vllm",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If server is vLLM",
    )
    parser.add_argument(
        "--semaphore_limit", type=int, default=16, help="Limit for concurrent requests"
    )
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
