import argparse
import asyncio
import base64
import csv
import json
import hashlib
import httpx
import os
import requests
from io import BytesIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from typing import List, Dict, Tuple, Union, Literal, Optional

import pandas as pd
from openai import AsyncOpenAI, OpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from mmred.const import (
    ROOMS,
    CHARS,
    NOBODY,
    SEQ_LENGTHS,
    AnswerTypePerson,
    AnswerTypeRoom,
    AnswerTypeNumber,
)


class RoomAnswer(BaseModel):
    # reasoning: Optional[str] = None
    answer: Literal[*ROOMS]


class NumberAnswer(BaseModel):
    # reasoning: Optional[str] = None
    answer: int = 0

class PersonAnswer(BaseModel):
    # reasoning: Optional[str] = None
    answer: Literal[*CHARS, NOBODY] = NOBODY


schemas = {
    AnswerTypeRoom: RoomAnswer.model_json_schema(),
    AnswerTypeNumber: NumberAnswer.model_json_schema(),
    AnswerTypePerson: PersonAnswer.model_json_schema(),
}

strict_schemas = {
    AnswerTypeRoom: {"schema": to_strict_json_schema(RoomAnswer)} | {"name": "room"},
    AnswerTypeNumber: {"schema": to_strict_json_schema(NumberAnswer)}
    | {"name": "number"},
    AnswerTypePerson: {"schema": to_strict_json_schema(PersonAnswer)}
    | {"name": "person"},
}

THINKING_PROMPT = """You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses.
First think about the reasoning process and then provide the user with the answer.\n
If room contains a ["?"], it's masked and you should infer information from surrounding elements of sequence.
Format your final answer with a {"answer": <value>}, where <value> is:
  - A **single room name** (e.g., 'Kitchen') for location answers.
  - A **number** (e.g., '3') for counting answers.
  - A **single person name** (e.g., 'Michael') for people answers or 'Nobody' if no person satisfies given conditions."""

SYSTEM_PROMPT = """You are an assistant that analyzes sequences of human agents moving in an environment.
If room contains a ["?"], it's masked and you should infer information from surrounding elements of sequence.
Format your response as a following json:
{ "answer": <value> }

Where <value> is:
- A **single room name** (e.g., "Kitchen") for location answers.
- A **number** (e.g., "3") for counting answers.
- A **single person name** (e.g., "Michael") for people answers or "Nobody" if no person satisfies given conditions."""

IN_CONTEXT_HEADER = (
    "Here are example sequences with their questions and answers for reference:"
)


def build_in_context_prompt(
    row: pd.Series,
    in_context_examples: Optional[List[Dict]],
    max_examples: int,
    max_len: int
) -> str:
    if not in_context_examples or max_examples <= 0:
        return ""

    qtype = row.get("qtype")
    filtered_examples = [ex for ex in in_context_examples if ex.get("qtype") == qtype and ex.get("seq_len", 0) == max_len]
    if not filtered_examples:
        filtered_examples = in_context_examples
    prompt_parts = [IN_CONTEXT_HEADER]
    try:
        for idx, example in enumerate(filtered_examples[:max_examples], 1):
            sequence_render = "\n".join(str(frame) for frame in example["sequence_json"])
            prompt_parts.append(
                "\n".join(
                    [
                        f"Example {idx} (len={example.get('seq_len', 'N/A')}):",
                        f"Question: {example.get('question', '')}",
                        f"Sequence:\n{sequence_render}",
                        f"Answer: {{'answer': '{example.get('answer', '')}'}}",
                    ]
                )
            )
    except Exception as e:
        print(e)
    return "\n\n".join(prompt_parts)


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


def get_text_sequence(row: pd.Series, key: str, prefix_question: bool = False) -> str:
    sequence_text = "\n".join(str(frame) for frame in row[key])
    question = row["question"]
    return (question + "\n" + sequence_text) if prefix_question else (sequence_text + "\n" + question)


_THINK_CLOSE_TAGS = ("</think>", "[/THINK]", "<channel|>")


def _is_gemma4_model(model_name: str) -> bool:
    m = (model_name or "").lower()
    return "gemma-4" in m or "gemma_4" in m or "gemma4" in m


_GEMMA4_CHANNEL_START = "<|channel>"
_GEMMA4_CHANNEL_END = "<channel|>"
_GEMMA4_THOUGHT_LABEL = "thought\n"


def _strip_gemma4_thought_label(text: str) -> str:
    """Strip Gemma-4 ``thought\\n`` role label after optional ``<|channel>`` (vLLM Gemma4ReasoningParser)."""
    if text.startswith(_GEMMA4_THOUGHT_LABEL):
        return text[len(_GEMMA4_THOUGHT_LABEL) :]
    return text


def _join_gemma4_reasoning_for_csv(reasoning: str, content: str) -> str:
    """Merge reasoning + answer for CSV using Gemma-4 ``<|channel>`` ... ``<channel|>`` markers."""
    r = reasoning or ""
    c = content or ""
    if _GEMMA4_CHANNEL_END in (r + c):
        return r + c
    body = r
    if body.startswith(_GEMMA4_CHANNEL_START):
        body = body[len(_GEMMA4_CHANNEL_START) :]
    body = _strip_gemma4_thought_label(body)
    if not body.strip():
        return _join_thinking_for_csv("", c)
    return (
        _GEMMA4_CHANNEL_START
        + _GEMMA4_THOUGHT_LABEL
        + body
        + _GEMMA4_CHANNEL_END
        + c
    )


def _join_thinking_for_csv(reasoning: str, content: str) -> str:
    """Merge reasoning_content + content for CSV; wrap only if neither part already has a known close tag."""
    r = reasoning or ""
    c = content or ""
    if any(tag in r for tag in _THINK_CLOSE_TAGS) or any(tag in c for tag in _THINK_CLOSE_TAGS):
        return r + c
    return "<think>" + r + "</think>" + c


_MINISTRAL_THINK_CLOSE = "[/THINK]"


def _mistral_completion_to_answer(text: str, thinking: bool) -> str:
    """Align ``/v1/completions`` output with chat assembly (``reasoning_content`` + ``content`` / ``_join_thinking_for_csv``)."""
    t = (text or "").strip()
    if not thinking:
        return t
    idx = t.rfind(_MINISTRAL_THINK_CLOSE)
    if idx == -1:
        return t
    reasoning_part = t[: idx + len(_MINISTRAL_THINK_CLOSE)]
    content_part = t[idx + len(_MINISTRAL_THINK_CLOSE) :].lstrip()
    return _join_thinking_for_csv(reasoning_part, content_part)


# Mirrors `default_system_message` in mistralai/Ministral-3-8B-Reasoning-2512 `chat_template.jinja`.
MINISTRAL_DEFAULT_SYSTEM_MESSAGE = (
    "# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive at a response. "
    "Format your response using Markdown, and use LaTeX for any mathematical equations. "
    "Write both your thoughts and the response in the same language as the input.\n\n"
    "Your thinking process must follow the template below:"
    "[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. "
    "Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]"
    "Here, provide a self-contained response."
) # TODO: dont use currently


def _ministral_content_is_empty(content) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        if not content:
            return True
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text" and str(block.get("text", "")).strip():
                return False
            if t == "thinking" and str(block.get("thinking", "")).strip():
                return False
            if t in ("image", "image_url"):
                return False
        return True
    return False


def _ministral_format_system_inner(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                parts.append(str(block.get("text", "")))
            elif t == "thinking":
                parts.append("[THINK]" + str(block.get("thinking", "")) + "[/THINK]")
            else:
                raise ValueError(
                    "Only text and thinking chunks are supported in system message contents."
                )
        return "".join(parts)
    raise TypeError("System content must be a string or a list of chunks.")


def _ministral_format_user_inst(content) -> str:
    if isinstance(content, str):
        return "[INST]" + content + "[/INST]"
    if not isinstance(content, list) or not content:
        raise ValueError("User message must have a string or a non-empty list of chunks in content.")
    blocks = sorted(content, key=lambda b: str(b.get("type", ""))) if len(content) == 2 else content
    inner: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            inner.append(str(block.get("text", "")))
        elif t in ("image", "image_url"):
            inner.append("[IMG]")
        else:
            raise ValueError(
                "Only text, image and image_url chunks are supported in user message content."
            )
    return "[INST]" + "".join(inner) + "[/INST]"


def _ministral_format_assistant_turn(message: dict, eos_token: str) -> str:
    parts: list[str] = []
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    if _ministral_content_is_empty(content) and not tool_calls:
        raise ValueError(
            "Assistant message must have a string or a list of chunks in content or a list of tool calls."
        )
    if isinstance(content, str) and content != "":
        parts.append(content)
    elif isinstance(content, list) and len(content) > 0:
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                parts.append(str(block.get("text", "")))
            elif t == "thinking":
                parts.append("[THINK]" + str(block.get("thinking", "")) + "[/THINK]")
            else:
                raise ValueError(
                    "Only text and thinking chunks are supported in assistant message contents."
                )
    for tool in tool_calls:
        fn = tool.get("function") or {}
        name = fn.get("name", "")
        arguments = fn.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif arguments == "":
            arguments = "{}"
        parts.append("[TOOL_CALLS]" + name + "[ARGS]" + arguments)
    parts.append(eos_token)
    return "".join(parts)


def _ministral_assert_alternating(loop_messages: list[dict]) -> None:
    """Same role-alternation check as in Ministral `chat_template.jinja` (user / assistant without tool_calls)."""
    ns_index = 0
    for message in loop_messages:
        role = message.get("role")
        tcalls = message.get("tool_calls")
        has_tc = bool(tcalls and len(tcalls) > 0)
        if role == "user" or (role == "assistant" and not has_tc):
            if (role == "user") != (ns_index % 2 == 0):
                raise ValueError(
                    "After the optional system message, conversation roles must alternate user and assistant "
                    "roles except for tool calls and results."
                )
            ns_index += 1


def _messages_to_plain_prompt(
    messages: list[dict],
    tools: Optional[list] = None,
) -> str:
    """
    Build a string prompt for ``/v1/completions`` matching ``chat_template.jinja`` from
    ``mistralai/Ministral-3-8B-Reasoning-2512`` (BOS, ``[SYSTEM_PROMPT]``, ``[INST]``, tool blocks, EOS after
    assistant turns). Used when vLLM cannot apply chat templates (e.g. Mistral-family tokenizers).

    Optional env: ``MISTRAL_PLAIN_BOS_TOKEN``, ``MISTRAL_PLAIN_EOS_TOKEN`` — set to tokenizer
    ``bos_token`` / ``eos_token`` if the server does not add them when encoding the prompt string.

    Leading ``system`` messages with empty content are dropped so the template's
    ``default_system_message`` branch applies (Jinja only injects it when the first message is not ``system``).
    """
    if not messages:
        return ""

    msgs = list(messages)
    while msgs and msgs[0].get("role") == "system" and _ministral_content_is_empty(msgs[0].get("content")):
        msgs = msgs[1:]

    bos_token = os.environ.get("MISTRAL_PLAIN_BOS_TOKEN", "")
    eos_token = os.environ.get("MISTRAL_PLAIN_EOS_TOKEN", "")

    out: list[str] = [bos_token]

    if msgs and msgs[0].get("role") == "system":
        out.append("[SYSTEM_PROMPT]")
        out.append(_ministral_format_system_inner(msgs[0]["content"]))
        out.append("[/SYSTEM_PROMPT]")
        loop_messages = msgs[1:]
    else:
        loop_messages = msgs
        if MINISTRAL_DEFAULT_SYSTEM_MESSAGE:
            out.append("[SYSTEM_PROMPT]")
            out.append(SYSTEM_PROMPT)
            out.append("[/SYSTEM_PROMPT]")

    if tools:
        out.append("[AVAILABLE_TOOLS]")
        out.append(json.dumps(tools, ensure_ascii=False))
        out.append("[/AVAILABLE_TOOLS]")

    _ministral_assert_alternating(loop_messages)

    for message in loop_messages:
        role = message.get("role")
        if role == "user":
            out.append(_ministral_format_user_inst(message.get("content")))
        elif role == "assistant":
            out.append(_ministral_format_assistant_turn(message, eos_token))
        elif role == "tool":
            out.append("[TOOL_RESULTS]" + str(message.get("content", "")) + "[/TOOL_RESULTS]")
        else:
            raise ValueError(f"Only user, assistant and tool roles are supported, got {role!r}.")

    return "".join(out)


async def process_row(
    row: pd.Series,
    data_path: Path,
    client: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    use_text_input: bool,
    timeout: int = 600,
    max_retries: int = 2,
    thinking: bool = False,
    prefix_question: bool = False,
    for_offline: bool = False,
    in_context_examples: Optional[List[Dict]] = None,
    max_in_context: int = 0,
    force_completions: bool = False,
    max_completion_tokens: int = 12000,
) -> Tuple[Dict, str]:
    # Prepare the content based on input type
    in_context_prompt = build_in_context_prompt(
        row, in_context_examples, max_in_context, min(row['seq_len'], 16)
    )
    if use_text_input:
        key = "sequence_json" if "sequence_json" in row else "sequence"
        user_text = get_text_sequence(row, key, prefix_question=prefix_question)
        if in_context_prompt:
            user_text = f"{in_context_prompt}\n\n{user_text}"
        if thinking:
            user_text = THINKING_PROMPT + '\n' + user_text
        user_content = [{"type": "text", "text": user_text}]
        row_dict = row.drop(key).to_dict()
    else:
        try:
            image_urls = get_image_urls(row, data_path)
            if not image_urls:
                return row.to_dict(), "Error: No images found for the given sequence"
            sequence = [url | {"type": "image_url"} for url in image_urls]
            question_text = THINKING_PROMPT + '\n' + row["question"] if thinking else row["question"]
            if in_context_prompt:
                question_text = f"{in_context_prompt}\n\n{question_text}"
            if prefix_question:
                user_content = [{"type": "text", "text": question_text}] + sequence
            else:
                user_content = sequence + [{"type": "text", "text": question_text}]
            row_dict = row.to_dict()
        except Exception as e:
            return row.to_dict(), f"Error preparing images: {str(e)}"

    messages = [
        {"role": "system", "content": "" if thinking else SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_content,
        },
    ]
    extra_kwargs = {}
    extra_body = {
        # "repetition_penalty": 1.1,
        # "frequency_penalty": 0.05,
        # "presence_penalty": 1.5,
        # "min_p": 0.05,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "skip_special_tokens": False,
    }
    completion_cap = max_completion_tokens if thinking else 50

    if not for_offline:
        # Prepare extra body parameters
        if row.get("atype") in schemas and not thinking:
            if "gemini" not in model_name and "4o" not in model_name and "Llama-3.2" not in model_name:
                extra_body.update(
                    {
                        "guided_json": schemas[row["atype"]],
                        # "guided_decoding_backend": "outlines",
                    }
                )
            else:
                extra_kwargs = extra_body.copy()
                extra_kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": strict_schemas[row["atype"]],
                        },
                    }
                )
                extra_body = None
        # Use semaphore for rate limiting
        async with semaphore:
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    if "mistral" in (model_name or "").lower():
                        prompt = _messages_to_plain_prompt(messages)
                        resp2 = await asyncio.wait_for(
                            client.completions.create(
                                model=model_name,
                                prompt=prompt,
                                temperature=0.6 if thinking else 0.0,
                                max_tokens=completion_cap,
                            ),
                            timeout=timeout,
                        )
                        text = (getattr(resp2.choices[0], "text", "") or "").strip()
                        return row_dict, _mistral_completion_to_answer(text, thinking)

                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=0.6 if thinking else 0.0,
                            max_completion_tokens=completion_cap,
                            extra_body=extra_body,
                            **extra_kwargs,
                        ),
                        timeout=timeout,
                    ) 
                    reasoning = getattr(response.choices[0].message, "reasoning_content", "") or ""
                    answer = getattr(response.choices[0].message, "content", "") or ""
                    if not thinking:
                        answer = reasoning + answer
                    else:
                        answer = (
                            _join_gemma4_reasoning_for_csv(reasoning, answer)
                            if _is_gemma4_model(model_name)
                            else _join_thinking_for_csv(reasoning, answer)
                        )
                    return row_dict, answer

                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 2**retry_count  # Exponential backoff
                        print(
                            f"Request for qid={row.get('qid', 'unknown')} timed out. Retrying in {wait_time} seconds..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        return (
                            row_dict,
                            "Error: Request timed out after multiple attempts",
                        )

                except Exception as e:
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 2**retry_count
                        print(
                            f"Error for qid={row.get('qid', 'unknown')}: {e}. Retrying in {wait_time} seconds..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        return row_dict, f"Error after {max_retries} retries: {str(e)}"
    else:
        # Offline batch body only; assistant text for CSV is assembled on the live path (see _join_thinking_for_csv).
        offline_task = {
            "custom_id": f"text_{use_text_input}-qid-{row['qid']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": messages,
                "temperature": 0.6 if thinking else 0.0,
                "max_completion_tokens": max_completion_tokens if thinking else 50,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": strict_schemas[row["atype"]],
                },
            },
        }
        return offline_task


async def test_connection(
    client: AsyncOpenAI,
    model_name: str,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    force_completions: bool = False,
):
    """Test the API connection and model availability with retries and backoff."""
    dummy_messages = [
        {
            "role": "system",
            "content": "Your only purpose is to answer 'I'm a teapot' to any message.",
        },
        {"role": "user", "content": "Who are you?"},
    ]
    extra_body = {
        # "repetition_penalty": 1.1,
        # "frequency_penalty": 0.05,
        # "presence_penalty": 1.5,
        "min_p": 0.0,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(
                f"Testing connection to model {model_name}... (attempt {retry_count+1})"
            )
            if "mistral" in (model_name or "").lower():
                prompt = _messages_to_plain_prompt(dummy_messages)
                response = await client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=25,
                    timeout=30,
                )
                reasoning = ""
                answer = getattr(response.choices[0], "text", "") or ""
            else:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=dummy_messages,
                    max_completion_tokens=25,
                    timeout=30,
                    extra_body=extra_body,
                )
                reasoning = getattr(response.choices[0].message, "reasoning_content", "") or ""
                answer = getattr(response.choices[0].message, "content", "") or ""
            # Probe uses enable_thinking=False; do not add thinking XML (would be empty blocks).
            answer = (reasoning or "") + (answer or "")
            print(
                f"Connection successful. Response: {answer}"
            )
            return True
        except Exception as e:
            retry_count += 1
            wait_time = backoff_factor**retry_count
            print(
                f"Connection test failed: {e}. Retrying in {wait_time:.1f} seconds..."
            )
            await asyncio.sleep(wait_time)

    print("Failed to establish connection after maximum retries.")
    return False


async def process_dataset(
    dataset: pd.DataFrame,
    data_path: Path,
    output_csv: str,
    client: AsyncOpenAI,
    model_name: str,
    semaphore_limit: int = 10,
    use_text_input: bool = False,
    batch_size: int = 128,
    timeout: int = 600,
    max_retries: int = 2,
    thinking: bool = False,
    prefix_question: bool = False,
    in_context_examples: Optional[List[Dict]] = None,
    max_in_context: int = 0,
    force_completions: bool = False,
    max_completion_tokens: int = 12000,
):

    semaphore = asyncio.Semaphore(semaphore_limit)
    # batch_size = min(batch_size, semaphore_limit)

    # Ensure output file exists with headers
    with open(output_csv, mode="a+", encoding="utf-8", newline="") as f_out:
        fieldnames = dataset.columns.tolist() + ["Predicted_Answer"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if os.path.getsize(output_csv) == 0:
            writer.writeheader()

    # Generate a unique position for the progress bar based on client base URL using SHA-256
    # This ensures parallel scripts don't have overlapping progress bars
    base_url_str = str(client.base_url)
    hash_url_str = (
        base_url_str + str(data_path) + str(thinking) + str(max_completion_tokens)
    )
    sha256_hash = hashlib.sha256(hash_url_str.encode()).hexdigest()
    position = int(sha256_hash, 16) % 16

    # Process in batches but maintain your parallel tqdm approach
    total_rows = len(dataset)
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataset.iloc[start_idx:end_idx]

        # Create tasks for this batch
        tasks = []
        for _, row in batch.iterrows():
            task = asyncio.create_task(
                process_row(
                    row,
                    data_path,
                    client,
                    model_name,
                    semaphore,
                    use_text_input,
                    timeout,
                    max_retries,
                    thinking,
                    prefix_question,
                    False,
                    in_context_examples,
                    max_in_context,
                    force_completions,
                    max_completion_tokens,
                )
            )
            tasks.append(task)

        # Use tqdm_asyncio.as_completed to maintain parallel progress bars
        batch_desc = (
            f"Batch {start_idx//batch_size + 1}/{(total_rows-1)//batch_size + 1}"
        )

        # Write results as they complete
        with open(output_csv, mode="a+", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)

            for coro in tqdm_asyncio.as_completed(
                tasks,
                position=position,
                total=len(tasks),
                desc=f"{batch_desc} ({base_url_str})",
                leave=False,
            ):
                qa_data, predicted_answer = await coro
                # try:
                #     out = await coro
                #     qa_data, predicted_answer = out
                # except Exception as e:
                #     # log whatever identifies this task
                #     print(e)
                #     # optionally continue with next task
                #     continue
                writer.writerow(qa_data | {"Predicted_Answer": predicted_answer})

            # Optional: add a small delay between batches to prevent API rate limits
            f_out.flush()


async def create_task_batch(
    dataset: pd.DataFrame,
    data_path: Path,
    output_csv: str,
    client: OpenAI,
    model_name: str,
    semaphore_limit: int = 10,
    use_text_input: bool = False,
    thinking: bool = False,
    prefix_question: bool = False,
    in_context_examples: Optional[List[Dict]] = None,
    max_in_context: int = 0,
    force_completions: bool = False,
    max_completion_tokens: int = 12000,
):
    tasks = []
    semaphore = asyncio.Semaphore(semaphore_limit)
    timeout, max_retries, for_offline = 0, 0, True
    for index, row in dataset.iterrows():
        task = asyncio.create_task(
            process_row(
                row,
                data_path,
                client,
                model_name,
                semaphore,
                use_text_input,
                timeout,
                max_retries,
                thinking,
                prefix_question,
                for_offline,
                in_context_examples,
                max_in_context,
                force_completions,
                max_completion_tokens,
            )
        )
        tasks.append(task)
    task_jsons = []
    task_file_name = (
        output_csv.split(".csv")[0] + f"_is_text_{use_text_input}" + "_tasks.jsonl"
    )
    for coro in tqdm_asyncio.as_completed(
        tasks,
        leave=False,
    ):
        task_json = await coro
        task_jsons.append(task_json)

    with open(task_file_name, "a+") as file:
        for task_json in task_jsons:
            file.write(json.dumps(task_json) + "\n")

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
                completed_seq_ids.append(str(row["qid"]).zfill(7))
    return completed_seq_ids


async def main_async(args):
    if "base_url" not in args:
        args.base_url = f"http://{args.host}:{args.port}{args.endpoint}"

    print(f"Connecting to API at {args.base_url}")
    if args.offline:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    else:
        client = AsyncOpenAI(
            http_client=(
                httpx.AsyncClient(
                    proxy="socks5://fb_lab:T2wO4gqgumHs@193.124.46.176:8080"
                )
                if "openai" in args.base_url
                else None
            ),
            api_key=args.api_key.replace("\xad", ""),
            base_url=args.base_url,
        )
    if args.model_name:
        model_name = args.model_name
    else:
        # Get available models
        try:
            model_names = await client.models.list()
            model_name = model_names.data[0].id
            print(f"Using model: {model_name}")
        except Exception as e:
            print(f"Error getting models: {e}")
            print("Using default model name...")
            model_name = "default_model"

    # Set output path
    use_text_input = args.text_json_path is not None
    if "output_csv" not in args:
        model_format = "_".join(model_name.split("/"))
        cap = int(getattr(args, "max_completion_tokens", 12000))
        mt_part = (
            f"_mt{cap}"
            if getattr(args, "thinking", False) and cap != 12000
            else ""
        )
        args.output_csv = os.path.join(
            "data_cache",
            args.exp_name,
            f"qa_pairs_answers_{model_format}_text_{use_text_input}_thinking_{args.thinking}{mt_part}_prefix_q_{args.prefix_question}.csv",
        )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if not args.offline:
        force_compl_env = os.getenv("FORCE_COMPLETIONS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        force_completions = bool(getattr(args, "force_completions", False) or force_compl_env)

        # Test connection before proceeding
        connection_ok = await test_connection(
            client, model_name, force_completions=force_completions
        )
        if not connection_ok:
            print("Connection test failed")
            return

    # Load dataset
    data_path = (
        args.text_json_path if use_text_input else Path(args.data_path) / args.exp_name
    )

    in_context_examples = None
    if args.in_context:
        default_ctx_path = Path(args.data_path) / args.exp_name / "in_context_examples.json"
        context_path = Path(args.in_context_path) if args.in_context_path else default_ctx_path
        if context_path.exists():
            print(f"Loading in-context examples from {context_path}")
            with open(context_path, "r") as f:
                in_context_examples = json.load(f)
            print(f"Loaded {len(in_context_examples)} in-context examples")
        else:
            print(
                f"In-context path {context_path} not found. Continuing without in-context examples."
            )

    full_dataset = _load_dataset(data_path, use_text_input)
    print(f"Total QA pairs to process: {len(full_dataset)}")

    # Check for already completed QA pairs
    completed_qids = _get_completed_qids(args.output_csv)
    print(f"Already processed {len(completed_qids)} QA pairs.")
    print(full_dataset.index[:50])
    # Filter out completed QIDs
    remaining_dataset = full_dataset.loc[
        ~full_dataset.index.isin(completed_qids)
    ].reset_index()
    print(f"QA pairs remaining to process: {len(remaining_dataset)}")
    print(f"Model type thinking: {args.thinking}")
    if args.thinking:
        print(f"max_completion_tokens (thinking cap): {args.max_completion_tokens}")

    # Process remaining QA pairs
    if len(remaining_dataset) > 0:
        if args.offline:
            print(f"Creating an offline jsonl batch for {args.output_csv}")
            await create_task_batch(
                remaining_dataset,
                data_path,
                args.output_csv,
                client,
                model_name,
                args.semaphore_limit,
                use_text_input,
                args.thinking,
                args.prefix_question,
                in_context_examples,
                args.in_context_examples,
                force_completions,
                args.max_completion_tokens,
            )
        else:
            print(f"Writing output to {args.output_csv}")
            await process_dataset(
                remaining_dataset,
                data_path,
                args.output_csv,
                client,
                model_name,
                args.semaphore_limit,
                use_text_input,
                args.batch_size,
                args.timeout,
                args.max_retries,
                args.thinking,
                args.prefix_question,
                in_context_examples,
                args.in_context_examples,
                force_completions,
                args.max_completion_tokens,
            )
            print(
                f"Processing complete. Answers have been written to {args.output_csv}"
            )
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
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the API")
    parser.add_argument("--endpoint", type=str, default="/v1", help="API endpoint")
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
        "--semaphore_limit", type=int, default=16, help="Limit for concurrent requests"
    )
    parser.add_argument(
        "--timeout", type=int, default=1200, help="Timeout in seconds for each request"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries for failed requests",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of items to process in each batch",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if connection test fails",
    )
    parser.add_argument(
        "--force_completions",
        action="store_true",
        default=False,
        help="Bypass /v1/chat/completions for Mistral-family models; use /v1/completions with a plain prompt (also FORCE_COMPLETIONS=1).",
    )
    parser.add_argument(
        "--thinking", action="store_true", default=False, help="If model is a thinker"
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=12000,
        help="When --thinking: max_completion_tokens (chat) / max_tokens (Mistral completions). "
        "When not thinking, generation still uses a short cap (50). Default filename includes _mt<N> only if thinking and N != 12000.",
    )
    parser.add_argument(
        "--prefix_question", action="store_true", default=False, help="If question is a prefix of sequence"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="If inference is an offline batch",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Model Name")
    parser.add_argument(
        "--in_context",
        action="store_true",
        help="Append in-context examples to each prompt",
    )
    parser.add_argument(
        "--in_context_path",
        type=str,
        default=None,
        help="Path to in-context examples json. Defaults to <data_path>/<exp_name>/in_context_examples.json",
    )
    parser.add_argument(
        "--in_context_examples",
        type=int,
        default=5,
        help="Number of in-context examples to include per request",
    )
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
