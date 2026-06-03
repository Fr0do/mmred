"""lm-eval task hooks for MMReD sparse benchmarks (MERA-style parsing, English gold)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmred.const import CHARS, NOBODY, ROOMS  # noqa: E402

try:
    from lm_eval.api.filter import Filter
except ImportError:
    from abc import ABC, abstractmethod

    class Filter(ABC):
        def __init__(self, **kwargs):
            pass

        @abstractmethod
        def apply(self, resps, docs):
            return resps


def _build_english_normalizations() -> dict[str, str]:
    normalizations: dict[str, str] = {
        "nobody": NOBODY,
        "no one": NOBODY,
        "none": NOBODY,
    }
    for room in ROOMS:
        normalizations[room.lower()] = room
    for char in CHARS:
        normalizations[char.lower()] = char
    return normalizations


_EN_NORMALIZATIONS = _build_english_normalizations()


def normalize_answer(answer: str, atype: str) -> str:
    """Normalize answer based on expected type (English canonical names)."""
    if not answer:
        return ""

    answer = answer.strip().rstrip(".,!?;:")

    if atype == "number":
        digits = re.sub(r"[^\d]", "", answer)
        return digits if digits else "0"

    answer = answer.lower().strip()
    return _EN_NORMALIZATIONS.get(answer, answer.capitalize())


def extract_answer(text: str, atype: str = "person") -> str:
    """Extract answer from model generation, handling reasoning chains."""
    if not text:
        return ""

    text = text.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    gemma_match = re.search(r"<channel\|>\s*(.+)", text, flags=re.DOTALL)
    if gemma_match:
        text = gemma_match.group(1)
    else:
        text = re.sub(r"<\|channel>thought.*?(?=<\|channel>|$)", "", text, flags=re.DOTALL)
    text = text.strip()

    if not text:
        return ""

    answer = None

    json_match = re.search(
        r'\{\s*["\']answer["\']\s*:\s*["\']?([^"\'}\s]+)["\']?\s*\}', text
    )
    if json_match:
        return normalize_answer(json_match.group(1), atype)

    json_block = re.search(r'\{[^{}]*"answer"[^{}]*\}', text)
    if json_block:
        try:
            obj = json.loads(json_block.group(0))
            if "answer" in obj:
                return normalize_answer(str(obj["answer"]), atype)
        except (json.JSONDecodeError, ValueError):
            pass

    patterns = [
        r"(?:the\s+)?(?:answer|result)\s*(?:is|:)\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
        r"(?:ответ|результат)\s*(?::|—|-)?\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1)
            break

    if not answer:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            answer = match.group(1)

    if not answer:
        match = re.search(r"\*\*([A-Za-zА-Яа-я0-9]+)\*\*", text)
        if match:
            answer = match.group(1)

    if not answer:
        if atype == "number":
            match = re.search(r"\d+", text)
            answer = match.group(0) if match else ""
        else:
            words = text.split()
            skip_words = {"the", "a", "an", "i", "think", "believe", "so", "therefore"}
            for word in words:
                clean = re.sub(r"[^\w]", "", word.lower())
                if clean and clean not in skip_words:
                    answer = word
                    break
            if not answer and words:
                answer = words[0]

    return normalize_answer(answer or "", atype)


def get_atype(doc: dict) -> str:
    meta = doc.get("meta") or {}
    categories = meta.get("categories") or {}
    return categories.get("atype", meta.get("atype", "person"))


def score_prediction(gold: str, pred: str, atype: str) -> bool:
    """MERA-style EM: extract/normalize prediction and compare to normalized gold."""
    gold_n = normalize_answer(str(gold), atype)
    pred_n = extract_answer(pred, atype)
    return gold_n.lower() == pred_n.lower()


class extract_answer_filter(Filter):
    """Filter class for lm-evaluation-harness that extracts answers from model outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self, resps: list, docs: list[dict]) -> list:
        extracted = []
        for resp, doc in zip(resps, docs):
            text = resp[0] if isinstance(resp, list) else resp
            atype = get_atype(doc)
            answer = extract_answer(text, atype)
            extracted.append([answer])
        return extracted


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Process results and compute exact match (plus sparse breakdown keys)."""
    meta = doc.get("meta") or {}
    atype = get_atype(doc)

    gold = normalize_answer(str(doc.get("gold", "")), atype)
    pred = normalize_answer(str(results[0]) if results else "", atype)

    em = float(gold.lower() == pred.lower())

    out: dict[str, float] = {"exact_match": em}
    if meta.get("k_target") is not None:
        out[f"exact_match.k{meta['k_target']}"] = em
    if meta.get("qtype"):
        out[f"exact_match.{meta['qtype']}"] = em
    return out
