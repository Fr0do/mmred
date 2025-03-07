import re
from typing import Dict, List, Union
import math
import string
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK resources (uncomment if not already available)
# nltk.download('punkt')

# Constants
PRINT_INTERVAL = 100  # For logging
MIN_ACCEPTABLE_REASONING_LENGTH = 50  # Characters
OPTIMAL_REASONING_LENGTH = 300  # Characters
MAX_REASONING_LENGTH = 1000  # Characters

# Define domain-specific knowledge
room_names = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
people_names = ["Nobody", "Daniel", "Mary", "Michael", "Sandra", "John"]
valid_names = set(room_names + people_names)

# Logical operators and reasoning indicators
reasoning_indicators = [
    "because",
    "therefore",
    "thus",
    "since",
    "as a result",
    "if",
    "then",
    "otherwise",
    "however",
    "although",
    "consequently",
    "given that",
    "first",
    "second",
    "third",
    "finally",
    "lastly",
    "consider",
    "assume",
    "suppose",
    "let's",
    "we know",
]


def extract_xml_template(text: str, template="answer", r1_format=False) -> str:
    try:
        if r1_format:
            match = re.search(r'\{"answer":\s*(.*?)\}', text, re.DOTALL)
        else:
            match = re.search(rf"<{template}>(.*?)</{template}>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # Fallback for malformed format
            if r1_format:
                parts = text.split('{"answer":')
                if len(parts) > 1:
                    template_part = parts[-1].split("}")[0]
                    return template_part.strip().replace('"', "")
            else:
                parts = text.split(f"<{template}>")
                if len(parts) > 1:
                    template_part = parts[-1].split(f"</{template}>")[0]
                    return template_part.strip()
            return ""
    except Exception:
        return ""


def correctness_reward(completions, answer, r1_format=False, **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        extract_xml_template(r, "answer", r1_format) for r in responses
    ]
    rewards = []
    for resp, ans in zip(extracted_responses, answer):
        resp, ans = resp.strip().replace('"', ""), ans.strip()
        if resp == ans:
            rewards.append(2.0)
        elif resp.lower() == ans.lower():
            rewards.append(1.5)
        elif resp in ans or ans in resp:
            rewards.append(1.0)
        else:
            similarity = string_similarity(resp, ans)
            if similarity > 0.8:
                rewards.append(0.75)
            else:
                rewards.append(0.0)
    return rewards


def atype_reward(completions, r1_format=False, **kwargs) -> List[float]:
    """Reward function for answer type validity with improved scoring."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        extract_xml_template(r, "answer", r1_format) for r in responses
    ]
    rewards = []
    for r in extracted_responses:
        r = r.strip().replace('"', "")
        if r.isdigit() or r in valid_names:
            rewards.append(0.5)
        elif any(name.lower() == r.lower() for name in valid_names):
            # Case-insensitive match
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards


def string_similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio using Levenshtein distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # Simple Levenshtein implementation
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0.0

    previous_row = range(len(b) + 1)
    for i, a_char in enumerate(a):
        current_row = [i + 1]
        for j, b_char in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (a_char != b_char)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    # Convert to similarity ratio
    distance = previous_row[-1]
    max_len = max(len(a), len(b))
    return 1 - (distance / max_len)


def format_reward(completions, r1_format=False, **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    if r1_format:
        strict_pattern = r"^.*?</think>\n\{\"answer\":\s.*?\}$"
        good_pattern = r".*?</think>\s*\{\"answer\":\s.*?\}"
        minimal_pattern = r".*?</think>.*?\{\"answer\":\s.*?\}"
    else:
        strict_pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        good_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        minimal_pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    rewards = []
    for r in responses:
        if re.match(strict_pattern, r, re.DOTALL | re.MULTILINE):
            rewards.append(1.0)
        elif re.match(good_pattern, r, re.DOTALL | re.MULTILINE):
            rewards.append(0.75)
        elif re.match(minimal_pattern, r, re.DOTALL | re.MULTILINE):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_quality_reward(completions, **kwargs) -> List[float]:
    """New reward function that evaluates the quality of reasoning."""
    responses = [completion[0]["content"] for completion in completions]
    thinking_parts = [extract_xml_template(r, "think") for r in responses]

    rewards = []
    for thinking in thinking_parts:
        score = 0.0

        # 1. Length check - neither too short nor too long
        length = len(thinking)
        if length < MIN_ACCEPTABLE_REASONING_LENGTH:
            length_score = 0.1  # Too short
        elif length < OPTIMAL_REASONING_LENGTH:
            length_score = 0.1 + 0.4 * (length - MIN_ACCEPTABLE_REASONING_LENGTH) / (
                OPTIMAL_REASONING_LENGTH - MIN_ACCEPTABLE_REASONING_LENGTH
            )
        elif length <= MAX_REASONING_LENGTH:
            length_score = 0.5 * (
                1
                - (length - OPTIMAL_REASONING_LENGTH)
                / (MAX_REASONING_LENGTH - OPTIMAL_REASONING_LENGTH)
            )
        else:
            length_score = 0.1  # Too long

        score += length_score

        # 2. Check for reasoning indicators
        indicator_count = sum(
            1 for indicator in reasoning_indicators if indicator in thinking.lower()
        )
        indicator_score = min(0.3, 0.05 * indicator_count)  # Cap at 0.3
        score += indicator_score

        # 3. Check for structure - multiple sentences/steps
        try:
            sentences = sent_tokenize(thinking)
            if len(sentences) >= 3:
                structure_score = 0.2
            elif len(sentences) == 2:
                structure_score = 0.1
            else:
                structure_score = 0.0
            score += structure_score
        except:
            # Fallback if nltk fails
            newlines = thinking.count("\n")
            score += min(0.2, 0.05 * newlines)

        rewards.append(score)

    return rewards


def consistency_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks consistency between thinking and answer."""
    responses = [completion[0]["content"] for completion in completions]

    rewards = []
    for r in responses:
        thinking = extract_xml_template(r, "think")
        answer = extract_xml_template(r, "answer")

        if not thinking or not answer:
            rewards.append(0.0)
            continue

        # Check if answer appears in thinking or is derived from it
        if answer in thinking:
            rewards.append(0.3)  # Direct appearance
        elif answer.lower() in thinking.lower():
            rewards.append(0.2)  # Case-insensitive appearance
        else:
            # Check if the last sentence of thinking leads to the answer
            try:
                sentences = sent_tokenize(thinking)
                if sentences and (answer in sentences[-1] or sentences[-1] in answer):
                    rewards.append(0.3)  # Answer consistent with final conclusion
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)

    return rewards


def combined_reward(
    completions: List[Dict],
    answer: List[str],
    weights: Dict[str, float] = None,
    **kwargs,
) -> List[float]:
    """
    Combined reward function with configurable weights.

    Args:
        completions: List of model completions
        answer: List of ground truth answers
        weights: Dictionary of reward function weights

    Returns:
        List of combined reward scores
    """
    if weights is None:
        weights = {
            "correctness": 1.0,
            "format": 0.5,
            "reasoning": 0.8,
            "consistency": 0.7,
            "atype": 0.3,
            "length": 0.4,
        }

    # Calculate individual rewards
    correct_rewards = correctness_reward(completions, answer)
    format_rewards = format_reward(completions)
    reasoning_rewards = reasoning_quality_reward(completions)
    consistency_rewards = consistency_reward(completions)
    atype_rewards = atype_reward(completions)

    # Calculate length-based rewards with the new cosine approach
    responses = [completion[0]["content"] for completion in completions]
    thinking_parts = [extract_xml_template(r, "think") for r in responses]

    length_rewards = []
    for thinking, is_correct in zip(thinking_parts, [r > 0 for r in correct_rewards]):
        length = len(thinking)

        if not is_correct:
            length_rewards.append(0.0)
            continue

        if length < MIN_ACCEPTABLE_REASONING_LENGTH:
            # Too short
            reward = 0.1
        elif length < OPTIMAL_REASONING_LENGTH:
            # Building up to optimal
            normalized = (length - MIN_ACCEPTABLE_REASONING_LENGTH) / (
                OPTIMAL_REASONING_LENGTH - MIN_ACCEPTABLE_REASONING_LENGTH
            )
            reward = 0.1 + 0.4 * normalized
        else:
            # Beyond optimal - use cosine decay
            normalized = min(
                1.0,
                (length - OPTIMAL_REASONING_LENGTH)
                / (MAX_REASONING_LENGTH - OPTIMAL_REASONING_LENGTH),
            )
            cosine = math.cos(normalized * (math.pi / 2))
            reward = 0.5 * cosine

        length_rewards.append(reward)

    # Combine rewards with weights
    combined_rewards = []
    for i in range(len(completions)):
        weighted_sum = (
            weights["correctness"] * correct_rewards[i]
            + weights["format"] * format_rewards[i]
            + weights["reasoning"] * reasoning_rewards[i]
            + weights["consistency"] * consistency_rewards[i]
            + weights["atype"] * atype_rewards[i]
            + weights["length"] * length_rewards[i]
        )
        combined_rewards.append(weighted_sum)

    return combined_rewards
