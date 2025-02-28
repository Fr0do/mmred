import re
from typing import Dict

global COUNTER
COUNTER = 0
global PRINT_EVERY
PRINT_EVERY = 128

room_names = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
people_names = ["Nobody", "Daniel", "Mary", "Michael", "Sandra", "John"]
valid_names = set(room_names + people_names)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward(completions, answer, prompts=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a.strip() else 0.0 for r, a in zip(extracted_responses, answer)]


def atype_reward(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [
        0.5 if r.isdigit() or r in valid_names else 0.0 for r in extracted_responses
    ]


def strict_format_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    return [0.25 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def len_reward(completions: list[Dict[str, str]], answer: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        answer: List of ground truth answers

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    # First check correctness of answers
    correctness = correctness_reward(completions, answer)

    # Calculate lengths
    lengths = [len(completion[0]["content"]) for completion in completions]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, correctness in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
        if correctness > 0:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)
        rewards.append(float(reward))

    return rewards
