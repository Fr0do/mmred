import re
from typing import Dict
import math

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
    if text.count("\n</answer>") == 1:
        count += 0.125
    count -= len(text.split("</answer>")[-1]) * 0.001
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


def cosine_length_correctness_reward(
    completions: list[dict],
    answer: list[str],
    len_cap: int = 64,
    min_value_correct: float = 0.25,
    max_value_correct: float = 1.0,
    penalty_incorrect: float = -0.5,
    **kwargs
) -> list[float]:
    """
    Compute rewards for a list of completions based on answer correctness and answer length,
    ensuring a minimum answer length (len_cap) for the highest reward.

    Correct answers:
      - For answers shorter than len_cap, the reward scales linearly from 0 up to max_value_correct.
      - For answers at least as long as len_cap, the reward decays from max_value_correct (at len_cap)
        to min_value_correct (for the longest answers) using a cosine transform.

    Incorrect answers receive a fixed penalty regardless of length.

    Args:
        completions: List of dictionaries, each containing a "content" key with the answer text.
        answer: The ground truth answer(s) (used by correctness_reward, assumed to return 1 for correct, 0 otherwise).
        len_cap: Minimum acceptable answer length for optimal reward.
        min_value_correct: Minimum reward for correct answers (when answer length is very long).
        max_value_correct: Maximum reward for correct answers (when answer length == len_cap).
        penalty_incorrect: Fixed penalty value for incorrect answers.

    Returns:
        A list of reward values corresponding to each completion.
    """
    # Assume correctness_reward returns a list of 1 (correct) or 0 (incorrect) for each completion.
    correctness = correctness_reward(
        completions, answer
    )  # This function is assumed to exist.

    # Determine maximum answer length from completions (for scaling longer answers)
    lengths = [len(completion[0]["content"]) for completion in completions]
    max_len = max(lengths)
    rewards = []

    for length, is_correct in zip(lengths, correctness):
        if is_correct:
            if length < len_cap:
                # Penalize answers that are too short even if correct.
                # Reward scales linearly with length up to len_cap.
                reward = max_value_correct * (length / len_cap)
            else:
                # For answers longer than or equal to len_cap, use cosine decay.
                # Normalize length between len_cap and max_len.
                if max_len > len_cap:
                    normalized = (length - len_cap) / (max_len - len_cap)
                else:
                    normalized = 0
                # Cosine: highest (1) at len_cap and decays toward 0 at the longest answer.
                cosine = math.cos(normalized * (math.pi / 2))
                reward = (
                    min_value_correct + (max_value_correct - min_value_correct) * cosine
                )
        else:
            reward = penalty_incorrect

        rewards.append(reward)

    return rewards
