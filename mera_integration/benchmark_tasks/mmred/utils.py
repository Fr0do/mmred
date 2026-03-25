"""Utility functions for MMReD MERA benchmark tasks.

This module provides:
- Answer extraction from model outputs (handling reasoning chains)
- Custom metric processing
- Image loading for multimodal evaluation
"""

import re
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================================
# Answer Extraction
# ============================================================================

def normalize_answer(answer: str, atype: str) -> str:
    """Normalize answer based on expected type.
    
    Args:
        answer: Raw answer string
        atype: Answer type ("person", "room", or "number")
        
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    answer = answer.strip().rstrip(".,!?;:")
    
    if atype == "number":
        # Extract digits only
        digits = re.sub(r"[^\d]", "", answer)
        return digits if digits else "0"
    
    # For person/room: capitalize first letter, handle common variations
    answer = answer.lower().strip()
    
    # Common answer normalizations
    normalizations = {
        "kitchen": "Kitchen",
        "bathroom": "Bathroom",
        "garden": "Garden",
        "office": "Office",
        "bedroom": "Bedroom",
        "hallway": "Hallway",
        "sandra": "Sandra",
        "mary": "Mary",
        "john": "John",
        "daniel": "Daniel",
        "michael": "Michael",
        "nobody": "Nobody",
        "no one": "Nobody",
        "none": "Nobody",
        # Russian normalizations
        "кухня": "Kitchen",
        "ванная": "Bathroom",
        "сад": "Garden",
        "офис": "Office",
        "спальня": "Bedroom",
        "коридор": "Hallway",
        "сандра": "Sandra",
        "мария": "Mary",
        "иван": "John",
        "даниил": "Daniel",
        "михаил": "Michael",
        "никто": "Nobody",
    }
    
    return normalizations.get(answer, answer.capitalize())


def extract_answer(text: str, atype: str = "person") -> str:
    """Extract answer from model generation, handling reasoning chains.
    
    Supports multiple extraction patterns for reasoning models:
    - "The answer is X"
    - "\\boxed{X}"
    - Last word/number in output
    
    Args:
        text: Model generated text
        atype: Expected answer type
        
    Returns:
        Extracted and normalized answer
    """
    if not text:
        return ""
    
    text = text.strip()
    answer = None
    
    # Pattern 1: "The answer is X" / "Answer: X" / "Result: X"
    patterns = [
        r"(?:the\s+)?(?:answer|result)\s*(?:is|:)\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
        r"(?:ответ|результат)\s*(?::|—|-)?\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1)
            break
    
    # Pattern 2: Boxed answer \boxed{X}
    if not answer:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            answer = match.group(1)
    
    # Pattern 3: Final answer marker **X** or *X*
    if not answer:
        match = re.search(r"\*\*([A-Za-zА-Яа-я0-9]+)\*\*", text)
        if match:
            answer = match.group(1)
    
    # Pattern 4: First word/number for short responses
    if not answer:
        if atype == "number":
            match = re.search(r"\d+", text)
            answer = match.group(0) if match else ""
        else:
            # Get first valid word (not common reasoning words)
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


def extract_answer_filter(resps, docs):
    """Filter function for lm-evaluation-harness.
    
    Extracts answers from model responses.
    
    Args:
        resps: List of model responses (each is a list with one response)
        docs: List of document dictionaries
        
    Returns:
        List of extracted answer strings
    """
    extracted = []
    for resp, doc in zip(resps, docs):
        text = resp[0] if isinstance(resp, list) else resp
        atype = doc.get("meta", {}).get("atype", "person")
        answer = extract_answer(text, atype)
        extracted.append([answer])  # Keep as list for pipeline
    return extracted


# ============================================================================
# Metrics
# ============================================================================

def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Process results and compute metrics per task/length.
    
    Args:
        doc: Document dictionary with meta information
        results: List of model predictions
        
    Returns:
        Dictionary of metric name to value
    """
    meta = doc.get("meta", {})
    task = meta.get("task", "unknown")
    seq_len = meta.get("seq_len", 0)
    atype = meta.get("atype", "person")
    
    gold = normalize_answer(str(doc.get("outputs", "")), atype)
    pred = normalize_answer(str(results[0]) if results else "", atype)
    
    em = float(gold.lower() == pred.lower())
    
    return {
        "exact_match": em,
        f"em.{task}": em,
        f"em.{task}.len{seq_len}": em,
        "em.dc_aggregate": em,  # For weighted aggregate
    }


def weighted_length_aggregate(items: list[dict]) -> float:
    """Aggregate metric with exponential weight on length in facts. 32 facts is base, 64 is 2x, 128 is 4x, etc.
    
    Args:
        items: List of result dictionaries
        
    Returns:
        Weighted average score
    """
    total_weight = 0
    weighted_sum = 0
    
    for item in items:
        score = item.get("em.dc_aggregate", 0)
        
        seq_len = item.get("seq_len", 0)
        weight = 2 ** (seq_len // 32)
        
        weighted_sum += score * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ============================================================================
# Multimodal Support
# ============================================================================

def doc_to_image(doc: dict) -> list:
    """Load images for multimodal evaluation.
    
    Args:
        doc: Document dictionary with image paths in meta
        
    Returns:
        List of PIL Image objects
    """
    if Image is None:
        raise ImportError("PIL is required for multimodal evaluation. Install with: pip install Pillow")
    
    meta = doc.get("meta", {})
    image_paths = meta.get("images", [])
    
    if not image_paths:
        return []
    
    images = []
    for path in image_paths:
        # Handle both local paths and HuggingFace dataset paths
        if isinstance(path, str):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {path}: {e}")
        elif hasattr(path, "convert"):
            # Already a PIL Image
            images.append(path)
    
    return images


def doc_to_text_with_images(doc: dict) -> str:
    """Format document text with image placeholders.
    
    Args:
        doc: Document dictionary
        
    Returns:
        Formatted prompt text
    """
    instruction = doc.get("instruction", "")
    inputs = doc.get("inputs", {})
    
    # Format instruction with inputs
    try:
        text = instruction.format(**inputs)
    except (KeyError, ValueError):
        text = instruction
    
    return text.strip()
