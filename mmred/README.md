# <span style="color:red">MMReD</span> Library Documentation

<span style="color:red">MMReD</span> (**M**ulti-**M**odal **R**easoning in **D**ense Context) is a library for generating and working with synthetic benchmarks for dense context reasoning evaluation.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Question Types](#question-types)
- [Controllable Generation](#controllable-generation)
- [Image Rendering](#image-rendering)
- [API Reference](#api-reference)

---

## Quick Start

### Generate a Dataset

```bash
# Generate with default settings (all question types, seq_lengths 1-128, 50 questions each)
python scripts/generate_dataset.py --output_path data/dataset.json

# Generate a smaller test dataset
python scripts/generate_dataset.py \
    --output_path data/test_dataset.json \
    --seq_lengths 4 8 16 \
    --n_questions 10 \
    --seed 42
```

### Programmatic Usage

```python
from mmred import GenerationConfig, generate_questions, save_dataset

# Create configuration
config = GenerationConfig(
    seed=42,
    seq_lengths=[8, 16, 32],
    n_questions=20,
    question_types=["spend_alone", "where_spend", "first_app"],
)

# Generate and save
samples = generate_questions(config)
save_dataset(samples, "my_dataset.json")
```

---

## Configuration

The `GenerationConfig` dataclass provides flexible configuration:

```python
from mmred import GenerationConfig

config = GenerationConfig(
    seed=0xBADFACE,           # Random seed for reproducibility
    seq_lengths=[1, 2, 4, 8, 16, 32, 64, 128],  # Sequence lengths
    n_questions=50,           # Questions per type per length
    question_types=None,      # None = all, or list of types
    rooms=["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"],
    chars=["Sandra", "Mary", "John", "Daniel", "Michael"],
    target_keyframes=None,    # For controllable generation
)
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_path` | str | required | Output JSON file path |
| `--seq_lengths` | int[] | [1,2,4,8,16,32,64,128] | Sequence lengths |
| `--n_questions` | int | 50 | Questions per type per length |
| `--question_types` | str[] | all | Specific question types |
| `--seed` | int | 0xBADFACE | Random seed |
| `--sequential` | flag | false | Disable parallelization |

---

## Data Format

Output is a JSON array of sample objects:

```json
[
  {
    "qid": "0000001",
    "seq_len": 8,
    "qtype": "spend_alone",
    "atype": "person",
    "question": "Who spent the most time alone in the rooms?",
    "answer": "Sandra",
    "sequence": [
      {
        "step_id": 1,
        "rooms": {
          "Kitchen": ["Sandra", "Daniel"],
          "Bathroom": ["Mary"],
          "Garden": [],
          "Office": ["John"],
          "Bedroom": ["Michael"],
          "Hallway": []
        }
      },
      "..."
    ],
    "metadata": [
      {
        "step_id": 1,
        "rooms": {
          "Kitchen": false,
          "Bathroom": true,
          "Garden": false,
          "Office": true,
          "Bedroom": true,
          "Hallway": false
        }
      },
      "..."
    ],
    "n_relevant_rooms_per_step": [3, 0, 1, 2, 1, 0, 1, 2],
    "n_relevant_rooms": 10
  }
]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `qid` | string | Unique question ID (7 digits, zero-padded) |
| `seq_len` | int | Number of steps in sequence |
| `qtype` | string | Question type identifier |
| `atype` | string | Answer type: `"person"`, `"room"`, or `"number"` |
| `question` | string | Question text |
| `answer` | str\|int | Correct answer |
| `sequence` | Step[] | Sequence of room occupancy states |
| `metadata` | MetadataStep[] | Room relevance per step for the answer |
| `n_relevant_rooms_per_step` | int[] | Per-step count of relevant rooms (same length as `metadata`) |
| `n_relevant_rooms` | int | Total relevant room-step pairs across the whole sequence |

### Metadata Fields

#### `metadata`

The `metadata` field indicates which rooms at which steps are relevant for computing the answer:

- `true` = This room at this step contributed to the answer
- `false` = This room at this step is not relevant

For example, in `q_spend_alone`, metadata marks `true` for rooms where the answer character was alone at each step.

#### `n_relevant_rooms_per_step`

A list of integers with one value per step — the count of rooms whose metadata flag is `true` at that step.
It has the same length as `metadata` and is a step-level summary of it.

```python
# Example: 8 steps, answer character was alone in 1 room at steps 1, 4, 5, 7, 8
#          and in 2 rooms at step 3 (never happened, just illustration)
sample["n_relevant_rooms_per_step"]
# → [1, 0, 0, 1, 1, 0, 1, 1]
```

#### `n_relevant_rooms`

A single integer — the global sum of all `true` flags across every step and every room.
Equivalent to `sum(n_relevant_rooms_per_step)`.

```python
sample["n_relevant_rooms"]
# → 5   (five room-step pairs were relevant for answering the question)
```

### Programmatic Access

```python
from mmred import aggregate_metadata_step, aggregate_metadata_global, MetadataStep

# Recompute from a list of MetadataStep objects
per_step = aggregate_metadata_step(metadata_list)   # list[int]
total    = aggregate_metadata_global(metadata_list) # int
```

---

## Question Types

### NIAH (Needle-in-a-Haystack) Questions

These questions require finding specific information at one or few steps.

| Type | Question Pattern | Answer |
|------|------------------|--------|
| `first_app` | In which room did [Person] first appear? | room |
| `final_app` | In which room was [Person] at the final step? | room |
| `char_at_frame` | In which room was [Person] at step X? | room |
| `char_on_char_first_app` | In which room was [Person1] when [Person2] first appeared in [Room]? | room |
| `char_on_char_final_app` | Same as above, but "final appearance" | room |
| `first_at_room` | Who was the first to appear in [Room]? | person |
| `last_at_room` | Who was the last to appear in [Room]? | person |
| `room_at_frame` | Who was in [Room] at step X? | person |
| `room_on_char_first_app` | Who was in [Room1] when [Person] first appeared in [Room2]? | person |
| `room_on_char_final_app` | Same as above, but "final appearance" | person |
| `char_on_char_at_frame` | Who was in the same room as [Person] at step X? | person |
| `n_room_on_char_first_app` | How many characters were in [Room1] when [Person] first appeared in [Room2]? | number |
| `n_room_on_char_final_app` | Same as above, but "final appearance" | number |
| `n_char_at_frame` | How many other characters were in the same room as [Person] at step X? | number |
| `n_empty` | How many rooms were empty at step X? | number |

### MMLong (Aggregation) Questions

These questions require aggregating information across multiple steps.

| Type | Question Pattern | Answer |
|------|------------------|--------|
| `room_empty` | Which room was empty for more/fewer steps than others? | room |
| `where_spend` | In which room did [Person] spend the most/least time? | room |
| `crowded_room` | Which room was crowded (N+ people) for the most steps? | room |
| `who_spend` | Who spent the most/least time in [Room]? | person |
| `spend_alone` | Who spent the most/least time alone in the rooms? | person |
| `spend_together` | With whom did [Person] spend the most/least time together? | person |
| `steps_in_room` | How many steps did [Person] spend in [Room]? | number |
| `rooms_visited` | How many different rooms did [Person] visit? | number |
| `crowd_count` | How many times did a crowd (N+ people) appear? | number |

---

## Controllable Generation

For research requiring specific keyframe counts, use `target_keyframes`:

```python
config = GenerationConfig(
    target_keyframes=8,  # Exactly 8 relevant steps per question
    seq_lengths=[128],
    question_types=["spend_alone"],
)
```

### Keyframe Options

| Value | Behavior |
|-------|----------|
| `None` | Random (default) |
| `int` | Exactly N keyframes |
| `(min, max)` | Random in range |
| `[4, 8, 16]` | Generate samples for each count |

---

## Image Rendering

Render images separately from text generation:

```bash
# Render as PNG frames
python scripts/render_images.py \
    --input_path data/dataset.json \
    --output_dir data/images/

# Render as animated GIFs
python scripts/render_images.py \
    --input_path data/dataset.json \
    --output_dir data/gifs/ \
    --format gif

# Render specific samples
python scripts/render_images.py \
    --input_path data/dataset.json \
    --output_dir data/images/ \
    --qids 0000001 0000002
```

### Programmatic Rendering

```python
from mmred.vgen.visualization import render_sequence_from_json

# Render a single sequence
render_sequence_from_json(
    sample["sequence"],
    output_path="output/frames/",
    as_gif=False,
)

# Create GIF
render_sequence_from_json(
    sample["sequence"],
    output_path="output/sample.gif",
    as_gif=True,
)
```

---

## API Reference

### `mmred.GenerationConfig`

Configuration dataclass for dataset generation.

```python
@dataclass
class GenerationConfig:
    seed: int = 0xBADFACE
    seq_lengths: list[int] = [1, 2, 4, 8, 16, 32, 64, 128]
    n_questions: int = 50
    question_types: list[str] | None = None
    rooms: list[str] = ["Kitchen", ...]
    chars: list[str] = ["Sandra", ...]
    target_keyframes: int | tuple | list | None = None
```

### `mmred.generate_questions(config)`

Generate questions with parallelization.

**Args:** `config: GenerationConfig`  
**Returns:** `list[dict]` — List of sample dictionaries

### `mmred.generate_questions_sequential(config)`

Generate questions without parallelization (for debugging).

**Args:** `config: GenerationConfig`  
**Returns:** `list[dict]` — List of sample dictionaries

### `mmred.save_dataset(samples, output_path)`

Save generated samples to JSON file.

**Args:**
- `samples: list[dict]` — Generated samples
- `output_path: str | Path` — Output file path

### `mmred.Sample`, `mmred.Step`, `mmred.MetadataStep`

Data classes for structured data. Each has a `.to_dict()` method for serialization
and a `.from_dict()` class method for deserialization.

### `mmred.aggregate_metadata_step(metadata)`

Compute per-step relevance counts.

**Args:** `metadata: list[MetadataStep]`  
**Returns:** `list[int]` — One integer per step, counting rooms with `True` relevance flag

### `mmred.aggregate_metadata_global(metadata)`

Compute the total number of relevant room-step pairs.

**Args:** `metadata: list[MetadataStep]`  
**Returns:** `int` — Scalar sum of all `True` flags across all steps and rooms

### `mmred.QUESTIONS`

Dictionary mapping question type names to generator functions.

```python
from mmred import QUESTIONS
print(list(QUESTIONS.keys()))
# ['first_app', 'final_app', 'char_on_char_first_app', ...]
```

---

## Migration from v1.x

Key changes in v2.0:

1. **Output format**: JSON array with inline sequences (no separate CSV files)
2. **New `metadata` field**: Tracks room relevance per step
3. **New `n_relevant_rooms_per_step` field**: Per-step count of relevant rooms
4. **New `n_relevant_rooms` field**: Global count of relevant room-step pairs
5. **No video generation by default**: Use `scripts/render_images.py` separately
6. **Dynamic configuration**: `GenerationConfig` replaces static `const.py`
7. **Parallelized generation**: Uses `ProcessPoolExecutor` for speed

### Before (v1.x)

```bash
python scripts/generate_dataset.py --base_path ./data --exp_name main
# Creates: data/main/len_X/questions.json + sequences/*.csv + videos/
```

### After (v2.0)

```bash
python scripts/generate_dataset.py --output_path ./data/dataset.json
# Creates: data/dataset.json (single file with everything)

python scripts/render_images.py --input_path ./data/dataset.json --output_dir ./data/images/
# Creates: data/images/QIDXXXX/frame_XXXX.png
```
