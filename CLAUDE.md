# CLAUDE.md — MMReD / Long-VQA

## Project Identity
**MMReD (Multimodal Reasoning Evaluation Dataset)**: A cross-modal benchmark for dense context reasoning in vision-language models. Evaluates long-context understanding where models must aggregate information across extended sequences.

Authors: Boris Shirokikh, Maxim Kurkin

## Repository Layout
```
long-vqa/
├── mmred/                    # Core library (pip-installable)
│   ├── config.py             # GenerationConfig dataclass
│   ├── const.py              # Colors, room/character names, dimensions
│   ├── data_model.py         # Sample, Step, MetadataStep dataclasses
│   ├── localization.py       # EN/RU question templates
│   ├── in_context.py         # Few-shot example generation
│   ├── qgen/                 # Question generation (24 types)
│   │   ├── questions.py      # All question implementations
│   │   ├── qgen.py           # Parallelized orchestration
│   │   └── utils.py
│   └── vgen/                 # Image/GIF rendering (matplotlib)
├── scripts/                  # CLI entry points
│   ├── generate_dataset.py   # Main dataset generation
│   ├── generate_mera_dataset.py  # MERA-format generation
│   ├── render_images.py      # PNG/GIF rendering
│   └── openai_server_inference.py
├── train/                    # Model training
│   ├── train_sft.py          # Supervised fine-tuning
│   ├── train_rmt_*.py        # RMT (Recurrent Memory Transformer)
│   ├── train_trl.py          # TRL-based training
│   ├── config_*.yaml         # Training configs (SFT, RMT, GRPO)
│   ├── modeling_rmt/         # Custom RMT model
│   └── rewards.py            # Reward functions
├── mera_integration/         # MERA leaderboard tasks (5 DC types × 3 lengths)
├── notebooks/                # Analysis notebooks
├── data/                     # Sample data + generated images
├── pyproject.toml            # Package metadata (v2.0.0)
└── test_integration.py       # Integration tests
```

## Question Taxonomy
**NIAH (15 types)**: Needle-in-a-Haystack — find info at specific steps
- first_app, final_app, char_at_frame, char_on_char_*, room queries, counting

**DC (9 types)**: Dense Context — aggregate across multiple steps
- spend_alone, where_spend, spend_together, who_spend, steps_in_room, rooms_visited, crowded_room, crowd_count, room_empty

## Key Technical Details
- Sequence lengths: [1, 2, 4, 8, 16, 32, 64, 128]
- 5 characters, 6 rooms, configurable
- Parallelized generation (ProcessPoolExecutor)
- Multilingual: EN + RU with proper grammar
- MERA evaluation: Exact Match, 2× weight on length-128

## Training Approaches
1. **SFT**: Qwen3-4B-Instruct, LoRA r=32, 5 epochs
2. **RMT**: Segment=256, memory_tokens=32, max_segments=16, curriculum learning
3. **GRPO**: 1.5B/3B/7B variants with domain-specific rewards

## Remote Counterpart
Full codebase also on **kurkin-1** at `/workspace-SR004.nfs2/kurkin/long-vqa`
Conda env: `kurkin_313_torch`

## How to Work on This Project
- **Dataset generation**: `python scripts/generate_dataset.py --seq_lengths 16 32 64 128 --n_questions 100`
- **Rendering**: `python scripts/render_images.py --input_path data/dataset.json --output_dir data/images/`
- **Training**: SSH to kurkin-1, activate env, run training scripts
- **MERA submission**: Use mera_integration/ configs

## Current Status
- Core library complete (v2.0.0)
- 24 question types implemented
- MERA integration with 15 task configs
- Training pipelines for SFT, RMT, GRPO ready
- Next: finalize MERA PR, run comprehensive evaluation

## Conventions
- Package installable via `pip install -e .`
- Core dependency: pandas only; training deps are optional
- Seed: 0xBADFACE (default reproducibility)
