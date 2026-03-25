# CLAUDE.md — MMReD

## What Is This
MMReD benchmark: Multi-Modal Reasoning over Documents. NeurIPS follow-up submission.
Authors: Boris Shirokikh, Maxim Kurkin.

## Cost Discipline

| Task type | Model | Examples |
|---|---|---|
| Planning, architecture, debugging | **Opus** (you) | Design decisions, complex reasoning, code review |
| Implementation (>20 lines) | **Sonnet subagent** | New files, scripts, refactoring, tests |
| Exploration, search, summarization | **Haiku subagent** | Codebase search, file exploration |

**Hard rule**: before writing >20 lines of code yourself, launch a Sonnet `Agent` subagent. No exceptions.

## Operational Rules
- **Issues before code**: create a GitHub issue FIRST before any feature/fix. No exceptions.
- Use `fixes #N` in commits to auto-close issues.
- Atomic commits; plan large, implement small.
- Reproducibility: config + seed + commit hash mandatory in experiment logs.

## Environment
- Dev install: `pip install -e .`
- Remote counterpart: `kurkin-1` at `/workspace-SR004.nfs2/kurkin/long-vqa`, conda env `kurkin_313_torch`

## Key Directories
| Path | Purpose |
|---|---|
| `qgen/` | Question generation |
| `vgen/` | Visualization generation |
| `scripts/` | Utility scripts & figures |
| `long-vqa/` | MERA integration, training |
| `train/` | Model training scripts |
| `mera_integration/` | MERA leaderboard tasks |
| `config.py` | Configuration |
| `data_model.py` | Data models |

## MERA Integration
- Dataset: `dondosss/mmred_mera` on HuggingFace
- Structure: 5 DC tasks × 3 lengths
- PR target: `MERA-Evaluation/MERA` branch `v2_dev`

## Design Style
NeurIPS publication-quality figures. Clean, serif fonts by default. xkcd style when requested.

## Git
- Work on feature branches. Commit & push by default.
- Prefix: `[feat]`, `[fix]`, `[fig]`, `[doc]`, `[data]`
