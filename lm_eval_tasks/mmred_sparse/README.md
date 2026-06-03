# MMReD sparse tasks for lm-evaluation-harness

Standalone task package (not MERA). Evaluates text JSONL exported from sequence-bundle JSON.

## Prerequisites

```bash
cd /path/to/mmred
python -m pip install -e ".[lm_eval,serving]"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

- GPU with vLLM (or set `BACKEND=hf` for slower CPU/GPU HF path).
- Use an **Instruct** checkpoint when using `--apply_chat_template`. Valid HF ids include `Qwen/Qwen3-4B-Instruct-2507` (not `Qwen/Qwen3-4B-Instruct`, which does not exist). Base checkpoint: `Qwen/Qwen3-4B`.

## Full scaling pipeline (k = 0 … seq_len)

One script loops **generate bundle → export JSONL → lm-eval → episode-strict accuracy vs k**:

```bash
bash lm_eval_tasks/mmred_sparse/run_sparse_scaling_pipeline.sh \
  Qwen/Qwen3-4B-Instruct-2507 0
```

Equivalent wrapper from repo root:

```bash
bash scripts/run_lm_eval_sparse_scaling.sh Qwen/Qwen3-4B-Instruct-2507 0
```

### Main environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEQ_LEN` | `16` | Sequence length |
| `K_START` | `0` | First target-question count |
| `K_END` | `SEQ_LEN` | Last k (inclusive) |
| `N_EPISODES` | `100` | Episodes per k |
| `SEED` | `12345` | Dataset seed |
| `TARGET_STEP_STRATEGY` | `random` | `prefix` or `random` |
| `QUESTION_TYPES` | `spend_alone_at_step crowded_room` | Target + fillers |
| `TARGET_QUESTION_TYPE` | `spend_alone_at_step` | Sparse target type |
| `PREFIX_QUESTION` | `1` | Question before sequence in prompt |
| `THINKING` | `0` | Thinking-style export prompt |
| `REWRITE_DATA_CACHE` | `1` | Regenerate JSON/JSONL if set |
| `SKIP_LM_EVAL` | `0` | `1` = export only, no harness |
| `LM_EVAL_LIMIT` | unset | `--limit N` per task (smoke) |
| `GPU_ID` / `TP_SIZE` | `0` / `1` | CUDA devices |
| `OUTPUT_DIR` | `data/lm_eval_results` | Harness output root |
| `BUNDLE_JSON_DIR` | `data_cache/bundles` | Native bundle JSON |
| `LM_EVAL_JSONL_DIR` | `data/lm_eval/mmred_bundle` | Exported JSONL |

### Smoke test (few samples, few k)

```bash
LM_EVAL_LIMIT=8 N_EPISODES=2 K_START=0 K_END=2 \
  bash lm_eval_tasks/mmred_sparse/run_sparse_scaling_pipeline.sh \
  Qwen/Qwen3-4B-Instruct-2507 0
```

Preview commands without running:

```bash
K_START=0 K_END=16 bash lm_eval_tasks/mmred_sparse/run_sparse_scaling_pipeline.sh \
  Qwen/Qwen3-4B-Instruct-2507 0 --dry-run
```

### Output layout

```text
data_cache/bundles/<model_slug>/seq16/bundle_*_k{k}_*.json
data/lm_eval/mmred_bundle/seq16_k{k}_seed12345_tposrandom/test.jsonl
lm_eval_tasks/mmred_sparse/generated/mmred_sparse_bundle_seq16_k{k}_*.yaml   # gitignored
data/lm_eval_results/<model_slug>/seq16_thinking0_prefix1_tposrandom_seed12345/
  k0/                          # lm-eval output + samples_*.jsonl
  k1/
  ...
  accuracy_vs_k_episode_strict.csv
  accuracy_vs_target_questions.png
logs/lm_eval_sparse_<model>_k{k}_*.log
```

**Summary CSV** columns: `seq_len,k,accuracy,n_episodes` (episode-strict, target rows only, accuracy as fraction 0–1).

## Inspect generations / parser

After a run, each `k*/samples_*.jsonl` line has `doc.prompt`, `doc.gold`, `resps` (raw), `filtered_resps` (parsed).

```bash
python <<'PY'
import json, glob
from pathlib import Path
from lm_eval_tasks.mmred_sparse.utils import extract_answer, get_atype, score_prediction

root = Path("data/lm_eval_results/Qwen_Qwen3-4B-Instruct-2507/seq16_thinking0_prefix1_tposrandom_seed12345/k1")
for path in sorted(root.glob("samples_*.jsonl")):
    for line in open(path):
        s = json.loads(line)
        doc = s["doc"]
        raw = s["resps"][0][0]
        atype = get_atype(doc)
        print(doc["doc_id"], doc["meta"]["qtype"], "hit=", score_prediction(doc["gold"], raw, atype))
        print("  gold:", doc["gold"], "| parsed:", extract_answer(raw, atype))
        print("  raw:", raw[:300], "...")
        break
    break
PY
```

## Manual steps (single k)

```bash
# 1. Generate bundle JSON
python scripts/generate_bundle_dataset.py \
  --output_path data_cache/bundles/MODEL/seq16/bundle_..._k8_....json \
  --seq_len 16 --k_target 8 --n_episodes 100 \
  --target_question_type spend_alone_at_step \
  --question_types spend_alone_at_step crowded_room \
  --target_step_strategy random --seed 12345

# 2. Export JSONL
python scripts/export_lm_eval_dataset.py --mode bundle \
  --input_json <bundle.json> \
  --output_dir data/lm_eval/mmred_bundle \
  --seq_len 16 --seed 12345 --k_target 8 \
  --target_step_strategy random --prefix_question

# 3. Run harness (one task)
bash scripts/run_lm_eval_sparse.sh Qwen/Qwen3-4B-Instruct-2507 0 \
  mmred_sparse_bundle_seq16_k8_tposrandom

# 4. Aggregate
python scripts/eval/lm_eval_aggregate_sparse.py \
  --results_dir data/lm_eval_results/Qwen_Qwen3-4B-Instruct-2507 \
  --target_only --episode_strict
```

## Hand-maintained tasks

| Task | Dataset |
|------|---------|
| `mmred_sparse_bundle_seq16_k1_tposrandom` | `data/lm_eval/mmred_bundle/seq16_k1_seed12345_tposrandom/test.jsonl` |
| `mmred_sparse_bundle_seq16_k8_tposrandom` | `data/lm_eval/mmred_bundle/seq16_k8_seed12345_tposrandom/test.jsonl` |
| `mmred_sparse_qsplit_seq16_k8` | `data/lm_eval/mmred_qsplit/seq16_k8_seed12345/test.jsonl` |

The scaling pipeline writes additional task YAMLs under `generated/` (one per k).

## Scoring

Per-question exact match: [`utils.py`](utils.py) (MERA-style parsing, English normalization).

Episode-strict accuracy vs k: `scripts/eval/lm_eval_aggregate_sparse.py --episode_strict --target_only`.

Legacy CSV benchmarks: `scripts/utils/parse_answers.py` (separate pydantic + Mistral strip path).
