#!/usr/bin/env bash
#
# Full sequence-bundle sparse scaling pipeline via lm-evaluation-harness.
# For each k in [K_START..K_END]: generate bundle JSON -> export JSONL ->
# write task YAML -> lm_eval --log_samples -> aggregate episode-strict accuracy.
#
# Usage:
#   bash lm_eval_tasks/mmred_sparse/run_sparse_scaling_pipeline.sh <model_id> <gpu_id> [--dry-run]
#
# Env (override as needed):
#   SEQ_LEN, K_START (default 0), K_END (default SEQ_LEN), N_EPISODES, SEED
#   TARGET_STEP_STRATEGY, QUESTION_TYPES, TARGET_QUESTION_TYPE
#   PREFIX_QUESTION, THINKING, REWRITE_DATA_CACHE, SKIP_LM_EVAL
#   LM_EVAL_LIMIT, TP_SIZE, BACKEND, OUTPUT_DIR, BUNDLE_JSON_DIR, LM_EVAL_JSONL_DIR
#   PLOT_ACCURACY_PNG, RESULTS_TAG (auto if unset)
#
set -euo pipefail

MODEL_ID="${MODEL_ID:-}"
GPU_ID="${GPU_ID:-0}"
SEQ_LEN="${SEQ_LEN:-16}"
K_START="${K_START:-0}"
K_END="${K_END:-$SEQ_LEN}"
N_EPISODES="${N_EPISODES:-100}"
SEED="${SEED:-12345}"
TARGET_STEP_STRATEGY="${TARGET_STEP_STRATEGY:-random}"
QUESTION_TYPES="${QUESTION_TYPES:-spend_alone_at_step crowded_room}"
TARGET_QUESTION_TYPE="${TARGET_QUESTION_TYPE:-spend_alone_at_step}"
PREFIX_QUESTION="${PREFIX_QUESTION:-1}"
THINKING="${THINKING:-0}"
REWRITE_DATA_CACHE="${REWRITE_DATA_CACHE:-1}"
SKIP_LM_EVAL="${SKIP_LM_EVAL:-0}"
TP_SIZE="${TP_SIZE:-1}"
BACKEND="${BACKEND:-vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-data/lm_eval_results}"
BUNDLE_JSON_DIR="${BUNDLE_JSON_DIR:-data_cache/bundles}"
LM_EVAL_JSONL_DIR="${LM_EVAL_JSONL_DIR:-data/lm_eval/mmred_sparsity_scaling}"
PLOT_ACCURACY_PNG="${PLOT_ACCURACY_PNG:-1}"
DRY_RUN=0

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$@"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_DIR="$PROJECT_DIR/lm_eval_tasks"
GENERATED_DIR="$SCRIPT_DIR/generated"
mkdir -p "$GENERATED_DIR" "$PROJECT_DIR/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      head -n 22 "$0" | tail -n +2
      exit 0
      ;;
    *)
      if [[ -z "$MODEL_ID" ]]; then
        MODEL_ID="$1"
      elif [[ "$GPU_ID" == "0" && "$1" =~ ^[0-9]+$ ]]; then
        GPU_ID="$1"
      else
        echo "Unknown argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: $0 <model_id> <gpu_id> [--dry-run]" >&2
  echo "  or set MODEL_ID env var." >&2
  exit 1
fi

MODEL_SLUG="$(echo "$MODEL_ID" | tr '/' '_')"
_qt_tag="$(echo "$QUESTION_TYPES" | tr -s ' ' '_' | sed 's/^_//;s/_$//' | cut -c1-80)"
_tpos_tag="tpos${TARGET_STEP_STRATEGY}"
RESULTS_TAG="${RESULTS_TAG:-seq${SEQ_LEN}_thinking${THINKING}_prefix${PREFIX_QUESTION}_${_tpos_tag}_seed${SEED}}"

_pf=""
_tf=""
[[ "$PREFIX_QUESTION" -eq 1 ]] && _pf="--prefix_question"
[[ "$THINKING" -eq 1 ]] && _tf="--thinking"

RUN_ROOT="$OUTPUT_DIR/$MODEL_SLUG/$RESULTS_TAG"
SUMMARY_CSV="$RUN_ROOT/accuracy_vs_k_episode_strict.csv"
mkdir -p "$RUN_ROOT"

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "seq_len,k,accuracy,n_episodes" > "$SUMMARY_CSV"
fi

GPU_LIST="$GPU_ID"
for ((i = 1; i < TP_SIZE; i++)); do
  GPU_LIST="$GPU_LIST,$((GPU_ID + i))"
done

if [[ "$BACKEND" = "vllm" ]]; then
  MODEL_ARGS="pretrained=${MODEL_ID},tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=0.9,max_model_len=16384,trust_remote_code=True"
else
  MODEL_ARGS="pretrained=${MODEL_ID},tensor_parallel_size=${TP_SIZE}"
fi

_LIMIT_ARG=""
if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
  _LIMIT_ARG="--limit ${LM_EVAL_LIMIT}"
fi

cd "$PROJECT_DIR"

echo "=== MMReD sparse scaling (lm-eval) ==="
echo "MODEL_ID=$MODEL_ID GPU=$GPU_LIST SEQ_LEN=$SEQ_LEN k=${K_START}..${K_END}"
echo "N_EPISODES=$N_EPISODES SEED=$SEED TARGET_STEP_STRATEGY=$TARGET_STEP_STRATEGY"
echo "THINKING=$THINKING PREFIX_QUESTION=$PREFIX_QUESTION"
echo "RUN_ROOT=$RUN_ROOT"
echo "LM_EVAL_LIMIT=${LM_EVAL_LIMIT:-<none>} SKIP_LM_EVAL=$SKIP_LM_EVAL"
echo

write_task_yaml() {
  local task_name="$1"
  local jsonl_rel="$2"
  local yaml_path="$GENERATED_DIR/${task_name}.yaml"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "$yaml_path"
    return 0
  fi
  cat > "$yaml_path" <<EOF
include: ../mmred_sparse_base.yaml
task: ${task_name}
dataset_kwargs:
  data_files:
    train: ${jsonl_rel}
metadata:
  benchmark: sequence_bundle
  seq_len: ${SEQ_LEN}
  k_target: ${k}
  target_step_strategy: ${TARGET_STEP_STRATEGY}
  seed: ${SEED}
  thinking: ${THINKING}
  prefix_question: ${PREFIX_QUESTION}
EOF
  echo "$yaml_path"
}

for k in $(seq "$K_START" "$K_END"); do
  echo "================ k_target=$k / seq_len=$SEQ_LEN ================"

  GEN_JSON="${BUNDLE_JSON_DIR}/${MODEL_SLUG}/seq${SEQ_LEN}/bundle_${_qt_tag}_nq${N_EPISODES}_k${k}_seed${SEED}_${_tpos_tag}.json"
  JSONL_DIR="${LM_EVAL_JSONL_DIR}/seq${SEQ_LEN}_k${k}_seed${SEED}_${_tpos_tag}"
  JSONL_PATH="${JSONL_DIR}/test.jsonl"
  JSONL_REL="${JSONL_PATH}"
  TASK_NAME="mmred_sparse_bundle_seq${SEQ_LEN}_k${k}_seed${SEED}_${_tpos_tag}"
  K_RESULTS_DIR="${RUN_ROOT}/k${k}"
  LOGFILE="$PROJECT_DIR/logs/lm_eval_sparse_${MODEL_SLUG}_k${k}_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -f "$GEN_JSON" ]] || [[ "$REWRITE_DATA_CACHE" -eq 1 ]]; then
    run "python scripts/generate_bundle_dataset.py \
      --output_path \"${GEN_JSON}\" \
      --seq_len \"${SEQ_LEN}\" \
      --k_target \"${k}\" \
      --n_episodes \"${N_EPISODES}\" \
      --target_question_type \"${TARGET_QUESTION_TYPE}\" \
      --question_types ${QUESTION_TYPES} \
      --target_step_strategy \"${TARGET_STEP_STRATEGY}\" \
      --seed \"${SEED}\""
  else
    echo "Using existing bundle JSON: $GEN_JSON"
  fi

  if [[ ! -f "$JSONL_PATH" ]] || [[ "$REWRITE_DATA_CACHE" -eq 1 ]]; then
    run "python scripts/export_lm_eval_dataset.py \
      --mode bundle \
      --input_json \"${GEN_JSON}\" \
      --output_dir \"${LM_EVAL_JSONL_DIR}\" \
      --seq_len \"${SEQ_LEN}\" \
      --seed \"${SEED}\" \
      --k_target \"${k}\" \
      --target_step_strategy \"${TARGET_STEP_STRATEGY}\" \
      ${_pf} ${_tf}"
  else
    echo "Using existing JSONL: $JSONL_PATH"
  fi

  YAML_PATH="$(write_task_yaml "$TASK_NAME" "$JSONL_REL")"
  echo "Task YAML: $YAML_PATH (task=$TASK_NAME)"

  if [[ "$SKIP_LM_EVAL" -eq 1 ]]; then
    echo "SKIP_LM_EVAL=1: skipping lm_eval for k=$k"
    continue
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ CUDA_VISIBLE_DEVICES=$GPU_LIST lm_eval --model $BACKEND --tasks $TASK_NAME --output_path $K_RESULTS_DIR $_LIMIT_ARG ..."
    echo "+ python scripts/eval/lm_eval_aggregate_sparse.py --results_dir $K_RESULTS_DIR ..."
    continue
  fi

  mkdir -p "$K_RESULTS_DIR"
  {
    echo "=== lm-eval k=$k task=$TASK_NAME ==="
    CUDA_VISIBLE_DEVICES="$GPU_LIST" lm_eval \
      --model "$BACKEND" \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK_NAME" \
      --include_path "$SCRIPT_DIR" \
      --batch_size auto \
      --output_path "$K_RESULTS_DIR" \
      --log_samples \
      --apply_chat_template \
      $_LIMIT_ARG
  } 2>&1 | tee -a "$LOGFILE"

  {
  run "python scripts/eval/lm_eval_aggregate_sparse.py \
    --results_dir \"${K_RESULTS_DIR}\" \
    --target_only \
    --episode_strict \
    --out \"${K_RESULTS_DIR}/accuracy_vs_k.csv\""

  python - "$K_RESULTS_DIR" "$SUMMARY_CSV" "$SEQ_LEN" "$k" << 'PY'
import sys
from pathlib import Path
import pandas as pd

k_dir, summary, seq_len, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ep_path = Path(k_dir) / "accuracy_vs_k_episode_strict.csv"
if not ep_path.exists():
    raise SystemExit(f"Missing {ep_path}")
ep = pd.read_csv(ep_path)
row = ep[ep["k"].astype(int) == k]
if row.empty:
    row = ep.iloc[[0]]
acc_pct = float(row["accuracy"].iloc[0])
n_ep = int(row["n_episodes"].iloc[0])
acc_frac = acc_pct / 100.0 if acc_pct > 1.0 else acc_pct
with open(summary, "a") as f:
    f.write(f"{seq_len},{k},{acc_frac:.6f},{n_ep}\n")
print(f"Appended k={k}: accuracy={acc_frac:.4f} n_episodes={n_ep}")
PY
  }
done

if [[ "$PLOT_ACCURACY_PNG" -eq 1 && "$DRY_RUN" -eq 0 && -f "$SUMMARY_CSV" ]]; then
  run "python - \"$SUMMARY_CSV\" \"${RUN_ROOT}/accuracy_vs_target_questions.png\" \"$MODEL_ID\" \"$THINKING\" \"$PREFIX_QUESTION\" << 'PLOTEOF'
import sys
import pandas as pd
try:
    import matplotlib
    matplotlib.use(\"Agg\")
    import matplotlib.pyplot as plt
except ImportError:
    print(\"matplotlib not found; skip plot.\", file=sys.stderr)
    sys.exit(0)

csv_path, png_path, title_model, thinking, prefix = sys.argv[1:6]
df = pd.read_csv(csv_path)
df = df.sort_values(\"k\")
slen = int(df[\"seq_len\"].iloc[0])
sub = f\"episode_strict, thinking={thinking}, prefix_q={prefix}\"

plt.figure(figsize=(7, 5))
plt.plot(df[\"k\"].astype(int), df[\"accuracy\"], marker=\"o\", linestyle=\"-\", linewidth=2, markersize=8)
plt.xlabel(\"Number of target questions (k)\")
plt.ylabel(\"Sequence accuracy (fraction)\")
plt.title(f\"{title_model}\\nseq_len={slen}, {sub}\")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()
print(\"Saved\", png_path)
PLOTEOF"
fi

echo "Done. Summary: $SUMMARY_CSV"
if [[ -f "${RUN_ROOT}/accuracy_vs_target_questions.png" ]]; then
  echo "Plot: ${RUN_ROOT}/accuracy_vs_target_questions.png"
fi
