#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# Configuration (override via env vars)
###############################################################################

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"

# You must have vLLM server running on this port (see vllm_servers.sh).
VLLM_PORT="${VLLM_PORT:-8003}"

# Root for inference CSVs and parse_answers inputs.
INPUT_DIR="${INPUT_DIR:-data_cache/${MODEL_NAME}}"

# Root for parse_answers outputs + plots.
RESULTS_DIR="${RESULTS_DIR:-mmred/results_${MODEL_NAME}}"

# Comma/space separated seq lengths.
SEQ_LENGTHS="${SEQ_LENGTHS:-16}"

SEED="${SEED:-12345}"
QUESTION_TYPES="${QUESTION_TYPES:-spend_alone_at_step crowded_room}"
TARGET_QUESTION_TYPE="${TARGET_QUESTION_TYPE:-spend_alone_at_step}"

# Optional: total questions per split JSON for each seq_len (must divide that seq_len).
# Unset => split_total questions per file (multiplier 1 in generate_dataset.py).

SEMAPHORE_LIMIT="${SEMAPHORE_LIMIT:-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREFIX_QUESTION="${PREFIX_QUESTION:-1}"
THINKING="${THINKING:-0}"

# Controls whether we overwrite generated split JSONs and inference CSVs.
# (This script always overwrites parse_answers outputs; inference can be skipped by existing CSVs.)
REWRITE_DATA_CACHE="${REWRITE_DATA_CACHE:-1}"

PLOT_ACCURACY_PNG="${PLOT_ACCURACY_PNG:-1}"

# If 1: force regeneration of split JSONs for each seq_len.
FORCE_SPLIT_REGEN="${FORCE_SPLIT_REGEN:-1}"

DRY_RUN=0

###############################################################################
# Helpers
###############################################################################

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Required env:
  - TARGET_QUESTION_TYPE
  - QUESTION_TYPES (must include TARGET_QUESTION_TYPE and at least one other)

Optional env:
  - MODEL_NAME (default: default_model) used for folder separation
  - VLLM_PORT (default: 8003)
  - SEQ_LENGTHS (default: "16 32 64 128 256")
  - SEED (default: 12345)
  - N_QUESTIONS (optional) total questions in each split JSON for that seq_len;
      must be divisible by seq_len. Unset => seq_len (same as multiplier 1).
  - SEMAPHORE_LIMIT, BATCH_SIZE
  - PREFIX_QUESTION, THINKING
  - REWRITE_DATA_CACHE (default: 0)
  - FORCE_SPLIT_REGEN (default: 0)
  - PLOT_ACCURACY_PNG (default: 1)
EOF
}

run() {
  echo "+ $*"
  [[ "$DRY_RUN" -eq 0 ]] && eval "$@"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

echo "=== Config ==="
echo "MODEL_NAME=$MODEL_NAME TARGET_QUESTION_TYPE=$TARGET_QUESTION_TYPE"
echo "QUESTION_TYPES=$QUESTION_TYPES"
echo "SEQ_LENGTHS=$SEQ_LENGTHS SEED=$SEED N_QUESTIONS=${N_QUESTIONS:-1200}"
echo "VLLM_PORT=$VLLM_PORT"
echo "INPUT_DIR=$INPUT_DIR RESULTS_DIR=$RESULTS_DIR"
echo "PLOT_ACCURACY_PNG=$PLOT_ACCURACY_PNG REWRITE_DATA_CACHE=$REWRITE_DATA_CACHE FORCE_SPLIT_REGEN=$FORCE_SPLIT_REGEN"
echo "THINKING=$THINKING PREFIX_QUESTION=$PREFIX_QUESTION"
echo

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Checking vLLM on port $VLLM_PORT ..."
  if ! curl -sS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Error: vLLM not reachable on port $VLLM_PORT. Start it first (e.g. vllm_servers.sh)." >&2
    exit 1
  fi
  echo "vLLM reachable."
  echo
fi

mkdir -p "$RESULTS_DIR"

_qt_tag="$(echo "$QUESTION_TYPES" | tr -s ' ' '_' | sed 's/^_//;s/_$//' | cut -c1-80)"

_pf=""
_tf=""
[[ "$PREFIX_QUESTION" -eq 1 ]] && _pf="--prefix_question"
[[ "$THINKING" -eq 1 ]] && _tf="--thinking"

###############################################################################
# Core plotting helper
###############################################################################

plot_for_seq_len() {
  local seq_len="$1"
  local csv_path="$2"
  local png_path="$3"
  local model_name="${4:-unknown_model}"

  if [[ "$PLOT_ACCURACY_PNG" -ne 1 ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" -ne 0 ]]; then
    return 0
  fi
  if [[ ! -f "$csv_path" ]]; then
    echo "Skip plot: missing $csv_path"
    return 0
  fi

  python - "$csv_path" "$png_path" "$model_name" << 'PLOTEOF'
import sys
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found; skip plot.", file=sys.stderr)
    sys.exit(0)

csv_path, png_path, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(csv_path)
df = df.sort_values("k")
slen = int(df["seq_len"].iloc[0])

plt.figure(figsize=(7, 5))
plt.plot(df["k"].astype(int), df["accuracy"], marker="o", linestyle="-", linewidth=2, markersize=8)
plt.xlabel("Number of target questions (k)")
plt.ylabel("Accuracy (%)")
plt.title(f"{model_name}: Accuracy vs target questions (seq_len={slen})")
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()
print("Saved", png_path)
PLOTEOF
}

###############################################################################
# Main loop: for each seq_len, generate question-split JSONs, run inference, parse,
# then plot accuracy(k).
###############################################################################

for SEQ in $SEQ_LENGTHS; do
  echo "================ seq_len=$SEQ ================"

  split_total="$SEQ"
  if [[ "$split_total" -lt 1 ]]; then
    echo "Error: seq_len must be >= 1 (got $split_total)" >&2
    exit 1
  fi

  Q_PER_SPLIT="${N_QUESTIONS:-$split_total}"
  if ! [[ "$Q_PER_SPLIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: N_QUESTIONS (questions per split file) must be a positive integer, got: '$Q_PER_SPLIT'" >&2
    exit 1
  fi
  if (( Q_PER_SPLIT % split_total != 0 )); then
    echo "Error: N_QUESTIONS=$Q_PER_SPLIT must be divisible by seq_len=$split_total (for integer split multiplier)." >&2
    exit 1
  fi
  NQ_MULTIPLIER=$(( Q_PER_SPLIT / split_total ))
  echo "Questions per split file: $Q_PER_SPLIT (generate_dataset --n_questions multiplier: $NQ_MULTIPLIER)"
  echo

  width="${#split_total}"
  [[ "$width" -lt 2 ]] && width=2

  # Base path: generator will emit:
  #   <base>_split_00.json, ... , <base>_split_${split_total}.json
  GENERATION_BASE="${INPUT_DIR}/generated_datasets/${MODEL_NAME}/seq${SEQ}/qsplit_${TARGET_QUESTION_TYPE}_nq${Q_PER_SPLIT}_${_qt_tag}.json"

  if [[ "$FORCE_SPLIT_REGEN" -eq 1 ]]; then
    rm -f "${GENERATION_BASE%.*}"_split_*.json >/dev/null 2>&1 || true
  fi

  if [[ ! -f "${GENERATION_BASE%.*}_split_00.json" || "$FORCE_SPLIT_REGEN" -eq 1 ]]; then
    run "python scripts/generate_dataset.py \
      --output_path \"${GENERATION_BASE}\" \
      --seq_lengths \"${SEQ}\" \
      --question_split \
      --target_question_type \"${TARGET_QUESTION_TYPE}\" \
      --question_types ${QUESTION_TYPES} \
      --split_total \"${split_total}\" \
      --n_questions \"${NQ_MULTIPLIER}\" \
      --seed \"${SEED}\""
  else
    echo "Split JSONs exist, skipping generation (set FORCE_SPLIT_REGEN=1 to regenerate)."
  fi

  # Accumulator for this seq_len
  SEQ_RESULTS_DIR="${RESULTS_DIR}/question_split/seq${SEQ}"
  mkdir -p "$SEQ_RESULTS_DIR"
  SUMMARY_CSV="${SEQ_RESULTS_DIR}/accuracy_vs_k.csv"
  echo "seq_len,k,accuracy" > "$SUMMARY_CSV"

  for k in $(seq 0 "$split_total"); do
    split_json="${GENERATION_BASE%.*}_split_$(printf "%0${width}d" "$k").json"
    if [[ ! -f "$split_json" ]]; then
      echo "Missing split json: $split_json" >&2
      exit 1
    fi

    # Unique exp name per split so parse_answers can glob safely.
    exp_name="qsplit_seq${SEQ}_k${k}_nq${Q_PER_SPLIT}_seed${SEED}_thinking${THINKING}_prefix${PREFIX_QUESTION}_${_qt_tag}"

    raw_csv="${INPUT_DIR}/${exp_name}/qa_pairs_answers_${k}_${TARGET_QUESTION_TYPE}.csv"
    results_csv="${RESULTS_DIR}/${exp_name}_newest_results.csv"

    mkdir -p "$(dirname "$raw_csv")"

    if [[ "$REWRITE_DATA_CACHE" -eq 1 && -f "$raw_csv" ]]; then
      run "rm -f \"${raw_csv}\""
    fi

    if [[ ! -f "$raw_csv" || "$REWRITE_DATA_CACHE" -eq 1 ]]; then
      run "python scripts/openai_server_inference.py \
        --port \"${VLLM_PORT}\" \
        --text_json_path \"${split_json}\" \
        --exp_name \"${exp_name}\" \
        --semaphore_limit \"${SEMAPHORE_LIMIT}\" \
        --batch_size \"${BATCH_SIZE}\" \
        ${_pf} ${_tf} \
        --output_csv \"${raw_csv}\""
    else
      echo "Inference CSV exists, skipping inference: $raw_csv"
    fi

    # Compute metrics for this split.
    run "python scripts/utils/parse_answers.py \
      --exp_name \"${exp_name}\" \
      --input_dir \"${INPUT_DIR}\" \
      --output_dir \"${RESULTS_DIR}\""

    if [[ -f "$results_csv" ]]; then
      # parse_answers outputs hit in percentage already (it multiplies by 100 internally).
      acc="$(python -c "import pandas as pd; df=pd.read_csv('${results_csv}'); print(df['hit'].mean())")"
      echo "${SEQ},${k},${acc}" >> "$SUMMARY_CSV"
      echo "k=${k} acc=${acc}%"
    else
      echo "Missing results csv: $results_csv" >&2
      exit 1
    fi
  done

  plot_for_seq_len "$SEQ" "$SUMMARY_CSV" "${SEQ_RESULTS_DIR}/accuracy_vs_target_questions.png" "$MODEL_NAME"
done

echo "Done."

