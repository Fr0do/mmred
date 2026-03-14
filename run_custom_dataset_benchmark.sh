#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# Configuration (override via env vars)
###############################################################################

DATASET_JSON="${DATASET_JSON:-data/custom_dataset.json}"
EXP_NAME="${EXP_NAME:-qwen3_instruct_4b}"

SEQ_LENGTHS="${SEQ_LENGTHS:-16 32 64 128 256 512}"
QUESTION_TYPES="${QUESTION_TYPES:-spend_alone_at_step}"
N_QUESTIONS="${N_QUESTIONS:-1200}"
SEED="${SEED:-12345}"

VLLM_PORT="${VLLM_PORT:-8003}"
SEMAPHORE_LIMIT="${SEMAPHORE_LIMIT:-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREFIX_QUESTION="${PREFIX_QUESTION:-1}"
THINKING="${THINKING:-0}"

MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"

INPUT_DIR="${INPUT_DIR:-data_cache/${MODEL_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-mmred/results_${MODEL_NAME}}"

REWRITE_DATA_CACHE="${REWRITE_DATA_CACHE:-0}"
BENCHMARK_PER_SEQ_LENGTH="${BENCHMARK_PER_SEQ_LENGTH:-0}"
PLOT_ACCURACY_PNG="${PLOT_ACCURACY_PNG:-1}"

DRY_RUN=0

###############################################################################
# Derived names
###############################################################################

_seq_tag="seq$(echo "$SEQ_LENGTHS" | tr -s ' ' '_' | tr -d ' ')"
_qt_tag="$(echo "$QUESTION_TYPES" | tr -s ' ' '_' | sed 's/^_//;s/_$//' | cut -c1-40)"
_run_label="${_seq_tag}_${N_QUESTIONS}q_${_qt_tag}_seed${SEED}_thinking${THINKING}_prefixq${PREFIX_QUESTION}"
RUN_LABEL="${RUN_LABEL:-$_run_label}"

[[ -z "$EXP_NAME" ]] && EXP_NAME="run_${RUN_LABEL}"
OUTPUT_DIR="${INPUT_DIR}/${EXP_NAME}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env: DATASET_JSON, EXP_NAME, SEQ_LENGTHS, QUESTION_TYPES, N_QUESTIONS, SEED,
     VLLM_PORT, MODEL_NAME, INPUT_DIR, RESULTS_DIR, REWRITE_DATA_CACHE,
     BENCHMARK_PER_SEQ_LENGTH (1=per seq length + plot), PLOT_ACCURACY_PNG.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

run() {
  echo "+ $*"
  [[ "$DRY_RUN" -eq 0 ]] && eval "$@"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Config ==="
echo "MODEL_NAME=$MODEL_NAME"
echo "DATASET_JSON=$DATASET_JSON EXP_NAME=$EXP_NAME RUN_LABEL=$RUN_LABEL"
echo "SEQ_LENGTHS=$SEQ_LENGTHS QUESTION_TYPES=$QUESTION_TYPES N_QUESTIONS=$N_QUESTIONS SEED=$SEED"
echo "VLLM_PORT=$VLLM_PORT RESULTS_DIR=$RESULTS_DIR BENCHMARK_PER_SEQ_LENGTH=$BENCHMARK_PER_SEQ_LENGTH"
echo

###############################################################################
# vLLM health check
###############################################################################
echo "Checking vLLM on port $VLLM_PORT ..."
if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! curl -sS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Error: vLLM not reachable on port $VLLM_PORT. Start it first (e.g. vllm_servers.sh)." >&2
    exit 1
  fi
fi
echo "vLLM reachable."
echo

if [[ "$BENCHMARK_PER_SEQ_LENGTH" -eq 1 ]]; then

###############################################################################
# Per-seq-length: generate -> inference -> parse per length, then plot
###############################################################################
  _qt_loop="$(echo "$QUESTION_TYPES" | tr -s ' ' '_' | sed 's/^_//;s/_$//' | cut -c1-40)"
  SUMMARY_CSV="${RESULTS_DIR}/accuracy_per_seq_length.csv"
  PLOT_PNG="${RESULTS_DIR}/accuracy_vs_seq_length.png"
  mkdir -p "$RESULTS_DIR"
  echo "seq_len,accuracy" > "$SUMMARY_CSV"

  for _seq in $SEQ_LENGTHS; do
    echo "========== seq_len=$_seq =========="
    D_JSON="data/custom_seq${_seq}.json"
    _rl="seq${_seq}_${N_QUESTIONS}q_${_qt_loop}_seed${SEED}_thinking${THINKING}_prefixq${PREFIX_QUESTION}"
    _exp="run_${_rl}"
    _out_dir="${INPUT_DIR}/${_exp}"
    _raw_csv="${_out_dir}/qa_pairs_answers_${_rl}.csv"
    _res_csv="${RESULTS_DIR}/${_exp}_newest_results.csv"

    if [[ -f "$D_JSON" && "${FORCE_REGEN:-0}" -ne 1 ]]; then
      echo "Dataset exists: $D_JSON"
    else
      run "python scripts/generate_dataset.py --output_path \"$D_JSON\" --seq_lengths $_seq --n_questions \"$N_QUESTIONS\" --question_types $QUESTION_TYPES --seed \"$SEED\""
    fi
    [[ "$DRY_RUN" -eq 0 && ! -f "$D_JSON" ]] && { echo "Missing $D_JSON" >&2; exit 1; }

    mkdir -p "$_out_dir"
    [[ "$REWRITE_DATA_CACHE" -eq 1 && -f "$_raw_csv" && "$DRY_RUN" -eq 0 ]] && rm -f "$_raw_csv"

    _pf="" _tf=""
    [[ "$PREFIX_QUESTION" -eq 1 ]] && _pf="--prefix_question"
    [[ "$THINKING" -eq 1 ]] && _tf="--thinking"

    run "python scripts/openai_server_inference.py --port \"$VLLM_PORT\" --text_json_path \"$D_JSON\" --exp_name \"$_exp\" --semaphore_limit \"$SEMAPHORE_LIMIT\" --batch_size \"$BATCH_SIZE\" $_pf $_tf --output_csv \"$_raw_csv\""
    run "python scripts/utils/parse_answers.py --exp_name \"$_exp\" --input_dir \"$INPUT_DIR\" --output_dir \"$RESULTS_DIR\" --debug"

    if [[ "$DRY_RUN" -eq 0 && -f "$_res_csv" ]]; then
      _acc=$(python -c "import pandas as pd; df=pd.read_csv('$_res_csv'); print(f\"{(df['hit'].mean()*100):.2f}\")" 2>/dev/null || echo "0")
      echo "$_seq,$_acc" >> "$SUMMARY_CSV"
      echo "Accuracy seq_len=$_seq: ${_acc}%"
    fi
    echo
  done

  if [[ "$PLOT_ACCURACY_PNG" -eq 1 && "$DRY_RUN" -eq 0 && -f "$SUMMARY_CSV" ]]; then
    echo "Plotting $PLOT_PNG ..."
    python - "$SUMMARY_CSV" "$PLOT_PNG" << 'PLOTEOF'
import sys
import pandas as pd
try:
  import matplotlib; matplotlib.use('Agg')
  import matplotlib.pyplot as plt
except ImportError:
  print("matplotlib not found; skip plot.", file=sys.stderr)
  sys.exit(0)
p, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(p)
if len(df) == 0: sys.exit(0)
df = df.sort_values('seq_len')
plt.figure(figsize=(6, 4))
plt.plot(df['seq_len'].astype(int), df['accuracy'], marker='o', linestyle='-')
plt.xlabel('Sequence length')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.title('Accuracy vs sequence length')
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print("Saved", out)
PLOTEOF
  fi

  echo "=== Summary (per-seq) ==="
  echo "Summary: $SUMMARY_CSV  Plot: $PLOT_PNG"
  [[ -f "$SUMMARY_CSV" ]] && cat "$SUMMARY_CSV"

else

###############################################################################
# Single run: generate -> inference -> parse -> accuracy summary
###############################################################################
  if [[ -f "$DATASET_JSON" && "${FORCE_REGEN:-0}" -ne 1 ]]; then
    echo "Dataset exists: $DATASET_JSON (set FORCE_REGEN=1 to regenerate)."
  else
    echo "Generating dataset: $DATASET_JSON"
    run "python scripts/generate_dataset.py --output_path \"$DATASET_JSON\" --seq_lengths $SEQ_LENGTHS --n_questions \"$N_QUESTIONS\" --question_types $QUESTION_TYPES --seed \"$SEED\""
  fi
  [[ "$DRY_RUN" -eq 0 && ! -f "$DATASET_JSON" ]] && { echo "Missing $DATASET_JSON" >&2; exit 1; }

  TEXT_JSON_PATH="${TEXT_JSON_PATH:-$DATASET_JSON}"
  CACHED_DATASET="${OUTPUT_DIR}/generated_dataset.json"
  mkdir -p "$OUTPUT_DIR"
  [[ "$DRY_RUN" -eq 0 ]] && cp "$DATASET_JSON" "$CACHED_DATASET"

  RAW_CSV="${OUTPUT_DIR}/qa_pairs_answers_${RUN_LABEL}.csv"
  [[ "$REWRITE_DATA_CACHE" -eq 1 && -f "$RAW_CSV" && "$DRY_RUN" -eq 0 ]] && rm -f "$RAW_CSV"

  _pf="" _tf=""
  [[ "$PREFIX_QUESTION" -eq 1 ]] && _pf="--prefix_question"
  [[ "$THINKING" -eq 1 ]] && _tf="--thinking"

  echo "Running inference -> $RAW_CSV"
  run "python scripts/openai_server_inference.py --port \"$VLLM_PORT\" --text_json_path \"$TEXT_JSON_PATH\" --exp_name \"$EXP_NAME\" --semaphore_limit \"$SEMAPHORE_LIMIT\" --batch_size \"$BATCH_SIZE\" $_pf $_tf --output_csv \"$RAW_CSV\""
  [[ "$DRY_RUN" -eq 0 && ! -f "$RAW_CSV" ]] && { echo "Missing $RAW_CSV" >&2; exit 1; }

  echo "Running parse_answers ..."
  run "python scripts/utils/parse_answers.py --exp_name \"$EXP_NAME\" --input_dir \"$INPUT_DIR\" --output_dir \"$RESULTS_DIR\" --debug"

  RESULTS_CSV="${RESULTS_DIR}/${EXP_NAME}_newest_results.csv"
  if [[ "$DRY_RUN" -eq 0 && -f "$RESULTS_CSV" ]]; then
    echo "Computing accuracy ..."
    python -c "
import pandas as pd
from pathlib import Path
p = Path('$RESULTS_CSV')
df = pd.read_csv(p)
if 'hit' in df.columns:
    acc = df['hit'].mean()
    print(f'Accuracy: {acc:.2f}% (n={len(df)})')
else:
    print('No hit column in', p)
"
  fi

  echo "=== Summary ==="
  echo "Dataset:  $DATASET_JSON"
  echo "Cache:    $CACHED_DATASET"
  echo "Raw CSV:  $RAW_CSV"
  echo "Results:  $RESULTS_CSV"
  echo "Exp:      $EXP_NAME"
fi

echo "Done."
