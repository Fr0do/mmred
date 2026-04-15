#!/usr/bin/env bash
#
# Sequence-bundle benchmark: shared-sequence episodes, sequence-level accuracy vs k.
#
# For each k in [K_START..K_END], generates N_EPISODES episodes; each episode has
# bundle_size questions (default bundle_size = SEQ_LEN): k target questions
# (spend_alone_at_step at steps 1..k) plus fillers on the same sequence.
# Accuracy = fraction of episodes where scoring passes (strict: all correct;
# at_least: >= BUNDLE_MIN_CORRECT correct).
#
# Env (override as needed):
#   MODEL_NAME, VLLM_PORT, INPUT_DIR, RESULTS_DIR
#   SEQ_LEN, N_EPISODES, SEED, QUESTION_TYPES, TARGET_QUESTION_TYPE
#   K_START, K_END (default 1 .. SEQ_LEN; set K_START=0 to include k=0)
#   BUNDLE_SIZE (optional; default equals SEQ_LEN)
#   BUNDLE_SCORING=strict|at_least, BUNDLE_MIN_CORRECT (required if at_least)
#   REWRITE_DATA_CACHE, PLOT_ACCURACY_PNG, SEMAPHORE_LIMIT, BATCH_SIZE
#   PREFIX_QUESTION, THINKING (included in exp_name and in
#     \$RESULTS_DIR/sequence_bundle/seq\${SEQ_LEN}_thinking\${THINKING}_prefix\${PREFIX_QUESTION}_<scoring>/)
#
# vLLM: start your server separately (e.g. edit and run vllm_servers.sh), or set
#   AUTOSTART_VLLM=1 VLLM_GPU=0  for a minimal single-GPU text serve (requires vllm CLI).
#
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
VLLM_PORT="${VLLM_PORT:-8003}"
INPUT_DIR="${INPUT_DIR:-data_cache/${MODEL_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-mmred/results_${MODEL_NAME}}"

SEQ_LEN="${SEQ_LEN:-16}"
N_EPISODES="${N_EPISODES:-100}"
SEED="${SEED:-12345}"
QUESTION_TYPES="${QUESTION_TYPES:-spend_alone_at_step crowded_room}"
TARGET_QUESTION_TYPE="${TARGET_QUESTION_TYPE:-spend_alone_at_step}"

K_START="${K_START:-1}"
K_END="${K_END:-$SEQ_LEN}"
BUNDLE_SIZE="${BUNDLE_SIZE:-}"

BUNDLE_SCORING="${BUNDLE_SCORING:-strict}"
BUNDLE_MIN_CORRECT="${BUNDLE_MIN_CORRECT:-}"

REWRITE_DATA_CACHE="${REWRITE_DATA_CACHE:-1}"
PLOT_ACCURACY_PNG="${PLOT_ACCURACY_PNG:-1}"
AUTOSTART_VLLM="${AUTOSTART_VLLM:-0}"
VLLM_GPU="${VLLM_GPU:-0}"

SEMAPHORE_LIMIT="${SEMAPHORE_LIMIT:-32}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PREFIX_QUESTION="${PREFIX_QUESTION:-1}"
THINKING="${THINKING:-1}"

DRY_RUN=0

run() {
  echo "+ $*"
  [[ "$DRY_RUN" -eq 0 ]] && eval "$@"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      head -n 25 "$0" | tail -n +2
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

echo "=== Sequence bundle benchmark ==="
echo "MODEL_NAME=$MODEL_NAME SEQ_LEN=$SEQ_LEN N_EPISODES=$N_EPISODES"
echo "K_START=$K_START K_END=$K_END BUNDLE_SCORING=$BUNDLE_SCORING"
echo "THINKING=$THINKING PREFIX_QUESTION=$PREFIX_QUESTION"
echo "QUESTION_TYPES=$QUESTION_TYPES TARGET=$TARGET_QUESTION_TYPE"
echo

if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ "$AUTOSTART_VLLM" -eq 1 ]]; then
    echo "AUTOSTART_VLLM=1: starting vllm on GPU $VLLM_GPU port $VLLM_PORT (background)..."
    export CUDA_VISIBLE_DEVICES="$VLLM_GPU"
    vllm serve "$MODEL_NAME" --port "$VLLM_PORT" --trust-remote-code --max-model-len 16384 &
    VLLM_PID=$!
    sleep 15
    trap '[[ -n "${VLLM_PID:-}" ]] && kill $VLLM_PID 2>/dev/null || true' EXIT
  fi
  echo "Checking vLLM on port $VLLM_PORT ..."
  if ! curl -sS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Error: vLLM not reachable on port $VLLM_PORT. Start the server first or use AUTOSTART_VLLM=1." >&2
    exit 1
  fi
  echo "vLLM reachable."
  echo
fi

mkdir -p "$INPUT_DIR/generated_bundles/$MODEL_NAME/seq${SEQ_LEN}"

_qt_tag="$(echo "$QUESTION_TYPES" | tr -s ' ' '_' | sed 's/^_//;s/_$//' | cut -c1-80)"
_model_slug="$(echo "$MODEL_NAME" | tr '/' '_')"

_pf=""
_tf=""
[[ "$PREFIX_QUESTION" -eq 1 ]] && _pf="--prefix_question"
[[ "$THINKING" -eq 1 ]] && _tf="--thinking"

# Disambiguate aggregate CSV/PNG from THINKING, PREFIX_QUESTION, and scoring mode.
# exp_name already includes t*/p* per run; generated JSON does not depend on inference flags.
_score_tag="${BUNDLE_SCORING}"
if [[ "$BUNDLE_SCORING" == "at_least" && -n "${BUNDLE_MIN_CORRECT:-}" ]]; then
  _score_tag="at_least_mc${BUNDLE_MIN_CORRECT}"
fi
SEQUENCE_BUNDLE_OUT="${RESULTS_DIR}/sequence_bundle/seq${SEQ_LEN}_thinking${THINKING}_prefix${PREFIX_QUESTION}_${_score_tag}"
mkdir -p "$SEQUENCE_BUNDLE_OUT"

SUMMARY_CSV="${SEQUENCE_BUNDLE_OUT}/accuracy_vs_k.csv"
if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "seq_len,k,accuracy,scoring,min_correct,n_episodes" > "$SUMMARY_CSV"
fi

plot_bundle_curve() {
  local csv_path="$1"
  local png_path="$2"
  local title_model="$3"
  local thinking_flag="$4"
  local prefix_flag="$5"
  if [[ "$PLOT_ACCURACY_PNG" -ne 1 ]] || [[ "$DRY_RUN" -ne 0 ]]; then
    return 0
  fi
  [[ -f "$csv_path" ]] || return 0
  python - "$csv_path" "$png_path" "$title_model" "$BUNDLE_SCORING" "${BUNDLE_MIN_CORRECT:-}" "$thinking_flag" "$prefix_flag" << 'PLOTEOF'
import sys
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found; skip plot.", file=sys.stderr)
    sys.exit(0)

csv_path, png_path, title_model, scoring, min_c, thinking, prefix = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
    sys.argv[7],
)
df = pd.read_csv(csv_path)
df = df.sort_values("k")
slen = int(df["seq_len"].iloc[0])
sub = f"{scoring}" + (f", min_correct={min_c}" if min_c and scoring == "at_least" else "")
sub = f"{sub}, thinking={thinking}, prefix_q={prefix}"

plt.figure(figsize=(7, 5))
plt.plot(df["k"].astype(int), df["accuracy"], marker="o", linestyle="-", linewidth=2, markersize=8)
plt.xlabel("Number of target questions (k)")
plt.ylabel("Sequence accuracy (fraction)")
plt.title(f"{title_model}\nseq_len={slen}, {sub}")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()
print("Saved", png_path)
PLOTEOF
}

for k in $(seq "$K_START" "$K_END"); do
  echo "================ k_target=$k / seq_len=$SEQ_LEN ================"

  GEN_JSON="${INPUT_DIR}/generated_bundles/$MODEL_NAME/seq${SEQ_LEN}/bundle_${_qt_tag}_nq${N_EPISODES}_k${k}_seed${SEED}.json"
  exp_name="bun_seq${SEQ_LEN}_k${k}_nq${N_EPISODES}_seed${SEED}_t${THINKING}_p${PREFIX_QUESTION}_${_qt_tag}"

  if [[ ! -f "$GEN_JSON" ]] || [[ "$REWRITE_DATA_CACHE" -eq 1 ]]; then
    if [[ -n "$BUNDLE_SIZE" ]]; then
      run "python scripts/generate_bundle_dataset.py \
        --output_path \"${GEN_JSON}\" \
        --seq_len \"${SEQ_LEN}\" \
        --k_target \"${k}\" \
        --n_episodes \"${N_EPISODES}\" \
        --target_question_type \"${TARGET_QUESTION_TYPE}\" \
        --question_types ${QUESTION_TYPES} \
        --seed \"${SEED}\" \
        --bundle_size \"${BUNDLE_SIZE}\""
    else
      run "python scripts/generate_bundle_dataset.py \
        --output_path \"${GEN_JSON}\" \
        --seq_len \"${SEQ_LEN}\" \
        --k_target \"${k}\" \
        --n_episodes \"${N_EPISODES}\" \
        --target_question_type \"${TARGET_QUESTION_TYPE}\" \
        --question_types ${QUESTION_TYPES} \
        --seed \"${SEED}\""
    fi
  else
    echo "Using existing dataset: $GEN_JSON"
  fi

  raw_csv="${INPUT_DIR}/${exp_name}/qa_pairs_answers_bundle.csv"
  mkdir -p "$(dirname "$raw_csv")"

  if [[ "$REWRITE_DATA_CACHE" -eq 1 && -f "$raw_csv" ]]; then
    run "rm -f \"${raw_csv}\""
  fi

  if [[ ! -f "$raw_csv" || "$REWRITE_DATA_CACHE" -eq 1 ]]; then
    run "python scripts/openai_server_inference.py \
      --port \"${VLLM_PORT}\" \
      --model_name \"${MODEL_NAME}\" \
      --text_json_path \"${GEN_JSON}\" \
      --exp_name \"${exp_name}\" \
      --semaphore_limit \"${SEMAPHORE_LIMIT}\" \
      --batch_size \"${BATCH_SIZE}\" \
      --output_csv \"${raw_csv}\" \
      ${_pf} ${_tf}"
  else
    echo "Inference CSV exists, skipping: $raw_csv"
  fi

  if [[ "$BUNDLE_SCORING" == "at_least" ]]; then
    if [[ -z "$BUNDLE_MIN_CORRECT" ]]; then
      echo "Error: BUNDLE_MIN_CORRECT required when BUNDLE_SCORING=at_least" >&2
      exit 1
    fi
    run "python scripts/utils/parse_answers.py \
      --exp_name \"${exp_name}\" \
      --input_dir \"${INPUT_DIR}\" \
      --output_dir \"${RESULTS_DIR}\" \
      --bundle_scoring \"${BUNDLE_SCORING}\" \
      --bundle_seq_len \"${SEQ_LEN}\" \
      --bundle_k_target \"${k}\" \
      --bundle_episode_column episode_id \
      --bundle_min_correct \"${BUNDLE_MIN_CORRECT}\""
  else
    run "python scripts/utils/parse_answers.py \
      --exp_name \"${exp_name}\" \
      --input_dir \"${INPUT_DIR}\" \
      --output_dir \"${RESULTS_DIR}\" \
      --bundle_scoring \"${BUNDLE_SCORING}\" \
      --bundle_seq_len \"${SEQ_LEN}\" \
      --bundle_k_target \"${k}\" \
      --bundle_episode_column episode_id"
  fi

  metrics_csv="${RESULTS_DIR}/${exp_name}_bundle_metrics.csv"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    python - "$metrics_csv" "$SUMMARY_CSV" << 'PY'
import sys
import pandas as pd
m_path, summary = sys.argv[1], sys.argv[2]
m = pd.read_csv(m_path).iloc[0]
mc = m.get("min_correct", "")
if pd.isna(mc) or str(mc).strip() == "":
    mc_str = ""
else:
    mc_str = str(int(mc)) if float(mc) == int(float(mc)) else str(mc)
with open(summary, "a") as f:
    f.write(
        f"{int(m['seq_len'])},{int(m['k_target'])},{m['accuracy']},"
        f"{m['scoring']},{mc_str},{int(m['n_episodes'])}\n"
    )
PY
    echo "Appended row for k=$k"
  fi
done

plot_bundle_curve "$SUMMARY_CSV" "${SEQUENCE_BUNDLE_OUT}/accuracy_vs_target_questions.png" "$MODEL_NAME" "$THINKING" "$PREFIX_QUESTION"

echo "Done. Summary: $SUMMARY_CSV"
echo "Plot: ${SEQUENCE_BUNDLE_OUT}/accuracy_vs_target_questions.png"
