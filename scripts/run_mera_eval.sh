#!/usr/bin/env bash
#
# run_mera_eval.sh — Run official MERA evaluation via lm-evaluation-harness.
#
# Usage:
#   bash scripts/run_mera_eval.sh <model_id> <gpu_id> [tp_size] [backend] [extra_args]
#
# Examples:
#   bash scripts/run_mera_eval.sh ai-sage/GigaChat3.1-10B-A1.8B 0
#   bash scripts/run_mera_eval.sh google/gemma-4-31B-it 0 2 sglang
#   bash scripts/run_mera_eval.sh Qwen/Qwen3.5-27B-FP8 2 1 vllm "dtype=float16"

set -eo pipefail

MODEL_ID="${1:?Usage: $0 <model_id> <gpu_id> [tp_size] [backend] [extra_args]}"
GPU_ID="${2:?Specify GPU ID}"
TP_SIZE="${3:-1}"
BACKEND="${4:-sglang}"
EXTRA_ARGS="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_DIR="$PROJECT_DIR/mera_integration/benchmark_tasks"
OUTPUT_DIR="$PROJECT_DIR/data/mera_results"

MODEL_SLUG="$(echo "$MODEL_ID" | tr '/' '_')"

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
LOGFILE="$PROJECT_DIR/logs/mera_eval_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S).log"

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }

{
echo "$(ts) === MERA evaluation: $MODEL_ID ==="
echo "$(ts) GPU: $GPU_ID, TP: $TP_SIZE, Backend: $BACKEND"
echo "$(ts) Log: $LOGFILE"

# Environment
eval "$(conda shell.bash hook)" && conda activate mmred
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export VLLM_DISABLED_KERNELS="deepgemm"

# Build GPU list for TP
GPU_LIST="$GPU_ID"
for ((i=1; i<TP_SIZE; i++)); do
    GPU_LIST="$GPU_LIST,$((GPU_ID + i))"
done

# Build model_args based on backend
if [ "$BACKEND" = "sglang" ]; then
    MODEL_ARGS="pretrained=$MODEL_ID,tp_size=$TP_SIZE"
    if [ -n "$EXTRA_ARGS" ]; then
        MODEL_ARGS="$MODEL_ARGS,$EXTRA_ARGS"
    fi
else
    MODEL_ARGS="pretrained=$MODEL_ID,tensor_parallel_size=$TP_SIZE,gpu_memory_utilization=0.9,max_model_len=32768,enforce_eager=True"
    if [ -n "$EXTRA_ARGS" ]; then
        MODEL_ARGS="$MODEL_ARGS,$EXTRA_ARGS"
    fi
fi

echo "$(ts) CUDA_VISIBLE_DEVICES=$GPU_LIST"
echo "$(ts) model_args: $MODEL_ARGS"

ALL_TASKS="mmred_dc_sa_c_32,mmred_dc_sa_c_64,mmred_dc_sa_c_128,mmred_dc_sr_i_32,mmred_dc_sr_i_64,mmred_dc_sr_i_128,mmred_dc_cc_i_32,mmred_dc_cc_i_64,mmred_dc_cc_i_128,mmred_dc_ws_r_32,mmred_dc_ws_r_64,mmred_dc_ws_r_128,mmred_dc_whs_c_32,mmred_dc_whs_c_64,mmred_dc_whs_c_128"

CUDA_VISIBLE_DEVICES="$GPU_LIST" lm_eval \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks $ALL_TASKS \
    --include_path "$TASK_DIR" \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/$MODEL_SLUG" \
    --log_samples \
    --apply_chat_template

echo "$(ts) === MERA evaluation complete: $MODEL_ID ==="
echo "$(ts) Results at: $OUTPUT_DIR/$MODEL_SLUG"

} 2>&1 | tee -a "$LOGFILE"
