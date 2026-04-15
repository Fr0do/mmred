#!/usr/bin/env bash
#
# run_all_evals.sh — Run full MMReD evaluation matrix
#
# Usage: bash scripts/run_all_evals.sh <gpu_ids> [backend]
# Example: bash scripts/run_all_evals.sh 0,1,2 vllm
#
set -eo pipefail

GPU_IDS="${1:?Usage: $0 <gpu_ids> [backend]}"
BACKEND="${2:-hf}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_DIR="$PROJECT_DIR/mera_integration/benchmark_tasks"
OUTPUT_DIR="$PROJECT_DIR/data/mera_results"
ALL_TASKS="mmred_dc_sa_c_32,mmred_dc_sa_c_64,mmred_dc_sa_c_128,mmred_dc_sr_i_32,mmred_dc_sr_i_64,mmred_dc_sr_i_128,mmred_dc_cc_i_32,mmred_dc_cc_i_64,mmred_dc_cc_i_128,mmred_dc_ws_r_32,mmred_dc_ws_r_64,mmred_dc_ws_r_128,mmred_dc_whs_c_32,mmred_dc_whs_c_64,mmred_dc_whs_c_128"

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"

run_eval() {
    local model_id="$1"
    local model_slug="$2"
    local suffix="$3"
    local extra_model_args="$4"
    local extra_cli_args="$5"
    local gpus="$6"

    local out_dir="$OUTPUT_DIR/${model_slug}_${suffix}"
    local logfile="$PROJECT_DIR/logs/eval_${model_slug}_${suffix}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date '+%H:%M:%S')] Starting: $model_slug ($suffix) on GPUs $gpus"

    if [ "$BACKEND" = "hf" ]; then
        local model_args="pretrained=$model_id,trust_remote_code=True${extra_model_args:+,$extra_model_args}"
    elif [ "$BACKEND" = "vllm" ]; then
        local model_args="pretrained=$model_id,gpu_memory_utilization=0.9,max_model_len=32768,enforce_eager=True${extra_model_args:+,$extra_model_args}"
    fi

    CUDA_VISIBLE_DEVICES="$gpus" lm_eval \
        --model "$BACKEND" \
        --model_args "$model_args" \
        --tasks $ALL_TASKS \
        --include_path "$TASK_DIR" \
        --batch_size auto \
        --output_path "$out_dir" \
        --log_samples \
        $extra_cli_args \
        2>&1 | tee "$logfile"

    echo "[$(date '+%H:%M:%S')] Done: $model_slug ($suffix)"
}

# ============================================================================
# Small models — HF backend, single GPU
# ============================================================================

if [ "$BACKEND" = "hf" ]; then
    # t5gemma-2-270m-270m (0.8B)
    run_eval "google/t5gemma-2-270m-270m" "t5gemma2_270m" "0shot" "" "" "$GPU_IDS"

    # t5gemma-2-1b-1b (2B) — if cached
    if [ -d "$HOME/.cache/huggingface/hub/models--google--t5gemma-2-1b-1b" ]; then
        run_eval "google/t5gemma-2-1b-1b" "t5gemma2_1b" "0shot" "" "" "$GPU_IDS"
    fi

    # t5gemma-2-4b-4b (9B) — if cached
    if [ -d "$HOME/.cache/huggingface/hub/models--google--t5gemma-2-4b-4b" ]; then
        run_eval "google/t5gemma-2-4b-4b" "t5gemma2_4b" "0shot" "" "" "$GPU_IDS"
    fi
fi

echo "[$(date '+%H:%M:%S')] All evaluations complete!"
