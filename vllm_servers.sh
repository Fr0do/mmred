#!/bin/bash
# Start the script in its own process group
set -m

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_LOGITS_PROCESSOR_THREADS=32
# export VLLM_ATTENTION_BACKEND="FLASHINFER"
# export VLLM_FLASHINFER_FORCE_TENSOR_CORES=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export WD="$(pwd)"

# Common configuration for all models
# --max-num-partial-prefills 16 --max-long-partial-prefills 16 --max-model-len 42000 --max-num-batched-tokens 8192  --long_prefill_token_threshold 8192
COMMON_ARGS="--no-enable-prefix-caching --enable-chunked-prefill True  --max-num-partial-prefills 4 --max-long-partial-prefills 2 --max-num-batched-tokens 16384  --long_prefill_token_threshold 8192 --max_num_seqs 16 --block-size 32 --guided_decoding_backend outlines --dtype bfloat16 --max-seq-len-to-capture 16384 --allowed-local-media-path / --trust-remote-code --disable-log-requests --limit-mm-per-prompt image=128,video=0 --max-model-len 35000"
MM_PROCESSOR_KWARGS='{"max_dynamic_patch": 1}'
COMMON_TEXT_ARGS="--enable-chunked-prefill --guided_decoding_backend outlines --max_num_seqs 40 --dtype bfloat16 --max-seq-len-to-capture 16384 --allowed-local-media-path / --max-model-len 16000 --max-num-batched-tokens 4096 --trust-remote-code --disable-log-requests"

(
    # CUDA_VISIBLE_DEVICES=6,7 OUTLINES_CACHE_DIR="$WD/cache/rhymes-ai-Aria" vllm serve rhymes-ai/Aria -tp 2 --gpu-memory-utilization 0.9 $COMMON_ARGS --port 8007 --max-model-len 35000 &

    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2-VL-72B-Instruct" vllm serve Qwen/Qwen2-VL-72B-Instruct -tp 4 --gpu-memory-utilization 0.95 $COMMON_ARGS --port 8008 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-num-batched-tokens 1300 --max-model-len 42000 --hf-overrides '{"max_position_embeddings": 42000}' &

    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/MiniCPM-o-2_6" vllm serve openbmb/MiniCPM-o-2_6 -tp 2 --gpu-memory-utilization 0.5 $COMMON_ARGS --port 8003 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/MiniCPM-V-2_6" vllm serve openbmb/MiniCPM-V-2_6 -tp 2 --gpu-memory-utilization 0.5 $COMMON_ARGS --port 8004 &

    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-4B-MPO" vllm serve OpenGVLab/InternVL2_5-4B-MPO -tp 2 --gpu-memory-utilization 0.4 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8009 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-4B" vllm serve OpenGVLab/InternVL2_5-4B -tp 2 --gpu-memory-utilization 0.4 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8010 &

    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-VL-72B-Instruct" vllm serve Qwen/Qwen2.5-VL-72B-Instruct $COMMON_ARGS -tp 4 --gpu-memory-utilization 0.8 --port 8011 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-model-len 42000 &
    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-VL-3B-Instruct" vllm serve Qwen/Qwen2.5-VL-3B-Instruct -tp 2 --gpu-memory-utilization 0.8 $COMMON_ARGS --port 8013 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-model-len 42000 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-VL-7B-Instruct" vllm serve Qwen/Qwen2.5-VL-7B-Instruct -tp 2 --gpu-memory-utilization 0.8 $COMMON_ARGS --port 8014 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-model-len 42000 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2-VL-7B-Instruct" vllm serve Qwen/Qwen2-VL-7B-Instruct -tp 4 --gpu-memory-utilization 0.35 $COMMON_ARGS --port 8015 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-num-batched-tokens 1300 --max-model-len 42000 --hf-overrides '{"max_position_embeddings": 42000}' &

    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-38B-MPO" vllm serve OpenGVLab/InternVL2_5-38B-MPO -tp 4 --gpu-memory-utilization 0.88 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8016 --hf-overrides '{"max_position_embeddings": 42000}' &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-38B" vllm serve OpenGVLab/InternVL2_5-38B -tp 4 --gpu-memory-utilization 0.9 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8017 --hf-overrides '{"max_position_embeddings": 42000}' &

    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-QVQ-72B-Preview" vllm serve Qwen/QVQ-72B-Preview -tp 4 --gpu-memory-utilization 0.95 $COMMON_ARGS --port 8018 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-num-batched-tokens 1300 --max-model-len 42000 &

    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-78B-MPO" vllm serve OpenGVLab/InternVL2_5-78B-MPO -tp 4 --gpu-memory-utilization 0.85 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8021 --hf-overrides '{"max_position_embeddings": 42000}' &
    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-78B" vllm serve OpenGVLab/InternVL2_5-78B -tp 4 --gpu-memory-utilization 0.85 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8022 --hf-overrides '{"max_position_embeddings": 42000}' &

    # CUDA_VISIBLE_DEVICES=4 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-4B-MPO" vllm serve OpenGVLab/InternVL2_5-4B-MPO -tp 1 --gpu-memory-utilization 0.8 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8023 &
    # CUDA_VISIBLE_DEVICES=5 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-4B" vllm serve OpenGVLab/InternVL2_5-4B -tp 1 --gpu-memory-utilization 0.8 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8024 &

    # CUDA_VISIBLE_DEVICES=6,7 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-2B-MPO" vllm serve OpenGVLab/InternVL2_5-2B-MPO -tp 2 --gpu-memory-utilization 0.25 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8025 &
    # CUDA_VISIBLE_DEVICES=1 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-2B" vllm serve OpenGVLab/InternVL2_5-2B -tp 1 --gpu-memory-utilization 0.45 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8026 &
    sleep 80
    # CUDA_VISIBLE_DEVICES=2 OUTLINES_CACHE_DIR="$WD/cache/InternVL2_5-8B" vllm serve OpenGVLab/InternVL2_5-8B -tp 1 --gpu-memory-utilization 0.85 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8005 &
    # CUDA_VISIBLE_DEVICES=6,7 OUTLINES_CACHE_DIR="$WD/cache/InternVL2_5-8B-MPO" vllm serve OpenGVLab/InternVL2_5-8B-MPO -tp 2 --gpu-memory-utilization 0.85 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8006 &

    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-7B-Instruct" vllm serve Qwen/Qwen2.5-7B-Instruct -tp 2 --gpu-memory-utilization 0.3 $COMMON_TEXT_ARGS --port 8002 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen/Qwen2.5-Coder-7B-Instruct" vllm serve Qwen/Qwen2.5-Coder-7B-Instruct -tp 2 --gpu-memory-utilization 0.3 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # sleep 75
    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-32B-Instruct" vllm serve Qwen/Qwen2.5-32B-Instruct -tp 2 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8002 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # sleep 75
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen/Qwen2.5-Coder-32B-Instruct" vllm serve Qwen/Qwen2.5-Coder-32B-Instruct -tp 2 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-72B-Instruct" vllm serve Qwen/Qwen2.5-72B-Instruct -tp 4 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Qwen-32B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -tp 4 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 8192 --max-model-len 12000 & 
    CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Llama-8B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B -tp 4 --gpu-memory-utilization 0.31 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 8192 --max-model-len 12000 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Qwen-14B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B -tp 4 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8010 --max-num-batched-tokens 8192 --max-model-len 12000 &
    wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT

wait
