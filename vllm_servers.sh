#!/bin/bash
# Start the script in its own process group
set -m

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export VLLM_USE_V1=1

# Common configuration for both models
COMMON_ARGS="--dtype bfloat16 --num-scheduler-steps 1 --max-seq-len-to-capture 32768 --limit-mm-per-prompt image=128,video=0 --allowed-local-media-path / --max-model-len 42000 --max-num-batched-tokens 42000 --trust-remote-code --disable-log-requests"
MM_PROCESSOR_KWARGS='{"max_dynamic_patch": 1}'
QWEN_ARGS=""

(
    # CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve rhymes-ai/Aria -tp 4 --gpu-memory-utilization 0.9 $COMMON_ARGS --port 8002 --max-model-len 35000 --max-num-batched-tokens 35000 &

    # CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen2-VL-72B-Instruct -tp 4 --gpu-memory-utilization 0.95 $COMMON_ARGS --port 8002 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 307200}' --max-num-batched-tokens 48000 --max-model-len 48000 --hf-overrides '{"max_position_embeddings": 48000}' &

    # sleep 60
    # CUDA_VISIBLE_DEVICES=0,1  vllm serve OpenGVLab/InternVL2_5-4B-MPO -tp 2 --gpu-memory-utilization 0.4  $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8005 &
    # CUDA_VISIBLE_DEVICES=2,3  vllm serve OpenGVLab/InternVL2_5-4B -tp 2 --gpu-memory-utilization 0.4  $COMMON_ARGS  --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8006 &

    # sleep 60
    # CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve Qwen/Qwen2-VL-2B-Instruct -tp 4 --gpu-memory-utilization 0.25 $COMMON_ARGS --port 8003 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 307200}' --max-num-batched-tokens 48000 --max-model-len 48000 --hf-overrides '{"max_position_embeddings": 48000}' &
    CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve Qwen/Qwen2-VL-7B-Instruct -tp 4 --gpu-memory-utilization 0.35 $COMMON_ARGS --port 8004 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 307200}' --max-num-batched-tokens 48000 --max-model-len 48000 --hf-overrides '{"max_position_embeddings": 48000}' &

    # CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve OpenGVLab/InternVL2_5-38B-MPO -tp 4 --gpu-memory-utilization 0.9 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8000 --hf-overrides '{"max_position_embeddings": 48000}' &
    # CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve OpenGVLab/InternVL2_5-38B -tp 4 --gpu-memory-utilization 0.9 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8000 --hf-overrides '{"max_position_embeddings": 48000}' &

    # sleep 60
    # CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve Qwen/QVQ-72B-Preview -tp 4 --gpu-memory-utilization 0.95 $COMMON_ARGS --port 8002 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 307200}' --max-num-batched-tokens 48000 --max-model-len 48000 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve Qwen/Qwen2.5-VL-3B-Instruct -tp 4 --gpu-memory-utilization 0.8 $COMMON_ARGS --port 8003 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 307200}' --max-num-batched-tokens 48000 --max-model-len 48000 --hf-overrides '{ "max_position_embeddings": 48000}' &
    # CUDA_VISIBLE_DEVICES=6,7  vllm serve Qwen/Qwen2.5-VL-7B-Instruct -tp 4 --gpu-memory-utilization 0.8 $COMMON_ARGS --port 8004 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 307200}' --max-num-batched-tokens 48000 --max-model-len 48000 --hf-overrides '{ "max_position_embeddings": 48000}' &
    
    # CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve OpenGVLab/InternVL2_5-78B-MPO -tp 4 --gpu-memory-utilization 0.9 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8000 --hf-overrides '{"max_position_embeddings": 48000}' &

    # CUDA_VISIBLE_DEVICES=4,5  vllm serve OpenGVLab/InternVL2_5-4B-MPO -tp 2 --gpu-memory-utilization 0.33  $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8007 &
    # CUDA_VISIBLE_DEVICES=6,7  vllm serve OpenGVLab/InternVL2_5-4B -tp 2 --gpu-memory-utilization 0.33  $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8008 &

    # sleep 35
    # CUDA_VISIBLE_DEVICES=4,5  vllm serve OpenGVLab/InternVL2_5-2B-MPO -tp 2 --gpu-memory-utilization 0.25  $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8005 &
    # CUDA_VISIBLE_DEVICES=6,7  vllm serve OpenGVLab/InternVL2_5-2B -tp 2 --gpu-memory-utilization 0.25  $COMMON_ARGS  --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8006 &

    # sleep 60
    # CUDA_VISIBLE_DEVICES=4,5  vllm serve OpenGVLab/InternVL2_5-8B -tp 2 --gpu-memory-utilization 0.4  $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8003 &
    # CUDA_VISIBLE_DEVICES=6,7  vllm serve OpenGVLab/InternVL2_5-8B-MPO -tp 2 --gpu-memory-utilization 0.4  $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8004 &

    wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT

wait