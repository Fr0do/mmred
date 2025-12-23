#!/bin/bash
# Start the script in its own process group
set -m

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_LOGITS_PROCESSOR_THREADS=128
export HF_HOME="/workspace-SR004.nfs2/.cache/huggingface"
export HF_TOKEN_PATH="/workspace-SR004.nfs2/kurkin/hf_key_read.txt"
export WD="$(pwd)"

# Common configuration for all models
#   --max-num-partial-prefills 128 --max-long-partial-prefills 16 --max-num-batched-tokens 5120  --long_prefill_token_threshold 4096
COMMON_ARGS="--enable-prefix-caching --max_num_seqs 128 --block-size 32 --allowed-local-media-path / --trust-remote-code --disable-log-requests --limit-mm-per-prompt image=128,video=0 --max-model-len 35000"
MM_PROCESSOR_KWARGS='{"max_dynamic_patch": 1}'
COMMON_TEXT_ARGS="--enable-prefix-caching --max_num_seqs 128  --block-size 32 --allowed-local-media-path / --trust-remote-code --max-model-len 16000 --max-num-batched-tokens 8192"

(
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/rhymes-ai-Aria" vllm serve unsloth/Llama-3.2-11B-Vision-Instruct -tp 4 --gpu-memory-utilization 0.85 $COMMON_ARGS --port 8007 --max-model-len 35000 --model-impl=transformers &
    # 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/rhymes-ai-Aria" vllm serve rhymes-ai/Aria -tp 4 --gpu-memory-utilization 0.95 $COMMON_ARGS --port 8007 --max-model-len 35000 &

    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2-VL-72B-Instruct" vllm serve Qwen/Qwen2-VL-72B-Instruct -tp 4 --gpu-memory-utilization 0.95 $COMMON_ARGS --port 8008 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-num-batched-tokens 1300 --max-model-len 42000 --hf-overrides '{"max_position_embeddings": 42000}' &

    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/MiniCPM-o-2_6" vllm serve openbmb/MiniCPM-o-2_6 -tp 2 --gpu-memory-utilization 0.85 $COMMON_ARGS --port 8003 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/MiniCPM-V-2_6" vllm serve openbmb/MiniCPM-V-2_6 -tp 2 --gpu-memory-utilization 0.85 $COMMON_ARGS --port 8004 &

    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-4B-MPO" vllm serve OpenGVLab/InternVL2_5-4B-MPO -tp 2 --gpu-memory-utilization 0.4 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8009 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/OpenGVLab-InternVL2_5-4B" vllm serve OpenGVLab/InternVL2_5-4B -tp 2 --gpu-memory-utilization 0.4 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8010 &

    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-VL-72B-Instruct" vllm serve Qwen/Qwen2.5-VL-72B-Instruct $COMMON_ARGS -tp 4 --gpu-memory-utilization 0.85 --port 8011 --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 262144}' --max-model-len 42000 &
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
    # sleep 80
    # CUDA_VISIBLE_DEVICES=2 OUTLINES_CACHE_DIR="$WD/cache/InternVL2_5-8B" vllm serve OpenGVLab/InternVL2_5-8B -tp 1 --gpu-memory-utilization 0.85 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8005 &
    # CUDA_VISIBLE_DEVICES=6,7 OUTLINES_CACHE_DIR="$WD/cache/InternVL2_5-8B-MPO" vllm serve OpenGVLab/InternVL2_5-8B-MPO -tp 2 --gpu-memory-utilization 0.85 $COMMON_ARGS --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" --port 8006 &

    # sleep 75
    # CUDA_VISIBLE_DEVICES=0 OUTLINES_CACHE_DIR="$WD/cache/gemma3-1b" vllm serve google/gemma-3-1b-it -tp 1 --gpu-memory-utilization 0.35 $COMMON_TEXT_ARGS --port 8004 --max-num-batched-tokens 16000 --max-model-len 16000 & 
    # CUDA_VISIBLE_DEVICES=1 OUTLINES_CACHE_DIR="$WD/cache/gemma3-4b" vllm serve google/gemma-3-4b-it -tp 1 --gpu-memory-utilization 0.85 $COMMON_ARGS --port 8005 --max-num-batched-tokens 16000 --max-model-len 16000 & 
    # CUDA_VISIBLE_DEVICES=3 OUTLINES_CACHE_DIR="$WD/cache/gemma3-12b" vllm serve google/gemma-3-12b-it -tp 1 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8006 --max-num-batched-tokens 12000 --max-model-len 12000 & 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/gemma3-27b" vllm serve google/gemma-3-27b-it -tp 4 --gpu-memory-utilization 0.85 $COMMON_ARGS --port 8005 --max-num-batched-tokens 16000 --max-model-len 16000 & 

    # CUDA_VISIBLE_DEVICES=2 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-7B-Instruct" vllm serve Qwen/Qwen2.5-3B-Instruct -tp 1 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8001 --max-num-batched-tokens 8192 --max-model-len 13000 &
    # CUDA_VISIBLE_DEVICES=3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-7B-Instruct" vllm serve Qwen/Qwen2.5-7B-Instruct -tp 1 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8006 --max-num-batched-tokens 8192 --max-model-len 13000 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen/Qwen2.5-Coder-7B-Instruct" vllm serve Qwen/Qwen2.5-Coder-7B-Instruct -tp 2 --gpu-memory-utilization 0.3 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # sleep 75
    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/Qwen-Qwen2.5-32B-Instruct" vllm serve Qwen/Qwen2.5-32B-Instruct -tp 2 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8002 --max-num-batched-tokens 8192 --max-model-len 13000 &
    # sleep 75
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen/Qwen2.5-Coder-32B-Instruct" vllm serve Qwen/Qwen2.5-Coder-32B-Instruct -tp 2 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-72B-Instruct" vllm serve Qwen/Qwen2.5-72B-Instruct -tp 4 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 4096 --max-model-len 13000 &
    # CUDA_VISIBLE_DEVICES=4,5,6,7 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Qwen-32B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -tp 4 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8004 --max-num-batched-tokens 8192 --max-model-len 13000 & 
    # CUDA_VISIBLE_DEVICES=0 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Llama-8B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B -tp 1 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 8192 --max-model-len 12000 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Llama-70B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B -tp 4 --gpu-memory-utilization 0.9 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 8192 --max-model-len 12000 &
    # CUDA_VISIBLE_DEVICES=1 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Qwen-7B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8004 --max-num-batched-tokens 8192 --max-model-len 12000 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Qwen-14B" vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B -tp 2 --gpu-memory-utilization 0.85 $COMMON_TEXT_ARGS --port 8010 --max-num-batched-tokens 8192 --max-model-len 12000 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-7B-Instruct-GRPO" vllm serve checkpoints/Qwen2.5-7B-Instruct-GRPO -tp 2 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8005 --max-num-batched-tokens 8192 --max-model-len 10000 & 
    # CUDA_VISIBLE_DEVICES=3 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-3B-Instruct-SFT-GRPO" vllm serve checkpoints/Qwen2.5-3B-Instruct-SFT-GRPO -tp 1 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8004 --max-num-batched-tokens 8192 --max-model-len 10000 & 
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-3B-Instruct-GRPO" vllm serve checkpoints/Qwen2.5-3B-Instruct-GRPO -tp 2 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8005 --max-num-batched-tokens 8192 --max-model-len 10000 & 
    # CUDA_VISIBLE_DEVICES=0,1 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-7B-Instruct-GRPO" vllm serve checkpoints/Qwen2.5-7B-Instruct-GRPO -tp 2 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8006 --max-num-batched-tokens 8192 --max-model-len 10000 & 
    # CUDA_VISIBLE_DEVICES=1 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-1.5B-Instruct-GRPO" vllm serve checkpoints/Qwen2.5-1.5B-Instruct-GRPO -tp 1 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8007 --max-num-batched-tokens 8192 --max-model-len 10000 & 
    # CUDA_VISIBLE_DEVICES=2 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-1.5B-Instruct-GRPO" vllm serve checkpoints/Qwen2.5-7B-Instruct-SFT-5-epochs --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8007 --max-num-batched-tokens 8192 --max-model-len 10000 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-1.5B-Instruct-GRPO" vllm serve checkpoints/DeepSeek-R1-Distill-Qwen-7B-GRPO -tp 4 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8007 --max-num-batched-tokens 8192 --max-model-len 10000 & 
    #CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/DeepSeek-R1-Distill-Llama-70B" vllm serve Qwen/QwQ-32B -tp 4 --gpu-memory-utilization 0.9 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 8192 --max-model-len 12000 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/r1" vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -tp 4 --gpu-memory-utilization 0.45 $COMMON_TEXT_ARGS --port 8004 --max-num-batched-tokens 8192 --max-model-len 13000 & 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/mistral" vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 -tp 4 --gpu-memory-utilization 0.65 $COMMON_TEXT_ARGS --port 8020 --max-num-batched-tokens 12000 --max-model-len 12000 &
    # sleep 50
    # VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=1 OUTLINES_CACHE_DIR="$WD/cache/mamba" vllm serve checkpoints/Falcon3-Mamba-7B-Instruct-SFT-5-epochs-stable --gpu-memory-utilization 0.25 --port 8004 --max-num-batched-tokens 12000 --max-model-len 12000 &
    # CUDA_VISIBLE_DEVICES=2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen2.5-3B-Instruct" vllm serve Qwen/Qwen2.5-3B-Instruct -tp 2 --gpu-memory-utilization 0.89 $COMMON_TEXT_ARGS --port 8005 --max-num-batched-tokens 8192 --max-model-len 10000 --max_lora_rank 32 --enable-lora --lora-modules mmlong-grpo-3b=checkpoints/mv1_grpo_qwen_3b_256 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen" vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking -tp 4 --gpu-memory-utilization 0.75 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 32000 --max-model-len 32768 --reasoning-parser qwen3 & 
    CUDA_VISIBLE_DEVICES=0 OUTLINES_CACHE_DIR="$WD/cache/Qwen" vllm serve checkpoints/sft_qwen_4b_full -tp 1 --gpu-memory-utilization 0.75 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 12000 & 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-32" vllm serve Qwen/Qwen3-32B --gpu-memory-utilization 0.75 -tp 4 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 32000 --max-model-len 12000 & 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-32" vllm serve tiiuae/Falcon-H1-34B-Instruct --gpu-memory-utilization 0.75 -tp 4 $COMMON_TEXT_ARGS --no-enable-prefix-caching  --port 8003 --max-num-batched-tokens 24000 --max-model-len 24000 & 
    # CUDA_VISIBLE_DEVICES=2 OUTLINES_CACHE_DIR="$WD/cache/Qwen-14" vllm serve Qwen/Qwen3-14B  --gpu-memory-utilization 0.75 $COMMON_TEXT_ARGS --port 8001 --max-num-batched-tokens 16384 --max-model-len 12000 & 
    # CUDA_VISIBLE_DEVICES=3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-4" vllm serve Qwen/Qwen3-4B --gpu-memory-utilization 0.75 --enable-reasoning --reasoning-parser deepseek_r1 $COMMON_TEXT_ARGS --port 8004 --max-num-batched-tokens 16384 --max-model-len 16384 & 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 OUTLINES_CACHE_DIR="$WD/cache/Qwen-3" vllm serve /home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/4c446470ba0aec43e22ac1128f9ffd915f338ba3 -tp 4 --gpu-memory-utilization 0.75 --enable-reasoning --reasoning-parser deepseek_r1 $COMMON_TEXT_ARGS --port 8003 --max-num-batched-tokens 16384 --max-model-len 16384 & 
    wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT

wait
