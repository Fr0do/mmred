export VLLM_USE_V1=0
export WANDB_PROJECT="mmlong"
export WANDB_API_KEY=$(cat /workspace-SR004.nfs2/kurkin/omni/LLaVA/wandb_key.txt)
export TOKENIZERS_PARALLELISM="false"
accelerate launch --config_file=train/deepspeed.yaml --num_processes 8 train/train_trl.py --config train/config_sft_mamba.yaml