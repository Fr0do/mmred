export VLLM_USE_V1=0
export WANDB_PROJECT="mmlong"
export WANDB_API_KEY=$(cat /workspace-SR004.nfs2/kurkin/omni/LLaVA/wandb_key.txt)
accelerate launch --config_file=train/deepspeed.yaml --num_processes 7 train/train_trl.py --config train/config_grpo_7b_r1.yaml