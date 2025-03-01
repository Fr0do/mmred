export VLLM_USE_V1=0
export WANDB_PROJECT="mmlong"
export WANDB_API_KEY=$(cat /workspace-SR004.nfs2/kurkin/omni/LLaVA/wandb_key.txt)
accelerate launch --config_file=train/zero1.yaml train/train_trl.py --config train/config.yaml