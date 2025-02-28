export VLLM_USE_V1=0
export WANDB_PROJECT="mmlong"
accelerate launch --config_file=train/zero1.yaml train/train_trl.py --config train/config.yaml