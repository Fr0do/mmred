export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/workspace-SR004.nfs2/kurkin/envs/kurkin_313_torch/bin/python"
export HF_HOME="/workspace-SR004.nfs2/.cache/huggingface"

export CLEARML_PROJECT="mmred"
export CLEARML_TASK="rmt"
export CLEARML_WEB_HOST="https://clearml.de.mlrnd.ru"
export CLEARML_API_HOST=https://clearml-api.de.mlrnd.ru
export CLEARML_FILES_HOST=https://clearml-files.de.mlrnd.ru
export CLEARML_API_ACCESS_KEY=H6TXVZH23GSUZCP0YP2R2J93F0WKR7
export CLEARML_API_SECRET_KEY=SpAFpItmbnQlwpqNPjyndcUOZ1fkRMrQgrUAF-DEhs7QEo4Wzc5EWdolmFVeAdEqzy0
export CLEARML_LOG_MODEL=False

source activate base
conda activate kurkin_313_torch

accelerate launch --config_file=train/deepspeed.yaml --num_processes 4 train/train_rmt_curriculum.py --config train/config_rmt.yaml
