# <span style="color:red">MMReD</span> Benchmark

Kurkin et al., *<span style="color:red">MMReD</span>: a **M**ulti-**M**odal **R**easoning in **D**ense Context Benchmark*

## 📦 Installation

Requires Python ≥ 3.8

```
git clone git@github.com:Fr0do/mmred.git
cd mmred
pip install -e .
```

## 📘 Project Structure

- `mmred/` — utility package for working with the benchmark ([detailed documentation](./mmred/README.md))
- `scripts/generate_dataset.py` — generate the benchmark dataset
- `scripts/render_images.py` — render images from generated dataset
- `vllm_servers.sh` — launch a vLLM model server
- `inference.sh` — run inference on the selected model
- `scripts/utils/parse_answers.py` — parse raw model outputs and save results in CSV format

## 🚀 How to Run

### 1. (Re)Generate the Dataset

```bash
# Generate with default settings
python scripts/generate_dataset.py --output_path data/dataset.json

# Generate with custom settings
python scripts/generate_dataset.py \
    --output_path data/dataset.json \
    --seq_lengths 16 32 64 128 \
    --n_questions 100 \
    --seed 42
```

### 2. (Optional) Render Images

```bash
python scripts/render_images.py \
    --input_path data/dataset.json \
    --output_dir data/images/
```

### 3. Launch vLLM Server

Edit vllm_servers.sh to specify your model(s), then run:

```bash
bash vllm_servers.sh
```

### 3. Run Inference

```bash
bash inference.sh
```

### 4. Parse Model Outputs

```bash
python scripts/utils/parse_answers.py --exp_name EXP_NAME --input_dir path/to/inference_outputs.csv --output_dir results.csv
```

You can then use the CSV file to compute evaluation metrics.


## MERA Evaluation With MLflow

`scripts/run_mera_mlflow.py` runs the MMReD MERA/lm-eval task configs and logs the run to MLflow. It stores the raw harness result JSON, sample logs, git state, package versions, and GPU diagnostics as MLflow artifacts.

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_mera_mlflow.py \
    --tasks mmred \
    --model vllm-vlm \
    --model-args pretrained=google/t5gemma-2-270m-270m,trust_remote_code=True,enforce_eager=True,gpu_memory_utilization=0.3,max_model_len=4096 \
    --gen-kwargs temperature=0.0,do_sample=False,max_gen_toks=100 \
    --limit 1 \
    --mlflow-tracking-uri file:///workspace-SR004.nfs2/kurkin/mlruns \
    --mlflow-experiment mera-mmred
```

The script registers `/workspace-SR004.nfs2/kurkin/vllm-t5gemma2-plugin` by default before running lm-eval. Use `--limit -1` for a full run after the smoke test passes.
