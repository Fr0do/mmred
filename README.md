# MMReD Benchmark

Kurkin et al., *MMReD: a Cross-Modal Benchmark for Dense Context Reasoning*

## 📦 Installation

Requires Python ≥ 3.8

```
git clone git@github.com:Fr0do/long-vqa.git
cd long-vqa
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
