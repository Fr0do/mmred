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

- `mmred/` — utility package for working with the benchmark
- `scripts/generate_dataset.py` — generate the benchmark dataset
- `vllm_servers.sh` — launch a vLLM model server
- `inference.sh` — run inference on the selected model
- `scripts/utils/parse_answers.py` — parse raw model outputs and save results in CSV format

## 🚀 How to Run

### 1. Generate the Dataset

```
python scripts/generate_dataset.py --base_path BASE_PATH --exp_name EXP_NAME
```

### 2. Launch vLLM Server

Edit vllm_servers.sh to specify your model(s), then run:

```
bash vllm_servers.sh
```

### 3. Run Inference

```
bash inference.sh
```

### 4. Parse Model Outputs

```
python scripts/utils/parse_answers.py --exp_name EXP_NAME --input_dir path/to/inference_outputs.csv --output_dir results.csv
```

You can then use the CSV file to compute evaluation metrics.
