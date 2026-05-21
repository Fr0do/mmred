#!/usr/bin/env python
"""Run MERA/lm-eval benchmarks with MLflow tracking.

The script keeps MERA/lm-eval as the evaluator and adds a callback layer that
logs run metadata, metrics, raw results, samples, and environment diagnostics
to MLflow.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCLUDE_PATH = REPO_ROOT / "mera_integration" / "benchmark_tasks"
DEFAULT_PLUGIN_PATH = Path("/workspace-SR004.nfs2/kurkin/vllm-t5gemma2-plugin")
DEFAULT_TRACKING_URI = "file:///workspace-SR004.nfs2/kurkin/mlruns"


def parse_key_value_string(value: str | None) -> dict[str, Any]:
    if not value:
        return {}

    parsed: dict[str, Any] = {}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            parsed[item] = True
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        lowered = raw.lower()
        if lowered == "true":
            parsed[key] = True
        elif lowered == "false":
            parsed[key] = False
        elif lowered == "none":
            parsed[key] = None
        else:
            try:
                parsed[key] = int(raw)
            except ValueError:
                try:
                    parsed[key] = float(raw)
                except ValueError:
                    parsed[key] = raw
    return parsed


def sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:/-]", "_", name).strip("_")[:250]


def flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task_name, task_metrics in results.get("results", {}).items():
        if not isinstance(task_metrics, dict):
            continue
        for metric_name, value in task_metrics.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                key = sanitize_metric_name(f"{task_name}/{metric_name}")
                metrics[key] = float(value)
    return metrics


def run_text_command(command: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def sample_rows(samples: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, task_samples in samples.items():
        if not isinstance(task_samples, list):
            continue
        for idx, sample in enumerate(task_samples):
            if not isinstance(sample, dict):
                continue
            row = {
                "task": task_name,
                "sample_index": idx,
                "doc_id": sample.get("doc_id"),
                "target": sample.get("target"),
                "filtered_resps": sample.get("filtered_resps"),
                "exact_match": sample.get("exact_match"),
                "arguments": sample.get("arguments"),
                "resps": sample.get("resps"),
                "doc": sample.get("doc"),
            }
            text = json.dumps(sample.get("resps"), ensure_ascii=False, default=str)
            row["response_chars"] = len(text)
            rows.append(row)
    return rows


@dataclass
class EvalConfig:
    tasks: list[str]
    model: str
    model_args: str
    gen_kwargs: dict[str, Any]
    batch_size: str
    limit: int | float | None
    num_fewshot: int | None
    include_path: Path
    output_dir: Path
    device: str | None
    cuda_visible_devices: str | None
    plugin_path: Path | None
    register_t5gemma2_plugin: bool


class MLflowMeraCallback:
    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        output_dir: Path,
        plugin_path: Path | None,
    ) -> None:
        import mlflow

        self.mlflow = mlflow
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.output_dir = output_dir
        self.plugin_path = plugin_path
        self.run = None

    def __enter__(self) -> "MLflowMeraCallback":
        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment)
        self.run = self.mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.mlflow.set_tag("status", "failed")
            self.mlflow.set_tag("failure_type", exc_type.__name__ if exc_type else "")
            self.mlflow.log_text(
                "".join(traceback.format_exception(exc_type, exc, tb)),
                "failure.txt",
            )
        self.mlflow.end_run()

    def on_start(self, config: EvalConfig) -> None:
        self.mlflow.set_tag("runner", "mera_mlflow")
        self.mlflow.set_tag("repo", str(REPO_ROOT))
        self.mlflow.set_tag("plugin_path", str(config.plugin_path or ""))
        self.mlflow.log_params(
            {
                "model": config.model,
                "model_args": config.model_args,
                "tasks": ",".join(config.tasks),
                "batch_size": config.batch_size,
                "limit": config.limit,
                "num_fewshot": config.num_fewshot,
                "include_path": str(config.include_path),
                "device": config.device,
                "cuda_visible_devices": config.cuda_visible_devices,
                "register_t5gemma2_plugin": config.register_t5gemma2_plugin,
            }
        )
        for key, value in config.gen_kwargs.items():
            self.mlflow.log_param(f"gen_kwargs.{key}", value)

        diagnostics = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "mmred_git_status": run_text_command(["git", "status", "--short"], REPO_ROOT),
            "mmred_git_rev": run_text_command(["git", "rev-parse", "HEAD"], REPO_ROOT),
            "mmred_git_diff": run_text_command(["git", "diff", "--stat"], REPO_ROOT),
            "nvidia_smi": run_text_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                REPO_ROOT,
            ),
            "pip_freeze": run_text_command([sys.executable, "-m", "pip", "freeze"], REPO_ROOT),
        }
        if self.plugin_path and self.plugin_path.exists():
            diagnostics.update(
                {
                    "plugin_git_status": run_text_command(
                        ["git", "status", "--short"], self.plugin_path
                    ),
                    "plugin_git_rev": run_text_command(
                        ["git", "rev-parse", "HEAD"], self.plugin_path
                    ),
                    "plugin_git_diff": run_text_command(
                        ["git", "diff", "--stat"], self.plugin_path
                    ),
                }
            )
        write_json(self.output_dir / "diagnostics.json", diagnostics)
        self.mlflow.log_artifact(str(self.output_dir / "diagnostics.json"))

    def on_results(self, results: dict[str, Any]) -> None:
        write_json(self.output_dir / "results.json", results)
        self.mlflow.log_artifact(str(self.output_dir / "results.json"))

        for key, value in flatten_metrics(results).items():
            self.mlflow.log_metric(key, value)

        samples = results.get("samples")
        if isinstance(samples, dict):
            rows = sample_rows(samples)
            write_jsonl(self.output_dir / "samples.jsonl", rows)
            self.mlflow.log_artifact(str(self.output_dir / "samples.jsonl"))

        if self.output_dir.exists():
            self.mlflow.log_artifacts(str(self.output_dir), artifact_path="harness_output")


def register_t5gemma2_plugin(plugin_path: Path | None) -> None:
    if plugin_path:
        plugin = str(plugin_path)
        sys.path.insert(0, plugin)
        pythonpath = os.environ.get("PYTHONPATH", "")
        paths = [path for path in pythonpath.split(os.pathsep) if path]
        if plugin not in paths:
            os.environ["PYTHONPATH"] = os.pathsep.join([plugin, *paths])
    from vllm_t5gemma2_plugin import register_t5gemma2_model

    register_t5gemma2_model()


def run_eval(config: EvalConfig) -> dict[str, Any]:
    import lm_eval
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=str(config.include_path))
    kwargs: dict[str, Any] = {
        "model": config.model,
        "model_args": config.model_args,
        "tasks": config.tasks,
        "num_fewshot": config.num_fewshot,
        "batch_size": config.batch_size,
        "limit": config.limit,
        "device": config.device,
        "task_manager": task_manager,
        "log_samples": True,
        "gen_kwargs": config.gen_kwargs,
    }
    signature = inspect.signature(lm_eval.simple_evaluate)
    if "confirm_run_unsafe_code" in signature.parameters:
        kwargs["confirm_run_unsafe_code"] = True
    kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters and v is not None}
    return lm_eval.simple_evaluate(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="mmred", help="Comma-separated MERA/lm-eval tasks.")
    parser.add_argument("--model", default="vllm-vlm", help="lm-eval model backend.")
    parser.add_argument(
        "--model-args",
        default=(
            "pretrained=google/t5gemma-2-270m-270m,"
            "trust_remote_code=True,enforce_eager=True,"
            "gpu_memory_utilization=0.3,max_model_len=4096"
        ),
        help="Comma-separated lm-eval model args.",
    )
    parser.add_argument(
        "--gen-kwargs",
        default="temperature=0.0,do_sample=False,max_gen_toks=100",
        help="Comma-separated generation kwargs.",
    )
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=float, default=1, help="Examples per task; omit with --limit -1.")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument("--include-path", type=Path, default=DEFAULT_INCLUDE_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace-SR004.nfs2/kurkin/mera_eval_outputs")
        / datetime.now(timezone.utc).strftime("mmred_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--mlflow-experiment", default="mera-mmred")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--plugin-path", type=Path, default=DEFAULT_PLUGIN_PATH)
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="Optional file containing a Hugging Face token for gated models.",
    )
    parser.add_argument("--no-register-t5gemma2-plugin", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_token_file:
        token = args.hf_token_file.read_text().strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            os.environ["HF_HUB_TOKEN"] = token

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    config = EvalConfig(
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        model=args.model,
        model_args=args.model_args,
        gen_kwargs=parse_key_value_string(args.gen_kwargs),
        batch_size=args.batch_size,
        limit=None if args.limit < 0 else args.limit,
        num_fewshot=args.num_fewshot,
        include_path=args.include_path,
        output_dir=args.output_dir,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        plugin_path=args.plugin_path,
        register_t5gemma2_plugin=not args.no_register_t5gemma2_plugin,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.register_t5gemma2_plugin:
        register_t5gemma2_plugin(config.plugin_path)

    run_name = args.run_name or f"t5gemma2-mmred-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    with MLflowMeraCallback(
        tracking_uri=args.mlflow_tracking_uri,
        experiment=args.mlflow_experiment,
        run_name=run_name,
        output_dir=config.output_dir,
        plugin_path=config.plugin_path,
    ) as callback:
        callback.on_start(config)
        results = run_eval(config)
        callback.on_results(results)

    print(f"MLflow tracking URI: {args.mlflow_tracking_uri}")
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    main()
