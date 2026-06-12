#!/usr/bin/env python
"""Evaluate Claude models on MMReD via the Claude Code CLI (subscription).

Protocol parity with the MERA lm-eval setup:
- prompt  = instruction.format(**inputs).strip()   (same as doc_to_text)
- answer extraction and EM scoring reuse mera_integration/benchmark_tasks/mmred/utils.py
  (extract_answer + process_results), so the metric is byte-identical to the
  lm-eval runs in PR #23.
- `--tools ""` disables all Claude Code tools: pure text generation, no bash.
- system prompt is overridden with a neutral one (the default Claude Code
  system prompt is a coding-agent persona and would distort the measurement).

Resume-safe: every result is appended to <out>/<model>.jsonl keyed by
(task, idx); on restart finished keys are skipped.

Usage:
    python scripts/eval_claude_cli.py --model sonnet [--limit N] [--concurrency 4]
    python scripts/eval_claude_cli.py --model sonnet --report   # only print table
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mera_integration" / "benchmark_tasks" / "mmred"))
from utils import extract_answer, process_results  # noqa: E402

DATASET = "dondosss/mmred_mera"
QTYPES = ["dc_whs_c", "dc_sa_c", "dc_ws_r", "dc_sr_i", "dc_cc_i"]
LENS = [32, 64, 128]
TASKS = [f"mmred_{q}_{n}" for q in QTYPES for n in LENS]
STOPS = ["\nВопрос:", "\nЗадача:"]
SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай на вопрос точно в требуемом формате."


def build_docs(limit: int | None):
    from datasets import load_dataset

    docs = []
    for task in TASKS:
        ds = load_dataset(DATASET, task, split="test")
        for idx, doc in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            docs.append((task, idx, doc))
    return docs


async def query_claude(model: str, prompt: str, timeout: float) -> str:
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", "--model", model, "--tools", "",
        "--system-prompt", SYSTEM_PROMPT, "--output-format", "text",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"timeout {timeout}s")
    if proc.returncode != 0:
        raise RuntimeError(f"rc={proc.returncode}: {err.decode()[:300]}")
    return out.decode()


async def run_one(sem, model, task, idx, doc, out_path, lock, timeout, log):
    prompt = doc["instruction"].format(**doc["inputs"]).strip()
    meta = doc.get("meta", {})
    categories = meta.get("categories", {})
    atype = categories.get("atype", meta.get("atype", "person"))

    async with sem:
        resp, error = "", None
        for attempt in range(4):
            try:
                t0 = time.monotonic()
                resp = await query_claude(model, prompt, timeout)
                dt = time.monotonic() - t0
                error = None
                break
            except RuntimeError as e:
                error = str(e)
                wait = 30 * (2 ** attempt)
                log(f"WARN {task}[{idx}] attempt {attempt + 1}: {error} -> sleep {wait}s")
                await asyncio.sleep(wait)
        if error is not None:
            log(f"FAIL {task}[{idx}]: {error}")
            return None

    for stop in STOPS:  # parity with lm-eval `until`
        pos = resp.find(stop)
        if pos != -1:
            resp = resp[:pos]
    pred = extract_answer(resp, atype)
    metrics = process_results(doc, [pred])

    rec = {
        "task": task, "idx": idx, "model": model,
        "gold": doc["outputs"], "pred": pred, "resp": resp,
        "exact_match": metrics["exact_match"],
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "latency_s": round(dt, 1),
    }
    async with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def report(out_path: Path):
    rows = [json.loads(line) for line in open(out_path)] if out_path.exists() else []
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["exact_match"])
    print(f"\n{out_path.name}: {len(rows)} samples")
    total = []
    for task in TASKS:
        ems = by_task.get(task, [])
        total += ems
        if ems:
            print(f"  {task:24s} n={len(ems):3d} EM={sum(ems) / len(ems):.2f}")
    if total:
        print(f"  {'MEAN':24s} n={len(total):3d} EM={sum(total) / len(total):.4f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="claude model alias: sonnet, haiku, opus, ...")
    ap.add_argument("--out-dir", default="/home/jovyan/kurkin/mera_eval_outputs/claude_cli")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N samples per task (smoke test)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--report", action="store_true", help="only print the score table")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model.replace('/', '_')}.jsonl"
    if args.report:
        report(out_path)
        return

    cli_ver = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "model": args.model, "claude_cli": cli_ver, "dataset": DATASET,
        "system_prompt": SYSTEM_PROMPT, "tools": "disabled", "stops": STOPS,
        "extraction": "mera utils.extract_answer + process_results",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / f"{args.model}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["task"], r["idx"]))

    docs = [(t, i, d) for t, i, d in build_docs(args.limit) if (t, i) not in done]
    print(f"{args.model}: {len(done)} done, {len(docs)} to go, concurrency={args.concurrency}", flush=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    tasks = [run_one(sem, args.model, t, i, d, out_path, lock, args.timeout, log) for t, i, d in docs]
    n_done = len(done)
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        n_done += 1
        if rec and n_done % 10 == 0:
            log(f"{n_done} done (last: {rec['task']}[{rec['idx']}] em={rec['exact_match']:.0f} {rec['latency_s']}s)")

    report(out_path)


if __name__ == "__main__":
    asyncio.run(main())
