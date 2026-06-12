#!/usr/bin/env python
"""Post-run validation and manifest for MMReD lm-eval runs.

lm-eval records the *intended* config (model_args, per-task generation_kwargs)
but not the *effective* procedure: silent defaults (e.g. the fewshot_delimiter
"\\n\\n" stop when `until` is absent), the chat template actually applied, or
whether a reasoning model actually reasoned. Every measurement bug we hit on
11-12.06.2026 (raw completion instead of chat template, thinking disabled by
the template default, generations cut after the first thought paragraph)
is detectable from the logged samples — this script makes those checks a
mandatory gate and writes an eval_manifest.json next to the results.

Usage:
    python scripts/validate_eval_run.py RUN_DIR --model MODEL_ID \\
        --expect-thinking {yes,no,any} [--tokenizer PATH] [--strict]

Exit code 1 when a gate fails (use in runner scripts to fail loudly).
"""

import argparse
import glob
import hashlib
import json
import statistics
import sys
from pathlib import Path

THINK_MARKERS = ("<think>", "</think>", "channel")
SHORT_RESPONSE_TOKENS = 32


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="Tokenizer path if overridden")
    ap.add_argument("--expect-thinking", choices=["yes", "no", "any"], default="any")
    ap.add_argument("--max-gen-toks", type=int, default=16384)
    ap.add_argument("--strict", action="store_true", help="Fail on warnings too")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    run_dir = Path(args.run_dir)

    sample_files = sorted(run_dir.rglob("samples_*.jsonl"))
    results_files = sorted(run_dir.rglob("results_*.json"))
    if not sample_files or not results_files:
        print(f"FAIL: нет samples/results в {run_dir}")
        return 1

    lens, em_sum = [], 0.0
    n = think_n = empty_extract = truncated = short = 0
    for f in sample_files:
        for line in open(f):
            d = json.loads(line)
            r = d["resps"][0]
            r = r[0] if isinstance(r, list) else r
            p = d["filtered_resps"][0]
            p = p[0] if isinstance(p, list) else p
            tl = len(tok.encode(r))
            n += 1
            lens.append(tl)
            em_sum += d.get("exact_match", 0)
            think_n += any(m in r for m in THINK_MARKERS)
            empty_extract += (not str(p).strip())
            truncated += tl >= args.max_gen_toks - 8
            short += tl < SHORT_RESPONSE_TOKENS

    lens.sort()
    stats = {
        "n_samples": n,
        "em_mean": round(em_sum / n, 4),
        "resp_tokens_median": lens[n // 2],
        "resp_tokens_p90": lens[int(0.9 * (n - 1))],
        "thinking_rate": round(think_n / n, 3),
        "empty_extraction_rate": round(empty_extract / n, 3),
        "truncation_rate": round(truncated / n, 3),
        "short_response_rate": round(short / n, 3),
    }

    results_cfg = json.loads(results_files[-1].read_text())
    task_cfgs = results_cfg.get("configs", {})
    effective_until = {
        t: c.get("generation_kwargs", {}).get("until", "ВНИМАНИЕ: не задан -> lm-eval стопится по fewshot_delimiter '\\n\\n'")
        for t, c in list(task_cfgs.items())[:1]
    }

    import lm_eval, vllm, transformers
    chat_template_path = None
    for cand in [Path(args.tokenizer or ""), ]:
        if cand and (cand / "chat_template.jinja").exists():
            chat_template_path = cand / "chat_template.jinja"
    manifest = {
        "model": args.model,
        "tokenizer_override": args.tokenizer,
        "chat_template_sha16": sha256_file(chat_template_path) if chat_template_path else None,
        "versions": {
            "lm_eval": lm_eval.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
        },
        "lm_eval_config": results_cfg.get("config", {}),
        "effective_until_sample": effective_until,
        "expect_thinking": args.expect_thinking,
        "sample_stats": stats,
    }
    (run_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    failures, warnings = [], []
    if args.expect_thinking == "yes" and stats["thinking_rate"] < 0.5:
        failures.append(f"ожидался thinking, а маркеры лишь в {stats['thinking_rate']:.0%} ответов")
    if args.expect_thinking == "no" and stats["thinking_rate"] > 0.5:
        failures.append(f"ожидался no-think, а маркеры в {stats['thinking_rate']:.0%}")
    if stats["empty_extraction_rate"] > 0.2:
        failures.append(f"пустая экстракция в {stats['empty_extraction_rate']:.0%}")
    if stats["short_response_rate"] > 0.5:
        failures.append(f"{stats['short_response_rate']:.0%} ответов короче {SHORT_RESPONSE_TOKENS} токенов — похоже на обрезку стоп-строкой")
    if stats["truncation_rate"] > 0.3:
        warnings.append(f"truncation rate {stats['truncation_rate']:.0%} — рассмотрите больший бюджет")
    if stats["em_mean"] == 0.0:
        failures.append("EM = 0.0 по всем сэмплам — замер сломан")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for w in warnings:
        print("WARN:", w)
    if failures or (args.strict and warnings):
        for f_ in failures:
            print("FAIL:", f_)
        return 1
    print("OK: прогон проходит санити-гейты, манифест записан")
    return 0


if __name__ == "__main__":
    sys.exit(main())
