#!/usr/bin/env python
"""Re-score lm-eval sample dumps offline with the current extract_answer.

When the extraction logic in utils.py changes, published EM numbers must be
recomputed from the raw responses (samples_*.jsonl keep `resps`); the
results_*.json written by lm-eval become stale. This recomputes per-task EM
and the mean for one or more run dirs and prints old vs new side by side.

Usage:
    python scripts/rescore_run.py RUN_DIR [RUN_DIR ...]
"""

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mera_integration" / "benchmark_tasks" / "mmred"))
from utils import extract_answer, process_results  # noqa: E402

ORDER = [f"mmred_{q}_{n}" for q in ["dc_whs_c", "dc_sa_c", "dc_ws_r", "dc_sr_i", "dc_cc_i"] for n in [32, 64, 128]]


def rescore(run_dir: str) -> dict[str, float]:
    new_em: dict[str, list[float]] = {}
    for f in sorted(glob.glob(f"{run_dir}/**/samples_mmred_*.jsonl", recursive=True)):
        task = re.search(r"samples_(mmred_\w+?)_\d{4}", Path(f).name)
        task = task.group(1) if task else Path(f).stem
        for line in open(f):
            d = json.loads(line)
            r = d["resps"][0]
            r = r[0] if isinstance(r, list) else r
            doc = d["doc"]
            atype = doc.get("meta", {}).get("categories", {}).get("atype", doc.get("meta", {}).get("atype", "person"))
            pred = extract_answer(r, atype)
            em = process_results(doc, [pred])["exact_match"]
            new_em.setdefault(task, []).append(em)
    return {t: sum(v) / len(v) for t, v in new_em.items()}


def old_results(run_dir: str) -> dict[str, float]:
    files = sorted(glob.glob(f"{run_dir}/**/results_*.json", recursive=True))
    if not files:
        return {}
    res = json.load(open(files[-1]))["results"]
    return {t: v["exact_match,scoring"] for t, v in res.items() if "exact_match,scoring" in v}


def main():
    for run_dir in sys.argv[1:]:
        old = old_results(run_dir)
        new = rescore(run_dir)
        print(f"\n=== {Path(run_dir).name}")
        olds, news = [], []
        for t in ORDER:
            if t not in new:
                continue
            o, n = old.get(t), new[t]
            olds.append(o if o is not None else float("nan"))
            news.append(n)
            delta = f"{n - o:+.2f}" if o is not None else "  n/a"
            flag = "  <-" if o is not None and abs(n - o) >= 0.005 else ""
            print(f"  {t:22s} old={o if o is not None else float('nan'):.2f} new={n:.2f} {delta}{flag}")
        if news:
            mo = sum(olds) / len(olds) if olds and old else float("nan")
            mn = sum(news) / len(news)
            print(f"  {'MEAN':22s} old={mo:.4f} new={mn:.4f} {mn - mo:+.4f}")


if __name__ == "__main__":
    main()
