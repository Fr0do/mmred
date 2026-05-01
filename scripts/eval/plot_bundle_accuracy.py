#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _infer_title_from_path(csv_path: Path) -> str:
    parts = list(csv_path.parts)
    if "results_Qwen" in parts:
        i = parts.index("results_Qwen")
        if i + 1 < len(parts):
            return parts[i + 1]
    return csv_path.parent.name


def _infer_thinking_prefix_from_path(csv_path: Path) -> tuple[str | None, str | None]:
    # Match ...thinking{0|1}_prefix{0|1}... anywhere in the directory path
    m = re.search(r"thinking(\d+)_prefix(\d+)", str(csv_path.parent))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot bundle accuracy_vs_k.csv to a PNG.")
    ap.add_argument("csv", type=Path, help="Path to accuracy_vs_k.csv")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: рядом с csv as accuracy_vs_target_questions.png)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Model name for title (default: inferred from path)",
    )
    ap.add_argument(
        "--ylabel",
        type=str,
        default="Sequence accuracy (fraction)",
        help="Y-axis label",
    )
    ap.add_argument(
        "--ylim",
        type=str,
        default="0,1",
        help="Y limits, e.g. '0,1' (fraction) or '0,100' (percent)",
    )
    ap.add_argument(
        "--reasoning_budget",
        type=int,
        default=12000,
        help="Reasoning token budget to annotate on the plot (e.g. max_completion_tokens for thinking).",
    )
    args = ap.parse_args()

    csv_path: Path = args.csv
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    out_path: Path = args.out or (csv_path.parent / "accuracy_vs_target_questions.png")

    df = pd.read_csv(csv_path).sort_values("k")
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    slen = int(df["seq_len"].iloc[0]) if "seq_len" in df.columns else None
    scoring = str(df["scoring"].iloc[0]) if "scoring" in df.columns else ""
    min_c = str(df["min_correct"].iloc[0]) if "min_correct" in df.columns else ""
    if min_c.lower() == "nan":
        min_c = ""

    thinking, prefix = _infer_thinking_prefix_from_path(csv_path)
    title_model = args.title or _infer_title_from_path(csv_path)

    sub = scoring
    if scoring == "at_least" and min_c:
        sub = f"{sub}, min_correct={min_c}"
    if thinking is not None and prefix is not None:
        sub = f"{sub}, thinking={thinking}, prefix_q={prefix}"
    if args.reasoning_budget is not None:
        sub = f"{sub}, reason_budget={int(args.reasoning_budget)}"

    y0_s, y1_s = (x.strip() for x in args.ylim.split(",", 1))
    y0, y1 = float(y0_s), float(y1_s)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(
        df["k"].astype(int),
        df["accuracy"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
    )
    plt.xlabel("Number of target questions (k)")
    plt.ylabel(args.ylabel)
    if slen is None:
        plt.title(f"{title_model}\n{sub}".rstrip(", "))
    else:
        plt.title(f"{title_model}\nseq_len={slen}, {sub}".rstrip(", "))
    plt.grid(True, alpha=0.3)
    plt.ylim(y0, y1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved", out_path)


if __name__ == "__main__":
    main()

