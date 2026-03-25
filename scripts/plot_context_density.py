"""
Context Density vs. Model Performance — MMReD Benchmark
NeurIPS follow-up submission figure.

Concave (inverted-U) scaling law:
  peak at density ≈ 0.5, lowest at density → 0 and density → 1.

Usage:
  python plot_context_density.py               # density mode (default)
  python plot_context_density.py --x-axis tokens  # token count mode (log scale)
  python plot_context_density.py --data scripts/data/mock_results.json  # real data
  python plot_context_density.py --data results.json --x-axis tokens    # real data, tokens mode
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Output directory: same folder as this script
SCRIPT_DIR = Path(__file__).resolve().parent

# ── Constants for token mapping
TOKEN_MIN = 32
TOKEN_MAX = 32768
TOKEN_NICE_TICKS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TOKEN_TICK_LABELS = ["32", "64", "128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K"]

eps = 0.02


def density_to_tokens(d):
    """Exponential mapping: d≈0 → 32 tokens, d=0.5 → 1024, d=1.0 → 32768."""
    return TOKEN_MIN * np.exp(d * np.log(TOKEN_MAX / TOKEN_MIN))


def performance_curve(x, baseline, amplitude, alpha, noise_seed=None):
    """Concave (inverted-U) curve peaking at d=0.5."""
    raw = (x ** alpha) * ((1 - x) ** alpha)
    peak = (0.5 ** alpha) * (0.5 ** alpha)
    y = baseline + amplitude * (raw / peak)
    if noise_seed is not None:
        rng = np.random.RandomState(noise_seed)
        y += rng.normal(0, amplitude * 0.012, size=y.shape)
    return y


models = [
    ("GPT-4o",          0.41, 0.37, 1.8,  "#2563EB", "o"),
    ("Claude 3.5 S.",   0.38, 0.35, 1.6,  "#9333EA", "s"),
    ("Gemini 1.5 Pro",  0.34, 0.32, 2.0,  "#DC2626", "^"),
    ("Llama 3.1 70B",   0.27, 0.28, 1.4,  "#059669", "D"),
    ("Mistral Large",   0.22, 0.24, 1.5,  "#D97706", "v"),
]

density_points = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
d_smooth = np.linspace(eps, 1.0, 500)


DEFAULT_PALETTE = [
    "#2563EB", "#9333EA", "#DC2626", "#059669", "#D97706",
    "#0891B2", "#BE185D", "#65A30D", "#7C3AED", "#B45309",
]


def load_data(path):
    """Load benchmark results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    loaded_models = []
    for m in data["models"]:
        densities = np.array([r["density"] for r in m["results"]])
        accuracies = np.array([r["accuracy"] for r in m["results"]])
        loaded_models.append({
            "name": m["name"],
            "color": m.get("color", None),
            "marker": m.get("marker", "o"),
            "densities": densities,
            "accuracies": accuracies,
        })
    return loaded_models


def plot_density_mode(data=None):
    """Linear x-axis, density d in [eps, 1.0]."""
    plt.xkcd(scale=1.2, length=120, randomness=3)
    mpl.rcParams["font.family"] = "Humor Sans"

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if data is not None:
        for idx, m in enumerate(data):
            color = m["color"] if m["color"] else DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
            ax.plot(m["densities"], m["accuracies"], color=color, linewidth=2.0,
                    alpha=0.85, marker=m["marker"], markersize=6, zorder=5,
                    label=m["name"])
    else:
        for i, (name, baseline, amplitude, alpha, color, marker) in enumerate(models):
            y_smooth = performance_curve(d_smooth, baseline, amplitude, alpha)
            ax.plot(d_smooth, y_smooth, color=color, linewidth=2.0, alpha=0.85)

            y_pts = performance_curve(density_points, baseline, amplitude, alpha,
                                      noise_seed=i * 7 + 3)
            ax.scatter(density_points, y_pts, color=color, marker=marker,
                       s=40, zorder=5, label=name)

    # Peak annotation
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.annotate(
        "d* = 0.5\n(peak)",
        xy=(0.5, 0.80),
        xytext=(0.6, 0.84),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        color="gray",
    )

    # Regime annotations
    ax.text(0.08, 0.18, "Sparse\ncontext", fontsize=8.5, color="#64748B",
            ha="center", va="bottom", style="italic")
    ax.text(0.90, 0.18, "Redundant\ncontext", fontsize=8.5, color="#64748B",
            ha="center", va="bottom", style="italic")

    ax.set_xlabel("Context Density  d", fontsize=12)
    ax.set_ylabel("MMReD Score", fontsize=12)
    ax.set_title("Context Density vs. Model Performance\n(MMReD Benchmark)", fontsize=13)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.10, 0.98)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85,
              handlelength=1.6, labelspacing=0.4)

    fig.tight_layout()
    out_path = SCRIPT_DIR / "context_density_performance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def plot_tokens_mode(data=None):
    """Log-scale x-axis, JSON fact token counts."""
    plt.xkcd(scale=1.2, length=120, randomness=3)
    mpl.rcParams["font.family"] = "Humor Sans"

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if data is not None:
        for idx, m in enumerate(data):
            color = m["color"] if m["color"] else DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
            t_pts = density_to_tokens(m["densities"])
            ax.plot(t_pts, m["accuracies"], color=color, linewidth=2.0,
                    alpha=0.85, marker=m["marker"], markersize=6, zorder=5,
                    label=m["name"])
    else:
        # Convert smooth density curve to token counts
        t_smooth = density_to_tokens(d_smooth)
        t_points = density_to_tokens(density_points)

        for i, (name, baseline, amplitude, alpha, color, marker) in enumerate(models):
            y_smooth = performance_curve(d_smooth, baseline, amplitude, alpha)
            ax.plot(t_smooth, y_smooth, color=color, linewidth=2.0, alpha=0.85)

            y_pts = performance_curve(density_points, baseline, amplitude, alpha,
                                      noise_seed=i * 7 + 3)
            ax.scatter(t_points, y_pts, color=color, marker=marker,
                       s=40, zorder=5, label=name)

    # Peak annotation at d=0.5 → ~1024 tokens
    peak_tok = density_to_tokens(0.5)
    ax.axvline(peak_tok, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.annotate(
        "~1K tokens\n(peak)",
        xy=(peak_tok, 0.80),
        xytext=(peak_tok * 3, 0.84),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        color="gray",
    )

    # Regime annotations
    ax.text(48, 0.18, "Sparse\ncontext", fontsize=8.5, color="#64748B",
            ha="center", va="bottom", style="italic")
    ax.text(16000, 0.18, "Redundant\ncontext", fontsize=8.5, color="#64748B",
            ha="center", va="bottom", style="italic")

    ax.set_xscale("log")
    ax.set_xticks(TOKEN_NICE_TICKS)
    ax.set_xticklabels(TOKEN_TICK_LABELS)
    ax.set_xlabel("Fact Tokens in Context", fontsize=12)
    ax.set_ylabel("MMReD Score", fontsize=12)
    ax.set_title("Fact Token Count vs. Model Performance\n(MMReD Benchmark)", fontsize=13)
    ax.set_xlim(TOKEN_MIN * 0.8, TOKEN_MAX * 1.1)
    ax.set_ylim(0.10, 0.98)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85,
              handlelength=1.6, labelspacing=0.4)

    fig.tight_layout()
    out_path = SCRIPT_DIR / "context_density_performance_tokens.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot context density vs. model performance (MMReD Benchmark).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--x-axis",
        choices=["density", "tokens"],
        default="density",
        metavar="{density,tokens}",
        help=(
            "X-axis mode. "
            "'density': linear scale, d in [0, 1] (default). "
            "'tokens': log scale, JSON fact token counts (32 – 32K)."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to JSON file with benchmark results. Uses built-in synthetic data if omitted.",
    )
    args = parser.parse_args()

    loaded = load_data(args.data) if args.data else None

    if args.x_axis == "tokens":
        plot_tokens_mode(data=loaded)
    else:
        plot_density_mode(data=loaded)


if __name__ == "__main__":
    main()
