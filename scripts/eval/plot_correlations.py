import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


def load_data(heatmap_data_path, opencompass_data_path):
    """Load and preprocess the data."""
    heatmap_data = pd.read_csv(heatmap_data_path)
    heatmap_data = heatmap_data.reset_index().set_index(["seq_len", "qtype", "model"])

    opencompass_df = pd.read_csv(opencompass_data_path, index_col=1)
    bench_cols = [
        "Avg. Score",
        "MMBench V1.1",
        "MMStar",
        "MMMU",
        "MathVista",
        "HallusionBench Avg.",
        "AI2D",
        "OCRBench",
        "MMVet",
    ]
    opencompass_df = opencompass_df[bench_cols].dropna()
    opencompass_df.index = opencompass_df.index.str.replace(".", "_", regex=False)

    return heatmap_data, opencompass_df


def find_model_match(row, df2):
    """Find matching models in df2 based on substring."""
    model = row["model"].split("/")[-1].strip("-Instruct")
    matches = [idx for idx in df2.index if model in idx]
    return sorted(matches, key=lambda x: len(x))[0] if matches else None


def prepare_data(heatmap_data, opencompass_df):
    """Prepare data for analysis."""
    df1_reset = (
        heatmap_data.loc[(heatmap_data != 0).any(axis=1)]
        .groupby(["seq_len", "model"])["hit"]
        .mean()
        .reset_index()
    )
    df1_reset["model_match"] = df1_reset.apply(
        lambda row: find_model_match(row, opencompass_df), axis=1
    )
    result = pd.merge(
        df1_reset, opencompass_df, left_on="model_match", right_index=True, how="left"
    )
    return result.drop(columns=["model_match"])


def plot_pairwise(df, output_dir):
    """Generate pairwise plots."""
    columns_to_compare = [
        "Avg. Score",
        "MMBench V1.1",
        "MMStar",
        "MMMU",
        "MathVista",
        "HallusionBench Avg.",
        "AI2D",
        "OCRBench",
        "MMVet",
    ]
    n_rows = len(df["seq_len"].dropna().unique())
    n_cols = len(columns_to_compare)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=True
    )
    axes = axes.flatten() if n_rows > 1 and n_cols > 1 else [axes]

    plot_idx = 0
    for n_step, group in df.dropna().groupby("seq_len"):
        for col in columns_to_compare:
            sns.regplot(
                data=group, x=col, y="hit", ax=axes[plot_idx], label=f"seq_len={n_step}"
            )
            axes[plot_idx].set_title(f"seq_len={n_step}, X={col}")
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("hit")
            plot_idx += 1

    plt.savefig(
        os.path.join(output_dir, "pairwise_plots.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_correlation(df, output_dir):
    """Plot Pearson correlation."""
    columns_to_compare = df.columns[df.columns.get_loc("hit") + 1 :]
    pearson_results = []
    for n_step, group in df.dropna().groupby("seq_len"):
        correlations = {"seq_len": n_step}
        hits = group.groupby("model")["hit"].mean().to_numpy()
        for col in columns_to_compare:
            bench = group.groupby("model")[col].mean().to_numpy()
            if len(bench) > 1:
                corr, _ = pearsonr(bench, hits)
            else:
                corr = 0.0
            correlations[col] = corr
            correlations[col + "_n"] = len(bench)
        pearson_results.append(correlations)

    pearson_df = pd.DataFrame(pearson_results)
    cols = [
        "Avg. Score",
        "MMBench V1.1",
        "MMStar",
        "MMMU",
        "MathVista",
        "HallusionBench Avg.",
        "AI2D",
        "OCRBench",
        "MMVet",
    ]
    colors = plt.cm.inferno(np.linspace(0.1, 0.9, len(cols)))

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(cols):
        plt.plot(
            pearson_df["seq_len"],
            pearson_df[col],
            color=colors[i],
            label=col,
            marker="o",
        )
        std = np.sqrt(
            (1 - pearson_df[col] ** 2) / (np.maximum(pearson_df[col + "_n"], 3) - 2)
        )
        plt.fill_between(
            pearson_df["seq_len"],
            pearson_df[col] - std,
            (pearson_df[col] + std).clip(0, 1),
            color=colors[i],
            alpha=0.4,
        )

    plt.title('Pearson Correlation of "hit" vs Other Columns by seq_len')
    plt.xlabel("seq_len")
    plt.xscale("log", base=2)
    plt.xticks(2 ** np.arange(0, 8))
    plt.ylabel("Pearson Correlation")
    plt.legend(title="Columns", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "corr.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from heatmap and OpenCompass data."
    )
    parser.add_argument(
        "--heatmap_data",
        type=str,
        default="results/newest_results.csv",
        help="Path to the heatmap data CSV file.",
    )
    parser.add_argument(
        "--opencompass_data",
        type=str,
        default="results/opencompass.csv",
        help="Path to the OpenCompass data CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the output plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    heatmap_data, opencompass_df = load_data(args.heatmap_data, args.opencompass_data)
    df = prepare_data(heatmap_data, opencompass_df)
    # plot_pairwise(df, args.output_dir)
    plot_correlation(df, args.output_dir)


if __name__ == "__main__":
    main()
