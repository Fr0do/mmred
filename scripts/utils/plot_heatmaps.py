import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(filepath):
    """
    Load and prepare the heatmap data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Prepared heatmap data with multi-indexing.
    """
    try:
        heatmap_data = pd.read_csv(filepath)
        heatmap_data = heatmap_data.reset_index().set_index(
            ["seq_len", "qtype", "model"]
        )
        return heatmap_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def plot_heatmap(
    data,
    xlabel,
    ylabel,
    title,
    output_path,
    annot=True,
    fmt="2.0f",
    cmap=None,
    vmin=0,
    vmax=100,
    rotation=90,
):
    """
    Generic function to plot and save a heatmap.

    Args:
        data (pd.DataFrame): Data to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        output_path (str): Path to save the plot.
        annot (bool): Whether to annotate the heatmap.
        fmt (str): Format for annotations.
        cmap: Colormap for the heatmap.
        vmin (int): Minimum value for the colormap.
        vmax (int): Maximum value for the colormap.
        rotation (int): Rotation angle for x-axis labels.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Accuracy"},
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.xticks(rotation=rotation)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def generate_heatmaps(heatmap_data, output_dir):
    """
    Generate and save multiple heatmaps based on the provided data.

    Args:
        heatmap_data (pd.DataFrame): Data for generating heatmaps.
        output_dir (str): Directory to save the heatmaps.
    """
    custom_cmap = sns.color_palette("RdYlGn", as_cmap=True)

    # Plot 1: Mean of all models
    mean_all_models = (
        heatmap_data.groupby(["seq_len", "qtype"])
        .mean()
        .reset_index()
        .pivot(index="seq_len", columns="qtype", values="hit")
    )
    plot_heatmap(
        mean_all_models,
        xlabel="Type of question",
        ylabel="Images in context",
        title="Mean of all models",
        output_path=os.path.join(output_dir, "mmlong_all_models.png"),
        cmap=custom_cmap,
    )

    # Plot 2: Mean of all lengths
    mean_all_lengths = (
        heatmap_data.groupby(["qtype", "model"])
        .mean()
        .reset_index()
        .pivot(index="model", columns="qtype", values="hit")
    )
    sorted_columns = mean_all_lengths.mean().sort_values(ascending=False).index
    mean_all_lengths = mean_all_lengths[sorted_columns]
    plot_heatmap(
        mean_all_lengths,
        xlabel="Type of question",
        ylabel="Model",
        title="Mean of all lengths",
        output_path=os.path.join(output_dir, "mmlong_all_lengths.png"),
        cmap=custom_cmap,
        rotation=-60,
    )

    # Plot 3: Mean of all questions
    mean_all_questions = (
        heatmap_data.groupby(["seq_len", "model"])
        .mean()
        .reset_index()
        .pivot(index="seq_len", columns="model", values="hit")
    )
    sorted_columns = mean_all_questions.mean().sort_values(ascending=False).index
    mean_all_questions = mean_all_questions[sorted_columns]
    plot_heatmap(
        mean_all_questions,
        xlabel="Model",
        ylabel="Images in context",
        title="Mean of all questions",
        output_path=os.path.join(output_dir, "mmlong_all_questions.png"),
        cmap=custom_cmap,
    )

    # Plot 4: Heatmaps for individual models
    for model in heatmap_data.index.get_level_values("model").unique():
        model_data = (
            heatmap_data.xs(model, level=2)
            .reset_index()
            .pivot(index="seq_len", columns="qtype", values="hit")
        )
        plot_heatmap(
            model_data,
            xlabel="Type of question",
            ylabel="Images in context",
            title=model,
            output_path=os.path.join(
                output_dir, f"mmlong_{"_".join(model.split("/"))}.png"
            ),
            cmap=custom_cmap,
        )


if __name__ == "__main__":
    data_path = "results/newest_results.csv"
    output_directory = "results"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        data = load_data(data_path)
        generate_heatmaps(data, output_directory)
    except Exception as e:
        print(f"An error occurred: {e}")
