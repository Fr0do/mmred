import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import hmean

qgen_map = {
    'first_app': 'FA-FA-R',
    'char_on_char_first_app': 'FA-CCFA-R',
    'first_at_room': 'FA-FR-C',
    'room_on_char_first_app': 'FA-RCFA-C',
    'n_room_on_char_first_app': 'FA-NRFA-I',
    'final_app': 'FI-FA-R',
    'char_on_char_final_app': 'FI-CCFA-R',
    'last_at_room': 'FI-LR-C',
    'room_on_char_final_app': 'FI-RCFA-C',
    'n_room_on_char_final_app': 'FI-NRFA-I',
    'char_at_frame': 'FX-CF-R',
    'room_at_frame': 'FX-RF-C',
    'char_on_char_at_frame': 'FX-CCF-C',
    'n_char_at_frame': 'FX-NCF-I',
    'n_empty': 'FX-NE-I',
    'room_empty': 'LC-RE-R',
    'where_spend': 'LC-WS-R',
    'crowded_room': 'LC-CR-R',
    'who_spend': 'LC-WHS-C',
    'spend_alone': 'LC-SA-C',
    'spend_together': 'LC-ST-C',
    'steps_in_room': 'LC-SR-I',
    'rooms_visited': 'LC-RV-I',
    'crowd_count': 'LC-CC-I'
}


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
        heatmap_data.loc[:, 'qtype'] = heatmap_data.qtype.apply(lambda x: qgen_map.get(x))
        heatmap_data = heatmap_data.reset_index().set_index(
            ["seq_len", "qtype", "model"]
        )
        return heatmap_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def filter_models(heatmap_data, models_list):
    """
    Filter the heatmap data to include only specified models.

    Args:
        heatmap_data (pd.DataFrame): Original heatmap data.
        models_list (list): List of model names to include.

    Returns:
        pd.DataFrame: Filtered heatmap data.
    """
    if not models_list:
        return heatmap_data

    # Get all available models
    available_models = heatmap_data.index.get_level_values("model").unique().tolist()

    # Check if specified models exist in the data
    valid_models = [model for model in models_list if model in available_models]

    if not valid_models:
        print(
            f"Warning: None of the specified models {models_list} found in data. Using all models."
        )
        return heatmap_data

    if len(valid_models) < len(models_list):
        missing_models = set(models_list) - set(valid_models)
        print(f"Warning: Some models not found in data: {missing_models}")

    # Filter data to include only specified models
    filtered_data = heatmap_data[
        heatmap_data.index.get_level_values("model").isin(valid_models)
    ]

    return filtered_data


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
    rotation=-90,
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
        annot_kws={'size': 16},
        cbar_kws={"label": "Accuracy"},
    )
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title)
    plt.tight_layout()
    plt.xticks(rotation=rotation)
    plt.savefig(output_path + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(output_path + ".pdf", bbox_inches="tight", dpi=300)
    plt.close()


def generate_heatmaps(heatmap_data, output_dir, exp_name):
    """
    Generate and save multiple heatmaps based on the provided data.

    Args:
        heatmap_data (pd.DataFrame): Data for generating heatmaps.
        output_dir (str): Directory to save the heatmaps.
        exp_name (str): Name of the experiment (used for plot titles).
    """
    custom_cmap = sns.color_palette("RdYlGn", as_cmap=True)
    # custom_cmap = sns.color_palette("PRGn", as_cmap=True)
    # custom_cmap = sns.color_palette("Spectral", as_cmap=True)

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
        ylabel="Steps in context",
        title=f"{exp_name} Mean of all models",
        output_path=os.path.join(output_dir, "mmlong_all_models"),
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
        title=f"{exp_name} Mean of all lengths",
        output_path=os.path.join(output_dir, "mmlong_all_lengths"),
        cmap=custom_cmap,
        rotation=-60,
    )

    # Plot 2.1: HMean of all lengths
    mean_all_lengths = (
        heatmap_data.groupby(["qtype", "model"])
        .agg(hmean)
        .reset_index()
        .pivot(index="model", columns="qtype", values="hit")
    )
    sorted_columns = mean_all_lengths.mean().sort_values(ascending=False).index
    mean_all_lengths = mean_all_lengths[sorted_columns]
    plot_heatmap(
        mean_all_lengths,
        xlabel="Type of question",
        ylabel="Model",
        title=f"{exp_name} Harmonic Mean of all lengths",
        output_path=os.path.join(output_dir, "mmlong_all_lengths_hmean"),
        cmap=custom_cmap,
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
        ylabel="Steps in context",
        title=f"{exp_name} Mean of all questions",
        output_path=os.path.join(output_dir, "mmlong_all_questions"),
        cmap=custom_cmap,
    )

    # Plot 3.1: HMean of all questions
    mean_all_questions = (
        heatmap_data.groupby(["seq_len", "model"])
        .agg(hmean)
        .reset_index()
        .pivot(index="seq_len", columns="model", values="hit")
    )
    sorted_columns = mean_all_questions.mean().sort_values(ascending=False).index
    mean_all_questions = mean_all_questions[sorted_columns]
    plot_heatmap(
        mean_all_questions,
        xlabel="Model",
        ylabel="Steps in context",
        title=f"{exp_name} Harmonic mean of all questions",
        output_path=os.path.join(output_dir, "mmlong_all_questions_hmean"),
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
            ylabel="Steps in context",
            title=f"{exp_name} {model}",
            output_path=os.path.join(
                output_dir, f"mmlong_{'_'.join(model.split('/'))}"
            ),
            cmap=custom_cmap,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps from CSV data.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="results.csv",
        help="Path to the CSV file containing the heatmap data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the generated heatmaps.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="main",
        help="Name of the experiment (used for naming output files).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to include in the analysis. If not specified, all models will be used.",
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.exp_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Load the data
        data = load_data(args.data_path)

        # Apply model filtering if specified
        if args.models:
            print(f"Filtering data to include only these models: {args.models}")
            filtered_data = filter_models(data, args.models)

            # Create a models suffix for output directory
            models_suffix = "_".join([m.split("/")[-1] for m in args.models[:3]])
            if len(args.models) > 3:
                models_suffix += "_etc"

            # Create a subdirectory for these specific models
            models_output_dir = os.path.join(output_dir, f"models_{models_suffix}")
            if not os.path.exists(models_output_dir):
                os.makedirs(models_output_dir)

            # Generate heatmaps with filtered data
            generate_heatmaps(filtered_data, models_output_dir, f"{args.exp_name}")

        # Always generate the full heatmaps as well
        generate_heatmaps(data, output_dir, args.exp_name)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()