import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Paths
base_dir = "/home/alextu/scratch/DeepNucNet_computecanada/evaluate_model_results_no_augmentations"
output_path = os.path.join(base_dir, "mean_metric_subplots_2x2_unet2_vs_unetR_labeled.png")

# Target models
target_models = ["model_unet2", "model_unetR"]
all_metrics = []

# Load and clean CSVs
for model_name in target_models:
    csv_path = os.path.join(base_dir, model_name, "test_metrics.csv")
    with open(csv_path, "r") as f:
        lines = f.readlines()
        valid_lines = [line for line in lines if not line.startswith("MEAN")]
    df = pd.read_csv(StringIO("".join(valid_lines)))
    df["model"] = model_name
    all_metrics.append(df)

# Combine data
combined_df = pd.concat(all_metrics, ignore_index=True)

# Plot setup
metrics = ["dice", "precision", "recall", "hausdorff"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
palette = {"model_unet2": "#4C72B0", "model_unetR": "#DD8452"}

for i, metric in enumerate(metrics):
    ax = axes[i]

    # Draw boxplot
    sns.boxplot(
        data=combined_df,
        x="model",
        y=metric,
        hue="model",
        palette=palette,
        ax=ax,
        showfliers=False,
        linewidth=1
    )

    # Draw swarmplot
    sns.swarmplot(
        data=combined_df,
        x="model",
        y=metric,
        hue="model",
        palette=palette,
        ax=ax,
        dodge=True,
        size=4,
        alpha=0.7,
        marker="o",
        linewidth=0.5,
        edgecolor="gray"
    )

    # Add median labels *above* the boxes
    grouped = combined_df.groupby("model")[metric].median()
    box_positions = {"model_unet2": 0, "model_unetR": 1}
    for model in ["model_unet2", "model_unetR"]:
        median_val = grouped[model]
        x = box_positions[model]
        ax.text(
            x,
            combined_df[metric].max() + 0.05,
            f"Median: {median_val:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color=palette[model]
        )

    ax.set_title(metric.capitalize())
    ax.set_ylabel("Value")
    ax.set_xlabel("")
    ax.set_ylim(0, combined_df[metric].max() + 0.15)

# Final layout
plt.suptitle("Metric Comparison: model_unet2 vs model_unetR", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_path)
print(f"Saved labeled 2x2 grid boxplot to: {output_path}")
