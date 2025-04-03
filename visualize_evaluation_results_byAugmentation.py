import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Paths
base_dirs = {
    "No Augmentations": "/home/alextu/scratch/DeepNucNet_computecanada/evaluate_model_results_no_augmentations/model_unetR/test_metrics.csv",
    "With Augmentations": "/home/alextu/scratch/DeepNucNet_computecanada/evaluate_model_results_with_augmentations/model_unetR/test_metrics.csv"
}
output_path = "/home/alextu/scratch/DeepNucNet_computecanada/unetR_augmentation_comparison_boxplot.png"

# Load and combine Unet2 results from both conditions
all_data = []

for aug_label, csv_path in base_dirs.items():
    with open(csv_path, "r") as f:
        lines = f.readlines()
        valid_lines = [line for line in lines if not line.startswith("MEAN")]
    df = pd.read_csv(StringIO("".join(valid_lines)))
    df["augmentation"] = aug_label
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# Plot setup
metrics = ["dice", "precision", "recall", "hausdorff"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
palette = {"No Augmentations": "#4C72B0", "With Augmentations": "#DD8452"}

for i, metric in enumerate(metrics):
    ax = axes[i]

    sns.boxplot(
        data=combined_df,
        x="augmentation",
        y=metric,
        palette=palette,
        ax=ax,
        showfliers=False,
        linewidth=1
    )

    sns.swarmplot(
        data=combined_df,
        x="augmentation",
        y=metric,
        palette=palette,
        ax=ax,
        dodge=True,
        size=4,
        alpha=0.7,
        marker="o",
        linewidth=0.5,
        edgecolor="gray"
    )

    # Add median labels above the boxes
    medians = combined_df.groupby("augmentation")[metric].median()
    for j, aug_label in enumerate(["No Augmentations", "With Augmentations"]):
        median_val = medians[aug_label]
        ax.text(
            j,
            combined_df[metric].max() + 0.05,
            f"Median: {median_val:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color=palette[aug_label]
        )

    ax.set_title(metric.capitalize())
    ax.set_ylabel("Value")
    ax.set_xlabel("")
    ax.set_ylim(0, combined_df[metric].max() + 0.15)

# Final layout
plt.suptitle("UNetR: Augmentation vs No Augmentation", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_path)
print(f"Saved UNet2 augmentation comparison plot to: {output_path}")
