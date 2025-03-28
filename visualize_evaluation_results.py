import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Paths
base_dir = "/home/alextu/scratch/DeepNucNet_computecanada/evaluate_model_results_with_augmentations"
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

# Combine and calculate means
combined_df = pd.concat(all_metrics, ignore_index=True)
mean_df = combined_df.groupby("model").mean(numeric_only=True).reset_index()
melted_mean = pd.melt(mean_df, id_vars="model", var_name="metric", value_name="mean_value")

# Plot setup
metrics = ["dice", "precision", "recall", "hausdorff"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
palette = {"model_unet2": "#4C72B0", "model_unetR": "#DD8452"}

for i, metric in enumerate(metrics):
    ax = axes[i]
    metric_data = melted_mean[melted_mean["metric"] == metric]

    sns.barplot(
        data=metric_data,
        x="model",
        y="mean_value",
        hue="model",
        palette=palette,
        ax=ax,
        legend=False
    )

    # Add value labels
    for idx, row in metric_data.iterrows():
        ax.text(
            idx % 2,
            row["mean_value"] + 0.01,
            f"{row['mean_value']:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_title(metric.capitalize())
    ax.set_ylabel("Mean Value")
    ax.set_xlabel("")
    ax.set_ylim(0, metric_data["mean_value"].max() + 0.2)

# Final layout
plt.suptitle("Mean Metric Comparison: model_unet2 vs model_unetR", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_path)
print(f"Saved labeled 2x2 grid bar plot to: {output_path}")
