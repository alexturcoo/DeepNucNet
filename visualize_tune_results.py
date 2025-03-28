import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base_dir = "/home/alextu/scratch/DeepNucNet_computecanada/tune_array_results_with_augmentations"
model_dirs = ["model_unet1", "model_unet2", "model_unet3", "model_unetR"]

# Load and combine
all_results = []

for model in model_dirs:
    file_path = os.path.join(base_dir, model, "results_combined.csv")
    df = pd.read_csv(file_path)
    df["model"] = model
    all_results.append(df)

combined_df = pd.concat(all_results, ignore_index=True)

# Save combined results
output_csv = os.path.join(base_dir, "all_models_results.csv")
combined_df.to_csv(output_csv, index=False)
print(f"Saved combined results to: {output_csv}")

# Prepare for plotting
sns.set(style="whitegrid")

# Convert params to string for categorical plotting
combined_df["lr_str"] = combined_df["lr"].astype(str)
combined_df["batch_size_str"] = combined_df["batch_size"].astype(str)
combined_df["epochs_str"] = combined_df["epochs"].astype(str)

# Sort categories
combined_df["lr_str"] = pd.Categorical(combined_df["lr_str"], 
                                       categories=sorted(combined_df["lr"].unique().astype(str)), ordered=True)
combined_df["batch_size_str"] = pd.Categorical(combined_df["batch_size_str"], 
                                               categories=sorted(combined_df["batch_size"].unique().astype(str)), ordered=True)
combined_df["epochs_str"] = pd.Categorical(combined_df["epochs_str"], 
                                           categories=sorted(combined_df["epochs"].unique().astype(str)), ordered=True)

### ----- Boxplots + Stripplots ----- ###

for param, label in zip(["lr_str", "batch_size_str", "epochs_str"], ["Learning Rate", "Batch Size", "Epochs"]):
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=combined_df, x=param, y="best_dice", hue="model")
    sns.stripplot(data=combined_df, x=param, y="best_dice", hue="model", 
                  dodge=True, color=".3", alpha=0.5, jitter=0.2, legend=False)
    plt.title(f"Best Dice by {label} (Grouped by Model)")
    plt.xlabel(label)
    plt.ylabel("Best Dice")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(base_dir, f"boxplot_dice_by_{param}.png")
    plt.savefig(plot_path)
    print(f"Saved boxplot to: {plot_path}")
    plt.close()

### ----- Heatmaps by Model (Batch Size × LR) ----- ###

grouped = (
    combined_df.groupby(["model", "batch_size", "lr"])
    .agg(mean_dice=("best_dice", "mean"))
    .reset_index()
)

for model in grouped["model"].unique():
    df_model = grouped[grouped["model"] == model]
    pivot = df_model.pivot(index="batch_size", columns="lr", values="mean_dice")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="magma", cbar_kws={"label": "Mean Dice"})
    plt.title(f"{model} - Mean Dice by Batch Size & Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Batch Size")
    plt.tight_layout()
    heatmap_path = os.path.join(base_dir, f"{model}_heatmap_dice_bs_lr.png")
    plt.savefig(heatmap_path)
    print(f"Saved heatmap to: {heatmap_path}")
    plt.close()

### ----- Top 20 Results ----- ###
top_performers = combined_df.sort_values("best_dice", ascending=False).head(20)
print("\nTop 20 parameter sets across all models:")
print(top_performers.to_string(index=False))
