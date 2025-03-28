import os
import pandas as pd

# Base directory where all model folders are located
base_dir = "/home/alextu/scratch/DeepNucNet_computecanada/tune_array_results_with_augmentations"

# List of model folders to process
model_dirs = ["model_unet1", "model_unet2", "model_unet3", "model_unetR"]

for model in model_dirs:
    model_path = os.path.join(base_dir, model)
    combined_df = []

    # Loop through subdirectories starting with "bs"
    for subdir in os.listdir(model_path):
        if subdir.startswith("bs"):
            result_file = os.path.join(model_path, subdir, "result.csv")
            if os.path.isfile(result_file):
                df = pd.read_csv(result_file)
                df["bs_dir"] = subdir  # optional: track which subdir it came from
                combined_df.append(df)

    # Combine all into one DataFrame
    if combined_df:
        result_df = pd.concat(combined_df, ignore_index=True)
        output_file = os.path.join(model_path, "results_combined.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Saved combined results to: {output_file}")
    else:
        print(f"No results found in: {model_path}")
