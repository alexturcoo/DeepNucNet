import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from transform_train_images import Dataset, PairedTransform


# Load your transformed dataset
dataset = torch.load("/home/alextu/scratch/DeepNucNet_computecanada/transformed_train_data_pth/train_data_transformed.pth", weights_only = False)

# Choose a sample to visualize
img, mask = dataset[22]  # Index can be changed to visualize other samples

# Convert image to [H, W, C] format and unnormalize (from [-1, 1] → [0, 1])
img_np = img.permute(1, 2, 0).numpy()
img_np = (img_np * 0.5) + 0.5  # Undo normalization if you used mean=0.5, std=0.5
img_np = np.clip(img_np, 0, 1)

# Convert mask to numpy and squeeze if needed
mask_np = mask.numpy()
if mask_np.ndim == 3:
    mask_np = mask_np.squeeze()

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_np, cmap='gray')  # Great for visualizing instance labels
plt.title("Segmentation Mask (Instances)")
plt.axis("off")

plt.tight_layout()

# Ensure output directory exists
save_dir = "/home/alextu/scratch/DeepNucNet_computecanada/results/training_data_visualisation"
os.makedirs(save_dir, exist_ok=True)

# Save and show
plt.savefig(os.path.join(save_dir, "row22.png"))
plt.show()