import torch
import numpy as np
import matplotlib.pyplot as plt 
# Import the Dataset class from the transformation script
from transform_train_images import Dataset  

dataset = torch.load("/home/alextu/scratch/DeepNucNet/transformed_train_data_pth/train_data_transformed.pth", weights_only=False)
print("Size of training dataset", len(dataset))

# Extract an image-mask pair from the dataset
img, mask = dataset[1]

# Print general dataset information
print(f"Dataset size: {len(dataset)} samples")
print(f"Image shape: {img.shape}")  # Shape in (C, H, W) format
print(f"Mask shape: {mask.shape}")  # Shape in (1, H, W) format

# Convert to NumPy arrays for further analysis
img_np = img.permute(1, 2, 0).numpy()  # Convert image from (C, H, W) -> (H, W, C)
mask_np = mask[0].numpy()  # Extract the single-channel mask

# Print mask statistics
unique_labels = np.unique(mask_np)  # Get unique labels present in the mask
num_masks = len(unique_labels) - 1  # Excluding background (assuming 0 is background)

print(f"Unique labels in mask: {unique_labels}")
print(f"Number of masks (segmentation objects): {num_masks}")

# Plot the image and mask
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img_np * 0.5 + 0.5)  # Denormalize and display image
plt.title("Image")

plt.subplot(122)
plt.imshow(mask_np, cmap="gray")  # Display mask
plt.title("Segmentation Mask")

plt.savefig("/home/alextu/scratch/DeepNucNet/results/training_data_visualisation/row1.png")
plt.show()