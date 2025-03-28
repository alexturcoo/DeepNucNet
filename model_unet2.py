# model.py
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the MONAI UNet on the given device
model = UNet(
    spatial_dims=2,         # 2D Image Segmentation
    in_channels=3,          # e.g. RGB images
    out_channels=1,         # Binary segmentation
    channels=(32, 64, 128, 256, 512),  # Increased number of filters by 2X vs model_unet1
    strides=(2, 2, 2, 2),   # Downsampling strides
    num_res_units=2,        # Number of residual units
    norm=Norm.BATCH,         # Batch Normalization
    dropout=0.2             # Added Dropout, perhaps to prevent overfitting
).to(device)

print("Model loaded from model_unet2.py:")
print(model)