import torch
from monai.networks.nets import UNETR


# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR( #Mostly using default UNETR model parameters
    in_channels=3,          # e.g. RGB images
    out_channels=1,         # Binary segmentation
    img_size=256,    # Image size
    feature_size=16,        # Number of features
    norm_name='instance',   # Normalization
    spatial_dims=2         # 2D Image Segmentation
).to(device)

print("Model loaded from model_unetR.py:")
print(model)