import torch
from monai.networks.nets import BasicUnetPlusPlus

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BasicUnetPlusPlus( #UNet++ model
    spatial_dims=2, 
    features=(64, 128, 256, 512, 1024, 128),
    in_channels=3,          # e.g. RGB images
    out_channels=1,         # Binary segmentation
    deep_supervision=False #Required here to output a single tensor instead of a list of tensors
).to(device)

print("Model loaded from model_unetPlusPlus.py:")
print(model)