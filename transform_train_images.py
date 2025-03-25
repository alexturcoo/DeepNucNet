import PIL
import torch
import numpy as np
from torchvision import transforms as tsf
import torchvision.transforms.functional as F

# Convert binary mask to float tensor in [0, 1]
def binary_mask_to_tensor(x):
    return torch.tensor(np.array(x), dtype=torch.float32) / 255.0

class PairedTransform:
    def __init__(self):
        self.resize = tsf.Resize((256, 256))
        self.normalize = tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.to_tensor = tsf.ToTensor()

    def __call__(self, img, mask):
        # Resize both image and mask
        img = self.resize(img)
        mask = self.resize(mask)

        # Image: to tensor and normalize to [-1, 1]
        img = self.to_tensor(img)
        img = self.normalize(img)

        # Mask: to float tensor in [0, 1]
        mask = binary_mask_to_tensor(mask)

        return img, mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.datas = data
        self.transform = transform

    def __getitem__(self, index):
        data = self.datas[index]

        # Convert image from [0,1] float → uint8 for PIL
        img = PIL.Image.fromarray((data['img'].numpy() * 255).astype(np.uint8))

        # Convert binary mask to PIL
        mask = PIL.Image.fromarray((data['mask'].numpy() * 255).astype(np.uint8))

        # Apply paired transforms
        img, mask = self.transform(img, mask)

        # Add channel dim to mask: [1, H, W]
        return img, mask.unsqueeze(0)

    def __len__(self):
        return len(self.datas)

# === Apply transform and save dataset ===
if __name__ == "__main__":
    paired_transform = PairedTransform()

    train_data = torch.load(
        "/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/train_data.pth",
        weights_only=False
    )

    dataset = Dataset(train_data, paired_transform)

    torch.save(
        dataset,
        "/home/alextu/scratch/DeepNucNet_computecanada/transformed_train_data_pth/train_data_transformed.pth"
    )

    print("Transformed dataset saved with masks shaped [1, 256, 256]. Ready for training.")