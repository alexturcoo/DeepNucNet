import torch
import PIL
import numpy as np
from torchvision import transforms as tsf
import monai.transforms as mt
import torchio as tio

# Convert binary mask to float tensor in [0, 1]
def binary_mask_to_tensor(x):
    return torch.tensor(np.array(x), dtype=torch.float32) / 255.0

# === Paired Transform ===
class PairedTransform:
    def __init__(self, augment=False):
        self.augment = augment
        self.resize = tsf.Resize((256, 256))
        self.to_tensor = tsf.ToTensor()
        self.normalize = tsf.Normalize(mean=[0.5]*3, std=[0.5]*3)

        # MONAI grayscale-only augmentations
        self.monai_augment = mt.Compose([
            mt.RandGaussianNoise(prob=0.15, mean=0.0, std=0.1),
            mt.RandShiftIntensity(offsets=0.1, prob=0.15),
            mt.RandAdjustContrast(prob=0.15, gamma=(0.95, 1.05)),
            mt.RandGaussianSmooth(prob=0.15),
            mt.RandGaussianSharpen(prob=0.15),
        ])

        # TorchIO affine (image + mask)
        self.torchio_transform = tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=10,
            translation=2,
            p=0.25
        )

    def __call__(self, img, mask):
        # Resize
        img = self.resize(img)
        mask = self.resize(mask)

        img_tensor = self.to_tensor(img)  # [3, H, W]
        mask_tensor = binary_mask_to_tensor(mask).unsqueeze(0)  # [1, H, W]

        if self.augment:
            # === Step 1: Grayscale conversion for MONAI augments ===
            img_np = np.array(img).astype(np.float32) / 255.0
            gray_np = np.mean(img_np, axis=2)  # [H, W]
            gray_tensor = torch.tensor(gray_np, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

            # Apply MONAI intensity/contrast augmentations
            gray_aug = self.monai_augment(gray_tensor)

            # Repeat to 3 channels
            img_tensor = gray_aug.repeat(3, 1, 1)  # [3, H, W]

            # === Step 2: TorchIO paired spatial transforms ===
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img_tensor.unsqueeze(0)),
                mask=tio.LabelMap(tensor=mask_tensor.repeat(3, 1, 1).unsqueeze(0))
            )
            subject = self.torchio_transform(subject)

            img_tensor = subject.image.data.squeeze(0)  # [3, H, W]
            mask_tensor = subject.mask.data.squeeze(0)[0].unsqueeze(0)  # [1, H, W]

        # Normalize image to [-1, 1]
        img_tensor = self.normalize(img_tensor)
        return img_tensor, mask_tensor.squeeze(0)

# === Dataset wrapper ===
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.datas = data
        self.transform = transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = PIL.Image.fromarray((data['img'].numpy() * 255).astype(np.uint8))
        mask = PIL.Image.fromarray((data['mask'].numpy() * 255).astype(np.uint8))
        img, mask = self.transform(img, mask)
        return img, mask.unsqueeze(0)  # Final mask shape: [1, H, W]

    def __len__(self):
        return len(self.datas)

# === Transform and save ===
if __name__ == "__main__":
    train_data = torch.load("/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/train_data.pth", weights_only=False)
    test_data = torch.load("/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/test_data.pth", weights_only=False)

    train_transform = PairedTransform(augment=True)
    test_transform = PairedTransform(augment=False)

    train_dataset = Dataset(train_data, train_transform)
    test_dataset = Dataset(test_data, test_transform)

    torch.save(train_dataset, "/home/alextu/scratch/DeepNucNet_computecanada/transformed_train_data_pth/train_data_augmented.pth")
    torch.save(test_dataset, "/home/alextu/scratch/DeepNucNet_computecanada/transformed_train_data_pth/test_data_augmented.pth")

    print("Augmented TRAIN and clean TEST datasets saved.")
