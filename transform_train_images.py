import PIL
import torch
from torchvision import transforms as tsf
import random

class Dataset():
    def __init__(self, data, source_transform, target_transform):
        self.datas = data
        self.s_transform = source_transform
        self.t_transform = target_transform

    def __getitem__(self, index):
        data = self.datas[index]

        # Convert image and mask to NumPy
        img = data['img'].numpy()
        mask = data['mask'][:, :, None].byte().numpy()

        # Apply transformations
        img = self.s_transform(img)
        mask = self.t_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.datas)

# **Updated Transformations for 256x256 & Data Augmentation**
s_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((256, 256)),  # Resize to 256x256
    tsf.RandomCrop(256),  # Keep 256x256 crops
    tsf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    tsf.RandomRotation(degrees=15),
    tsf.ToTensor(),
    tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

t_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((256, 256), interpolation=PIL.Image.NEAREST),  # Resize to 256x256
    tsf.RandomCrop(256),
    tsf.ToTensor(),
])

# **Load dataset and save transformed version**
train_data = torch.load("/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/train_data.pth", weights_only=False)
dataset = Dataset(train_data, s_trans, t_trans)
torch.save(dataset, "/home/alextu/scratch/DeepNucNet_computecanada/transformed_train_data_pth/train_data_transformed.pth")