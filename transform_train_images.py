import PIL
import torch
from torchvision import transforms as tsf

class Dataset():
    def __init__(self,data,source_transform,target_transform):
        self.datas = data
#         self.datas = train_data
        self.s_transform = source_transform
        self.t_transform = target_transform
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()
        img = self.s_transform(img)
        mask = self.t_transform(mask)
        return img, mask
    def __len__(self):
        return len(self.datas)
s_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128)),
    tsf.ToTensor(),
    tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
]
)
t_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128),interpolation=PIL.Image.NEAREST),
    tsf.ToTensor(),]
)

train_data = torch.load("/home/alextu/scratch/DeepNucNet/train_test_data_pth/train_data.pth", weights_only=False)  # Ensure you have this file
dataset = Dataset(train_data,s_trans,t_trans)
torch.save(dataset, "/home/alextu/scratch/DeepNucNet/transformed_train_data_pth/train_data_transformed.pth")  # saves the entire Dataset object