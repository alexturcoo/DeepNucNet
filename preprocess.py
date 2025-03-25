import os
import numpy as np
import torch as t
from pathlib import Path
from skimage import io
from tqdm import tqdm

# Define paths
TRAIN_PATH = '/home/alextu/scratch/DeepNucNet_computecanada/data/stage1_train/'
TEST_PATH = '/home/alextu/scratch/DeepNucNet_computecanada/data/stage1_test/'

# Output paths for saving .pth files
TRAIN_PTH = '/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/train_data.pth'
TEST_PTH = '/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/test_data.pth'

def process_dataset(file_path, has_mask=True):
    """Processes image dataset and saves as a .pth file."""
    file_path = Path(file_path)
    files = sorted(list(file_path.iterdir()))
    datas = []

    for file in tqdm(files, desc=f"Processing {'Train' if has_mask else 'Test'} Dataset"):
        item = {}
        imgs = []

        # Load image
        for image in (file / 'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs) == 1
        img = imgs[0]

        # Remove alpha channel if present
        if img.shape[2] > 3:
            assert (img[:, :, 3] != 255).sum() == 0
        img = img[:, :, :3]  # Keep only RGB

        # Normalize image to (0,1)
        item['img'] = t.from_numpy(img).float() / 255.0
        item['name'] = file.name

        # Process binary masks
        if has_mask:
            mask_files = list((file / 'masks').iterdir())
            masks = None
            for ii, mask_path in enumerate(mask_files):
                mask = io.imread(mask_path)
                assert (mask[(mask != 0)] == 255).all()  # Make sure masks are binary (0 or 255)

                if masks is None:
                    H, W = mask.shape
                    masks = np.zeros((len(mask_files), H, W), dtype=np.uint8)

                masks[ii] = (mask / 255).astype(np.uint8)

            # Combine masks into a single binary mask: 1 where any nucleus is present
            combined_mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)

            # Save as float tensor for use with BCEWithLogitsLoss or DiceLoss(sigmoid=True)
            item['mask'] = t.from_numpy(combined_mask).float()

        datas.append(item)

    return datas

# Process and save datasets as .pth files
train_data = process_dataset(TRAIN_PATH, has_mask=True)
t.save(train_data, TRAIN_PTH)
print(f"Train dataset saved to {TRAIN_PTH}")

test_data = process_dataset(TEST_PATH, has_mask=False)
t.save(test_data, TEST_PTH)
print(f"Test dataset saved to {TEST_PTH}")