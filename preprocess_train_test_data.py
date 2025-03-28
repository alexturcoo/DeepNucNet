import os
import numpy as np
import torch as t
import pandas as pd
from pathlib import Path
from skimage import io
from tqdm import tqdm

# Define paths
TRAIN_PATH = '/home/alextu/scratch/DeepNucNet_computecanada/data/stage1_train/'
TEST_PATH = '/home/alextu/scratch/DeepNucNet_computecanada/data/stage1_test/'

# CSV with RLE masks for test set
TEST_CSV = '/home/alextu/scratch/DeepNucNet_computecanada/data/stage1_solution.csv'  # Update path if needed

# Output paths for saving .pth files
TRAIN_PTH = '/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/train_data.pth'
TEST_PTH = '/home/alextu/scratch/DeepNucNet_computecanada/train_test_data_pth/test_data.pth'

# ------------------- RLE Decoding Function -------------------

def rle_decode(rle_str, height, width):
    """Decode a run-length encoded string into a binary mask."""
    s = list(map(int, rle_str.split()))
    starts, lengths = s[::2], s[1::2]
    starts = [x - 1 for x in starts]
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape((height, width), order='F')

# ------------------- Load RLE Mask CSV -------------------

# Format: {image_id: {'rles': [...], 'height': H, 'width': W}}
rle_dict = {}
if os.path.exists(TEST_CSV):
    rle_df = pd.read_csv(TEST_CSV)
    for _, row in rle_df.iterrows():
        img_id = row['ImageId']
        rle = row['EncodedPixels']
        h, w = int(row['Height']), int(row['Width'])

        if img_id not in rle_dict:
            rle_dict[img_id] = {'rles': [], 'height': h, 'width': w}
        rle_dict[img_id]['rles'].append(rle)

# ------------------- Dataset Processing Function -------------------

def process_dataset(file_path, has_mask=True, use_rle_dict=None):
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
            # If we have RLE masks (e.g., test set with CSV)
            if use_rle_dict is not None and file.name in use_rle_dict:
                info = use_rle_dict[file.name]
                h, w = info['height'], info['width']
                masks = []
                for rle in info['rles']:
                    decoded = rle_decode(rle, h, w)
                    masks.append(decoded)
                combined_mask = (np.sum(np.stack(masks), axis=0) > 0).astype(np.uint8)
                item['mask'] = t.from_numpy(combined_mask).float()
            else:
                # Regular training set with image masks
                mask_files = list((file / 'masks').iterdir())
                masks = None
                for ii, mask_path in enumerate(mask_files):
                    mask = io.imread(mask_path)
                    assert (mask[(mask != 0)] == 255).all()  # Make sure masks are binary (0 or 255)

                    if masks is None:
                        H, W = mask.shape
                        masks = np.zeros((len(mask_files), H, W), dtype=np.uint8)

                    masks[ii] = (mask / 255).astype(np.uint8)

                combined_mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)
                item['mask'] = t.from_numpy(combined_mask).float()

        datas.append(item)

    return datas

# ------------------- Process and Save -------------------

train_data = process_dataset(TRAIN_PATH, has_mask=True)
t.save(train_data, TRAIN_PTH)
print(f"Train dataset saved to {TRAIN_PTH}")

test_data = process_dataset(TEST_PATH, has_mask=True, use_rle_dict=rle_dict)
t.save(test_data, TEST_PTH)
print(f"Test dataset saved to {TEST_PTH}")
