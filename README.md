# DeepNucNet
Nuclei detection and segmentation deep learning project for graduate level AI course (Biomedical Applications of Artificial Intelligence)

# Step 1: Downloading the Data
Go to the following link https://www.kaggle.com/competitions/data-science-bowl-2018/data and download the 7 zip files to wherever you have 385.48 MB

# Step 2: Processing the Data
To process the Images and corresponding masks (labels) for the nuclei image. Use the script
`preprocess.py` - This script prepares the image data. Loads images from downloaded data, prepares segmentation masks as labelled tensors.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

```python
# Define paths 
TRAIN_PATH = 'your_file_path/stage1_train/'    
TEST_PATH = 'your_file_path/stage1_test/'

# Output paths for saving .pth files
TRAIN_PTH = 'your_file_path/train_test_pth_data/train_data.pth'
TEST_PTH = 'your_file_path/DeepNucNet/train_test_pth_datatest_data.pth'`
```

This will output 2 .pth files, one for train data (`train_data.pth`), and one for test data (`test_data.pth`).

# Step 3: Transforming images for training
To transform the images prior to training, Use the script
`transform_train_images.py`

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

```python
train_data = torch.load("your_file_path/train_test_data_pth/train_data.pth", weights_only=False)  # From Step 1
dataset = Dataset(train_data,s_trans,t_trans)
torch.save(dataset, "your_file_path/transformed_train_data_pth/train_data_transformed.pth")  # saves the entire Dataset object
```

This will output the transformed saved training data as `train_data_transformed.pth` 

# Step 4: Visualize Train Data
If you are interested in seeing what the training data looks like (images and corresponding masks), you can use the following code
`visualize_train_data.py` 

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

```python
dataset = torch.load("your_file_path/transformed_train_data_pth/train_data_transformed.pth", weights_only=False)
```

# Step 5: Training the Model
The script below will run model training using the processed and transformed training data created above.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

```python
DATASET_PATH="your_file_path/transformed_train_data_pth/train_data_transformed.pth"
BATCH_SIZE=16
LEARNING_RATE=0.0001
NUM_WORKERS=0
EPOCHS=300
TRAIN_RATIO=0.8
VAL_INTERVAL=2
```





