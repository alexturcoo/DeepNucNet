# DeepNucNet
Nuclei detection and segmentation deep learning project for graduate level AI course (Biomedical Applications of Artificial Intelligence)

# Step 1: Downloading the Data
Go to the following link https://www.kaggle.com/competitions/data-science-bowl-2018/data and download the 7 zip files to wherever you have 385.48 MB

# Step 2: Processing the Data
To process the Images and corresponding masks (labels) for the nuclei image. Use the script
`preprocess_train_test_data.py` - This script prepares the image data. Loads images from downloaded data, prepares segmentation masks as labelled tensors. Also prepares the test set masks (which must be processed from RLE format)

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

```python
# Define paths for data from Kaggle
TRAIN_PATH = 'your_file_path/stage1_train/'    
TEST_PATH = 'your_file_path/stage1_test/'

# CSV with RLE masks for test set, solution from https://bbbc.broadinstitute.org/BBBC038
TEST_CSV = 'your_file_path/stage1_solution.csv'

# Output paths for saving .pth files
TRAIN_PTH = 'your_file_path/train_test_pth_data/train_data.pth'
TEST_PTH = 'your_file_path/train_test_data_pth/test_data.pth'
```

This will output 2 .pth files, one for train data (`train_data.pth`), and one for test data (`test_data.pth`).

# Step 3: Transforming images for training
To transform the images prior to training, Use the scripts
`transform_train_test_images_no_augmentations.py` or `transform_train_test_images_with_augmentations.py` for transforming train images without or with image augmentations respectively.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPTS*

`transform_train_test_images_no_augmentations.py`:
```python
# Within "__main__"
train_data = torch.load("your_file_path/train_test_data_pth/train_data.pth", weights_only=False)  # From Step 2: Processing the Data
torch.save(train_dataset, "your_file_path/transformed_train_data_pth/train_data_transformed.pth")  # saves the entire Dataset object

test_data = torch.load("your_file_path/transformed_train_data_pth/test_data.pth", weights_only=False)
torch.save(test_dataset, "your_file_path/transformed_train_data_pth/test_data_transformed.pth")
```
This will output the transformed saved training and test data as `train_data_transformed.pth` and `test_data_transformed.pth`

---

`transform_train_test_images_with_augmentations.py`:
```python
# Within "__main__"
train_data = torch.load("your_file_path/train_test_data_pth/train_data.pth", weights_only=False) 
test_data = torch.load("your_file_path/train_test_data_pth/test_data.pth", weights_only=False)

torch.save(train_dataset, "your_file_path/transformed_train_data_pth/train_data_augmented.pth")
torch.save(test_dataset, "your_file_path/transformed_train_data_pth/test_data_augmented.pth")
```
This will output the transformed *augmented training* and *clean test* data as `train_data_augmented.pth` and `test_data_augmented.pth`

# Step 4: Visualize Train Data
If you are interested in seeing what the training data looks like (images and corresponding masks), you can use the following code
`visualize_train_data.py` 

![Visualization of Training Data Image and Masks.](/results/training_data_visualisation/row1.png)

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

```python
dataset = torch.load("your_file_path/transformed_train_data_pth/train_data_transformed.pth", weights_only=False)
```

# Step 5: Training the Model
The script `train.sh` will run model training using the processed and transformed training data created above.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

```python
DATASET_PATH="your_file_path/transformed_train_data_pth/train_data_transformed.pth"
BATCH_SIZE=16
LEARNING_RATE=0.0001
NUM_WORKERS=0
EPOCHS=200
TRAIN_RATIO=0.8
VAL_INTERVAL=2
OUTPUT_DIR="your_file_path/DeepNucNet/results"
```
Training will output the following plots to assess performance

![Training Loss and Validation Loss Curves.](/results/model_training_images/training_metrics.png)


# Step 6: Hyperparameter Tuning
The script `tune.sh` uses the generated `param_list.txt` to run a SLURM job array for to train and find the optimally performing model. The script will run 60 jobs, one per line of hyperparameters from `param_list.txt` - was performed with 1 GPU and 40G of memory.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*
`tune.sh`
```python
DATASET_PATH="your_file_path/transformed_train_data_pth/train_data_[transformed] OR [augmented].pth"
OUTPUT_DIR="your_file_path/tune_array_results/model_unet[model desired]/bs${BATCH_SIZE}_lr${LEARNING_RATE}_ep${EPOCHS}" #change directory according to the model being trained
```

If you would like to change the hyperparameters tested, edit the batch_sizes, learning_rates or epoch_list in `generate_parameter_list.py`

### Step 6.1: Process Tuning Results
After all model training is complete, use `process_tune_results.py` to process each model's training results and combine into a single csv. 

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*
```python
base_dir = "your_file_path/tune_array_results/"
model_dirs = ["model_unet1", "model_unet2", "model_unet3", "model_unetR"] #list of your models trained
```

# Step 7: Evaluation Plots
## Visualize Hyperparameter Tuning Results
Following Step 6.1, use `visualize_tune_results.py` to visualize hyperparameters versus mean dice score for each trained model.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*
```python
base_dir = "your_file_path/tune_array_results/"
model_dirs = ["model_unet1", "model_unet2", "model_unet3", "model_unetR"] #list of your models trained
```

Outputted heatmap plots can be used to determine which combination of batch size, learning rate hyperparameters yield the best mean dice scores.
<!-- ADD IMAGE OF TUNING PLOTS -->
![Visualization of Hyperparameter Tuning.](/results/tune_images/model_unet2_heatmap_dice_bs_lr.png)


## Visualize Image Segmentation Results 
Visualize model inferences on the sample images from the test dataset. Each test image is evaluated individually for the metrics of interest (Dice, Precision, Recall and Hausdorff)

Using `evaluate.sh`, the script will run `evaluate.py` which will generate an images to compare Test Image, Ground Truth Mask and Inference Result outputting the metrics and

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*

`evaluate.sh`
```sh
python your_file_path/evaluate.py 
  --model_path your_file_path/tune_array_results/model_unetR/bs8_lr0.0001_ep300/best_model/best_metric_model.pth #directory to the desired model
  --test_data_path your_file_path/transformed_train_data_pth/test_data_augmented.pth #test data directory
  --output_dir your_file_path/evaluate_model_results_with_augmentations/model_unetR
```

<!-- ADD IMAGE OF IMAGE INFERENCE/MASK OVERLAYS -->
![Image, Ground Truth Masks, and Inference Results.](/results/evaluation_imgs/test_result_2.png)



## Plotting Performance Metrics
Outputs boxplots and swarm plots for each metrics (Dice, Precision, Recall and Hausdorff)

`visualize_evaluation_results.py`

Compares evaluation results between models.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*
```python
base_dir = "your_file_path/output_dir" #output_dir used from evaluate.sh
```
<!-- ADD IMAGE OF between model boxplots -->
![Comparing model performance.](/results/evaluation_metrics/mean_metric_subplots_2x2_unet2_vs_unetR_labeled.png)

---

`visualize_evaluation_results_byAugmentation.py`

Compares evaluation results within a model type, between being trained using augmented and unaugmented training images.

*IMPORTANT PARAMETERS TO CHANGE IN SCRIPT*
```python
base_dirs = {
    "No Augmentations": "your_file_path/evaluate_model_results_no_augmentations/model_unetR/test_metrics.csv",
    "With Augmentations": "your_file_path/evaluate_model_results_with_augmentations/model_unetR/test_metrics.csv"
} #showing example of directories for model_unetR
output_path = "your_file_path/unetR_augmentation_comparison_boxplot.png"
```

<!-- ADD IMAGE OF within model, augmented vs unaugment boxplot -->
![Augmented vs Unaugmented training results.](/results/evaluation_metrics/unet2_augmentation_comparison_boxplot.png)
