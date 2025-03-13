#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1 #specify number of nodes
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G # MEMORY PER NODE
#SBATCH --time=1:00:00 #hrs:mins:secs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexanderturco1@gmail.com #sends me an email once done
#SBATCH --job-name=train
#SBATCH --output=train.o
#SBATCH --error=train.e

module load python/3.11


# Set default values
DATASET_PATH="/home/alextu/scratch/DeepNucNet/transformed_train_data_pth/train_data_transformed.pth"
BATCH_SIZE=16
LEARNING_RATE=0.0001
NUM_WORKERS=0
EPOCHS=10
TRAIN_RATIO=0.8
VAL_INTERVAL=2
OUTPUT_DIR="/home/alextu/scratch/DeepNucNet/results"

# Display information
echo "[INFO] Starting training script..."
echo "Dataset: $DATASET_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Workers: $NUM_WORKERS"
echo "Epochs: $EPOCHS"
echo "Train Ratio: $TRAIN_RATIO"
echo "Validation Interval: $VAL_INTERVAL"

# Run the training script
python train.py \
    --dataset_path "$DATASET_PATH" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --train_ratio "$TRAIN_RATIO" \
    --val_interval "$VAL_INTERVAL" \
    --output_dir "$OUTPUT_DIR"

echo "[INFO] Training completed."