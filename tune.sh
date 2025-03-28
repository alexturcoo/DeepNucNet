#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexanderturco1@gmail.com
#SBATCH --job-name=tune_array
#SBATCH --output=tune_%A_%a.out
#SBATCH --error=tune_%A_%a.err
#SBATCH --array=0-59  # Now running 60 jobs, one per line in param_list.txt

module load python/3.11

# Load the correct hyperparams from the param list
PARAM_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" param_list.txt)
IFS=',' read -r BATCH_SIZE LEARNING_RATE EPOCHS <<< "$PARAM_LINE"

# Fixed params
DATASET_PATH="/home/alextu/scratch/DeepNucNet_computecanada/transformed_train_data_pth/train_data_augmented.pth"
OUTPUT_DIR="/home/alextu/scratch/DeepNucNet_computecanada/tune_array_results_with_augmentations/model_unetR/bs${BATCH_SIZE}_lr${LEARNING_RATE}_ep${EPOCHS}"
TRAIN_RATIO=0.8
VAL_INTERVAL=2
NUM_WORKERS=0
PATIENCE=10

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Running config: BS=$BATCH_SIZE LR=$LEARNING_RATE EP=$EPOCHS"

python train.py \
    --dataset_path "$DATASET_PATH" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --val_interval "$VAL_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    --patience "$PATIENCE"

echo "[INFO] Task $SLURM_ARRAY_TASK_ID completed."
