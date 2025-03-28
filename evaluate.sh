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
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate.o
#SBATCH --error=evaluate.e

module load python/3.11
module load scipy-stack

python /home/alextu/scratch/DeepNucNet/evaluate.py \
  --model_path /home/alextu/scratch/DeepNucNet_computecanada/tune_array_results_with_augmentations/model_unetR/bs8_lr0.0001_ep300/best_model/best_metric_model.pth \
  --test_data_path /home/alextu/scratch/DeepNucNet_computecanada/transformed_train_data_pth/test_data_augmented.pth \
  --output_dir /home/alextu/scratch/DeepNucNet_computecanada/evaluate_model_results_with_augmentations/model_unetR \
  --batch_size 1
