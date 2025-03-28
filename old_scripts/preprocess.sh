#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1 #specify number of nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G # MEMORY PER NODE
#SBATCH --time=2:00:00 #hrs:mins:secs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexanderturco1@gmail.com #sends me an email once done
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.o
#SBATCH --error=preprocess.e

module load python/3.11

python /home/alextu/scratch/DeepNucNet/preprocess.py