#!/bin/bash
#SBATCH --job-name=orca
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --output=chemprop.out
#SBATCH --error=chemprop.err

module load anaconda/2023a
source activate chemprop_train

python fetch_data_and_train_chemprop.py