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

module load cuda/11.8
module load anaconda/2023a
conda activate chemprop_train

python test_cuda.py
