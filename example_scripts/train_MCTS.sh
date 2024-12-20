#!/bin/bash
#SBATCH --job-name=train_mcts
#SBATCH --partition=xeon-p8
#SBATCH -o train_mcts-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

module load anaconda/2023a
source activate chemprop_train

# export TMPDIR=/state/partition1/user/$USER

output_dir="$1"
iter="$2"
environment="$3"
python train_mcts.py --output_dir "${output_dir}" --iter "${iter}" --environment "${environment}"
