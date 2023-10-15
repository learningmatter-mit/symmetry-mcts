#!/bin/bash
#SBATCH -J automation
#SBATCH -o MCTS_grid_search-%j.out
#SBATCH -t 2-00:00:00
#SBATCH --mem=50gb
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-8

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
cat $0
echo ""

source /etc/profile
module load anaconda/2021a
source activate chemprop

python driver.py --sweep_step $SLURM_ARRAY_TASK_ID