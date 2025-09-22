#! /bin/bash
# ====================================
#SBATCH --job-name=cont_full_unit_train
#SBATCH --output=unit_anat_train_1-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
# ====================================

# Activate Conda and then the environment.
source ~/software/init-conda
conda activate cyclegan

python train_3d.py

