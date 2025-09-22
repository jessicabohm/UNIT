#! /bin/bash
# ====================================
#SBATCH --job-name=train_ae
#SBATCH --output=ae_anat_mouse_train_3-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
# ====================================

# Activate Conda and then the environment.
source ~/software/init-conda
conda activate cyclegan

python -u train_update_2.py

