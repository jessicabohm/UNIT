#! /bin/bash
# ====================================
#SBATCH --job-name=train_ae
#SBATCH --output=train_vae_anat_human_64_2-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=124GB
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=crc
# ====================================

# Activate Conda and then the environment.
source ~/software/init-conda
conda activate cyclegan

# Print GPU usage every 10 min in the background
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 600 &
GPU_MONITOR_PID=$!

python -u train_VAE.py

# Kill background GPU monitor when job finishes
kill $GPU_MONITOR_PID

