#!/usr/bin/bash -l
#SBATCH --job-name=dataproc
#SBATCH --partition=scalar6000q
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=./logs/train-%j.log
#SBATCH --error=./logs/train-%j.err

#print GPU info
echo "Job started on $(hostname) at $(data)"
nvidia-smi

#Activate virtul environment 

source ~/Project/translation/venv/bin/activate

cd ~/Project/translation

python train.py

echo "Job finished at $(date)"

