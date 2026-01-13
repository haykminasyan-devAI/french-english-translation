#!/usr/bin/bash -l
#SBATCH --job-name=dataproc
#SBATCH --partition=scalar6000q
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=./logs/model1_%j.log
#SBATCH --error=./logs/model1_%j.log

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Load modules if needed
# module load cuda/11.8  # Uncomment if needed

# Navigate to project directory
cd /home/hayk.minasyan/Project/translation

# Activate virtual environment
source venv/bin/activate

# Show Python and PyTorch info
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Run training
echo "Starting Model 1 training..."
python train__model_1.py

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
