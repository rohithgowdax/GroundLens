#!/bin/bash
#SBATCH --job-name=masters
#SBATCH --account=nn8104k
#SBATCH --time=0-12:00:00
#SBATCH --partition=accel 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=60G
#SBATCH --gpus-per-node=1 

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# Export paths
echo "Setting up project area path"
export PROJECT_AREA_PATH="/cluster/work/projects/nn8104k/kjetil_thesis"
export HF_HOME="$PROJECT_AREA_PATH/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# NRIS Setup
echo "Setting up NRIS"
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
module --force purge
module load NRIS/GPU
module load hpc-container-wrapper

# print gpu info
echo -e "\n"
nvidia-smi || echo "[warn] nvidia-smi not available (skipping)"
echo -e "\n"

# Run
python -m src.scripts.evaluation.eval "$@"