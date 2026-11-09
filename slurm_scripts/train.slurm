#!/bin/bash
#SBATCH --job-name=masters_ddp
#SBATCH --account=nn8104k
#SBATCH --time=2-05:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=240G 

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# Export paths
echo "Setting up project area path"
export PROJECT_AREA_PATH="/cluster/work/projects/nn8104k/kjetil_thesis"
export HF_HUB_VERBOSITY=error
export HF_HOME="$PROJECT_AREA_PATH/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NRIS Setup
echo "Setting up NRIS"
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
module --force purge
module load NRIS/GPU
module load hpc-container-wrapper

# Check python version from the container
echo "Sanity: GPU from this python"
python - <<'PY'
import torch, platform
print("arch:", platform.machine())
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "built_with_cuda:", bool(torch.version.cuda))
print("cuda.is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
if torch.cuda.is_available(): print("device0:", torch.cuda.get_device_name(0))
PY

# Log gpu info
echo -e "\n"
nvidia-smi || echo "[warn] nvidia-smi not available (skipping)"
echo -e "\n"

# Run with torchrun for distributed training (single-node)
echo "Launching distributed training with 4 GPUs"
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m src.scripts.training.train "$@"
