#!/bin/bash
#SBATCH --job-name=masters
#SBATCH --account=ec35
#SBATCH --time=00:50:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpu-2
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=12

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# unload all current modules (just in case) and load only the necessary ones
module purge

# Load python module and source the virtual environment
module load Python/3.10.8-GCCcore-12.2.0
source /cluster/work/projects/ec35/ec-kjetiki-master-thesis/.venv/bin/activate
pip install --no-cache-dir -q --progress-bar off wandb hf_xet >/dev/null 2>&1

# print information (optional)
echo "submission directory: ${SUBMITDIR}"

# print gpu info
echo -e "\n"
nvidia-smi || echo "[warn] nvidia-smi not available (skipping)"
echo -e "\n"

# by default, pass on any remaining command-line options
python3 -m src.scripts.evaluation.eval ${@}