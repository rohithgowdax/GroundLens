#!/bin/bash
#SBATCH --job-name=masters
#SBATCH --account=ec35
#SBATCH --time=00:40:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=gpu-4
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=10

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

# by default, pass on any remaining command-line options
python3 -m src.scripts.evaluation.eval_answer_mask_pix2struct ${@}