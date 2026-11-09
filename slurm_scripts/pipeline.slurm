#!/bin/bash
#SBATCH --job-name=masters
#SBATCH --account=ec35
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=36G
#SBATCH --cpus-per-task=4

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# unload all current modules (just in case) and load only the necessary ones
module purge

# Load python module and source the virtual environment
module load Python/3.10.8-GCCcore-12.2.0
source /cluster/work/projects/ec35/ec-kjetiki-master-thesis/.venv/bin/activate

# print information (optional)
echo "submission directory: ${SUBMITDIR}"


# envs
export WORK_SCRATCH="/cluster/work/projects/ec35/ec-kjetiki-master-thesis/"
export HF_DATASETS_CACHE="$WORK_SCRATCH/hf_datasets"
export HF_HOME="$WORK_SCRATCH/hf_home"
export HUGGINGFACE_HUB_CACHE="$WORK_SCRATCH/hf_hub"
export TRANSFORMERS_CACHE="$WORK_SCRATCH/transformers"

# Should make transfers faster
export HF_HUB_ENABLE_HF_TRANSFER=1

# Make sure to set cores as expected 
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export BLIS_NUM_THREADS=${SLURM_CPUS_PER_TASK}


# Install extra packages
echo "installing paddle ocr, hf_transfer and rapidfuzz..."
pip install --no-cache-dir -q --progress-bar off paddlepaddle paddleocr rapidfuzz hf_transfer >/dev/null 2>&1

# by default, pass on any remaining command-line options
python3 -m src.run_pipeline ${@}