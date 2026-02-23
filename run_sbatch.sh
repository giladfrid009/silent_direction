#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# ---------------------------------------------------------------------------
# Generic SLURM wrapper script.
# Usage (submitted via sbatch):
#   sbatch [sbatch-options] run_sbatch.sh <python_script> [script_args ...]
#
# Example:
#   sbatch --job-name my_job --gres=gpu:A100:1 \
#       run_sbatch.sh scripts/run_norm.py --model meta-llama/Llama-2-7b-chat-hf
# ---------------------------------------------------------------------------

set -euo pipefail

# Create slurm output directory (referenced in #SBATCH --output/--error)
mkdir -p slurm

# Activate virtual environment if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

echo "=== SLURM Job Info ==="
echo "Job ID   : ${SLURM_JOB_ID:-N/A}"
echo "Job Name : ${SLURM_JOB_NAME:-N/A}"
echo "Node     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Working dir: $(pwd)"
echo "Command  : python $*"
echo "======================"

# Run the python script with all forwarded arguments
python "$@"

