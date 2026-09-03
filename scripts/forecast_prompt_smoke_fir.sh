#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00

set -euo pipefail

echo "hostname: $(hostname)"
echo "date:     $(date --iso-8601=seconds)"
srun nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
srun uv run python scripts/forecast_prompt_smoke.py
