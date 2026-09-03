#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:05:00

set -euo pipefail

echo "hostname: $(hostname)"
echo "date:     $(date --iso-8601=seconds)"

srun nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader

srun uv run python -c '
import torch

assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"
device = torch.device("cuda")
values = torch.arange(1, 6, device=device, dtype=torch.float32)
result = (values * values).sum().item()
assert result == 55.0, result

print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"gpu_smoke_test_result={result}")
'
