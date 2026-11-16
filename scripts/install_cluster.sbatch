#!/bin/bash -l
# Build the training environment on the anvme workspace: a venv with this repo installed (editable) and the
# Cosmos3-Edge base checkpoint. Needs no GPU, only the proxy: `bash scripts/install_cluster.sbatch` on a login
# node works too.
#   sbatch scripts/install_cluster.sbatch
#SBATCH --job-name=cosmos3-install
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=cosmos3-install-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PATH=$HOME/.local/bin:$PATH

REPO=${REPO:-$HOME/Projects/cosmos3-wam}
WS=${WS:-$(ws_find univtac)}
ROOT=${COSMOS3_ROOT:-$WS/cosmos3}
VENV=${VENV:-$ROOT/venv-wam}
export HF_HOME=${HF_HOME:-$WS/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WS/uv-cache} UV_LINK_MODE=copy

mkdir -p "$ROOT/ckpt" "$ROOT/outputs"
[ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.12
VIRTUAL_ENV=$VENV uv pip install -q -e "$REPO" --torch-backend=cu128
(cd /tmp && "$VENV/bin/python" -c "
import torch, lerobot
from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3EdgePolicy
print('torch', torch.__version__, 'lerobot package', lerobot.__file__)")

# Base checkpoint (about 9 GB, bfloat16): weights and tokenizer. On NVMe: mmap reads from $WORK (NFS) can
# stall at a few MB/s.
[ -f "$ROOT/ckpt/Cosmos3-Edge-hub/model.safetensors" ] || \
    "$VENV/bin/hf" download lerobot/Cosmos3-Edge --local-dir "$ROOT/ckpt/Cosmos3-Edge-hub"
[ -f "$ROOT/ckpt/univtac_resnet18_20251128-125750.pth" ] || \
    echo "[install] copy the UniVTAC tactile encoder (encoder/checkpoints/resnet18/20251128-125750/best.pth)" \
         "to $ROOT/ckpt/univtac_resnet18_20251128-125750.pth"
echo "INSTALL DONE"
