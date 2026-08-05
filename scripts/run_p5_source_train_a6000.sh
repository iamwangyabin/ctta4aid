#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/yabin/ctta4aid-arrow}"
UFD_FORENSYNTHS_ARROW_ROOT="${UFD_FORENSYNTHS_ARROW_ROOT:-/home/yabin/ctta4aid_assets/data/df_arrow_20260717/ForenSynths}"
CONFIG="${1:-configs/train/universalfake_progan_resnet50_arrow.yaml}"

if [[ -f /home/home/yabin/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/home/home/yabin/miniconda3
elif [[ -f /home/yabin/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/home/yabin/miniconda3
else
  echo "Cannot locate the server Conda installation" >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate cl

cd "${PROJECT_ROOT}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export UFD_FORENSYNTHS_ARROW_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python train_source.py --config "${CONFIG}"
