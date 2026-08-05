#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <single_target|continual> <0|1|2>" >&2
  exit 2
fi

SETTING="$1"
SEED="$2"
case "${SETTING}" in
  single_target)
    ENTRYPOINT=run_single_target.py
    CONFIG="configs/p5/universalfake_arrow_single_target_seed${SEED}.yaml"
    ;;
  continual)
    ENTRYPOINT=run_continual_stream.py
    CONFIG="configs/p5/universalfake_arrow_continual_seed${SEED}.yaml"
    ;;
  *)
    echo "unknown setting: ${SETTING}" >&2
    exit 2
    ;;
esac
case "${SEED}" in
  0|1|2) ;;
  *)
    echo "seed must be 0, 1, or 2" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/yabin/ctta4aid-arrow}"
UFD_SOURCE_CHECKPOINT="${UFD_SOURCE_CHECKPOINT:-/home/yabin/ctta4aid_assets/weights/p5/ufd_progan_resnet50_source_seed0.pt}"

if [[ -f /home/home/yabin/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/home/home/yabin/miniconda3
elif [[ -f /home/yabin/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/home/yabin/miniconda3
else
  echo "Cannot locate the server Conda installation" >&2
  exit 1
fi

if [[ -z "${UFD_FORENSYNTHS_ARROW_ROOT:-}" ]]; then
  for root in \
    /home/yabin/ctta4aid_assets/data/df_arrow_20260717 \
    /home/yabin/ctta4aid_assets/data/df_arrow_20260716 \
    /home/yabin/ctta4aid_assets/data/df_arrow_remote_3070; do
    if [[ -f "${root}/ForenSynths/state.json" && -f "${root}/Ojha/state.json" ]]; then
      UFD_FORENSYNTHS_ARROW_ROOT="${root}/ForenSynths"
      UFD_OJHA_ARROW_ROOT="${root}/Ojha"
      break
    fi
  done
fi

if [[ -z "${UFD_FORENSYNTHS_ARROW_ROOT:-}" || -z "${UFD_OJHA_ARROW_ROOT:-}" ]]; then
  echo "Cannot locate both UFD Arrow roots" >&2
  exit 1
fi
if [[ ! -r "${UFD_SOURCE_CHECKPOINT}" ]]; then
  echo "Cannot read source checkpoint: ${UFD_SOURCE_CHECKPOINT}" >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate cl

cd "${PROJECT_ROOT}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export UFD_SOURCE_CHECKPOINT
export UFD_FORENSYNTHS_ARROW_ROOT
export UFD_OJHA_ARROW_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -d /home/yabin/ctta4aid_assets/driver_compat/nvidia-580.159.03/rootfs/usr/lib/x86_64-linux-gnu ]]; then
  export LD_LIBRARY_PATH="/home/yabin/ctta4aid_assets/driver_compat/nvidia-580.159.03/rootfs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec python "${ENTRYPOINT}" --config "${CONFIG}"
