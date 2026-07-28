#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 VARIANT RANK [RANK ...] -- DOMAIN [DOMAIN ...]" >&2
  exit 2
fi

variant=$1
shift

views=32
tta_steps=2
selection_count=6
tta_entropy=averaged
ois=true

case $variant in
  views8) views=8 ;;
  views16) views=16 ;;
  steps1) tta_steps=1 ;;
  select2) selection_count=2 ;;
  select4) selection_count=4 ;;
  select8) selection_count=8 ;;
  select12) selection_count=12 ;;
  pointwise) tta_entropy=pointwise ;;
  ois_off) ois=false ;;
  baseline) ;;
  steps3) tta_steps=3 ;;
  *) echo "Unknown P4 variant: $variant" >&2; exit 2 ;;
esac

output_dir=${IAPL_OUTPUT_DIR:?IAPL_OUTPUT_DIR is required}
host_name=$(hostname -s | tr -cd '[:alnum:]_.-')

export IAPL_VIEWS=$views
export IAPL_TTA_STEPS=$tta_steps
export IAPL_SELECTION_COUNT=$selection_count
export IAPL_SELECTION_P=0.2
export IAPL_TTA_ENTROPY=$tta_entropy
export IAPL_OIS=$ois
export IAPL_PROFILE_PATH=${IAPL_PROFILE_PATH:-$output_dir/profile.json}
export IAPL_GPU_MONITOR_PATH=${IAPL_GPU_MONITOR_PATH:-$output_dir/gpu_monitor_${host_name}.csv}

exec "${PROJECT_ROOT:-/home/yabin/ctta4aid-arrow}/scripts/run_iapl_manual_ranks.sh" "$@"
