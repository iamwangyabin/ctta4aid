#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 {genimage|progan} SEED [SEED ...]" >&2
  exit 2
fi

track=$1
shift
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
log_root=${RUN_LOG_ROOT:-$project_root/run_logs/caidbench/$track}
mkdir -p "$log_root"

case "$track" in
  genimage)
    : "${CAID_GENIMAGE_SOURCE_CHECKPOINT:?set CAID_GENIMAGE_SOURCE_CHECKPOINT}"
    ;;
  progan)
    : "${CAID_PROGAN_SOURCE_CHECKPOINT:?set CAID_PROGAN_SOURCE_CHECKPOINT}"
    ;;
  *)
    echo "unknown track: $track" >&2
    exit 2
    ;;
esac
: "${CAIDBENCH_ROOT:?set CAIDBENCH_ROOT}"

cd "$project_root"
for seed in "$@"; do
  single_config="configs/caidbench_${track}_single_target_seed${seed}.yaml"
  continual_config="configs/caidbench_${track}_continual_seed${seed}.yaml"
  if [[ ! -f "$single_config" || ! -f "$continual_config" ]]; then
    echo "missing frozen configs for seed $seed" >&2
    exit 2
  fi

  "$python_bin" -u run_single_target.py --config "$single_config" \
    2>&1 | tee "$log_root/seed${seed}_single_target.log"
  "$python_bin" -u run_continual_stream.py --config "$continual_config" \
    2>&1 | tee "$log_root/seed${seed}_continual.log"
done
