#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ $1 != arrow && $1 != imagefolder ]]; then
  echo "Usage: $0 {arrow|imagefolder}" >&2
  exit 2
fi

backend=$1
project_root=${PROJECT_ROOT:-/home/yabin/ctta4aid-arrow}
python=${IAPL_PYTHON:-/home/yabin/miniconda3/envs/cl/bin/python}
arrow_root=${UFD_ARROW_ROOT:-/home/yabin/ctta4aid_assets/data/df_arrow_20260716}
imagefolder_root=${P0_IMAGEFOLDER_ROOT:-/home/yabin/ctta4aid_assets/data/p0_5domain_imagefolder}
domains=(crn guided imle san seeingdark)

if [[ $backend == arrow ]]; then
  dataset_path="hf_arrow://$arrow_root/ForenSynths|$arrow_root/Ojha"
else
  manifest="$imagefolder_root/export_manifest.json"
  if [[ ! -f $manifest ]]; then
    export_args=()
    for domain in "${domains[@]}"; do
      export_args+=(--domain "$domain")
    done
    PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$python" "$project_root/scripts/export_hf_arrow_imagefolder.py" \
      --root "$arrow_root/ForenSynths" \
      --root "$arrow_root/Ojha" \
      "${export_args[@]}" \
      --output-root "$imagefolder_root"
  fi
  dataset_path=$imagefolder_root
fi

output_dir=${IAPL_OUTPUT_DIR:-$project_root/outputs/iapl_official/p0_single_$backend}
mkdir -p "$output_dir/predictions"

export PROJECT_ROOT="$project_root"
export IAPL_PYTHON="$python"
export IAPL_DATASET_PATH="$dataset_path"
export IAPL_OUTPUT_DIR="$output_dir"
export IAPL_PREDICTION_DIR="$output_dir/predictions"
export NODE_RANK=0
export NNODES=1
export NPROC_PER_NODE=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29640}

exec "$project_root/scripts/run_iapl_multinode_node.sh" "${domains[@]}"
