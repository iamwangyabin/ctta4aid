#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: WORLD_SIZE=... $0 RANK [RANK ...] -- DOMAIN [DOMAIN ...]" >&2
  exit 2
}

ranks=()
while (( $# > 0 )) && [[ $1 != -- ]]; do
  ranks+=("$1")
  shift
done
(( ${#ranks[@]} > 0 )) || usage
(( $# > 0 )) || usage
shift
domains=("$@")
(( ${#domains[@]} > 0 )) || usage

world_size=${WORLD_SIZE:?WORLD_SIZE is required}
project_root=${PROJECT_ROOT:-/home/yabin/ctta4aid-arrow}
iapl_repo=${IAPL_REPO_PATH:-/home/yabin/ctta4aid_assets/external/IAPL}
python=${IAPL_PYTHON:-/home/yabin/miniconda3/envs/caid-gemini-compat/bin/python}
dataset_path=${IAPL_DATASET_PATH:-${IAPL_GENIMAGE_ROOT:-/home/yabin/ctta4aid_assets/data/genimage_official_20260718/GenImage}}
pretrained_model=${IAPL_GENIMAGE_CHECKPOINT:-/home/yabin/ctta4aid_assets/weights/iapl/checkpoint_best_acc_sd14.pth}
clip_path=${CLIP_VIT_L14_CHECKPOINT:-/home/yabin/.cache/clip/ViT-L-14.pt}
output_dir=${IAPL_OUTPUT_DIR:-$project_root/outputs/iapl_official/genimage_manual_ranks}
prediction_dir=${IAPL_PREDICTION_DIR:-$output_dir/predictions}
master_addr=${MASTER_ADDR:?MASTER_ADDR is required}
master_port=${MASTER_PORT:-29631}
nccl_lib_dir=${IAPL_NCCL_LIB_DIR:-}
nvidia_compat_lib_dir=${IAPL_NVIDIA_COMPAT_LIB_DIR:-}
distributed_timeout_seconds=${IAPL_DISTRIBUTED_TIMEOUT_SECONDS:-7200}
seed=${IAPL_SEED:-100}
num_workers=${IAPL_NUM_WORKERS:-}

if [[ ! -x $python ]]; then
  echo "IAPL Python is not executable: $python" >&2
  exit 1
fi
if [[ ! -d $iapl_repo ]]; then
  echo "IAPL repository is missing: $iapl_repo" >&2
  exit 1
fi
if [[ $dataset_path == hf_arrow://* ]]; then
  arrow_roots=${dataset_path#hf_arrow://}
  IFS='|' read -r -a arrow_root_list <<<"$arrow_roots"
  if (( ${#arrow_root_list[@]} < 1 || ${#arrow_root_list[@]} > 2 )); then
    echo "IAPL_DATASET_PATH must contain one or two Arrow roots" >&2
    exit 1
  fi
  for arrow_root in "${arrow_root_list[@]}"; do
    if [[ ! -f $arrow_root/state.json ]]; then
      echo "GenImage Arrow state is missing: $arrow_root/state.json" >&2
      exit 1
    fi
  done
  dataset_format=hf_arrow
  num_workers=${num_workers:-0}
else
  if [[ ! -d $dataset_path/test ]]; then
    echo "Official GenImage test root is missing: $dataset_path/test" >&2
    exit 1
  fi
  if [[ ! -f $dataset_path/extract_manifest.json ]]; then
    echo "Official GenImage extraction manifest is missing: $dataset_path/extract_manifest.json" >&2
    exit 1
  fi
  dataset_format=imagefolder
  num_workers=${num_workers:-8}
fi
if [[ ! -f $pretrained_model ]]; then
  echo "IAPL checkpoint is missing: $pretrained_model" >&2
  exit 1
fi
if [[ ! -f $clip_path ]]; then
  echo "CLIP checkpoint is missing: $clip_path" >&2
  exit 1
fi
if [[ -n $nvidia_compat_lib_dir ]] && {
  [[ ! -f $nvidia_compat_lib_dir/libcuda.so.1 ]] ||
    [[ ! -f $nvidia_compat_lib_dir/libnvidia-ml.so.1 ]]
}; then
  echo "NVIDIA compatibility libraries are incomplete: $nvidia_compat_lib_dir" >&2
  exit 1
fi

if [[ -z $nccl_lib_dir ]]; then
  nccl_lib_dir=$("$python" -c '
import pathlib
import site

for root in site.getsitepackages():
    candidate = pathlib.Path(root) / "nvidia" / "nccl" / "lib"
    if (candidate / "libnccl.so.2").is_file():
        print(candidate)
        break
' 2>/dev/null || true)
fi
if (( ${#ranks[@]} > 1 )); then
  if [[ ! -f $nccl_lib_dir/libnccl.so.2 ]]; then
    echo "Multiple ranks share one GPU, but NCCL 2.30+ was not found" >&2
    exit 1
  fi
  nccl_version=$("$python" -c '
import ctypes
import sys

library = ctypes.CDLL(sys.argv[1])
version = ctypes.c_int()
if library.ncclGetVersion(ctypes.byref(version)) != 0:
    raise SystemExit("ncclGetVersion failed")
print(version.value)
' "$nccl_lib_dir/libnccl.so.2")
  if (( nccl_version < 23000 )); then
    echo "NCCL $nccl_version is too old for multiple ranks on one GPU" >&2
    exit 1
  fi
  echo "Using NCCL $nccl_version from $nccl_lib_dir"
fi

if [[ ${IAPL_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'python=%s\ndataset_path=%s\ndataset_format=%s\nnum_workers=%s\niapl_repo=%s\npretrained_model=%s\nclip_path=%s\nnvidia_compat_lib_dir=%s\ndistributed_timeout_seconds=%s\nseed=%s\n' \
    "$python" "$dataset_path" "$dataset_format" "$num_workers" "$iapl_repo" \
    "$pretrained_model" "$clip_path" "$nvidia_compat_lib_dir" \
    "$distributed_timeout_seconds" "$seed"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ -n $nvidia_compat_lib_dir ]]; then
  export LD_LIBRARY_PATH="$nvidia_compat_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -n $nccl_lib_dir ]]; then
  export LD_LIBRARY_PATH="$nccl_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export NCCL_MULTI_RANK_GPU_ENABLE=${NCCL_MULTI_RANK_GPU_ENABLE:-1}
export NCCL_MAX_CTAS=${NCCL_MAX_CTAS:-2}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export IAPL_DISTRIBUTED_TIMEOUT_SECONDS="$distributed_timeout_seconds"
export IAPL_PREDICTION_DIR="$prediction_dir"
mkdir -p "$output_dir" "$prediction_dir"
cd "$iapl_repo"

pids=()
terminate_children() {
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_children INT TERM EXIT

for rank in "${ranks[@]}"; do
  if ! [[ $rank =~ ^[0-9]+$ ]] || (( rank >= world_size )); then
    echo "Invalid rank $rank for WORLD_SIZE=$world_size" >&2
    exit 2
  fi
  log="$output_dir/rank${rank}.log"
  env \
    WORLD_SIZE="$world_size" \
    RANK="$rank" \
    LOCAL_RANK=0 \
    MASTER_ADDR="$master_addr" \
    MASTER_PORT="$master_port" \
    "$python" main.py \
      --batchsize 32 \
      --evalbatchsize 32 \
      --dataset_path "$dataset_path" \
      --train_selected_subsets SDv14 \
      --test_selected_subsets "${domains[@]}" \
      --lr 0.005 \
      --model_name tta \
      --dataset GenImage \
      --epoch 1 \
      --lr_drop 10 \
      --gate True \
      --condition True \
      --pretrained_model "$pretrained_model" \
      --clip_path "$clip_path" \
      --tta True \
      --tta_steps 2 \
      --selection_p 0.2 \
      --ois True \
      --smooth True \
      --num_workers "$num_workers" \
      --seed "$seed" \
      --output_dir "$output_dir" \
      --eval \
      >"$log" 2>&1 &
  pids+=("$!")
  echo "$!" >"$output_dir/rank${rank}.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
trap - INT TERM EXIT
exit "$status"
