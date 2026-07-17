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
python=${IAPL_PYTHON:-/home/yabin/venvs/iapl-cu121/bin/python}
dataset_path=${IAPL_DATASET_PATH:-hf_arrow:///home/yabin/ctta4aid_assets/data/df_arrow_20260717/ForenSynths\|/home/yabin/ctta4aid_assets/data/df_arrow_20260717/Ojha}
pretrained_model=${IAPL_UFD_CHECKPOINT:-/home/yabin/ctta4aid_assets/weights/iapl/checkpoint_best_acc_progan.pth}
clip_path=${CLIP_VIT_L14_CHECKPOINT:-/home/yabin/.cache/clip/ViT-L-14.pt}
output_dir=${IAPL_OUTPUT_DIR:-$project_root/outputs/iapl_official/manual_ranks}
master_addr=${MASTER_ADDR:?MASTER_ADDR is required}
master_port=${MASTER_PORT:-29621}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# NCCL 2.30+ can intentionally place multiple communicator ranks on one GPU.
# This keeps the released eight-process sampler/seed semantics on three GPUs.
export NCCL_MULTI_RANK_GPU_ENABLE=${NCCL_MULTI_RANK_GPU_ENABLE:-1}
export NCCL_MAX_CTAS=${NCCL_MAX_CTAS:-2}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
mkdir -p "$output_dir"
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
      --train_selected_subsets car cat chair horse \
      --test_selected_subsets "${domains[@]}" \
      --lr 0.005 \
      --model_name tta \
      --dataset UniversalFakeDetect \
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
      --num_workers 0 \
      --seed 100 \
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
