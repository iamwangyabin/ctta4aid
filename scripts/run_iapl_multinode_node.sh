#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "Usage: NODE_RANK=... $0 DOMAIN [DOMAIN ...]" >&2
  exit 2
fi

project_root=${PROJECT_ROOT:-/home/yabin/ctta4aid-arrow}
iapl_repo=${IAPL_REPO_PATH:-/home/yabin/ctta4aid_assets/external/IAPL}
python=${IAPL_PYTHON:-/home/yabin/miniconda3/envs/cl/bin/python}
dataset_path=${IAPL_DATASET_PATH:-hf_arrow:///home/yabin/ctta4aid_assets/data/df_arrow_20260716/ForenSynths\|/home/yabin/ctta4aid_assets/data/df_arrow_20260716/Ojha}
pretrained_model=${IAPL_UFD_CHECKPOINT:-/home/yabin/ctta4aid_assets/weights/iapl/checkpoint_best_acc_progan.pth}
clip_path=${CLIP_VIT_L14_CHECKPOINT:-/home/yabin/.cache/clip/ViT-L-14.pt}
output_dir=${IAPL_OUTPUT_DIR:-$project_root/outputs/iapl_official/multinode}

nnodes=${NNODES:-2}
node_rank=${NODE_RANK:?NODE_RANK is required}
nproc_per_node=${NPROC_PER_NODE:-1}
master_addr=${MASTER_ADDR:-192.168.10.188}
master_port=${MASTER_PORT:-29620}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output_dir"
cd "$iapl_repo"

exec "$python" -m torch.distributed.run \
  --nnodes "$nnodes" \
  --node_rank "$node_rank" \
  --nproc_per_node "$nproc_per_node" \
  --master_addr "$master_addr" \
  --master_port "$master_port" \
  main.py \
  --batchsize 32 \
  --evalbatchsize 32 \
  --dataset_path "$dataset_path" \
  --train_selected_subsets car cat chair horse \
  --test_selected_subsets "$@" \
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
  --eval
