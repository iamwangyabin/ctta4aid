#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || ! [[ $1 =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 SEED" >&2
  exit 2
fi
seed=$1

project_root=${PROJECT_ROOT:-/home/yabin/ctta4aid-arrow}
iapl_repo=${IAPL_REPO_PATH:-/home/yabin/ctta4aid_assets/external/IAPL}
python=${IAPL_PYTHON:-python}
dataset_path=${IAPL_DATASET_PATH:-}
clip_path=${CLIP_VIT_L14_CHECKPOINT:-/home/yabin/.cache/clip/ViT-L-14.pt}
output_root=${IAPL_TRAIN_OUTPUT_ROOT:-$project_root/outputs/iapl_official/p3_genimage_train}
model_name=${IAPL_MODEL_NAME:-genimage_sd14_seed${seed}}
master_port=${MASTER_PORT:-29711}

if [[ $dataset_path != hf_arrow://* ]]; then
  echo "IAPL_DATASET_PATH must be a verified hf_arrow:// URI" >&2
  exit 1
fi
if ! command -v "$python" >/dev/null 2>&1 && [[ ! -x $python ]]; then
  echo "IAPL Python is not executable: $python" >&2
  exit 1
fi
if [[ ! -d $iapl_repo ]]; then
  echo "IAPL repository is missing: $iapl_repo" >&2
  exit 1
fi
if [[ ! -f $clip_path ]]; then
  echo "CLIP checkpoint is missing: $clip_path" >&2
  exit 1
fi

if [[ ${IAPL_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'python=%s\ndataset_path=%s\niapl_repo=%s\nclip_path=%s\noutput_root=%s\nmodel_name=%s\nseed=%s\n' \
    "$python" "$dataset_path" "$iapl_repo" "$clip_path" "$output_root" \
    "$model_name" "$seed"
  exit 0
fi

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$output_root"
cd "$iapl_repo"

exec "$python" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port "$master_port" \
  main.py \
  --batchsize 32 \
  --evalbatchsize 32 \
  --dataset_path "$dataset_path" \
  --train_selected_subsets SDv14 \
  --test_selected_subsets \
    ADM BigGAN glide Midjourney stable_diffusion_v_1_4 \
    stable_diffusion_v_1_5 VQDM wukong \
  --lr 0.00005 \
  --model_name "$model_name" \
  --dataset GenImage \
  --epoch 1 \
  --lr_drop 10 \
  --gate True \
  --condition True \
  --smooth True \
  --clip_path "$clip_path" \
  --seed "$seed" \
  --output_dir "$output_root"
