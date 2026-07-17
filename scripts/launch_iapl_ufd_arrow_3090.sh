#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/home/yabin/projects/ctta4aid-arrow}
python=${IAPL_PYTHON:-/home/yabin/miniconda3/envs/iapl/bin/python}
config=${IAPL_CONFIG:-configs/iapl_official_ufd_arrow_1gpu.yaml}
output_root=${IAPL_OUTPUT_ROOT:-$project_root/outputs/iapl_official/universalfake_arrow_1gpu}
campaign_log="$output_root/campaign.log"

export IAPL_REPO_PATH=${IAPL_REPO_PATH:-/home/yabin/projects/IAPL-GenImage}
export UFD_FORENSYNTHS_ARROW_ROOT=${UFD_FORENSYNTHS_ARROW_ROOT:-/data/DF-arrow-data/ForenSynths}
export UFD_OJHA_ARROW_ROOT=${UFD_OJHA_ARROW_ROOT:-/data/DF-arrow-data/Ojha}
export IAPL_UFD_CHECKPOINT=${IAPL_UFD_CHECKPOINT:-/home/yabin/projects/IAPL-GenImage/pretrained/checkpoint_best_acc_progan.pth}
export CLIP_VIT_L14_CHECKPOINT=${CLIP_VIT_L14_CHECKPOINT:-/home/yabin/.cache/clip/ViT-L-14.pt}
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

if (( $# )); then
  domains=("$@")
else
  domains=(
    crn cyclegan dalle biggan deepfake gaugan
    glide_50_27 glide_100_10 glide_100_27 guided imle
    ldm_100 ldm_200 ldm_200_cfg progan san seeingdark stargan stylegan
  )
fi

wait_for_path=${WAIT_FOR_PATH:-}
wait_for_tmux_session=${WAIT_FOR_TMUX_SESSION-genimage_fsd_campaign_watchdog}

mkdir -p "$output_root"
exec 9>"$output_root/campaign.lock"
if ! flock -n 9; then
  echo "An IAPL UFD Arrow campaign is already running." >&2
  exit 1
fi

cd "$project_root"
while [[ -n "$wait_for_path" && ! -f "$wait_for_path" ]]; do
  printf '%s waiting for data sentinel=%s\n' \
    "$(date -Is)" "$wait_for_path" | tee -a "$campaign_log"
  sleep 300
done
if [[ "${CHECK_ARROW_DATA:-0}" == 1 ]]; then
  "$python" scripts/check_hf_arrow_datasets.py ufd \
    "$UFD_FORENSYNTHS_ARROW_ROOT" "$UFD_OJHA_ARROW_ROOT" \
    --output "$output_root/ufd_arrow_data_check.json" \
    >>"$campaign_log" 2>&1
fi
for domain in "${domains[@]}"; do
  metrics="$output_root/shards/$domain/official_iapl_metrics.json"
  if [[ -s "$metrics" ]]; then
    printf '%s skip completed domain=%s\n' "$(date -Is)" "$domain" | tee -a "$campaign_log"
    continue
  fi

  if [[ -n "$wait_for_tmux_session" ]]; then
    while tmux has-session -t "$wait_for_tmux_session" 2>/dev/null; do
      printf '%s waiting for tmux=%s domain=%s\n' \
        "$(date -Is)" "$wait_for_tmux_session" "$domain" | tee -a "$campaign_log"
      sleep 300
    done
  fi
  while true; do
    used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    if (( used_memory < 4096 )); then
      break
    fi
    printf '%s waiting for GPU memory domain=%s used_mib=%s\n' \
      "$(date -Is)" "$domain" "$used_memory" | tee -a "$campaign_log"
    sleep 300
  done

  printf '%s start domain=%s\n' "$(date -Is)" "$domain" | tee -a "$campaign_log"
  "$python" run_iapl_official.py --config "$config" --domains "$domain" \
    >>"$campaign_log" 2>&1
  printf '%s complete domain=%s\n' "$(date -Is)" "$domain" | tee -a "$campaign_log"
done

merge_shards=${MERGE_SHARDS:-auto}
if [[ "$merge_shards" == auto ]]; then
  if (( ${#domains[@]} == 19 )); then
    merge_shards=1
  else
    merge_shards=0
  fi
fi
if [[ "$merge_shards" == 1 ]]; then
  metric_files=()
  for domain in "${domains[@]}"; do
    metric_files+=("$output_root/shards/$domain/official_iapl_metrics.json")
  done
  "$python" merge_iapl_shards.py \
    --config "$config" \
    --metrics "${metric_files[@]}" \
    --output "$output_root/official_iapl_metrics_merged.json" \
    >>"$campaign_log" 2>&1
fi
printf '%s campaign complete domains=%s merged=%s\n' \
  "$(date -Is)" "${#domains[@]}" "$merge_shards" | tee -a "$campaign_log"
