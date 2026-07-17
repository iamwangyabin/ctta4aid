#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/home/yabin/projects/ctta4aid-arrow}
python=${IAPL_PYTHON:-/home/yabin/miniconda3/envs/iapl/bin/python}
remote_output=/home/yabin/ctta4aid-arrow/outputs/iapl_official/universalfake_arrow_1gpu
merge_root="$project_root/outputs/iapl_official/universalfake_arrow_dual4090"
log="$merge_root/merge_campaign.log"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

host_1=${IAPL_HOST_1:-yabin@192.168.10.188}
host_2=${IAPL_HOST_2:-yabin@192.168.10.187}
domains_1=(crn gaugan progan deepfake biggan dalle san seeingdark glide_50_27)
domains_2=(cyclegan imle stylegan stargan guided glide_100_10 glide_100_27 ldm_100 ldm_200 ldm_200_cfg)

mkdir -p "$merge_root/4090-1" "$merge_root/4090-2"
exec 9>"$merge_root/merge_campaign.lock"
if ! flock -n 9; then
  echo "A dual-4090 merge coordinator is already running." >&2
  exit 1
fi

wait_for_metrics() {
  local host=$1
  shift
  local domains=("$@")
  while true; do
    local missing=0
    for domain in "${domains[@]}"; do
      if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
        "test -s '$remote_output/shards/$domain/official_iapl_metrics.json'"; then
        missing=$((missing + 1))
      fi
    done
    if (( missing == 0 )); then
      return
    fi
    printf '%s waiting host=%s missing_domains=%s\n' \
      "$(date -Is)" "$host" "$missing" | tee -a "$log"
    sleep 600
  done
}

wait_for_metrics "$host_1" "${domains_1[@]}"
wait_for_metrics "$host_2" "${domains_2[@]}"

metric_files=()
for domain in "${domains_1[@]}"; do
  destination="$merge_root/4090-1/$domain.json"
  rsync -a "$host_1:$remote_output/shards/$domain/official_iapl_metrics.json" "$destination"
  metric_files+=("$destination")
done
for domain in "${domains_2[@]}"; do
  destination="$merge_root/4090-2/$domain.json"
  rsync -a "$host_2:$remote_output/shards/$domain/official_iapl_metrics.json" "$destination"
  metric_files+=("$destination")
done

set +e
cd "$project_root"
"$python" merge_iapl_shards.py \
  --config configs/iapl_official_ufd_arrow_1gpu.yaml \
  --metrics "${metric_files[@]}" \
  --output "$merge_root/official_iapl_metrics_merged.json" \
  >>"$log" 2>&1
merge_status=$?
set -e

rsync -a "$merge_root/official_iapl_metrics_merged.json" \
  "$host_1:$remote_output/official_iapl_metrics_dual4090.json"
rsync -a "$merge_root/official_iapl_metrics_merged.json" \
  "$host_2:$remote_output/official_iapl_metrics_dual4090.json"
printf '%s merge complete status=%s\n' "$(date -Is)" "$merge_status" | tee -a "$log"
exit "$merge_status"
