#!/usr/bin/env bash
set -euo pipefail

destination=${1:-weights/iapl}
profile=${2:-all}
base_url=https://modelscope.cn/models/yihengli/IAPL_pretrain/resolve/master
mkdir -p "$destination"

download() {
  local filename=$1
  local expected_sha256=$2
  local path="$destination/$filename"

  if [[ -f "$path" ]] && printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum --check --status; then
    echo "$filename already verified"
    return
  fi

  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$path" "$base_url/$filename"
  printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum --check
}

case "$profile" in
  all)
    download checkpoint_best_acc_sd14.pth \
      4f86a1e3e93bfa8c0cf1099413f872dccd34b6ec3f26dec2162bcf2430552018
    download checkpoint_best_acc_progan.pth \
      1e04047b74d287ba2f3682cde84246688dfa486354a5677b5147e677bc2a3f81
    ;;
  genimage)
    download checkpoint_best_acc_sd14.pth \
      4f86a1e3e93bfa8c0cf1099413f872dccd34b6ec3f26dec2162bcf2430552018
    ;;
  progan)
    download checkpoint_best_acc_progan.pth \
      1e04047b74d287ba2f3682cde84246688dfa486354a5677b5147e677bc2a3f81
    ;;
  *)
    echo "unknown profile: $profile" >&2
    exit 2
    ;;
esac
