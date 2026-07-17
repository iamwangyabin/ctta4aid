#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 TARGET_HOST DESTINATION_ROOT" >&2
  exit 2
fi

target_host=$1
destination_root=$2
source_root=${UFD_ARROW_SOURCE_ROOT:-/data/DF-arrow-data}

ssh -o BatchMode=yes -o ConnectTimeout=10 "$target_host" \
  "mkdir -p '$destination_root'"
ionice -c 3 nice -n 10 rsync \
  -a --partial --append-verify --info=progress2 \
  "$source_root/ForenSynths" \
  "$source_root/Ojha" \
  "$target_host:$destination_root/"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$target_host" \
  "touch '$destination_root/.transfer_complete'"
