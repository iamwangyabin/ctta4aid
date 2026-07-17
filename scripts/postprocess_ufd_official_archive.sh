#!/usr/bin/env bash
set -euo pipefail

archive_root=${1:?Usage: $0 ARCHIVE_ROOT}
archive=${UFD_OFFICIAL_ARCHIVE:-$archive_root/CNN_synth_testset.zip}
expected_size=${UFD_OFFICIAL_ARCHIVE_SIZE:-20052866587}
download_pid_file=${UFD_OFFICIAL_DOWNLOAD_PID_FILE:-$archive_root/download.pid}
extract_root=${UFD_OFFICIAL_EXTRACT_ROOT:-$archive_root/extracted}

if [[ -f $download_pid_file ]]; then
  download_pid=$(<"$download_pid_file")
  while kill -0 "$download_pid" 2>/dev/null; do
    echo "Waiting for download process $download_pid at $(date --iso-8601=seconds)"
    sleep 30
  done
fi

if [[ ! -f $archive ]]; then
  echo "Official archive is missing: $archive" >&2
  exit 1
fi
actual_size=$(stat -c %s "$archive")
if [[ $actual_size != "$expected_size" ]]; then
  echo "Archive size mismatch: expected $expected_size, got $actual_size" >&2
  exit 1
fi

low_io=()
if command -v ionice >/dev/null 2>&1; then
  low_io=(ionice -c3)
fi

echo "Computing SHA-256 at $(date --iso-8601=seconds)"
"${low_io[@]}" nice -n 19 sha256sum "$archive" | tee "$archive.sha256"

echo "Testing ZIP structure at $(date --iso-8601=seconds)"
"${low_io[@]}" nice -n 19 unzip -tq "$archive"

mkdir -p "$extract_root"
echo "Extracting to $extract_root at $(date --iso-8601=seconds)"
"${low_io[@]}" nice -n 19 unzip -q -o "$archive" -d "$extract_root"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "archive_size=$actual_size"
  echo "archive_sha256=$(cut -d ' ' -f 1 "$archive.sha256")"
  echo "extract_root=$extract_root"
} >"$archive_root/postprocess_complete.txt"
echo "Official archive post-processing completed"
