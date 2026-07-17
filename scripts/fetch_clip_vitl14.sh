#!/usr/bin/env bash
set -euo pipefail

destination=${1:-weights/clip/ViT-L-14.pt}
sha256=b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
url="https://openaipublic.azureedge.net/clip/models/$sha256/ViT-L-14.pt"
mkdir -p "$(dirname "$destination")"

if [[ -f "$destination" ]] && printf '%s  %s\n' "$sha256" "$destination" | sha256sum --check --status; then
  echo "ViT-L-14.pt already verified"
  exit 0
fi

curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
  --output "$destination" "$url"
printf '%s  %s\n' "$sha256" "$destination" | sha256sum --check
