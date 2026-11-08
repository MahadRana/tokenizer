#!/usr/bin/env bash
# Download cl100k_base.tiktoken into ~/.cache. The file is not committed.
# Tests that need it call curl themselves if this has not been run.
set -euo pipefail

url="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
dest="${HOME:?HOME is not set}/.cache/tokenizer/cl100k_base.tiktoken"

mkdir -p "$(dirname "$dest")"

if [[ -f "$dest" ]]; then
  got="$(sha256sum "$dest" | awk '{print $1}')"
  if [[ "$got" == "$hash" ]]; then
    echo "cached $dest"
    exit 0
  fi
  echo "hash mismatch at $dest, downloading again" >&2
fi

tmp="${dest}.partial"
curl -fsSL --retry 3 -o "$tmp" "$url"
got="$(sha256sum "$tmp" | awk '{print $1}')"
if [[ "$got" != "$hash" ]]; then
  echo "downloaded hash $got, expected $hash" >&2
  rm -f "$tmp"
  exit 1
fi
mv "$tmp" "$dest"
echo "wrote $dest"
