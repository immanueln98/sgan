#!/usr/bin/env bash
# Download ETH/UCY trajectory datasets in the SGAN format.
# Source: https://github.com/agrimgupta92/sgan (download_data.sh)
#
# Layout produced:
#   datasets/
#     eth/{train,val,test}/*.txt
#     hotel/{train,val,test}/*.txt
#     univ/{train,val,test}/*.txt
#     zara1/{train,val,test}/*.txt
#     zara2/{train,val,test}/*.txt
#
# Run from the navigan_training/ repo root.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/datasets"
ZIP_URL="https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=1"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

if [[ -d zara1 && -d eth ]]; then
    echo "[download_data] Datasets appear to exist already at ${DATA_DIR}. Skipping."
    exit 0
fi

echo "[download_data] Fetching SGAN dataset bundle (~25 MB)..."
if command -v curl >/dev/null 2>&1; then
    curl -L -o datasets.zip "${ZIP_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -O datasets.zip "${ZIP_URL}"
else
    echo "Need curl or wget" >&2
    exit 1
fi

echo "[download_data] Unzipping..."
unzip -q datasets.zip
# SGAN zip extracts to ./datasets/<scene>/{train,val,test} — flatten it
if [[ -d datasets ]]; then
    mv datasets/* .
    rmdir datasets
fi
rm -f datasets.zip

echo "[download_data] Done. Scenes available:"
ls -1 "${DATA_DIR}"
