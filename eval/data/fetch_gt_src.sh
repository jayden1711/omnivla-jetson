#!/bin/bash
# Fetch the FrodoBots converter modules that gt_extract.py uses (convert_to_hf.py, filtering.py, interpolation_utils.py) from
# github.com/catglossop/frodo_dataset at a pinned commit. They are not included here because that repository has no license.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COMMIT=bf2c4b82951e6b9dd8b9ab6a9421de9ddb0eb02c
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git clone -q https://github.com/catglossop/frodo_dataset "$TMP/frodo_dataset"
git -C "$TMP/frodo_dataset" checkout -q "$COMMIT"
mkdir -p "$HERE/gt_src"
cp "$TMP/frodo_dataset/"{convert_to_hf.py,filtering.py,interpolation_utils.py} "$HERE/gt_src/"
echo "gt_src ready at $HERE/gt_src (frodo_dataset @ ${COMMIT:0:7})"
