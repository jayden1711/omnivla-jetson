#!/bin/bash
# determinism.sh N - N fresh processes of determinism.py (from the repo root on the Jetson), then the comparison.
set -u
N=${1:-50}; cd "$(dirname "$0")/../.."
rm -rf results/determinism
for i in $(seq 1 "$N"); do
  ./deploy/launch.sh eval/jetson/determinism.py "$i" 2>&1 | grep -E "\[DET\]|Error|Traceback" || echo "[DET] run $i FAILED"
done
./deploy/venv/bin/python eval/jetson/determinism_compare.py
