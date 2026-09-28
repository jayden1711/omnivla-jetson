#!/bin/bash
# Create (or update) your own private Kaggle dataset <KAGGLE_USER>/omnivla-frodobots-frames, which build_model.sh and
# the Kaggle studies read: OmniVLA inference code, harness.py, FrodoBots-2K test frames, held-out calibration samples
# and ground truth.
#   KAGGLE_USER=<you> ./build/make_build_dataset.sh [--dry-run]
# Steps: clone OmniVLA @5182600 into build/work/OmniVLA and apply deploy/patches/omnivla.patch; run the eval/data
# scripts whose outputs are missing (a few GB of FrodoBots-2K range downloads, a few hours); pack kaggle_ds/; upload.
# --dry-run checks tools, Python packages and the Kaggle login, and prints what would run. It uploads and downloads nothing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
KAGGLE="${KAGGLE:-kaggle}"
PY="${PYTHON:-python3}"
USER_K="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
DATASET="$USER_K/omnivla-frodobots-frames"
SRC="$HERE/work/OmniVLA"
OMNIVLA_COMMIT=5182600cb4a9ee07684e17cdd2a6cbafc56b8a68
cd "$ROOT"
say() { echo "[dataset] $*"; }
run() { if [ $DRY = 1 ]; then echo "  would run: $*"; else "$@"; fi; }

for t in git ffmpeg "$KAGGLE" "$PY"; do command -v "$t" >/dev/null || { say "missing tool: $t"; exit 2; }; done
"$PY" -c "import numpy, pandas, PIL, remotezip, utm, scipy, geomag" 2>/dev/null \
  || { say "missing Python packages: $PY -m pip install -r eval/requirements-eval.txt"; exit 2; }
"$KAGGLE" datasets list --user "$USER_K" --csv > /dev/null 2>&1 \
  || { say "kaggle CLI not logged in (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)"; exit 2; }
if "$KAGGLE" datasets status "$DATASET" > /dev/null 2>&1; then EXISTS=1; say "$DATASET exists: a new version will be uploaded"
else EXISTS=0; say "$DATASET does not exist yet: it will be created (private)"; fi

if [ -d "$SRC/.git" ]; then
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$OMNIVLA_COMMIT" ] || { say "$SRC is not at $OMNIVLA_COMMIT"; exit 1; }
  git -C "$SRC" apply --reverse --check "$ROOT/deploy/patches/omnivla.patch" 2>/dev/null \
    || run git -C "$SRC" apply "$ROOT/deploy/patches/omnivla.patch"
  say "OmniVLA clone ok: $SRC"
else
  run git clone -q https://github.com/NHirose/OmniVLA.git "$SRC"
  run git -C "$SRC" checkout -q "$OMNIVLA_COMMIT"
  run git -C "$SRC" apply "$ROOT/deploy/patches/omnivla.patch"
fi

# eval/data chain; each script is skipped when its output exists
for step in "results/frames_manifest.csv extract_frames.py" "results/gt_frodobots.npz gt_extract.py" \
            "results/gt_tests.npz gt_tests.py" "results/heldout.npz heldout_extract.py"; do
  set -- $step
  if [ -f "$1" ]; then say "have $1"; else run "$PY" "eval/data/$2"; fi
done

run env OMNIVLA_SRC="$SRC" KAGGLE_USER="$USER_K" "$ROOT/eval/kaggle/make_kaggle_bundle.sh"
if [ $EXISTS = 1 ]; then run "$KAGGLE" datasets version -p kaggle_ds -m "omnivla-jetson build inputs"
else run "$KAGGLE" datasets create -p kaggle_ds; fi
[ $DRY = 1 ] && say "dry run: nothing uploaded" || say "done: $DATASET (use it with KAGGLE_USER=$USER_K ./build/build_model.sh)"
