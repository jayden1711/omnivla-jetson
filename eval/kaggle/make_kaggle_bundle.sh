#!/bin/bash
# Build kaggle_ds/omnivla_bundle.tar.gz for the private Kaggle dataset that the Kaggle studies and build/build_model.sh read:
# OmniVLA inference code (patched clone), harness.py, the FrodoBots-2K frames (CC BY-SA 4.0) and the ground-truth files
# made by eval/data/*.py. Run from the repo root after the eval/data scripts.
#   OMNIVLA_SRC=<OmniVLA clone @5182600 with deploy/patches/omnivla.patch applied> KAGGLE_USER=<you> eval/kaggle/make_kaggle_bundle.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${OMNIVLA_SRC:?set OMNIVLA_SRC to a patched OmniVLA clone}"
USER_K="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
B=kaggle_ds/bundle
rm -rf "$B"; mkdir -p "$B"
(cd "$SRC" && tar czf - --exclude=__pycache__ prismatic inference/run_omnivla.py inference/goal_img.jpg inference/current_img.jpg LICENSE) \
  | COPYFILE_DISABLE=1 tar xzf - -C "$B"
cp "$HERE/harness.py" "$B/"
[ -f results/verify_frames.txt ] || cp "$HERE/verify_frames.txt" results/verify_frames.txt
cp -r data_frames/frames "$B/frames"
for f in frames_manifest.csv verify_frames.txt gt_frodobots.npz gt_tests.npz gt_tests.csv shuffle_map.csv heldout.npz heldout_frames.csv results_idea1.npz; do
  if [ -f "results/$f" ]; then cp "results/$f" "$B/"; else echo "note: results/$f not found, skipped"; fi
done
cp -r data_frames/goal_img3m "$B/goal_img3m"
cp -r data_heldout/frames "$B/heldout_frames"; cp -r data_heldout/goals_img "$B/heldout_goals_img"
COPYFILE_DISABLE=1 tar czf kaggle_ds/omnivla_bundle.tar.gz -C "$B" .
rm -rf "$B"
cat > kaggle_ds/dataset-metadata.json <<JSON
{"title": "omnivla-frodobots-frames", "id": "$USER_K/omnivla-frodobots-frames",
 "licenses": [{"name": "CC-BY-SA-4.0"}],
 "description": "224x224 front-camera frames from FrodoBots-2K (CC BY-SA 4.0, FrodoBots Lab) plus the OmniVLA inference code (MIT, NHirose/OmniVLA)."}
JSON
ls -la kaggle_ds
