#!/bin/bash
# readme_rerun.sh TARBALL [REF_IMAGES_DIR] - the README's Jetson workflow from scratch, timed: unpack the repo (a git
# export of the tree under test), setup_jetson.sh (fresh venv, fresh OmniVLA clone, weights from Hugging Face, bit-exact
# check), launch.sh smoke test; then the same for the CAST weights (setup_jetson.sh --cast, OMNIVLA_WEIGHTS=weights_cast).
# Uses its own root (/mnt/nvme/omnivla/readme_rerun) so the working deployment is not touched. System settings are left
# as they are (--no-system; this Jetson is already configured), runtime steps are accepted (--yes-runtime). Without
# ffmpeg on the Jetson the README says to copy the 20 reference images from a PC: REF_IMAGES_DIR (frames/ and goals/).
# Every output line is prefixed with the seconds since the start; step lines start with "[RERUN] ===".
set -u
TAR=$1; REFIMG=${2:-}
R=/mnt/nvme/omnivla/readme_rerun
T0=$(date +%s)
stamp() { while IFS= read -r l; do printf '%6d %s\n' $(( $(date +%s) - T0 )) "$l"; done; }
step() { echo "[RERUN] === $1" | stamp; }
{
step "0 clean root"; rm -rf "$R"; mkdir -p "$R"
step "1 get the repo (git export, as a clone would)"; mkdir -p "$R/omnivla-jetson" && tar -xzf "$TAR" -C "$R/omnivla-jetson" && echo "files: $(find "$R/omnivla-jetson" -type f | wc -l)"
if [ -n "$REFIMG" ]; then
  step "1b reference images copied from a PC (no ffmpeg on this Jetson)"
  cp -r "$REFIMG/frames" "$REFIMG/goals" "$R/omnivla-jetson/deploy/tests/reference/" && ls "$R/omnivla-jetson/deploy/tests/reference/frames" | wc -l
fi
cd "$R/omnivla-jetson"
step "2 setup_jetson.sh (default weights)"; ./deploy/setup_jetson.sh --root "$R" --no-system --yes-runtime; echo "setup exit $?"
step "3 launch.sh smoke test (default weights)"; ./deploy/launch.sh; echo "smoke exit $?"
step "4 setup_jetson.sh --cast"; ./deploy/setup_jetson.sh --cast --root "$R" --no-system --yes-runtime; echo "setup --cast exit $?"
step "5 launch.sh smoke test (CAST weights)"; OMNIVLA_WEIGHTS=weights_cast ./deploy/launch.sh; echo "smoke cast exit $?"
step "DONE"
} 2>&1 | stamp
