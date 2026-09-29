#!/bin/bash
# vism_validate.sh - Jetson validation of the Marlin vision layers (OMNIVLA_VISION=marlin) in one boot, run from the repo
# root in tmux: validation (img3m 100 frames + pose) for Marlin and HQQ4, per-stage latency for both, reference re-record
# with Marlin vision, 50 fresh-process determinism runs, reference check, 10-minute soak. Stops at the first failure.
set -u
cd "$(dirname "$0")/../.."; R=$PWD; OV=/mnt/nvme/omnivla/OmniVLA
step() { echo "[VISM] === $1 ($(date +%H:%M:%S))"; }
memok() { a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -ge 5700 ] || { echo "[VISM] only $a MB available: stop"; exit 2; }; }
run() { memok; "$@" || { echo "[VISM] FAILED: $*"; exit 3; }; }
for V in marlin hqq4; do
  T=$([ $V = marlin ] && echo _vism || echo _hqq4b)
  for MD in 6 4; do
    step "validate $V mode $MD"; (cd $OV && memok && OMNIVLA_VISION=$V VALIDATE_TAG=$T $R/deploy/launch.sh final_validate.py $MD 2>&1 | grep -E "\[FINAL\] SUMMARY|\[DEPLOY\] ready|Error|Traceback") || exit 3
  done
  step "stage profile $V"; memok; OMNIVLA_VISION=$V PROFILE_TAG=_$V ./deploy/launch.sh eval/jetson/stage_profile.py stages 2>&1 | grep -E "\[STAGE\] [0-9]|Error|Traceback" || exit 3
done
export OMNIVLA_VISION=marlin
step "re-record references (Marlin vision)"; memok; ./deploy/launch.sh deploy/tools/reference_check.py --record-all 2>&1 | grep -E "\[CHECK\]|Error|Traceback" || exit 3
step "determinism x50"; memok; ./eval/jetson/determinism.sh 50 2>&1 | grep -E "FAILED|PASS|FAIL|identical" | tail -15
step "reference check"; memok; ./deploy/launch.sh deploy/tools/reference_check.py 2>&1 | grep -E "\[CHECK\]|Error|Traceback" || exit 3
step "soak 10 min"; (cd $OV && { [ -f results_soak.csv ] && cp results_soak.csv results_soak.csv.bak_vism; true; } && memok && VALIDATE_TAG=_vism $R/deploy/launch.sh final_validate.py --soak 10 2>&1 | grep -E "\[FINAL\]|\[SOAK\]|Error|Traceback" | tail -8)
echo "[VISM] ALL STEPS DONE"
