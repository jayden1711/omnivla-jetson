#!/bin/bash
# run_power_sweep.sh - power_measure.py at 15W, 25W and MAXN_SUPER, then back to the original power mode.
#   ./eval/jetson/run_power_sweep.sh [N_PER_CASE]          (on the Jetson, from the repo root)
# Changing the power mode needs sudo (nvpmodel) and asks first. If nvpmodel says a reboot is needed, the script stops:
# reboot, then rerun it (modes that already have results/power_<mode>.json are skipped). Summary: power_analysis.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; N="${1:-50}"
CONF=/etc/nvpmodel.conf
ORIG=$(nvpmodel -q | sed -n 's/NV Power Mode: *//p')
id_of() { sed -n "s/.*POWER_MODEL ID=\([0-9]*\) NAME=$1 .*/\1/p" "$CONF" | head -1; }
for NAME in 15W 25W MAXN_SUPER; do
  [ -f "$ROOT/results/power_$NAME.json" ] && { echo "[sweep] $NAME: done, skipping"; continue; }
  ID=$(id_of "$NAME"); [ -n "$ID" ] || { echo "[sweep] $NAME not in $CONF, skipping"; continue; }
  CUR=$(nvpmodel -q | sed -n 's/NV Power Mode: *//p')
  if [ "$CUR" != "$NAME" ]; then
    read -r -p "[sweep] switch power mode $CUR -> $NAME (sudo nvpmodel -m $ID)? [y/N] " a; [ "$a" = y ] || exit 1
    OUT=$(yes no | sudo nvpmodel -m "$ID" 2>&1 || true); echo "$OUT"
    [ "$(nvpmodel -q | sed -n 's/NV Power Mode: *//p')" = "$NAME" ] || { echo "[sweep] $NAME needs a reboot: reboot, then rerun"; exit 2; }
  fi
  "$ROOT/deploy/launch.sh" "$ROOT/eval/jetson/power_measure.py" "$N"
done
CUR=$(nvpmodel -q | sed -n 's/NV Power Mode: *//p')
if [ "$CUR" != "$ORIG" ]; then
  read -r -p "[sweep] restore power mode $ORIG? [y/N] " a
  [ "$a" = y ] && { yes no | sudo nvpmodel -m "$(id_of "$ORIG")" || true; }
fi
python3 "$ROOT/eval/analysis/power_analysis.py"
