#!/bin/bash
# End-to-end test of deploy/setup_jetson.sh, run from the Mac/PC: a fresh venv, OmniVLA clone and deploy folder under
# $OMNIVLA_ROOT/setup_test on the Jetson, with the weights from build_model.sh. An existing deployment in
# $OMNIVLA_ROOT/deploy is not modified. System settings are never changed (--no-system); runtime-only steps are accepted
# (--yes-runtime). The script runs setup twice (idempotency) and compares the weights with an existing deployment if there is one.
#   build/jetson_e2e_test.sh          weights from build/out/weights (build_model.sh)
#   build/jetson_e2e_test.sh --hf     weights downloaded by setup_jetson.sh from Hugging Face (pinned revision), as a new user
set -uo pipefail
HF=0; [ "${1:-}" = --hf ] && HF=1
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
H="${JETSON_HOST:-jetson}"
R="${OMNIVLA_ROOT:-/mnt/nvme/omnivla}"
T=$R/setup_test
ssh -o ConnectTimeout=10 "$H" true || { echo "Jetson unreachable ($H)"; exit 2; }
ssh "$H" "test ! -e $T || { echo '$T exists: remove it first (rm -rf $T) for a fresh test'; exit 1; }" || exit 1
ssh "$H" "mkdir -p $T/deploy $T/weights_src"
rsync -a --exclude venv --exclude weights --exclude 'weights_*' --exclude '__pycache__' "$ROOT/deploy/" "$H:$T/deploy/"
if [ $HF = 1 ]; then WS=""; else rsync -a "$ROOT/build/out/weights/" "$H:$T/weights_src/"; WS="--weights-src $T/weights_src"; fi
ssh "$H" "sudo -n /usr/bin/systemctl stop docker docker.socket containerd snapd snapd.socket jtop; free -h | sed -n 2p
  cd $T/deploy && tmux new -d -s setuptest './setup_jetson.sh --root $T $WS --no-system --yes-runtime > $T/setup_run1.log 2>&1; echo RUN1_EXIT \$? >> $T/setup_run1.log; ./setup_jetson.sh --root $T --no-system --yes-runtime > $T/setup_run2.log 2>&1; echo RUN2_EXIT \$? >> $T/setup_run2.log'"
JETSON_HOST="$H" "$ROOT/eval/jetson/wait_job.sh" setuptest "$T/setup_run*.log $T/deploy/setup_jetson.log" "RUN2_EXIT" 90
ssh "$H" "grep -h 'ERROR\|CHECK\]\|SUMMARY\|correctness check:\|NOT DONE\|RUN._EXIT\|ok: venv\|ok: weights' $T/setup_run1.log $T/setup_run2.log
  if [ -d $R/deploy/weights ]; then
    echo '--- weights vs existing deployment (file level)'; cd $T/deploy/weights && for f in base.safetensors model/config.json model/action_head--120000_checkpoint.pt SHARDS.json; do
      a=\$(sha256sum \$f | cut -c1-16); b=\$(sha256sum $R/deploy/weights/\$f | cut -c1-16); echo \"\$f \$a \$b \$([ \$a = \$b ] && echo SAME || echo DIFF)\"; done
  fi"
