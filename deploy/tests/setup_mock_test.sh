#!/bin/bash
# setup_mock_test.sh - control-flow test of setup_jetson.sh without a Jetson (any Unix). Jetson commands are stubbed
# with their state kept in files, so reruns see earlier changes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; DEPLOY_SRC="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; mkdir -p "$T/bin"
PASS=0; FAILN=0
check() { if eval "$2"; then echo "  PASS $1"; PASS=$((PASS+1)); else echo "  FAIL $1"; FAILN=$((FAILN+1)); fi; }
mkstate() {  # fresh "Jetson": nothing configured
  rm -rf "$T/state" "$T/root" "$T/deploy"; mkdir -p "$T/state" "$T/root" "$T/deploy" "$T/bin"
  echo graphical.target > "$T/state/default"; echo "NV Power Mode: 15W" > "$T/state/pm"; : > "$T/state/swap"; : > "$T/state/sudo.log"
  cp -R "$DEPLOY_SRC/setup_jetson.sh" "$DEPLOY_SRC/constraints.txt" "$DEPLOY_SRC/requirements-deploy.txt" "$DEPLOY_SRC/patches" "$T/deploy/"
  mkdir -p "$T/deploy/tools"; printf '#!/bin/sh\nexit 0\n' > "$T/deploy/tools/reference_check.py"
  printf '#!/bin/sh\necho "launch.sh $*" >> "%s/state/launch.log"; exit ${CHECK_RC:-0}\n' "$T" > "$T/deploy/launch.sh"; chmod +x "$T/deploy/launch.sh"
}
stub() { printf '#!/bin/bash\n%s\n' "$2" > "$T/bin/$1"; chmod +x "$T/bin/$1"; }
S="$T/state"
stub uname 'echo aarch64'
stub findmnt 'echo /dev/nvme0n1p1'
stub df 'printf "Avail\n 120G\n"'
stub nvpmodel "cat $S/pm"
stub python3.10 'if [ "$1" = -m ] && [ "$2" = venv ]; then mkdir -p "$3/bin"; cp '"$T"'/bin/fakepy "$3/bin/python"; fi'
stub fakepy "echo \"py \$*\" >> $S/py.log
case \"\$1\" in *download_weights.py) [ -d $S/hfsrc ] || exit 1; mkdir -p \"\$4\"; cp -R $S/hfsrc/. \"\$4\"/;; esac
exit 0"
stub systemctl "case \"\$1\" in get-default) cat $S/default;; set-default) echo \$2 > $S/default;; esac"
stub swapon "if [ \"\$1\" = --show=NAME ]; then cat $S/swap; else echo \$1 > $S/swap; fi"
stub fallocate "touch \"\${@: -1}\""; stub mkswap "true"; stub chmod "true"
stub sudo "echo \"sudo \$*\" >> $S/sudo.log
if [ \"\$1\" = -n ] && [ \"\$2\" = -l ]; then [ -f $S/sudoers ]; exit \$?; fi
[ \"\$1\" = -n ] && shift
case \"\$1\" in
  nvpmodel) echo 'NV Power Mode: MAXN_SUPER' > $S/pm;;
  systemctl) \"\$@\";;
  install) touch $S/sudoers;;
  visudo) true;;
  tee) case \"\$*\" in *fstab*) cat > /dev/null; touch $S/fstab;; *) cat > \"\${@: -1}\"; touch $S/dropcache;; esac;;
  fallocate|mkswap|chmod) true;;
  swapon) echo \$2 > $S/swap;;
  /usr/bin/jetson_clocks) echo MaxFreq;;
  *) \"\$@\";;
esac"
stub git "case \"\$*\" in *clone*) mkdir -p \"\${@: -1}/.git\"; echo 5182600cb4a9ee07684e17cdd2a6cbafc56b8a68 > \"\${@: -1}/.git/HEAD_SHA\";;
  *rev-parse*) cat \"\$2/.git/HEAD_SHA\" 2>/dev/null || echo 0;;
  *'apply --reverse --check'*) [ -f \"\$2/.patched\" ];;
  *apply*) [ -f $S/badpatch ] && exit 1; touch \"\$2/.patched\";;
  *diff*) [ ! -f $S/dirty ];; *) true;; esac"
# the real script uses a fixed dropcache path and SWAPF; point them at the sandbox
run() {  # run setup_jetson.sh with stubs; extra args follow
  sed -e "s#/usr/local/bin/omni-dropcache#$S/dropcache_bin#g" -e "s#SWAPF=/mnt/nvme/swapfile#SWAPF=$S/swapfile#" \
      -e "s#grep -q \"^\$SWAPF \" /etc/fstab#[ -f $S/fstab ]#" "$T/deploy/setup_jetson.sh" > "$T/deploy/setup_run.sh"
  [ -f $S/dropcache ] && { printf 'drop_caches\n' > $S/dropcache_bin; chmod +x $S/dropcache_bin; }
  chmod +x "$T/deploy/setup_run.sh"
  PATH="$T/bin:$PATH" "$T/deploy/setup_run.sh" --root "$T/root" "$@" < /dev/null > "$S/out.log" 2>&1; echo $? > "$S/rc"
}
mkweights() { mkdir -p "$1/marpc"; echo a > "$1/marpc/shard_000.pt"; echo b > "$1/base.safetensors"; (cd "$1" && shasum -a 256 marpc/shard_000.pt base.safetensors > SHA256SUMS); }
stub sha256sum 'if [ "$1" = -c ]; then shift; [ "$1" = --quiet ] && shift; shasum -a 256 -c "$@" > /dev/null; else shasum -a 256 "$@"; fi'

echo "[T1] fresh Jetson, --no-system, no weights source"
mkstate; run --no-system
check "aborts (weights missing) with nonzero exit" "[ \$(cat $S/rc) != 0 ] && grep -q 'weights/ missing' $S/out.log"
check "no persistent change made" "[ \$(cat $S/default) = graphical.target ] && [ ! -s $S/swap ] && [ ! -f $S/sudoers ] && [ ! -f $S/dropcache ]"
check "every missing system step reported" "[ \$(grep -c 'NOT DONE (--no-system)' $S/out.log) -eq 5 ]"
check "venv + repo + patch done anyway" "[ -x $T/deploy/venv/bin/python ] && [ -f $T/root/OmniVLA/.patched ]"

echo "[T2] fresh Jetson, --yes, weights from a source folder"
mkstate; mkweights "$T/wsrc"; run --yes --weights-src "$T/wsrc"
check "exit 0" "[ \$(cat $S/rc) = 0 ]"
check "each system step applied once" "grep -q MAXN $S/pm && [ \$(cat $S/default) = multi-user.target ] && [ -s $S/swap ] && [ -f $S/sudoers ] && [ -f $S/dropcache ] && [ -f $S/fstab ]"
check "weights copied and verified" "[ -f $T/deploy/weights/SHA256SUMS ] && grep -q 'weights/ copied and verified' $S/out.log"
check "correctness check ran through launch.sh" "grep -q reference_check $S/launch.log"
echo "[T3] rerun after T2 (default mode + --yes-runtime: no persistent questions)"
cp $S/sudo.log $S/sudo_before.log; run --yes-runtime
check "exit 0, CHECK PASS" "[ \$(cat $S/rc) = 0 ] && grep -q 'correctness check: PASS' $S/out.log"
check "asked nothing persistent / changed nothing" "! grep -q 'NOT DONE\|\[y/N\]' $S/out.log && ! diff <(grep -v ' -l \| -n /usr/bin/jetson_clocks\| -n /usr/bin/systemctl' $S/sudo_before.log) <(grep -v ' -l \| -n /usr/bin/jetson_clocks\| -n /usr/bin/systemctl' $S/sudo.log) | grep -q '^>'"
check "weights not re-copied" "grep -q 'weights/ verified' $S/out.log"
echo "[T4] corrupted weights source"
mkstate; mkweights "$T/wbad"; echo tampered >> "$T/wbad/base.safetensors"; run --no-system --weights-src "$T/wbad"
check "rejected (nonzero, 'fail SHA256SUMS', no weights/ installed)" "[ \$(cat $S/rc) != 0 ] && grep -q 'fail SHA256SUMS' $S/out.log && [ ! -e $T/deploy/weights ]"
echo "[T5] patch does not apply"
mkstate; touch $S/badpatch; run --no-system --weights-src "$T/wsrc"
check "aborts on patch failure" "[ \$(cat $S/rc) != 0 ] && grep -q 'does not apply' $S/out.log"
echo "[T6] correctness check fails"
mkstate; run --yes --weights-src "$T/wsrc" ; : ; CHECK_RC=1 run --yes-runtime --weights-src "$T/wsrc"
check "nonzero exit and FAIL in summary" "[ \$(cat $S/rc) != 0 ] && grep -q 'correctness check: FAIL' $S/out.log"
echo "[T7] --no-system: runtime steps asked separately; without --yes-runtime the check is NOT RUN"
mkstate; run --no-system --weights-src "$T/wsrc"
check "check not run, reported" "grep -q 'correctness check: NOT RUN' $S/out.log && ! [ -f $S/launch.log ]"
mkstate; run --no-system --yes-runtime --weights-src "$T/wsrc"
check "--yes-runtime runs the check without persistent changes" "grep -q 'correctness check: PASS' $S/out.log && [ \$(cat $S/default) = graphical.target ]"
echo "[T8] default: weights downloaded from Hugging Face"
mkstate; mkweights "$S/hfsrc"; run --no-system --yes-runtime
check "downloaded, verified, check PASS" "[ \$(cat $S/rc) = 0 ] && grep -q 'weights/ downloaded and verified' $S/out.log && grep -q 'download_weights.py jayden1711/omnivla-7b-jetson-int4 ' $S/py.log"
echo "[T9] corrupted Hugging Face download"
mkstate; mkweights "$S/hfsrc"; echo tampered >> "$S/hfsrc/base.safetensors"; run --no-system --yes-runtime
check "rejected, nothing installed, partial download removed" "[ \$(cat $S/rc) != 0 ] && grep -q 'downloaded weights fail SHA256SUMS' $S/out.log && [ ! -e $T/deploy/weights ] && [ ! -e $T/deploy/weights.incoming ]"
echo "[mock] $PASS passed, $FAILN failed"
[ $FAILN = 0 ]
