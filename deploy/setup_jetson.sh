#!/bin/bash
# setup_jetson.sh - set up a fresh Jetson Orin Nano 8 GB (JetPack 6.2) to run the deployed OmniVLA.
#
#   ./setup_jetson.sh [--weights-src DIR|HOST:DIR] [--hf-repo ID] [--hf-revision REV] [--root DIR] [--venv DIR] [--repo DIR]
#                     [--no-system | --yes] [--yes-runtime] [--cast]
# --cast: install the omnivla-finetuned-cast weights (CAST-style instructions) into weights_cast/ instead of weights/;
#         run them with OMNIVLA_WEIGHTS=weights_cast. Both folders can be installed side by side.
#
# Run from this deploy/ folder on the Jetson. Safe to rerun: every step checks first.
# Persistent system changes (power mode, boot target, swap + fstab, page-cache helper, sudoers) are made only after
# asking; --no-system never makes them, --yes accepts all. Runtime-only changes (stopping services, max clocks during
# the correctness check) are asked separately; --yes-runtime accepts them.
# Steps: 1 preflight, 2 system settings, 3 venv (Jetson torch 2.8.0, transformers fork, pinned packages, Marlin for
# sm_87), 4 OmniVLA @5182600 + patches/omnivla.patch, 5 weights (download from Hugging Face, or copy from --weights-src;
# then SHA256SUMS), 6 correctness check
# (tools/reference_check.py: 10 frames x 2 modes must match the reference outputs bit-exactly).
set -uo pipefail
DEPLOY="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT=/mnt/nvme/omnivla; VENV=""; REPO=""; WSRC=""; MODE=ask; RT=ask
HF_REPO="${OMNIVLA_HF_REPO:-jayden1711/omnivla-7b-jetson-int4}"; HF_REV="${OMNIVLA_HF_REVISION:-bd51d775e220495625e43f312bfe5fd85395e371}"
CAST_REPO=jayden1711/omnivla-7b-cast-jetson-int4; CAST_REV=2c15a98a5f09e5efe823da7d474c28d22426ee2c; WNAME=weights; CAST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;; --venv) VENV="$2"; shift 2;; --repo) REPO="$2"; shift 2;;
    --weights-src) WSRC="$2"; shift 2;; --no-system) MODE=no; shift;; --yes) MODE=yes; RT=yes; shift;;
    --yes-runtime) RT=yes; shift;;
    --hf-repo) HF_REPO="$2"; shift 2;; --hf-revision) HF_REV="$2"; shift 2;;
    --cast) CAST=1; shift;;
    -h|--help) sed -n 2,17p "$0"; exit 0;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
VENV="${VENV:-$DEPLOY/venv}"; REPO="${REPO:-$ROOT/OmniVLA}"; SRC="$ROOT/src"
if [ "$CAST" = 1 ]; then          # an explicit --hf-repo / --hf-revision still wins
  WNAME=weights_cast; [ -z "${OMNIVLA_HF_REPO:-}" ] && HF_REPO=$CAST_REPO; [ -z "${OMNIVLA_HF_REVISION:-}" ] && HF_REV=$CAST_REV
fi
OMNIVLA_COMMIT=5182600cb4a9ee07684e17cdd2a6cbafc56b8a68
TF_FORK="transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab"
MARLIN_COMMIT=1f25790
TORCH_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
LOG="$DEPLOY/setup_jetson.log"; exec > >(tee -a "$LOG") 2>&1
SKIPPED=(); FAILED=0
say() { echo "[setup] $*"; }
ok() { echo "[setup]   ok: $*"; }
die() { echo "[setup] ERROR: $*"; exit 1; }
ask() {  # ask "question" -> 0 = yes
  case "$MODE" in
    yes) say "$1 -> yes (--yes)"; return 0;;
    no) say "$1 -> NOT DONE (--no-system)"; SKIPPED+=("$1"); return 1;;
  esac
  local a; read -r -p "[setup] $1 [y/N] " a < /dev/tty || a=n
  [[ "$a" =~ ^[Yy] ]] && return 0; SKIPPED+=("$1"); return 1
}
ask_rt() {  # runtime (non-persistent) system change
  [ "$RT" = yes ] && { say "$1 -> yes (--yes-runtime)"; return 0; }
  local a; read -r -p "[setup] $1 [y/N] " a < /dev/tty || a=n
  [[ "$a" =~ ^[Yy] ]] && return 0; SKIPPED+=("$1"); return 1
}
say "$(date '+%Y-%m-%dT%H:%M:%S') deploy=$DEPLOY root=$ROOT venv=$VENV repo=$REPO system-changes=$MODE"

# ---------- 1. preflight ----------
[ "$(uname -m)" = aarch64 ] || die "not an aarch64 Jetson"
REL=$(head -1 /etc/nv_tegra_release 2>/dev/null)
echo "$REL" | grep -q "R36 (release), REVISION: 4" && ok "L4T $REL" || say "WARNING: expected L4T R36.4 (JetPack 6.2), found: ${REL:-none}"
command -v python3.10 >/dev/null && ok "python3.10" || die "python3.10 missing (JetPack 6.2 ships it)"
mkdir -p "$ROOT" || die "cannot create $ROOT"
SRCDEV=$(findmnt -n -o SOURCE -T "$ROOT")
case "$SRCDEV" in /dev/nvme*) ok "$ROOT is on $SRCDEV";;
  *) die "$ROOT is on '$SRCDEV', not the NVMe SSD. Mount the NVMe (e.g. at /mnt/nvme, via /etc/fstab) and rerun; this script does not edit fstab.";; esac
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 30 ] && ok "$FREE_GB GB free on $ROOT" || die "need >= 30 GB free on $ROOT (venv ~8 GB, weights 3.9 GB, swap 16 GB)"

# ---------- 2. system settings (each asks) ----------
PM=$(nvpmodel -q 2>/dev/null | grep -i "power mode" | head -1)
if echo "$PM" | grep -qi "MAXN_SUPER\|MAXN SUPER"; then ok "power mode: $PM"
else say "power mode: ${PM:-unknown (nvpmodel -q needs root on some images)}"
  ask "set power mode 2 (MAXN SUPER) with 'sudo nvpmodel -m 2' (persistent)" && sudo nvpmodel -m 2; fi
if [ "$(systemctl get-default)" = multi-user.target ]; then ok "headless boot (multi-user.target)"
else ask "boot headless: 'sudo systemctl set-default multi-user.target' (saves ~0.8 GB; needs a reboot; undo: graphical.target)" \
  && sudo systemctl set-default multi-user.target && say "headless boot set: REBOOT before running the model"; fi
SWAPF=/mnt/nvme/swapfile
if swapon --show=NAME --noheadings | grep -q "^$SWAPF$"; then ok "NVMe swap $SWAPF active"
else
  if ask "create a 16 GB swap file $SWAPF, enable it and add it to /etc/fstab"; then
    [ -f $SWAPF ] || { sudo fallocate -l 16G $SWAPF && sudo chmod 600 $SWAPF && sudo mkswap $SWAPF; }
    sudo swapon $SWAPF; grep -q "^$SWAPF " /etc/fstab || echo "$SWAPF none swap sw,nofail 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi
fi
DC=/usr/local/bin/omni-dropcache
if [ -x $DC ] && grep -q "drop_caches" $DC; then ok "page-cache helper $DC"
else ask "install the page-cache helper $DC (root loop: sync; drop_caches; sleep 1)" && {
  printf '#!/bin/sh\nwhile true; do sync; echo 3 > /proc/sys/vm/drop_caches; sleep 1; done\n' | sudo tee $DC >/dev/null && sudo chmod 755 $DC; }; fi
need_sudo=0
for c in "/usr/bin/jetson_clocks" "$DC" "/usr/bin/pkill -f omni-dropcache" "/usr/bin/tegrastats" "/usr/bin/systemctl stop docker" "/usr/sbin/reboot"; do
  sudo -n -l $c >/dev/null 2>&1 || { need_sudo=1; say "no passwordless sudo for: $c"; }
done
if [ $need_sudo = 0 ]; then ok "sudoers entries (jetson_clocks, omni-dropcache, pkill, tegrastats, systemctl stop, reboot)"
else
  if ask "write /etc/sudoers.d/omnivla (passwordless: jetson_clocks, omni-dropcache, pkill -f omni-dropcache, tegrastats, systemctl stop *, reboot) for $USER"; then
    T=$(mktemp); echo "$USER ALL=(root) NOPASSWD: /usr/bin/jetson_clocks, $DC, /usr/bin/pkill -f omni-dropcache, /usr/bin/tegrastats, /usr/bin/systemctl stop *, /usr/sbin/reboot" > "$T"
    sudo visudo -cf "$T" && sudo install -m 440 -o root -g root "$T" /etc/sudoers.d/omnivla || say "sudoers entry NOT installed (visudo check failed)"; rm -f "$T"
  fi
fi

# ---------- 3. venv ----------
[ -x "$VENV/bin/python" ] || { say "creating venv $VENV"; python3.10 -m venv "$VENV" || die "venv"; }
PY="$VENV/bin/python"; PIP="$PY -m pip"
if $PY -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.8.0') and torch.cuda.is_available() else 1)" 2>/dev/null; then ok "torch 2.8.0 with CUDA"
else
  $PIP install -q --upgrade pip wheel
  $PIP install -q torch==2.8.0 torchvision==0.23.0 --index-url $TORCH_INDEX || die "torch install"
  $PY -c "import torch; assert torch.__version__.startswith('2.8.0') and torch.cuda.is_available()" || die "torch 2.8.0 without CUDA"
  ok "torch installed"
fi
if $PY -c "import inspect, transformers.models.llama.modeling_llama as m, sys; sys.exit(0 if 'is_causal=False' in inspect.getsource(m.LlamaSdpaAttention.forward) else 1)" 2>/dev/null; then
  ok "transformers openvla-oft fork"
else $PIP install -q -c "$DEPLOY/constraints.txt" "$TF_FORK" || die "transformers fork"; fi
$PIP install -q -c "$DEPLOY/constraints.txt" -r "$DEPLOY/requirements-deploy.txt" || die "requirements-deploy.txt"
if $PY -c "import marlin; marlin.Layer" 2>/dev/null; then ok "marlin"
else
  [ -d "$SRC/marlin/.git" ] || git clone -q https://github.com/IST-DASLab/marlin.git "$SRC/marlin"
  git -C "$SRC/marlin" checkout -q $MARLIN_COMMIT || die "marlin checkout"
  TORCH_CUDA_ARCH_LIST=8.7 $PIP install -q -c "$DEPLOY/constraints.txt" --no-build-isolation "$SRC/marlin" || die "marlin build"
fi
$PY - <<'PYC' || die "venv check"
import inspect, torch, numpy, transformers, hqq, gemlite, marlin
import transformers.models.llama.modeling_llama as ml
assert torch.__version__.startswith("2.8.0") and torch.cuda.is_available(), torch.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
assert "is_causal=False" in inspect.getsource(ml.LlamaSdpaAttention.forward), "stock transformers: need the openvla-oft fork"
print(f"[setup]   ok: venv torch {torch.__version__} numpy {numpy.__version__} transformers {transformers.__version__} (fork) hqq {hqq.__version__}")
PYC

# ---------- 4. OmniVLA repo ----------
[ -d "$REPO/.git" ] || git clone -q https://github.com/NHirose/OmniVLA.git "$REPO" || die "clone"
HEAD=$(git -C "$REPO" rev-parse HEAD)
if [ "$HEAD" != $OMNIVLA_COMMIT ]; then
  git -C "$REPO" diff --quiet || die "$REPO has local changes on another commit; not touching it"
  git -C "$REPO" checkout -q $OMNIVLA_COMMIT || die "checkout $OMNIVLA_COMMIT"
fi
if git -C "$REPO" apply --reverse --check "$DEPLOY/patches/omnivla.patch" 2>/dev/null; then ok "OmniVLA @${OMNIVLA_COMMIT:0:7}, patch applied"
else git -C "$REPO" apply "$DEPLOY/patches/omnivla.patch" && ok "patch applied" || die "patches/omnivla.patch does not apply"; fi

# ---------- 5. weights ----------
verify_weights() { [ -f "$1/SHA256SUMS" ] && (cd "$1" && sha256sum -c --quiet SHA256SUMS >/dev/null 2>&1); }
if verify_weights "$DEPLOY/$WNAME"; then ok "$WNAME/ verified ($(wc -l < "$DEPLOY/$WNAME/SHA256SUMS") files)"
else
  if [ -n "$WSRC" ]; then
    say "copying weights from $WSRC"
    rm -rf "$DEPLOY/$WNAME.incoming"; rsync -a "${WSRC%/}/" "$DEPLOY/$WNAME.incoming/" || die "copy"
    verify_weights "$DEPLOY/$WNAME.incoming" || die "copied weights fail SHA256SUMS"
    HOW="copied"
  else
    say "downloading weights from https://huggingface.co/$HF_REPO (revision $HF_REV, 4.1 GB)"
    "$PY" "$DEPLOY/tools/download_weights.py" "$HF_REPO" "$HF_REV" "$DEPLOY/$WNAME.incoming" \
      || die "$WNAME/ missing: download from Hugging Face failed (see above); or pass --weights-src DIR"
    verify_weights "$DEPLOY/$WNAME.incoming" || { rm -rf "$DEPLOY/$WNAME.incoming"; die "downloaded weights fail SHA256SUMS"; }
    HOW="downloaded"
  fi
  [ -e "$DEPLOY/$WNAME" ] && mv "$DEPLOY/$WNAME" "$DEPLOY/$WNAME.old.$(date +%s)"
  mv "$DEPLOY/$WNAME.incoming" "$DEPLOY/$WNAME"; ok "$WNAME/ $HOW and verified"
fi

# ---------- 6. runtime + correctness check ----------
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null); AVAIL=${AVAIL:-0}
if [ "$AVAIL" -lt 5700 ]; then
  say "only $AVAIL MB available (need >= 5700)"
  ask_rt "stop memory-holding services now (sudo systemctl stop docker docker.socket containerd snapd snapd.socket jtop; until reboot)" \
    && sudo -n /usr/bin/systemctl stop docker docker.socket containerd snapd snapd.socket jtop 2>/dev/null
  AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null); AVAIL=${AVAIL:-0}
fi
[ "$AVAIL" -ge 5700 ] && ok "$AVAIL MB available" || say "WARNING: $AVAIL MB available; the 7B model may not fit (reboot headless?)"
sudo -n /usr/bin/jetson_clocks --show 2>/dev/null | grep -q "MaxFreq" && ok "jetson_clocks available (launch.sh sets max clocks for every run)"
if ask_rt "run the correctness check now (launch.sh sets max clocks with jetson_clocks until reboot and runs the page-cache helper while the model runs)"; then
  if OMNIVLA_WEIGHTS="$WNAME" OMNIVLA_VENV="$VENV" OMNIVLA_REPO="$REPO" "$DEPLOY/launch.sh" "$DEPLOY/tools/reference_check.py"; then CHECK=PASS; else CHECK=FAIL; FAILED=1; fi
else CHECK="NOT RUN"; FAILED=1; fi
echo; say "================ SUMMARY ================"
say "correctness check: $CHECK"
[ ${#SKIPPED[@]} -gt 0 ] && { say "system steps NOT done (answer yes or run them by hand):"; for s in "${SKIPPED[@]}"; do say "  - $s"; done; }
say "log: $LOG"
exit $FAILED
