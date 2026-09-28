#!/bin/bash
# build_model.sh - build the deployable weights/ folder from NHirose/omnivla-original on a Kaggle T4 (~0.4 GPU-hours).
#   KAGGLE_USER=<you> ./build/build_model.sh [--dry-run] [OUT_DIR]          (default OUT_DIR = build/out)
# GPTQ int4 LLM for Marlin + HQQ4 vision, shards, SHA256SUMS and per-tensor hashes; downloads and verifies OUT_DIR/weights.
# The notebook compares every tensor with build/reference_tensor_hashes.json (the validated deployment).
# Needs the kaggle CLI, a phone-verified Kaggle account (GPU + internet) and your dataset from build/make_build_dataset.sh.
# --dry-run writes the notebook and its metadata and checks the dataset, without pushing (no GPU time).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
OUT="${1:-$HERE/out}"
KAGGLE="${KAGGLE:-kaggle}"
USER_K="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
SLUG="${KAGGLE_SLUG:-omnivla-build}"
DATASET="${KAGGLE_DATASET:-$USER_K/omnivla-frodobots-frames}"
TIMEOUT_MIN="${TIMEOUT_MIN:-150}"
K="$HERE/kaggle_$SLUG"; mkdir -p "$K" "$OUT"

python3 - "$ROOT" "$K" "$SLUG" "$USER_K" "$DATASET" <<'PY'
import json, os, sys
root, kdir, slug, user, dataset = sys.argv[1:6]
pins = ("git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab "
        "tokenizers==0.19.1 timm==0.9.10 accelerate==0.30.1 peft==0.11.1 draccus==0.8.0 json-numpy==2.1.1 jsonlines==4.0.0 "
        "utm==0.9.0 sentencepiece==0.2.2 hqq==0.2.8.post1 termcolor==3.3.0 huggingface_hub==0.36.2 safetensors==0.8.0 wheel")
cells = [("markdown", "# OmniVLA deployable weights: GPTQ int4 LLM (Marlin) + HQQ4 vision (GemLite), built from NHirose/omnivla-original"),
         ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv\n!pip install -q " + pins +
                  "\n!TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=4 pip install -q --no-build-isolation git+https://github.com/IST-DASLab/marlin.git@1f25790"
                  "\n!python -c \"import torch, transformers, hqq, marlin; print('torch', torch.__version__, 'transformers', transformers.__version__, 'hqq', hqq.__version__)\"")]
for f in ("build/kaggle_compress.py", "build/tensor_hashes.py", "build/kaggle_build.py", "build/reference_tensor_hashes.json"):
    p = os.path.join(root, f)
    if os.path.exists(p):
        cells.append(("code", f"%%writefile /kaggle/working/{os.path.basename(f)}\n" + open(p).read()))
cells.append(("code", "!bash -c 'set -o pipefail; cd /kaggle/working && python kaggle_build.py 2>&1 | grep -v Warning'"))   # a failing build fails the notebook
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": [s], **({"execution_count": None, "outputs": []} if t == "code" else {})}
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(f"{kdir}/{slug}.ipynb", "w"), indent=1)
json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "dataset_sources": [dataset],
           "competition_sources": [], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}, open(f"{kdir}/kernel-metadata.json", "w"), indent=1)
PY

"$KAGGLE" datasets status "$DATASET" > /dev/null 2>&1 || { echo "[build_model] dataset $DATASET not found: run build/make_build_dataset.sh"; [ $DRY = 1 ] || exit 2; }
if [ $DRY = 1 ]; then
  python3 -c "import json,sys; nb=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); print('[build_model] notebook cells', len(nb['cells']), '| id', m['id'], '| dataset', m['dataset_sources'], '| kernel_sources', m['kernel_sources'], '| gpu', m['enable_gpu'], '| private', m['is_private'])" "$K/$SLUG.ipynb" "$K/kernel-metadata.json"
  echo "[build_model] dry run: would run: $KAGGLE kernels push -p $K"; exit 0
fi
echo "[build_model] pushing $USER_K/$SLUG"
PUSH=$("$KAGGLE" kernels push -p "$K" 2>&1 | tail -1); echo "$PUSH"
echo "$PUSH" | grep -qi "successfully pushed" || { echo "[build_model] push refused (GPU session limit or quota?) - stopping"; exit 3; }
start=$(date +%s)
while true; do                                  # bounded wait (TIMEOUT_MIN), stops on any terminal status
  sleep 60
  s=$("$KAGGLE" kernels status "$USER_K/$SLUG" 2>&1 || true)
  case "$s" in *RUNNING*|*QUEUED*) ;; *) break;; esac
  [ $(( $(date +%s) - start )) -ge $(( TIMEOUT_MIN * 60 )) ] && { echo "[build_model] timeout after $TIMEOUT_MIN min: $s"; exit 2; }
done
echo "[build_model] $s"
rm -rf "$OUT/weights" "$OUT/kaggle"; mkdir -p "$OUT/kaggle"
"$KAGGLE" kernels output "$USER_K/$SLUG" -p "$OUT/kaggle" > "$OUT/download.log" 2>&1
grep -h "\[BUILD\]\|Error\|Traceback" "$OUT"/kaggle/*.log 2>/dev/null | tail -12 || true
case "$s" in *COMPLETE*) ;; *) echo "[build_model] kernel did not complete"; exit 4;; esac
grep -qh "\[BUILD\] DONE" "$OUT"/kaggle/*.log || { echo "[build_model] build log has no '[BUILD] DONE': failed"; exit 5; }
mv "$OUT/kaggle/weights" "$OUT/weights"
(cd "$OUT/weights" && shasum -a 256 -c SHA256SUMS --quiet) && echo "[build_model] SHA256SUMS OK (after download)"
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps({k: m[k] for k in m if k != 'shards'}, indent=1))" "$OUT/weights/BUILD_MANIFEST.json"
echo "[build_model] weights ready: $OUT/weights  (copy to the Jetson; setup_jetson.sh --weights-src verifies them again)"
