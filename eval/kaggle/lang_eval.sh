#!/bin/bash
# lang_eval.sh - language-goal test (CAST) on a Kaggle T4, plus the deployed int4 config on all driving tests (~1 GPU-hour).
#   KAGGLE_USER=<you> eval/kaggle/lang_eval.sh [--dry-run] [OUT_DIR]        (default OUT_DIR = build/kaggle_lang_out)
# Needs: your dataset from build/make_build_dataset.sh and a kernel <you>/omnivla-cast-extract whose output is
# cast_lang.npz (eval/data/cast_extract.py, run on a Kaggle CPU kernel with internet: no GPU time).
# Runs kaggle_compress.py: STUDY=lang (fp16 with blank / shuffled image / shuffled instruction controls, bf16 reference),
# then STUDY=gptqx with the deployed settings (GPTQ per-channel int4 LLM, HQQ4 vision, 75% token pruning) on lang, img3m,
# pose5 and pose20. The int4 LLM weights are compared tensor by tensor with build/reference_tensor_hashes.json, so the
# numbers are for the deployed weights; the kernels are not the Jetson's (fp16 fake-quant instead of Marlin/GemLite).
# LANG_TEST=lelan runs the object-goal test instead (eval/data/lelan_extract.py) plus the demo clips' image-goal
# predictions (eval/demo/seq_demo_inputs.py); both npz files go in a private dataset <you>/omnivla-lelan-demo-inputs.
# LANG_TEST=ablation (~0.8 GPU-hour): object-goal test for the pruning ablation: fp16 with 75% pruning (no quantization),
# then the deployed int4 weights (GPTQ calibrated as built) evaluated with 0%, 50% and 75% image-token pruning.
# LANG_TEST=promptprune (~0.7 GPU-hour): object-goal test with prompt-aware pruning (deploy/prompt_prune.py) at 25/50/75%
# with the fine-tuned SigLIP, 75% with the original SigLIP image tower, and uniform 25%; deployed int4 weights.
# LANG_TEST=vismarlin (~0.6-1 GPU-hour): deployed int4 LLM (hash-checked) with the vision linears as Marlin per-channel
# int4 (HQQ4 baseline, RTN all, RTN SigLIP MLP only, GPTQ all) on img3m (75% pruning) and the object-goal test (no
# pruning, as deployed); exports prequant_vis_{rtn,gptq}.pt. LANG_TEST=vismarlin_smoke: 3 frames, 4 calibration samples.
# LANG_TEST=visgptqx (~0.2 GPU-hour): only the vision GPTQ export (prequant_vis_gptq*.pt), same calibration as vismarlin.
# LANG_TEST=xmode (~0.45 GPU-hour): omnivla-finetuned-cast in bf16 on img3m, pose5 and the object-goal test (real inputs,
# no pruning), then NHirose/omnivla-original in fp16: latency per mode on the Kaggle GPU, as released and with elision.
# LANG_TEST=castgptq (~0.75 GPU-hour): NHirose/omnivla-finetuned-cast (pinned revision, heads at step 210000), language
# mode, no pruning. build/cast_split.json (eval/data/cast_split.py) splits cast_lang.npz by recording: GPTQ per-channel
# int4 for the vision linears and the LLM is calibrated on the calibration episodes; bf16 and int4 are evaluated on the
# held-out episodes with blank / shuffled image / shuffled instruction controls. Exports prequant_marpcg_cast*.pt and
# prequant_vis_cast_gptq*.pt. LANG_TEST=castgptq_smoke: 4 calibration samples, 3 frames, no packing.
set -euo pipefail
LANG_TEST="${LANG_TEST:-lang}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
OUT="${1:-$ROOT/build/kaggle_${LANG_TEST}_out}"
KAGGLE="${KAGGLE:-kaggle}"
USER_K="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
SLUG="${KAGGLE_SLUG:-omnivla-${LANG_TEST//_/-}-eval}"   # Kaggle slugs: hyphens only
TIMEOUT_MIN="${TIMEOUT_MIN:-150}"
K="$ROOT/build/kaggle_$SLUG"; mkdir -p "$K" "$OUT"

python3 - "$ROOT" "$K" "$SLUG" "$USER_K" "$LANG_TEST" <<'PY'
import json, os, sys
root, kdir, slug, user, test = sys.argv[1:6]
pins = ("git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab "
        "tokenizers==0.19.1 timm==0.9.10 accelerate==0.30.1 peft==0.11.1 draccus==0.8.0 json-numpy==2.1.1 jsonlines==4.0.0 "
        "utm==0.9.0 sentencepiece==0.2.2 hqq==0.2.8.post1 termcolor==3.3.0 huggingface_hub==0.36.2 safetensors==0.8.0 pandas wheel")
run = lambda env: ("import subprocess  # streamed; a failing step stops the notebook (a '!' line would not)\n"
                   f"p = subprocess.Popen('cd /kaggle/working && {env} python kaggle_compress.py', shell=True, text=True, "
                   "stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n"
                   "for line in p.stdout:\n    if 'Warning' not in line: print(line, end='', flush=True)\n"
                   "assert p.wait() == 0, 'step failed'")
check = r'''import json, torch, os
from tensor_hashes import th
R = json.load(open("reference_tensor_hashes.json"))
g = torch.load("prequant_marpcg.pt", map_location="cpu", weights_only=False)
same = sum(1 for n in g if n in R and {k: th(v) for k, v in sorted(g[n].items())} == R[n])
print(f"[LANG] int4 LLM vs deployed weights: {same}/{len(g)} layers identical", flush=True)
for f in ("prequant_marpcg.pt", "prequant_marpcg_check.pt"):
    os.remove(f)
assert same == len(g) == 224, "int4 weights differ from the deployment"'''
cells = [("markdown", "# OmniVLA language-goal test (CAST) and deployed int4 config on all driving tests"),
         ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv\n!pip install -q " + pins +
                  "\n!TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=4 pip install -q --no-build-isolation git+https://github.com/IST-DASLab/marlin.git@1f25790")]
if test == "promptprune":                                    # SigLIP text tower (open_clip 2.24.0 keeps timm 0.9.10)
    cells.append(("code", "!pip install -q --no-deps open_clip_torch==2.24.0 && pip install -q ftfy regex"))
for f in ("build/kaggle_compress.py", "build/tensor_hashes.py", "build/reference_tensor_hashes.json", "deploy/prompt_prune.py"):
    cells.append(("code", f"%%writefile /kaggle/working/{os.path.basename(f)}\n" + open(os.path.join(root, f)).read()))
if test == "promptprune":
    cells += [("code", run("STUDY=gptqx PRUNE=spatial75 N_CALIB=64 TESTS_ONLY=lelan BLIND=none,blank,shuffled,lang_shuffled "
                           "BLIND_TESTS=lelan EVAL_PRUNES=spatial25,prompt25,prompt50,prompt75,promptorig75 OUT_SUFFIX=_promptprune"))]
elif test in ("vismarlin", "vismarlin_smoke"):
    smoke = "SMOKE=1 N_CALIB=4" if test.endswith("smoke") else "N_CALIB=64"
    cells += [("code", run(f"STUDY=vismarlin PRUNE=spatial75 {smoke} TESTS_ONLY=img3m,lelan"))]
elif test in ("castgptq", "castgptq_smoke"):
    cells.append(("code", "%%writefile /kaggle/working/cast_split.json\n" + open(os.path.join(root, "build/cast_split.json")).read()))
    base = ("STUDY=castgptq MODEL_ID=NHirose/omnivla-finetuned-cast MODEL_REV=7d3744a72cd89218be4d223783f0742819b6c6db "
            "HEAD_STEP=210000 CAST_SPLIT=/kaggle/working/cast_split.json TESTS_ONLY=lang EXPORT_TAG=_cast")
    cells += [("code", run(base + (" SMOKE=1 N_CALIB=4 NO_EXPORT=1" if test.endswith("smoke") else " N_CALIB=64")))]
elif test in ("xmode", "xmode_smoke"):                       # CAST bf16 on img3m/pose5/lelan, then fp16 latency (original)
    sm = " SMOKE=1" if test.endswith("smoke") else ""
    cells += [("code", run(f"STUDY=castxm MODEL_ID=NHirose/omnivla-finetuned-cast "
                           f"MODEL_REV=7d3744a72cd89218be4d223783f0742819b6c6db HEAD_STEP=210000 TESTS_ONLY=img3m,pose5,lelan{sm}")),
              ("code", "!rm -rf /tmp/omnivla-finetuned-cast && df -h /tmp | tail -1"),
              ("code", run(f"STUDY=t4lat TESTS_ONLY=img3m,pose5,lelan{sm}"))]
elif test == "visgptqx":                                     # vision-only GPTQ export (~0.2 GPU-hour)
    cells += [("code", run("STUDY=visgptqx N_CALIB=64"))]
elif test == "ablation":
    cells += [("code", run("STUDY=lelan_p75 PRUNE=spatial75 TESTS_ONLY=lelan")),
              ("code", run("STUDY=gptqx PRUNE=spatial75 N_CALIB=64 TESTS_ONLY=lelan BLIND=none,blank,shuffled,lang_shuffled "
                           "BLIND_TESTS=lelan EVAL_PRUNES=none,spatial50,spatial75 OUT_SUFFIX=_ablation"))]
else:
    extra = "img3m,pose5,pose20" if test == "lang" else "seq"
    cells += [("code", run(f"STUDY={test} TESTS_ONLY={test}")),
              ("code", run(f"STUDY=gptqx PRUNE=spatial75 N_CALIB=64 TESTS_ONLY={test},{extra} "
                           f"BLIND=none,blank,shuffled,lang_shuffled BLIND_TESTS={test} OUT_SUFFIX=_{test}"))]
if test not in ("vismarlin_smoke", "visgptqx", "castgptq", "castgptq_smoke", "xmode", "xmode_smoke"):                                # the smoke run calibrates on 4 samples: weights differ by design
    cells += [("code", "%cd /kaggle/working\n" + check)]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": [s], **({"execution_count": None, "outputs": []} if t == "code" else {})}
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(f"{kdir}/{slug}.ipynb", "w"), indent=1)
json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [os.environ.get("KAGGLE_DATASET", f"{user}/omnivla-frodobots-frames")]
                              + ([f"{user}/omnivla-lelan-demo-inputs"] if test in ("lelan", "ablation", "promptprune", "vismarlin", "vismarlin_smoke", "xmode", "xmode_smoke") else []), "competition_sources": [],
           "kernel_sources": [f"{user}/omnivla-cast-extract"] if test in ("lang", "castgptq", "castgptq_smoke") else [], "machine_shape": "NvidiaTeslaT4"}, open(f"{kdir}/kernel-metadata.json", "w"), indent=1)
PY

if [ $DRY = 1 ]; then
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print('[lang_eval] id', m['id'], '| dataset', m['dataset_sources'], '| kernel_sources', m['kernel_sources'], '| gpu', m['enable_gpu'])" "$K/kernel-metadata.json"
  echo "[lang_eval] dry run: would run: $KAGGLE kernels push -p $K"; exit 0
fi
PUSH=$("$KAGGLE" kernels push -p "$K" 2>&1 | tail -1); echo "$PUSH"
echo "$PUSH" | grep -qi "successfully pushed" || { echo "[lang_eval] push refused - stopping"; exit 3; }
start=$(date +%s)
while true; do                                  # bounded wait, stops on any terminal status
  sleep 60
  s=$("$KAGGLE" kernels status "$USER_K/$SLUG" 2>&1 || true)
  case "$s" in *RUNNING*|*QUEUED*) ;; *) break;; esac
  [ $(( $(date +%s) - start )) -ge $(( TIMEOUT_MIN * 60 )) ] && { echo "[lang_eval] timeout after $TIMEOUT_MIN min: $s"; exit 2; }
done
echo "[lang_eval] $s"
"$KAGGLE" kernels output "$USER_K/$SLUG" -p "$OUT" > "$OUT/download.log" 2>&1
grep -h "\[LANG\]\|\[C\]\|Error\|Traceback" "$OUT"/*.log 2>/dev/null | tail -15 || true
case "$s" in *COMPLETE*) ;; *) echo "[lang_eval] kernel did not complete"; exit 4;; esac
case "$LANG_TEST" in vismarlin_smoke|visgptqx|castgptq*|xmode*) ;; *) false;; esac || grep -qh "layers identical" "$OUT"/*.log || { echo "[lang_eval] no weight check in the log: failed"; exit 5; }
case "$LANG_TEST" in lang) A=lang_analysis.py;; vismarlin*) A=vismarlin_analysis.py;; castgptq*) A=cast_gptq_analysis.py;; *) A=lelan_analysis.py;; esac
echo "[lang_eval] outputs in $OUT ($(cd "$OUT" && ls compress_*.npz | tr '\n' ' ')); next: eval/analysis/$A"
