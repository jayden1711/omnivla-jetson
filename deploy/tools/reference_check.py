# reference_check.py - the installed runtime + weights must reproduce the reference outputs bit-exactly
# (10 frames, pose-goal and image-goal mode). Run: ./launch.sh tools/reference_check.py   (exit 0 = PASS)
import os, subprocess, sys
import numpy as np
from PIL import Image

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

REF = os.path.join(D, "tests", "reference")
# reference.npz: default runtime (Marlin vision). reference_hqq4.npz: HQQ4 vision (OMNIVLA_VISION=hqq4, or weights without
# vis_marpc, e.g. the Hugging Face weights before vis_marpc was added): the outputs of the previous default.
# reference_cast.npz: weights built from omnivla-finetuned-cast (OMNIVLA_WEIGHTS=weights_cast); same frames and goals.
m = OmniVLADeploy(D)
_man = os.path.join(m.weights_dir, "BUILD_MANIFEST.json")
IS_CAST = os.path.exists(_man) and "finetuned-cast" in open(_man).read()
REF_FILE = "reference_cast.npz" if IS_CAST else ("reference.npz" if m.vision_kernel == "marlin" else "reference_hqq4.npz")
RECORD_ALL = "--record-all" in sys.argv
R = np.load(os.path.join(REF, REF_FILE if os.path.exists(os.path.join(REF, REF_FILE)) or not RECORD_ALL else "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
missing = [p for f in frames for p in (os.path.join(REF, "frames", f + ".jpg"), os.path.join(REF, "goals", f + ".jpg")) if not os.path.exists(p)]
if missing:
    print(f"[CHECK] {len(missing)} reference images missing: downloading them from FrodoBots-2K", flush=True)
    if subprocess.run([sys.executable, os.path.join(D, "tools", "fetch_reference_images.py")]).returncode != 0:
        print("[CHECK] FAIL: reference images unavailable (see tests/reference/README.md)", flush=True)
        sys.exit(2)
# Language modes (7, 8): checked when tests/reference holds them. `--record-lang` records them (only after modes 4 and 6
# pass in the same run), with fixed prompts, into reference.npz.
RECORD = "--record-lang" in sys.argv
# --record-all: re-record every mode from this runtime (keeps the previous file as reference_<date>.npz). Use only after
# checking that the runtime reproduces the previous runtime bit for bit (see tests/reference/README.md).
if (RECORD or RECORD_ALL) and REF_FILE == "reference_hqq4.npz":
    sys.exit("[CHECK] recording is not supported for the HQQ4 fallback (its references are fixed)")
LANG = {7: "move toward the bench", 8: "stay on the path"}
modes = [md for md in (6, 4, 7, 8) if md in (4, 6) or RECORD or RECORD_ALL or f"act_m{md}__{frames[0]}" in R.files]
m.warmup(modes=tuple(modes))
# The references were recorded by eval/jetson/final_validate.py, which warms up on the first frame itself, so the first
# image-goal frame was a goal-cache hit (goal vision features reused; current image encoded alone). Same protocol here:
# a cache miss encodes both images together, and that kernel shape rounds differently (up to ~0.004 action units).
m.predict(Image.open(os.path.join(REF, "frames", frames[0] + ".jpg")).convert("RGB"), mode=6,
          goal_image=Image.open(os.path.join(REF, "goals", frames[0] + ".jpg")).convert("RGB"))
worst, exact, n, lat, new = 0.0, 0, 0, {md: [] for md in modes}, {}
for mode in modes:
    for f in frames:
        img = Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB")
        kw = dict(goal_image=Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB")) if mode == 6 else dict(goal_pose=R[f"goal__{f}"])
        if mode in (7, 8):
            kw["lang"] = str(R["lang__m%d" % mode]) if f"lang__m{mode}" in R.files else LANG[mode]
        o = m.predict(img, mode=mode, **kw)
        lat[mode].append(o["t_fwd"])
        if RECORD_ALL or (RECORD and mode in (7, 8)):
            new[f"act_m{mode}__{f}"] = o["actions"]; continue
        d = float(np.abs(o["actions"] - R[f"act_m{mode}__{f}"]).max())
        worst, exact, n = max(worst, d), exact + (d == 0), n + 1
        if d:
            print(f"[CHECK] mismatch mode {mode} {f}: max |d| {d:.3g}", flush=True)
ok = exact == n
if RECORD_ALL:
    import shutil, time
    keep = os.path.join(REF, time.strftime(f"{REF_FILE[:-4]}_%Y-%m-%d.npz"))
    if os.path.exists(os.path.join(REF, REF_FILE)):
        shutil.copy(os.path.join(REF, REF_FILE), keep)
    out = {k: R[k] for k in R.files if k.startswith("goal__")}; out.update(new)
    out.update({f"lang__m{md}": np.array(LANG[md]) for md in (7, 8)})
    np.savez(os.path.join(REF, REF_FILE), **out)
    print(f"[CHECK] re-recorded {len(new)} outputs (modes {modes}) into {REF_FILE}"
          + (f"; previous file kept as {os.path.basename(keep)}" if os.path.exists(keep) else ""), flush=True)
    sys.exit(0)
print(f"[CHECK] {REF_FILE}: {exact}/{n} predictions bit-identical to the recorded outputs, modes {[md for md in modes if not (RECORD and md in (7, 8))]} "
      f"(max |d| {worst:.3g}); median latency " + ", ".join(f"mode {md} {np.median(v) * 1000:.0f} ms" for md, v in lat.items()), flush=True)
if RECORD:
    if not ok:
        print("[CHECK] not recording language references: modes 4/6 do not match", flush=True)
    else:
        out = dict(R); out.update(new); out.update({f"lang__m{md}": np.array(LANG[md]) for md in (7, 8)})
        np.savez(os.path.join(REF, REF_FILE), **out)
        print(f"[CHECK] recorded {len(new)} language-mode references into tests/reference/{REF_FILE}", flush=True)
print(f"[CHECK] {'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
