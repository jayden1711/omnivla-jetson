# reference_check.py - the installed runtime + weights must reproduce the reference outputs bit-exactly
# (10 frames, pose-goal and image-goal mode). Run: ./launch.sh tools/reference_check.py   (exit 0 = PASS)
import os, subprocess, sys
import numpy as np
from PIL import Image

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

REF = os.path.join(D, "tests", "reference")
R = np.load(os.path.join(REF, "reference.npz"))
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
RECORD_ALL = "--record-all" in sys.argv
LANG = {7: "move toward the bench", 8: "stay on the path"}
modes = [md for md in (6, 4, 7, 8) if md in (4, 6) or RECORD or RECORD_ALL or f"act_m{md}__{frames[0]}" in R.files]
m = OmniVLADeploy(D)
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
    keep = os.path.join(REF, time.strftime("reference_%Y-%m-%d.npz"))
    shutil.copy(os.path.join(REF, "reference.npz"), keep)
    out = {k: R[k] for k in R.files if k.startswith("goal__")}; out.update(new)
    out.update({f"lang__m{md}": np.array(LANG[md]) for md in (7, 8)})
    np.savez(os.path.join(REF, "reference.npz"), **out)
    print(f"[CHECK] re-recorded {len(new)} outputs (modes {modes}); previous file kept as {os.path.basename(keep)}", flush=True)
    sys.exit(0)
print(f"[CHECK] {exact}/{n} predictions bit-identical to the recorded outputs, modes {[md for md in modes if not (RECORD and md in (7, 8))]} "
      f"(max |d| {worst:.3g}); median latency " + ", ".join(f"mode {md} {np.median(v) * 1000:.0f} ms" for md, v in lat.items()), flush=True)
if RECORD:
    if not ok:
        print("[CHECK] not recording language references: modes 4/6 do not match", flush=True)
    else:
        out = dict(R); out.update(new); out.update({f"lang__m{md}": np.array(LANG[md]) for md in (7, 8)})
        np.savez(os.path.join(REF, "reference.npz"), **out)
        print(f"[CHECK] recorded {len(new)} language-mode references into tests/reference/reference.npz", flush=True)
print(f"[CHECK] {'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
