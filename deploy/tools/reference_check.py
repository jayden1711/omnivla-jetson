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
m = OmniVLADeploy(D)
m.warmup(modes=(4, 6))
worst, exact, n, lat = 0.0, 0, 0, {4: [], 6: []}
for mode in (6, 4):
    for f in frames:
        img = Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB")
        kw = dict(goal_image=Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB")) if mode == 6 else dict(goal_pose=R[f"goal__{f}"])
        o = m.predict(img, mode=mode, **kw)
        d = float(np.abs(o["actions"] - R[f"act_m{mode}__{f}"]).max())
        worst, exact, n = max(worst, d), exact + (d == 0), n + 1
        lat[mode].append(o["t_fwd"])
ok = exact == n
print(f"[CHECK] {exact}/{n} predictions bit-identical to the validated deployment (max |d| {worst:.3g}); "
      f"median latency pose {np.median(lat[4]) * 1000:.0f} ms, image goal {np.median(lat[6]) * 1000:.0f} ms", flush=True)
print(f"[CHECK] {'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
