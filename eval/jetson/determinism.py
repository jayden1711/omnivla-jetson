# Determinism of the image-goal path across fresh processes. One process = reference_check.py's protocol: warm-up (modes
# 6, 4, 7, 8), one prediction on the first frame (a goal-cache miss, saved as "miss__<frame>"), then all 10 reference
# frames in mode 6 (the first is a goal-cache hit). Run it N times with determinism.sh; determinism_compare.py checks that
# every run is bit-identical to the first and to tests/reference.
#   ./deploy/launch.sh eval/jetson/determinism.py RUN_ID      -> results/determinism/run_<id>.npz
import os, sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "deploy"); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

REF = os.path.join(D, "tests", "reference"); R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
ld = lambda s, f: Image.open(os.path.join(REF, s, f + ".jpg")).convert("RGB")
m = OmniVLADeploy(D, verbose=False)
m.warmup(modes=(6, 4, 7, 8))
out = {f"miss__{frames[0]}": m.predict(ld("frames", frames[0]), mode=6, goal_image=ld("goals", frames[0]))["actions"]}
for f in frames:
    out[f"act_m6__{f}"] = m.predict(ld("frames", f), mode=6, goal_image=ld("goals", f))["actions"]
os.makedirs(os.path.join(ROOT, "results", "determinism"), exist_ok=True)
np.savez(os.path.join(ROOT, "results", "determinism", f"run_{int(sys.argv[1]):03d}.npz"), **out)
print(f"[DET] run {sys.argv[1]} saved", flush=True)
