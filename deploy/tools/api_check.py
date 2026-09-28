# api_check.py - the omnivla_jetson package must return exactly what the runtime returns: its waypoints are compared
# with the validated deployment's recorded outputs (tests/reference, the same data as reference_check.py) and its
# command with OmniVLADeploy.actions_to_cmd. Run on the Jetson: ./launch.sh tools/api_check.py   (exit 0 = PASS)
import os, sys
import numpy as np
from PIL import Image

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(D)); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy
from omnivla_jetson import OmniVLAJetson

REF = os.path.join(D, "tests", "reference")
R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
if not all(os.path.exists(os.path.join(REF, s, f + ".jpg")) for s in ("frames", "goals") for f in frames):
    sys.exit("[API] reference images missing: run ./launch.sh tools/reference_check.py first (it downloads them)")
m = OmniVLAJetson(os.path.join(D, os.environ.get("OMNIVLA_WEIGHTS", "weights")))
m.warmup()
n = exact = 0
for f in frames:
    img = os.path.join(REF, "frames", f + ".jpg")
    for mode, p in ((4, m.predict(img, goal_pose=R[f"goal__{f}"])),
                    (6, m.predict(img, goal_image=Image.open(os.path.join(REF, "goals", f + ".jpg"))))):
        ref = R[f"act_m{mode}__{f}"]
        ok = p.mode == mode and np.array_equal(p.waypoints, ref) and p.command == tuple(float(x) for x in OmniVLADeploy.actions_to_cmd(ref))
        n, exact = n + 1, exact + ok
        if not ok:
            print(f"[API] {f} mode {mode}: max |d| {np.abs(p.waypoints - ref).max():.3g}", flush=True)
print(f"[API] {exact}/{n} predictions identical to the validated deployment (waypoints and command)")
print(f"[API] {'PASS' if exact == n else 'FAIL'}")
sys.exit(0 if exact == n else 1)
