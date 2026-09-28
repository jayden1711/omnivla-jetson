# Goal-cache refresh sweep on the deployed runtime: image-goal mode on the sequential clips (seq_extract.py), every
# STRIDE-th frame, goal_refresh=REFRESH (1 = no K/V reuse; 999 = refresh only on goal change).
# Output: mf_results/dcache{VALIDATE_TAG}_r{REFRESH}_s{STRIDE}/<frame>.npz (actions, latency, reuse flag, cache age).
# Run in the OmniVLA clone: $OMNIVLA_ROOT/deploy/launch.sh deploy_refresh_sweep.py STRIDE REFRESH
import csv, os, sys
import numpy as np
from PIL import Image
ROOT = os.environ.get("OMNIVLA_ROOT", "/mnt/nvme/omnivla")
sys.path.insert(0, f"{ROOT}/deploy")
from omnivla_deploy import OmniVLADeploy

STRIDE, REFRESH = int(sys.argv[1]), int(sys.argv[2])
OUT = f"mf_results/dcache{os.environ.get('VALIDATE_TAG', '')}_r{REFRESH}_s{STRIDE}"; os.makedirs(OUT, exist_ok=True)
m = OmniVLADeploy(f"{ROOT}/deploy", goal_refresh=REFRESH)
m.warmup(modes=(6,))
rows = list(csv.DictReader(open("seq_frames.csv")))
T = []
for c in sorted({r["clip"] for r in rows}):
    fr = [r["frame"] for r in sorted((r for r in rows if r["clip"] == c), key=lambda r: int(r["idx"])) if int(r["idx"]) % STRIDE == 0]
    g = Image.open(f"seq/goals/{c}.jpg").convert("RGB")
    for f in fr:
        o = m.predict(Image.open(f"seq/{f}.jpg").convert("RGB"), mode=6, goal_image=g, goal_id=c)
        cc = o["cache"]
        np.savez(f"{OUT}/{f}.npz", act=o["actions"], t_fwd=o["t_fwd"], kv_reused=cc["kv_reused"], age=cc["age"],
                 age_s=cc["age"] * STRIDE / 20.0, goal_vision_reused=cc["goal_vision_reused"])
        T.append((o["t_fwd"], cc["kv_reused"]))
T = np.array(T, dtype=float)
print(f"[DRS] stride {STRIDE} refresh {REFRESH}: {len(T)} frames | fwd median {np.median(T[:, 0]) * 1000:.0f} ms | "
      f"K/V reused on {T[:, 1].mean() * 100:.0f}% of frames", flush=True)
