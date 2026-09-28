# Reproduce every chunk of an image-goal ROS replay (results/ros_replay6_chunks.json) with the deploy runtime.
# A reuse chunk depends only on (kv_src frame, frame, goal), so it is rebuilt as predict(kv_src) then predict(frame).
# Expect bit-identical actions. Run in the OmniVLA clone: $OMNIVLA_ROOT/deploy/launch.sh verify_replay6.py [MAX_CHUNKS] [CHUNKS_JSON]
import json, os, sys
import numpy as np
from PIL import Image
ROOT = os.environ.get("OMNIVLA_ROOT", "/mnt/nvme/omnivla")
sys.path.insert(0, f"{ROOT}/deploy")
from omnivla_deploy import OmniVLADeploy

MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
C = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "results/ros_replay6_chunks.json"))[:MAXC]
import csv
CLIPS = sorted({r["clip"] for r in csv.DictReader(open("seq_frames.csv"))})
m = OmniVLADeploy(f"{ROOT}/deploy", goal_refresh=10 ** 6)
m.warmup(modes=(6,))
img = lambda f: Image.open(f"seq/{f}.jpg").convert("RGB")
gimg = lambda gid: Image.open(f"seq/goals/{CLIPS[gid - G0]}.jpg").convert("RGB")   # the goal the node held (goal_id),
n_race = 0                                                 # not the frame's clip: a clip's last frame can meet the next goal
d, uid = {"full": [], "reuse": []}, 0
G0 = min((c["goal_id"] for c in C if "goal_id" in c), default=0)
n_missing = 0
for c in C:
    if not all(k in c for k in ("goal_id", "kv_reused", "kv_src")):   # not produced by a mode-6 call: cannot be right
        n_missing += 1; d["full"].append(float("inf")); continue
    f, g = c["frame_id"], gimg(c["goal_id"]); uid += 1
    n_race += CLIPS[c["goal_id"] - G0] != f.rsplit("_", 1)[0]
    if c["kv_reused"]:
        m.predict(img(c["kv_src"]), mode=6, goal_image=g, goal_id=f"v{uid}")
        o = m.predict(img(f), mode=6, goal_image=g, goal_id=f"v{uid}"); assert o["cache"]["kv_reused"]
        d["reuse"].append(float(np.abs(o["actions"] - np.array(c["actions"])).max()))
    else:
        o = m.predict(img(f), mode=6, goal_image=g, goal_id=f"v{uid}"); assert not o["cache"]["kv_reused"]
        d["full"].append(float(np.abs(o["actions"] - np.array(c["actions"])).max()))
for k, v in d.items():
    print(f"[VR6] {k} chunks: {len(v)} | max|d| {max(v) if v else float('nan'):.3g} | exact {sum(x == 0 for x in v)}/{len(v)}", flush=True)
print(f"[VR6] {n_race} chunks paired a clip's frame with the next clip's goal (camera/goal topic race; reproduced as the node ran them)", flush=True)
print(f"[VR6] {n_missing} chunks lacked the goal-cache fields (counted as mismatches)", flush=True)
ok = bool(C) and all(x == 0 for v in d.values() for x in v)
print(f"[VR6] {'ALL CHUNKS REPRODUCED BIT-EXACTLY' if ok else 'MISMATCH'}", flush=True)
