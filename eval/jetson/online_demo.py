# Online-footage demo on the Jetson: the deployed runtime replays each segment of eval/demo/online_demo_plan.json as a
# live camera. The next prediction always takes the frame the video has reached when the previous one finished
# (video time advances by the measured duration of each call, preprocessing included), so the prediction rate is the
# model's real rate. Before a segment's clock starts, its instruction (or goal) is run twice on the first frame, as a
# robot would warm up for a new goal (the second call captures that prompt's CUDA graph).
#   ./deploy/launch.sh eval/jetson/online_demo.py FRAMES_DIR   (frames from eval/demo/online_frames.py) -> results/online_demo.npz
import json, os, sys, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "deploy"))
from omnivla_deploy import OmniVLADeploy

FR = sys.argv[1]
P = json.load(open(os.path.join(ROOT, "eval", "demo", "online_demo_plan.json")))
FM = json.load(open(os.path.join(FR, "frames_meta.json")))
m = OmniVLADeploy(os.path.join(ROOT, "deploy"))
m.warmup(modes=(4, 6, 7))
out = {}
for s in P["segments"]:
    sid, fps, n = s["id"], FM[s["id"]]["fps"], FM[s["id"]]["n_frames"]
    ld = lambda i: Image.open(os.path.join(FR, "model", sid, f"{i:05d}.jpg")).convert("RGB")
    kw = dict(mode=s["mode"])
    if s["mode"] == 7:
        kw["lang"] = s["lang"]
    elif s["mode"] == 6:
        kw["goal_image"] = Image.open(os.path.join(FR, "goal", sid + ".jpg")).convert("RGB")
    for _ in range(2):
        m.predict(ld(0), **kw)
    t_video, rows = 0.0, []
    while True:
        i = int(t_video * fps)
        if i >= n:
            break
        img = ld(i)                                           # decode = the camera callback's work, counted below
        t = time.perf_counter(); o = m.predict(img, **kw); dt = time.perf_counter() - t
        v, w = m.actions_to_cmd(o["actions"])
        rows.append((i, dt, o["t_fwd"], v, w)); out[f"act__{sid}__{len(rows) - 1}"] = o["actions"]
        t_video += dt
    r = np.array(rows)
    out[f"rows__{sid}"] = r                                   # frame index, call s, forward s, v, w
    print(f"[ONLINE] {sid} (mode {s['mode']}): {len(rows)} predictions over {n / fps:.1f} s of video | call median "
          f"{np.median(r[:, 1]) * 1000:.0f} ms, max {r[:, 1].max() * 1000:.0f} ms", flush=True)
out["meta"] = np.array(json.dumps(dict(vision=m.vision_kernel, weights=os.path.basename(m.weights_dir), head_step=m.head_step)))
np.savez(os.path.join(ROOT, "results", "online_demo.npz"), **out)
print("[ONLINE] DONE", flush=True)
