# Past frames for OmniVLA-edge, which takes 5 earlier frames plus the current one (context_size 5, context_spacing 3 at
# 10 Hz in OmniVLA's FrodoBots loader -> 0.3 s apart). The 7B model uses only the current frame.
# For every test frame in results/gt_frames.csv: frames at t - 0.3 k (k = 1..5) from the local recordings (data_raw/,
# written by gt_extract.py), same 224x224 bicubic resize and JPEG q95 as the test frames. If the history reaches before
# the first local segment of the ride, the earliest available frame is repeated (listed in the output csv).
# Output: data_frames/history/<frame>_h<k>.jpg, results/history_frames.csv. Run from the repo root.
import glob, os, re, subprocess
import numpy as np
import pandas as pd
from PIL import Image

RAW = "data_raw"                                           # as in gt_extract.py
def seg_start(name):                                       # gt_extract.seg_start
    s = re.search(r"_video_(\d{17})\.ts$", name)[1]
    return pd.Timestamp(f"{s[:8]}T{s[8:14]}.{s[14:]}", tz="UTC").timestamp()

K, DT, FPS, W, H = 5, 0.3, 20, 1024, 576
os.makedirs("data_frames/history", exist_ok=True)
gtf = pd.read_csv("results/gt_frames.csv")

def decode(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)

rows, cache = [], {}
for (part, ride), fr in gtf.groupby(["part", "episode"]):
    segs = sorted((seg_start(os.path.basename(f)), f) for f in glob.glob(f"{RAW}/output_rides_{part}/{ride}/recordings/*.ts")
                  if "uid_s_1000__uid_e_video_" in f)
    for _, r in fr.iterrows():
        filled = []
        for k in range(1, K + 1):
            t = r.t_frame - DT * k
            cand = [(s0, f) for s0, f in segs if s0 <= t]
            if not cand:
                filled.append(k); continue
            s0, f = cand[-1]
            if f not in cache:
                cache.clear(); cache[f] = decode(f)
            i = int(round((t - s0) * FPS))
            if i >= len(cache[f]):
                filled.append(k); continue
            Image.fromarray(cache[f][i]).resize((224, 224), Image.BICUBIC).save(f"data_frames/history/{r.frame}_h{k}.jpg", quality=95)
        for k in filled:                                  # repeat the earliest frame that exists (or the current frame)
            src = next((f"data_frames/history/{r.frame}_h{j}.jpg" for j in range(k - 1, 0, -1)
                        if os.path.exists(f"data_frames/history/{r.frame}_h{j}.jpg")), f"data_frames/frames/{r.frame}.jpg")
            Image.open(src).save(f"data_frames/history/{r.frame}_h{k}.jpg", quality=95)
        rows.append(dict(frame=r.frame, repeated=len(filled)))
pd.DataFrame(rows).to_csv("results/history_frames.csv", index=False)
print(f"[HIST] {len(rows)} frames, {sum(x['repeated'] > 0 for x in rows)} with repeated history")
