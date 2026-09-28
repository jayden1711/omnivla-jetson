# Sequential test set for temporal-reuse experiments: consecutive frames at the native 20 fps from rides that are
# neither test nor held-out rides. Two 5 s clips per ride (101 frames) with one fixed image goal (the frame at t0+6 s).
# Ground truth as in gt_extract.py. Output: data_seq/frames/*.jpg, data_seq/goals/*.jpg, results/seq.npz,
# results/seq_frames.csv. Run from the repo root: python eval/data/seq_extract.py [N_RIDES]
import os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from remotezip import RemoteZip
import gt_extract as G
import heldout_extract as HX

N_RIDES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CLIP_T0, CLIP_LEN, GOAL_DT = (1.0, 8.0), 5.0, 6.0
os.makedirs("data_seq/frames", exist_ok=True); os.makedirs("data_seq/goals", exist_ok=True)
ho_rides = set(pd.read_csv("results/heldout_frames.csv").episode)

def decode(path, n_max):
    w, h = 1024, 576
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(n_max), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    n = len(p.stdout) // (w * h * 3)
    return [np.frombuffer(p.stdout, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3) for i in range(n)]

def coverage(clips, s0):
    """fraction of the segment's clip frames that fall inside a training clip with a full 2.4 s window, and the
    EKF path length over the two clip windows (the rover must be moving)"""
    inside, length = 0, 0.0
    ts = [s0 + t0 + n / 20 for t0 in CLIP_T0 for n in range(0, int(CLIP_LEN * 20) + 1, 10)]
    for t in ts:
        for c in clips:
            j = int(np.argmin(np.abs(c["abs_t"] - t)))
            if abs(c["abs_t"][j] - t) <= 0.051 and j + G.H * G.STEP < len(c["abs_t"]):
                inside += 1; break
    for t0 in CLIP_T0:
        for c in clips:
            w = (c["abs_t"] >= s0 + t0) & (c["abs_t"] <= s0 + t0 + CLIP_LEN)
            if w.sum() > 1:
                P = c["observation.filtered_position"][w]; length += float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return inside / len(ts), length

def process(part, ride, front):
    from PIL import Image
    G.fetch(part, ride, [r"gps_data_.*", r"imu_data_.*", r"front_camera_timestamps_.*", r"control_data_.*"])
    dfs = {k: G.csv(part, ride, k) for k in ("control_data", "imu_data", "gps_data", "front_camera_timestamps")}
    raw = G.lowdim(dfs)
    clips = [G.filter_data_postprocess(c) for c in G.clips_with_abs_time(G.filter_data_preprocess(raw))]
    if not clips:
        return [], {}
    cand = []
    for f in front:
        cov, length = coverage(clips, G.seg_start(os.path.basename(f)))
        if cov >= 0.9 and length >= 4.0:                       # >= ~0.4 m/s average over the two clips
            cand.append((cov, length, f))
    if not cand:
        return [], {}
    seg = sorted(cand, key=lambda x: (-x[0], abs(front.index(x[2]) - len(front) // 2)))[0][2]   # full coverage, nearest the middle
    if not os.path.exists(f"{G.RAW}/{seg}"):
        os.makedirs(os.path.dirname(f"{G.RAW}/{seg}"), exist_ok=True)
        with RemoteZip(HX.BASE.format(part)) as z:
            z.extract(seg, G.RAW)
    cam = dfs["front_camera_timestamps"]["timestamp"].values.astype(np.float64)
    s0 = G.seg_start(os.path.basename(seg))
    fr = decode(f"{G.RAW}/{seg}", int(20 * (max(CLIP_T0) + GOAL_DT)) + 2)
    rows, arr = [], {}
    for ci, t0 in enumerate(CLIP_T0):
        gi = int(round((t0 + GOAL_DT) * 20))
        if gi >= len(fr):
            continue
        cid = f"sq_p{part:02d}_{ride.split('_')[1]}_{ci}"
        Image.fromarray(fr[gi]).resize((224, 224), Image.BICUBIC).save(f"data_seq/goals/{cid}.jpg", quality=95)
        t_goal = cam[np.argmin(np.abs(cam - (s0 + gi / 20)))]
        for n in range(int(CLIP_LEN * 20) + 1):
            i = int(round(t0 * 20)) + n
            Image.fromarray(fr[i]).resize((224, 224), Image.BICUBIC).save(f"data_seq/frames/{cid}_{n:03d}.jpg", quality=95)
            t = cam[np.argmin(np.abs(cam - (s0 + i / 20)))]
            hit = None
            for c in clips:
                j = int(np.argmin(np.abs(c["abs_t"] - t)))
                if abs(c["abs_t"][j] - t) <= 0.051 and j + G.H * G.STEP < len(c["abs_t"]):
                    hit = (c, j); break
            ok, disp = False, np.nan
            if hit is not None:
                c, j = hit
                P, TH = c["observation.filtered_position"], c["observation.filtered_heading"]
                acts = np.stack([G.rel_pose(P[j], TH[j], P[j + k * G.STEP], TH[j + k * G.STEP]) for k in range(1, G.H + 1)])
                acts[:, :2] /= G.SPACING
                span = c["abs_t"][j + G.H * G.STEP] - c["abs_t"][j]
                disp = float(np.linalg.norm(acts[-1, :2]) * G.SPACING)
                ekf_gps = np.linalg.norm((c["observation.utm_position"][j:j + G.H * G.STEP + 1] - c["observation.utm_position"][j])
                                         - (P[j:j + G.H * G.STEP + 1] - P[j]), axis=1).max()
                ok = disp >= 0.25 and abs(span - G.H * G.STEP * 0.1) <= 0.25 and ekf_gps <= 8.0
                arr[f"act__{cid}_{n:03d}"] = acts.astype(np.float32)
            rows.append(dict(clip=cid, idx=n, frame=f"{cid}_{n:03d}", part=part, episode=ride, t_frame=t, t_goal=t_goal,
                             goal_ahead_s=t_goal - t, gt=hit is not None, reliable=bool(ok), disp_m=disp))
    try:
        os.remove(f"{G.RAW}/{seg}")
    except OSError:
        pass
    return rows, arr

if __name__ == "__main__":
    rides = [r for r in HX.ride_list() if r[1] not in ho_rides]
    print(f"[SEQ] candidate fresh rides: {len(rides)}", flush=True)
    rows, arr, used = [], {}, 0
    for part, ride, front in rides:
        if used >= N_RIDES:
            break
        try:
            r, a = process(part, ride, front)
        except Exception as e:
            print(f"[SEQ] {ride}: skipped ({type(e).__name__}: {e})", flush=True); continue
        good = [x for x in r if x["reliable"]]
        per_clip = pd.DataFrame(r).groupby("clip").reliable.mean() if r else []
        print(f"[SEQ] {ride}: {len(good)}/{len(r)} reliable; per clip {dict(per_clip.round(2)) if len(r) else {}}", flush=True)
        if len(good) < 60:                                      # mostly paused / outside training clips: drop the ride
            for x in r:
                for p in (f"data_seq/frames/{x['frame']}.jpg",):
                    if os.path.exists(p): os.remove(p)
            for cid in {x["clip"] for x in r}:
                if os.path.exists(f"data_seq/goals/{cid}.jpg"): os.remove(f"data_seq/goals/{cid}.jpg")
            continue
        rows += r; arr.update(a); used += 1
        np.savez("results/seq.npz", **arr); pd.DataFrame(rows).to_csv("results/seq_frames.csv", index=False)
    print(f"[SEQ] done: {used} rides, {len(rows)} frames, {sum(x['reliable'] for x in rows)} with reliable ground truth", flush=True)
