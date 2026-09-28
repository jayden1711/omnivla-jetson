# Held-out FrodoBots calibration/training samples (never from a test ride) + image goals for the test frames.
# Per ride: 8 front-camera segments, samples 2/6/10 s into each so the image goal (5.1 s later) is in the same segment.
# Ground truth from gt_extract.py. Output: data_heldout/{frames,goals_img}/*.jpg, results/heldout.npz,
# results/heldout_frames.csv, data_frames/goal_images/*.jpg. Run from the repo root: python eval/data/heldout_extract.py
import os, random, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from remotezip import RemoteZip
import gt_extract as G

N_RIDES, SEGS, OFFS = 60, 8, (2.0, 6.0, 10.0)
PARTS = (0, 21, 22, 23)
BASE = "https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/output_rides_{}.zip"
os.makedirs("data_heldout/frames", exist_ok=True); os.makedirs("data_heldout/goals_img", exist_ok=True)
os.makedirs("data_frames/goal_images", exist_ok=True)
test_rides = set(pd.read_csv("results/frames_manifest.csv").episode)
rng = random.Random(1)

def frame_at(path, t_rel):
    """224x224 bicubic squash (same as the test set) of the frame nearest t_rel seconds into the segment (20 fps)"""
    w, h = 1024, 576
    i = int(round(t_rel * 20))
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(i + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    n = len(p.stdout) // (w * h * 3)
    if n <= i:
        return None, None
    a = np.frombuffer(p.stdout, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3)
    return Image.fromarray(a).resize((224, 224), Image.BICUBIC), i

def frames_at(path, t_rels):
    """decode the segment once; return {t_rel: (PIL 224x224, frame index) or (None, None)}"""
    w, h = 1024, 576
    idx = {t: int(round(t * 20)) for t in t_rels}
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(max(idx.values()) + 1), "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    n = len(p.stdout) // (w * h * 3)
    out = {}
    for t, i in idx.items():
        if i < n:
            a = np.frombuffer(p.stdout, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3)
            out[t] = (Image.fromarray(a).resize((224, 224), Image.BICUBIC), i)
        else:
            out[t] = (None, None)
    return out

def ride_list():
    out = []
    for part in PARTS:
        with RemoteZip(BASE.format(part)) as z:
            by = defaultdict(list)
            for inf in z.infolist():
                m = re.match(rf"output_rides_{part}/(ride_[^/]+)/(.+)$", inf.filename)
                if m and not inf.is_dir():
                    by[m[1]].append(inf.filename)
        for ride, files in by.items():
            front = sorted(f for f in files if re.search(r"uid_s_1000__uid_e_video_\d+\.ts$", f))
            if ride not in test_rides and len(front) >= 12 and any("/control_data_" in f for f in files):
                out.append((part, ride, front))
    rng.shuffle(out)
    return out

def process_ride(part, ride, front):
    url = BASE.format(part)
    segs = [front[i] for i in sorted(set(np.linspace(0, len(front) - 1, SEGS).round().astype(int)))]
    need = [f for f in segs if not os.path.exists(f"{G.RAW}/{f}")]
    for f in need:
        os.makedirs(os.path.dirname(f"{G.RAW}/{f}"), exist_ok=True)
    def dl(f):
        with RemoteZip(url) as zz:
            zz.extract(f, G.RAW)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(dl, need))
    G.fetch(part, ride, [r"gps_data_.*", r"imu_data_.*", r"front_camera_timestamps_.*", r"control_data_.*"])
    dfs = {k: G.csv(part, ride, k) for k in ("control_data", "imu_data", "gps_data", "front_camera_timestamps")}
    raw = G.lowdim(dfs)
    clips = [G.filter_data_postprocess(c) for c in G.clips_with_abs_time(G.filter_data_preprocess(raw))]
    cam = dfs["front_camera_timestamps"]["timestamp"].values.astype(np.float64)
    gps_t, gps_u = raw["observation.utm_position"].timestamps, raw["observation.utm_position"].data
    rows, arr = [], {}
    for seg in segs:
        s0 = G.seg_start(os.path.basename(seg))
        try:
            fr = frames_at(f"{G.RAW}/{seg}", [o for o in OFFS] + [o + 5.1 for o in OFFS])
        except subprocess.CalledProcessError:
            continue
        for off in OFFS:
            img, i = fr[off]
            gimg, gi = fr[off + 5.1]
            if img is None or gimg is None:
                continue
            t = cam[np.argmin(np.abs(cam - (s0 + i / 20)))]
            hit = None
            for c in clips:
                j = int(np.argmin(np.abs(c["abs_t"] - t)))
                if abs(c["abs_t"][j] - t) <= 0.051 and j + G.GOAL_STEPS * G.STEP < len(c["abs_t"]):
                    hit = (c, j); break
            if hit is None:
                continue
            c, j = hit
            P, TH = c["observation.filtered_position"], c["observation.filtered_heading"]
            acts = np.stack([G.rel_pose(P[j], TH[j], P[j + k * G.STEP], TH[j + k * G.STEP]) for k in range(1, G.H + 1)]); acts[:, :2] /= G.SPACING
            nd = j + G.GOAL_STEPS * G.STEP
            goal = G.rel_pose(P[j], TH[j], P[nd], TH[nd]); goal[:2] /= G.SPACING
            gspan = c["abs_t"][nd] - c["abs_t"][j]
            w = (gps_t >= t - 1) & (gps_t <= t + gspan + 1)
            v = np.linalg.norm(np.diff(gps_u[w], axis=0), axis=1) / np.maximum(np.diff(gps_t[w]), 1e-3) if w.sum() > 1 else np.array([np.nan])
            disp = float(np.linalg.norm(acts[-1, :2]) * G.SPACING)
            ok = disp >= 0.25 and (gspan - G.GOAL_STEPS * G.STEP * 0.1) <= 0.25 and np.nanmax(v) <= 4.0 and \
                (np.diff(gps_t[w]).max() if w.sum() > 1 else 9) <= 3.0 and \
                np.linalg.norm((c["observation.utm_position"][j:nd + 1] - c["observation.utm_position"][j]) - (P[j:nd + 1] - P[j]), axis=1).max() <= 8.0
            key = f"ho_p{part:02d}_{ride.split('_')[1]}_{os.path.basename(seg)[-21:-3]}_{int(off)}"
            if ok:
                img.save(f"data_heldout/frames/{key}.jpg", quality=95); gimg.save(f"data_heldout/goals_img/{key}.jpg", quality=95)
                arr[f"act__{key}"] = acts.astype(np.float32); arr[f"goal__{key}"] = goal.astype(np.float32)
            rows.append(dict(frame=key, part=part, episode=ride, t_frame=t, reliable=bool(ok), disp_m=disp))
    for f in segs:                                                     # keep disk use bounded
        try: os.remove(f"{G.RAW}/{f}")
        except OSError: pass
    return rows, arr

def test_goal_images():
    """image goal for each test frame = the front frame 5.1 s later (downloads the next segment if needed)"""
    gtf = pd.read_csv("results/gt_frames.csv"); man = pd.read_csv("results/frames_manifest.csv").set_index(man_key := "filename")
    done = 0
    for _, r in gtf[gtf.in_training_clip & gtf.window_complete.fillna(False).astype(bool)].iterrows():
        out = f"data_frames/goal_images/{r.frame}.jpg"
        if os.path.exists(out):
            done += 1; continue
        m = man.loc[r.frame + ".jpg"]
        rec = f"{G.RAW}/output_rides_{r.part}/{r.episode}/recordings"
        segs = sorted(x for x in os.listdir(rec) if "uid_s_1000__uid_e_video_" in x and x.endswith(".ts")) if os.path.isdir(rec) else []
        seg = m.segment
        t_rel = r.t_frame + 5.1 - G.seg_start(seg)
        path = f"{rec}/{seg}"
        if t_rel > 16.3:                                               # in the next segment: find and download it
            with RemoteZip(BASE.format(r.part)) as z:
                cands = sorted(i.filename for i in z.infolist() if f"/{r.episode}/recordings/" in i.filename
                               and "uid_s_1000__uid_e_video_" in i.filename and i.filename.endswith(".ts"))
                nxt = [f for f in cands if G.seg_start(os.path.basename(f)) > G.seg_start(seg)][0]
                if not os.path.exists(f"{G.RAW}/{nxt}"):
                    z.extract(nxt, G.RAW)
            path = f"{G.RAW}/{nxt}"; t_rel = r.t_frame + 5.1 - G.seg_start(os.path.basename(nxt))
        img, _ = frame_at(path, t_rel)
        if img is not None:
            img.save(out, quality=95); done += 1
    print(f"[HO] test image goals: {done}", flush=True)

if __name__ == "__main__":
    test_goal_images()
    rides = ride_list()
    print(f"[HO] candidate held-out rides: {len(rides)}", flush=True)
    rows, arr = [], {}
    for part, ride, front in rides[:N_RIDES]:
        try:
            r, a = process_ride(part, ride, front)
        except Exception as e:
            print(f"[HO] {ride}: skipped ({type(e).__name__}: {e})", flush=True); continue
        rows += r; arr.update(a)
        print(f"[HO] {ride}: {sum(x['reliable'] for x in r)}/{len(r)} reliable; total {len(arr)//2}", flush=True)
        np.savez("results/heldout.npz", **arr); pd.DataFrame(rows).to_csv("results/heldout_frames.csv", index=False)
    print(f"[HO] done: {len(arr)//2} reliable held-out samples from {len({r['episode'] for r in rows})} rides", flush=True)
