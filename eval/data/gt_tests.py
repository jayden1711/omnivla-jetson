# Perception-dependent driving tests for the test frames:
#  img3m : image goal = the frame where the rover's actual (EKF) path first reaches 3.0 m (capped at 6 s of motion)
#  pose20: pose goal 67 steps (20.1 s) ahead; pose5: the 5.1 s pose goal from gt_frodobots.npz
# Also writes a shuffled-image map (each frame -> a frame from another ride) for the blind-model control.
# Output: results/gt_tests.npz, results/gt_tests.csv, results/shuffle_map.csv, data_frames/goal_img3m/*.jpg. Run from the repo root.
import os, re
import numpy as np
import pandas as pd
from remotezip import RemoteZip
import gt_extract as G
from heldout_extract import frames_at, BASE

TARGET_M, CAP_STEPS, FAR = 3.0, 60, 67          # 3 m of travel, cap 60 samples (6 s), far goal 67 steps x 0.3 s
os.makedirs("data_frames/goal_img3m", exist_ok=True)
gtf = pd.read_csv("results/gt_frames.csv")
gtf = gtf[gtf.in_training_clip & gtf.window_complete.fillna(False).astype(bool)]
LIST = {}
def ride_segments(part, ride):
    if part not in LIST:
        with RemoteZip(BASE.format(part)) as z:
            LIST[part] = [i.filename for i in z.infolist() if "uid_s_1000__uid_e_video_" in i.filename and i.filename.endswith(".ts")]
    return sorted((G.seg_start(os.path.basename(f)), f) for f in LIST[part] if f"/{ride}/" in f)

rows, arr = [], {}
for (part, ride), fr in gtf.groupby(["part", "episode"]):
    dfs = {k: G.csv(part, ride, k) for k in ("control_data", "imu_data", "gps_data", "front_camera_timestamps")}
    raw = G.lowdim(dfs)
    clips = [G.filter_data_postprocess(c) for c in G.clips_with_abs_time(G.filter_data_preprocess(raw))]
    segs = ride_segments(part, ride)
    for _, r in fr.iterrows():
        c = j = None
        for cc in clips:
            jj = int(np.argmin(np.abs(cc["abs_t"] - r.t_frame)))
            if abs(cc["abs_t"][jj] - r.t_frame) <= 0.051:
                c, j = cc, jj; break
        P, TH, AT = c["observation.filtered_position"], c["observation.filtered_heading"], c["abs_t"]
        row = dict(frame=r.frame, part=part, episode=ride,
                   paused_8=bool((AT[j + G.H * G.STEP] - AT[j]) - G.H * G.STEP * 0.1 > 0.25),
                   stationary_8=bool(r.gt_disp_8 < 0.25), ekf_bad=bool(r.flag_ekf_gps))
        # ---- image goal ~3 m of travel ----
        seglen = np.linalg.norm(np.diff(P[j:], axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seglen)])
        k = int(np.searchsorted(cum, TARGET_M)); k = min(k, CAP_STEPS, len(cum) - 1)
        T = AT[j + k]
        row.update(img_lookahead_s=float(T - r.t_frame), img_motion_s=k * 0.1, img_path_m=float(cum[k]),
                   img_straight_m=float(np.linalg.norm(P[j + k] - P[j])), img_capped=bool(cum[k] < TARGET_M - 0.05))
        seg = [f for s0, f in segs if s0 <= T]
        ok_img = False
        if seg:
            f = seg[-1]
            if not os.path.exists(f"{G.RAW}/{f}"):
                with RemoteZip(BASE.format(part)) as z:
                    z.extract(f, G.RAW)
            got = frames_at(f"{G.RAW}/{f}", [T - G.seg_start(os.path.basename(f))])
            img = list(got.values())[0][0]
            if img is not None:
                img.save(f"data_frames/goal_img3m/{r.frame}.jpg", quality=95); ok_img = True
        row["img_ok"] = ok_img
        g = G.rel_pose(P[j], TH[j], P[j + k], TH[j + k]); g[:2] /= G.SPACING
        arr[f"imggoalpose__{r.frame}"] = g.astype(np.float32)          # oracle pose of the goal frame (baselines only)
        # ---- far pose goal ----
        if j + FAR * G.STEP < len(AT):
            gf = G.rel_pose(P[j], TH[j], P[j + FAR * G.STEP], TH[j + FAR * G.STEP]); gf[:2] /= G.SPACING
            arr[f"goal20__{r.frame}"] = gf.astype(np.float32)
            row.update(far_ok=True, far_dist_m=float(np.linalg.norm(gf[:2]) * G.SPACING),
                       far_span_s=float(AT[j + FAR * G.STEP] - AT[j]))
        else:
            row.update(far_ok=False)
        # current speed (past information only) for the straight-ahead baseline
        jp = max(j - 5, 0)
        row["speed_now_mps"] = float(np.linalg.norm(P[j] - P[jp]) / max(AT[j] - AT[jp], 1e-3))
        rows.append(row)
        print(f"[TESTS] {r.frame}: img goal {row['img_path_m']:.2f} m path / {row['img_straight_m']:.2f} m straight, "
              f"{row['img_lookahead_s']:.1f} s (ok {ok_img}) | far goal {row.get('far_dist_m', float('nan')):.1f} m", flush=True)

df = pd.DataFrame(rows)
df["reliable_path"] = ~(df.paused_8 | df.stationary_8 | df.ekf_bad)
df.to_csv("results/gt_tests.csv", index=False)
np.savez("results/gt_tests.npz", **arr)
# shuffled image map: each frame gets a frame from a different ride (deterministic)
rng = np.random.default_rng(7)
frames = df.frame.tolist(); rides = dict(zip(df.frame, df.episode))
perm = frames[:]
for _ in range(1000):
    rng.shuffle(perm)
    if all(rides[a] != rides[b] for a, b in zip(frames, perm)):
        break
pd.DataFrame(dict(frame=frames, shuffled_from=perm)).to_csv("results/shuffle_map.csv", index=False)
v = df[df.reliable_path]
print(f"[TESTS] {len(df)} frames, reliable path {len(v)}; image goal ok {int(v.img_ok.sum())} "
      f"(path {v.img_path_m.median():.2f} m median, straight {v.img_straight_m.median():.2f} m, lookahead {v.img_lookahead_s.median():.1f} s, "
      f"capped {int(v.img_capped.sum())}); far goal ok {int(v.far_ok.sum())} (median {v.far_dist_m.median():.1f} m)")
