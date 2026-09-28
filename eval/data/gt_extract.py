# Ground-truth future trajectories for the test frames, in OmniVLA's action frame, reproducing OmniVLA's FrodoBots
# training pipeline: catglossop/frodo_dataset converter (10 Hz clips + forward/backward EKF; see fetch_gt_src.sh), then
# frodobots_dataset.py: 8 steps x 0.3 s, [x_fwd, y_left] / 0.25 m, cos/sin dyaw; pose goal 17 steps (5.1 s) ahead.
# Output: results/gt_frodobots.npz, results/gt_frames.csv (per-frame flags).
# Run from the repo root: python eval/data/gt_extract.py (fetches gt_src with fetch_gt_src.sh on first use)
import ast, io, os, re, subprocess, sys, types
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import pandas as pd
from PIL import Image
from remotezip import RemoteZip

GT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_src")
if not all(os.path.exists(os.path.join(GT_SRC, f)) for f in ("convert_to_hf.py", "filtering.py", "interpolation_utils.py")):
    subprocess.run([os.path.join(os.path.dirname(GT_SRC), "fetch_gt_src.sh")], check=True)

# frodo_dataset's `data` module imports torch, zarr and others; only TimestampedData is needed
@dataclass
class TimestampedData:
    data: Any
    timestamps: np.ndarray
shim = types.ModuleType("data"); shim.TimestampedData = TimestampedData; shim.TimestampedDataDict = Dict[str, TimestampedData]
sys.modules["data"] = shim
sys.path.insert(0, GT_SRC)
from filtering import filter_data_preprocess, filter_data_postprocess      # noqa: E402
from interpolation_utils import cut_interp_clips  # noqa: E402
from utm import from_latlon  # noqa: E402


def _from_convert_to_hf(names):
    """Load functions from frodo_dataset's convert_to_hf.py by source: importing the module needs lerobot, torch etc."""
    tree = ast.parse(open(os.path.join(GT_SRC, "convert_to_hf.py")).read())
    ns = dict(np=np, pd=pd, re=re, from_latlon=from_latlon, Dict=Dict, TimestampedData=TimestampedData,
              TimestampedDataDict=Dict[str, TimestampedData])
    for node in ast.walk(tree):                     # max_diff_fn is nested in main()
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "convert_to_hf.py", "exec"), ns)
    assert all(n in ns for n in names), names
    return ns


_U = _from_convert_to_hf({"fast_parse_string_to_numpy", "get_control_data", "get_imu_data", "get_gps_data", "max_diff_fn"})
LOWDIM_KEYS = ("action", "observation.wheel_rpm", "observation.magnetometer", "observation.latitude",
               "observation.longitude", "observation.utm_position", "observation.utm_zone_number", "observation.utm_zone_letter")


def lowdim(dfs):
    d = {**_U["get_control_data"](dfs), **_U["get_imu_data"](dfs), **_U["get_gps_data"](dfs)}
    return {k: d[k] for k in LOWDIM_KEYS}          # accelerometer and gyroscope are not used by the filters


def clips_with_abs_time(all_data):
    """frodo_dataset's cut_interp_clips plus an 'abs_t' channel (absolute time; exact under linear interpolation, and
    with the action timestamps it adds no gap cuts) so that frames can be located in the clips."""
    a = all_data["action"]
    t = np.asarray(a.timestamps, np.float64)
    return cut_interp_clips({**all_data, "abs_t": TimestampedData(t, a.timestamps)}, 0, max_diff_fn=_U["max_diff_fn"])


RAW = "data_raw"
SPACING = 0.25          # frodobots_dataset.py:586
STEP = 3                # action_spacing=3 at 10 Hz -> 0.3 s per chunk step
H = 8                   # action_horizon
GOAL_STEPS = 17         # pose goal 17 steps x 0.3 s = 5.1 s ahead
man = pd.read_csv("results/frames_manifest.csv")

def fetch(part, ride, patterns):
    """download the ride's small CSVs (gps, imu, front camera timestamps, control) if missing"""
    url = f"https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/output_rides_{part}.zip"
    base = f"{RAW}/output_rides_{part}/{ride}"
    have = os.listdir(base) if os.path.isdir(base) else []
    if all(any(re.match(p, f) for f in have) for p in patterns):
        return
    with RemoteZip(url) as z:
        for i in z.infolist():
            m = re.match(rf"output_rides_{part}/{ride}/([^/]+\.csv)$", i.filename)
            if m and any(re.match(p, m[1]) for p in patterns):
                os.makedirs(os.path.dirname(f"{RAW}/{i.filename}"), exist_ok=True)
                z.extract(i.filename, RAW)

def csv(part, ride, prefix):
    base = f"{RAW}/output_rides_{part}/{ride}"
    f = [x for x in os.listdir(base) if x.startswith(prefix) and x.endswith(".csv")]
    assert len(f) == 1, (ride, prefix, f)
    return pd.read_csv(f"{base}/{f[0]}")

def rel_pose(p0, th0, p1, th1):
    """frodobots_dataset.to_local_coords_yaw -> [x, y, cos dyaw, sin dyaw] (non-flipped branch)"""
    c, s = np.cos(th0), np.sin(th0)
    d = p1 - p0
    x, y = d[0] * c + d[1] * s, -d[0] * s + d[1] * c
    return np.array([x, y, np.cos(th1 - th0), np.sin(th1 - th0)])

def seg_start(name):
    s = re.search(r"_video_(\d{17})\.ts$", name)[1]          # YYYYmmddHHMMSS + milliseconds, UTC
    return pd.Timestamp(f"{s[:8]}T{s[8:14]}.{s[14:]}", tz="UTC").timestamp()

def pin_frame_time(part, ride, seg, approx_off, saved_jpg, cam_ts):
    """match the saved 224x224 frame to the exact 20 fps frame near approx_off; snap to the logged camera time"""
    path = f"{RAW}/output_rides_{part}/{ride}/recordings/{seg}"
    w, h = 1024, 576
    last = int((approx_off + 1.5) * 20) + 1
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(last), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    n = len(p.stdout) // (w * h * 3)
    ref = np.asarray(Image.open(saved_jpg).convert("RGB"), np.float32)
    lo, hi = max(0, int((approx_off - 1.5) * 20)), min(n, int((approx_off + 1.5) * 20) + 1)
    best = (1e18, -1)
    for i in range(lo, hi):
        a = np.frombuffer(p.stdout, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3)
        im = np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC), np.float32)
        best = min(best, (float(((im - ref) ** 2).mean()), i))
    t_est = seg_start(seg) + best[1] / 20.0
    t_cam = cam_ts[np.argmin(np.abs(cam_ts - t_est))]
    return t_cam, best[0], best[1], n

def main():
    rows, G = [], {}
    for (part, ride), fr in man.groupby(["source_part", "episode"]):
        fetch(part, ride, [r"gps_data_.*", r"imu_data_.*", r"front_camera_timestamps_.*", r"control_data_.*"])
        dfs = {k: csv(part, ride, k) for k in ("control_data", "imu_data", "gps_data", "front_camera_timestamps")}
        raw = lowdim(dfs)
        gps_t, gps_u = raw["observation.utm_position"].timestamps, raw["observation.utm_position"].data
        print(f"[GT] {ride}: control {raw['action'].timestamps[[0, -1]]} gps {gps_t[[0, -1]]} "
              f"imu {raw['observation.magnetometer'].timestamps[[0, -1]]}", flush=True)
        clips = [filter_data_postprocess(c) for c in clips_with_abs_time(filter_data_preprocess(raw))]
        cam_ts = dfs["front_camera_timestamps"]["timestamp"].values.astype(np.float64)
        for _, r in fr.iterrows():
            key = r.filename[:-4]
            t, mse, idx, nfr = pin_frame_time(part, ride, r.segment, float(r.offset_in_segment_s) - 0.5,
                                              f"data_frames/frames/{r.filename}", cam_ts)
            row = dict(frame=key, part=part, episode=ride, t_frame=t, match_mse=mse, t_shift_s=t - pd.Timestamp(r.timestamp_utc).timestamp())
            hit = None
            for ci, c in enumerate(clips):
                j = int(np.argmin(np.abs(c["abs_t"] - t)))
                if abs(c["abs_t"][j] - t) <= 0.051:
                    hit = (ci, j); break
            row["in_training_clip"] = hit is not None
            if hit:
                c, j = clips[hit[0]], hit[1]
                need = j + GOAL_STEPS * STEP
                row["window_complete"] = need < len(c["abs_t"])
                if row["window_complete"]:
                    P, TH = c["observation.filtered_position"], c["observation.filtered_heading"]
                    acts = np.stack([rel_pose(P[j], TH[j], P[j + k * STEP], TH[j + k * STEP]) for k in range(1, H + 1)])
                    acts[:, :2] /= SPACING
                    goal = rel_pose(P[j], TH[j], P[need], TH[need]); goal[:2] /= SPACING
                    span = c["abs_t"][j + H * STEP] - c["abs_t"][j]
                    gspan = c["abs_t"][need] - c["abs_t"][j]
                    # reliability flags
                    wmask = (gps_t >= t - 1) & (gps_t <= t + gspan + 1)
                    gw, gtw = gps_u[wmask], gps_t[wmask]
                    if len(gw) > 1:
                        v = np.linalg.norm(np.diff(gw, axis=0), axis=1) / np.maximum(np.diff(gtw), 1e-3)
                        row["gps_max_speed"] = float(v.max()); row["gps_max_gap_s"] = float(np.diff(gtw).max())
                    else:
                        row["gps_max_speed"] = np.nan; row["gps_max_gap_s"] = np.inf
                    raw_rel = c["observation.utm_position"][j:need + 1] - c["observation.utm_position"][j]
                    f_rel = P[j:need + 1] - P[j]
                    row.update(gt_disp_8=float(np.linalg.norm(acts[-1, :2]) * SPACING), window_span_s=span, goal_span_s=gspan,
                               ekf_vs_gps_m=float(np.linalg.norm(raw_rel - f_rel, axis=1).max()),
                               gt_turn_deg=float(np.degrees(np.arctan2(acts[-1, 3], acts[-1, 2]))))
                    G[f"act__{key}"] = acts.astype(np.float32); G[f"goal__{key}"] = goal.astype(np.float32)
            rows.append(row)
            print(f"[GT] {key}: pinned t {t:.3f} (shift {row['t_shift_s']:+.2f} s, mse {mse:.1f}) clip {hit is not None} "
                  f"complete {row.get('window_complete')} disp {row.get('gt_disp_8', float('nan')):.2f} m", flush=True)

    df = pd.DataFrame(rows)
    df["flag_not_in_clip"] = ~df.in_training_clip | ~df.window_complete.fillna(False).astype(bool)
    df["flag_stationary"] = df.gt_disp_8 < 0.25                              # < 1 action unit moved in 2.4 s
    df["flag_paused"] = (df.goal_span_s - GOAL_STEPS * STEP * 0.1) > 0.25   # removed stationary samples inside the window
    df["flag_gps_jump"] = df.gps_max_speed > 4.0                             # implied speed > 4 m/s between fixes
    df["flag_gps_gap"] = df.gps_max_gap_s > 3.0
    df["flag_ekf_gps"] = df.ekf_vs_gps_m > 8.0
    fl = [c for c in df.columns if c.startswith("flag_")]
    df["reliable"] = ~df[fl].fillna(True).any(axis=1)
    df.to_csv("results/gt_frames.csv", index=False)
    np.savez("results/gt_frodobots.npz", **G)
    print(df[fl + ["reliable"]].sum().to_string())
    print(f"[GT] saved results/gt_frodobots.npz ({len(G)//2} frames with GT), results/gt_frames.csv; reliable {df.reliable.sum()}/{len(df)}")


if __name__ == "__main__":
    main()
