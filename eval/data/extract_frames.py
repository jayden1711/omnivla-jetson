# Build a ~200-frame 224x224 test set from FrodoBots-2K (CC BY-SA 4.0) using HTTP range requests only
# (front-camera HLS segments + CSVs of a few rides per part; no full-part downloads).
# Output: data_frames/frames/*.jpg, results/frames_manifest.csv, results/frames_contact_sheet.jpg.
# Run from the repo root: python eval/data/extract_frames.py
import csv, io, os, random, re, subprocess, datetime as dt
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from remotezip import RemoteZip
from concurrent.futures import ThreadPoolExecutor

BASE = "https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/output_rides_{}.zip"
LICENSE = "CC BY-SA 4.0 (FrodoBots-2K, FrodoBots Lab)"
PARTS = {0: 70, 22: 70, 23: 60}     # part -> target frames (rides of ~10 frames each -> ~20 rides, 200 frames)
MIN_PER_RIDE = 5                     # skip low-diversity rides (e.g. robot mostly stationary)
PER_RIDE = 10
MIN_FRONT_SEGS, SEGS_PER_RIDE = 12, 12   # rides >= ~3.3 min; download 12 segments (~16.5 s, ~3 MB each) spread over the ride
RAW, OUT = "data_raw", "data_frames/frames"
os.makedirs(OUT, exist_ok=True); os.makedirs("results", exist_ok=True)
rng = random.Random(0)

def seg_start(name):   # ..._video_20240504065503993.ts -> UTC epoch seconds
    s = re.search(r"_video_(\d{17})\.ts$", name)[1]
    return dt.datetime.strptime(s, "%Y%m%d%H%M%S%f").replace(tzinfo=dt.timezone.utc).timestamp() if len(s) == 20 else \
        dt.datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp() + int(s[14:]) / 1000

def decode_1fps(path):
    """yield (offset_s, 224x224 RGB PIL) at 1 fps (frames at t = 0.5, 1.5, ...)."""
    w, h = 1024, 576
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "fps=1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    buf = p.stdout
    n = len(buf) // (w * h * 3)
    for k in range(n):
        a = np.frombuffer(buf, np.uint8, w * h * 3, k * w * h * 3).reshape(h, w, 3)
        yield k + 0.5, Image.fromarray(a).resize((224, 224), Image.BICUBIC)

def thumb(img):
    return np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), np.float32) / 255

rows = []
for part, target in PARTS.items():
    url = BASE.format(part)
    with RemoteZip(url) as z:
        by_ride = defaultdict(list)
        for i in z.infolist():
            m = re.match(rf"output_rides_{part}/(ride_[^/]+)/(.+)$", i.filename)
            if m and not i.is_dir():
                by_ride[m[1]].append(i.filename)
        ok = []
        for ride, files in by_ride.items():
            front = sorted(f for f in files if re.search(r"uid_s_1000__uid_e_video_\d+\.ts$", f))
            ctrl = [f for f in files if "/control_data_" in f]
            if MIN_FRONT_SEGS <= len(front) and ctrl:
                ok.append((ride, front, ctrl[0]))
        print(f"[FRAMES] part {part}: {len(by_ride)} rides, {len(ok)} with >= {MIN_FRONT_SEGS} front segments")
        rng.shuffle(ok)
        got = 0
        for ride, front, ctrl in ok:
            if got >= target:
                break
            first_seg = front[0]
            front = [front[i] for i in sorted(set(np.linspace(0, len(front) - 1, SEGS_PER_RIDE).round().astype(int)))]
            need = [f for f in front + [ctrl] if not os.path.exists(os.path.join(RAW, f))]
            def fetch(f):
                with RemoteZip(url) as zz:
                    zz.extract(f, RAW)
            for f in need:                              # zipfile's makedirs races under threads
                os.makedirs(os.path.join(RAW, os.path.dirname(f)), exist_ok=True)
            with ThreadPoolExecutor(6) as ex:
                list(ex.map(fetch, need))
            c = np.genfromtxt(os.path.join(RAW, ctrl), delimiter=",", names=True)
            c = np.atleast_1d(c)
            # candidates
            cand = []
            t0 = seg_start(first_seg)
            for f in front:
                s0 = seg_start(f)
                try:
                    for off, img in decode_1fps(os.path.join(RAW, f)):
                        t = s0 + off
                        if t - t0 < 3:                      # skip start-up
                            continue
                        g = np.asarray(img.convert("L"), np.float32)
                        if g.mean() < 8 or g.std() < 4:      # black / blank / corrupt
                            continue
                        j = np.abs(c["timestamp"] - t)
                        near = j < 0.75
                        ang = float(np.mean(c["angular"][near])) if near.any() else float("nan")
                        lin = float(np.mean(c["linear"][near])) if near.any() else float("nan")
                        cand.append(dict(t=t, seg=os.path.basename(f), off=off, img=img, th=thumb(img),
                                         ang=ang, lin=lin, bright=float(g.mean())))
                except subprocess.CalledProcessError as e:
                    print(f"[FRAMES] decode failed {f}: {e.stderr[:200]}")
            if len(cand) < PER_RIDE:
                print(f"[FRAMES] {ride}: only {len(cand)} candidates, skipping"); continue
            # one frame per downloaded segment, far from already-picked frames of this ride
            picked = []
            for seg in [os.path.basename(f) for f in front]:
                if len(picked) == PER_RIDE:
                    break
                inbin = [x for x in cand if x["seg"] == seg]
                if not inbin:
                    continue
                def score(x):
                    return min([np.abs(x["th"] - p["th"]).mean() for p in picked], default=1.0)
                best = max(inbin, key=score)
                if score(best) < 0.02:          # near-duplicate of an earlier pick (e.g. robot stationary)
                    continue
                picked.append(best)
            if len(picked) < MIN_PER_RIDE:
                print(f"[FRAMES] part {part} {ride}: only {len(picked)} distinct frames, skipping ride"); continue
            picked = picked[:target - got]
            got += len(picked)
            for x in picked:
                rel = x["t"] - t0
                fn = f"fb_p{part:02d}_{ride.split('_')[1]}_t{int(round(rel)):05d}.jpg"
                x["img"].save(os.path.join(OUT, fn), "JPEG", quality=95)
                turning = (not np.isnan(x["ang"])) and abs(x["ang"]) > 0.3
                rows.append(dict(filename=fn, source_part=part, episode=ride, segment=x["seg"], offset_in_segment_s=x["off"],
                                 t_from_episode_start_s=round(rel, 1),
                                 timestamp_utc=dt.datetime.fromtimestamp(x["t"], dt.timezone.utc).isoformat(timespec="milliseconds"),
                                 control_linear=round(x["lin"], 3), control_angular=round(x["ang"], 3),
                                 turning=int(turning), brightness=round(x["bright"], 1), license=LICENSE,
                                 source_url=url))
            print(f"[FRAMES] part {part} {ride}: {len(cand)} candidates -> {len(picked)} frames")

with open("results/frames_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
eps = {r["episode"] for r in rows}
print(f"[FRAMES] total {len(rows)} frames from {len(eps)} episodes; turning {sum(r['turning'] for r in rows)}; "
      f"brightness min/median/max {min(r['brightness'] for r in rows):.0f}/"
      f"{np.median([r['brightness'] for r in rows]):.0f}/{max(r['brightness'] for r in rows):.0f}")

# contact sheet: 40 frames sampled evenly across the manifest (8 x 5)
idx = np.linspace(0, len(rows) - 1, 40).round().astype(int)
S, cols = 224, 8
sheet = Image.new("RGB", (cols * S, 5 * (S + 18)), (252, 252, 251))
d = ImageDraw.Draw(sheet)
for k, i in enumerate(idx):
    r = rows[i]
    x, y = (k % cols) * S, (k // cols) * (S + 18)
    sheet.paste(Image.open(os.path.join(OUT, r["filename"])), (x, y))
    d.text((x + 3, y + S + 3), f"{r['episode'].split('_')[1]} t={r['t_from_episode_start_s']:.0f}s"
           + (" TURN" if r["turning"] else ""), fill=(40, 40, 40))
sheet.save("results/frames_contact_sheet.jpg", quality=90)
print("[FRAMES] saved results/frames_manifest.csv, results/frames_contact_sheet.jpg")
