# Demo video and GIF: OmniVLA-7B image-goal predictions with this repo's deployed int4 weights, drawn over FrodoBots-2K
# driving clips (CC BY-SA 4.0, FrodoBots Lab). Uses saved predictions only (no model is run here): by default the
# Kaggle run of eval/kaggle/lang_eval.sh with LANG_TEST=lelan (deployed GPTQ int4 LLM + HQQ4 vision + 75% token pruning,
# fp16 kernels on a T4; matches the Jetson to about 0.003 action units), on the sequential clips (eval/data/seq_extract.py;
# rides outside the test and calibration sets), one prediction every 21 frames (1.05 s, the Jetson's image-goal latency).
# The inset is a top-down view in the robot frame at the prediction's input frame: predicted path vs the path the
# human driver took (0.25 m per action unit, FrodoBots' spacing). Clips are chosen by the largest heading change in
# the driven path, not by prediction accuracy.
#   python eval/demo/render_demo.py PRED [N_CLIPS]    (from a folder with results/seq_frames.csv, results/seq.npz,
#   data_seq/goals/). PRED = compress_gptqx_lelan.npz (keys gptqx_pc4__seq__<frame>), or a folder of per-frame npz files
#   from eval/jetson/jetson_cache.py (then set DEMO_WEIGHTS_NOTE). Needs ffmpeg, remotezip. Writes docs/media/demo.{mp4,gif}.
import glob, os, re, subprocess, sys, tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from remotezip import RemoteZip

PRED, NCLIP = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = os.environ.get("DEMO_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "media"))
BASE = "https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/output_rides_{}.zip"
RAW, W, H, FPS, STRIDE, SP = "data_raw", 1024, 576, 20, 21, 0.25
WEIGHTS_NOTE = os.environ.get("DEMO_WEIGHTS_NOTE", "deployed int4 weights, simulated on a GPU")

def seg_start(name):
    s = re.search(r"_video_(\d{17})\.ts$", name)[1]
    return pd.Timestamp(f"{s[:8]}T{s[8:14]}.{s[14:]}", tz="UTC").timestamp()

S = pd.read_csv("results/seq_frames.csv"); GT = np.load("results/seq.npz")
if PRED.endswith(".npz"):
    _z = np.load(PRED, allow_pickle=True)
    PR = {k.split("__")[2]: np.asarray(_z[k]).reshape(8, 4) for k in _z.files if k.startswith("gptqx_pc4__seq__")}
else:
    PR = {os.path.basename(p)[:-4]: np.load(p)["act"] for p in glob.glob(f"{PRED}/*.npz") if not p.endswith("_summary.npz")}
print(f"[DEMO] predictions: {len(PR)}", flush=True)
turn = {}
for c, g in S.groupby("clip"):
    if not all(f in PR for f in g[g.idx % STRIDE == 0].frame) or not g.reliable.all():
        continue
    h = [np.degrees(np.arctan2(GT[f"act__{f}"][-1, 3], GT[f"act__{f}"][-1, 2])) for f in g.frame]
    turn[c] = (np.abs(h).max(), g.episode.iloc[0])
clips, rides = [], set()
for c, (t, ride) in sorted(turn.items(), key=lambda x: -x[1][0]):
    if ride not in rides:
        clips.append(c); rides.add(ride)
    if len(clips) == NCLIP:
        break
if not clips:
    sys.exit("[DEMO] 0 clips with predictions for every frame - check PRED")
print(f"[DEMO] clips (largest turns): {clips}", flush=True)

def segment_frames(part, ride, t_first, n, first_224):
    """native-resolution frames of the clip, from the ride's video segment (range requests); aligned to the saved
    224x224 first frame (search +-3 frames)"""
    with RemoteZip(BASE.format(part)) as z:
        segs = sorted((seg_start(os.path.basename(i.filename)), i.filename) for i in z.infolist()
                      if f"/{ride}/" in i.filename and re.search(r"uid_s_1000__uid_e_video_\d+\.ts$", i.filename))
        s0, f = [x for x in segs if x[0] <= t_first + 1e-3][-1]
        if not os.path.exists(f"{RAW}/{f}"):
            z.extract(f, RAW)
    i0 = int(round((t_first - s0) * FPS))
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{RAW}/{f}", "-frames:v", str(i0 + n + 4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    v = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
    ref = np.asarray(first_224, np.float32)
    err = {j: float(((np.asarray(Image.fromarray(v[j]).resize((224, 224), Image.BICUBIC), np.float32) - ref) ** 2).mean())
           for j in range(max(0, i0 - 3), min(len(v), i0 + 4))}
    j = min(err, key=err.get)
    assert err[j] < 30, f"clip start not found in {f} (best mse {err[j]:.1f})"
    return v[j:j + n]

FONT = next((ImageFont.truetype(p, 18) for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
             if os.path.exists(p)), ImageFont.load_default())
SMALL = FONT.font_variant(size=14) if hasattr(FONT, "font_variant") else FONT
BLUE, WHITE, GOLD = (77, 163, 255), (240, 240, 240), (255, 196, 61)

def inset(pred, gt):
    """top-down view in the robot frame at the prediction's input frame, x forward (up), y left; 2.5 m x 2.5 m"""
    S_, M = 260, 2.5
    im = Image.new("RGBA", (S_, S_), (20, 22, 26, 235)); d = ImageDraw.Draw(im)
    px = lambda x, y: (S_ / 2 - y * SP / M * S_, S_ - 18 - x * SP / M * (S_ - 30))
    for m in (0.5, 1.0, 1.5, 2.0):
        yy = px(m / SP, 0)[1]; d.line([(8, yy), (S_ - 8, yy)], fill=(70, 72, 78), width=1); d.text((10, yy - 16), f"{m:g} m", font=SMALL, fill=(150, 150, 155))
    d.text((S_ - 10 - d.textlength("top view", font=SMALL), 6), "top view", font=SMALL, fill=(150, 150, 155))
    for path, col, w in ((gt, WHITE, 3), (pred, BLUE, 4)):
        pts = [px(0, 0)] + [px(x, y) for x, y in path[:, :2]]
        d.line(pts, fill=col, width=w, joint="curve")
        for q in pts[1:]:
            d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=col)
    x0, y0 = px(0, 0); d.polygon([(x0, y0 - 9), (x0 - 7, y0 + 6), (x0 + 7, y0 + 6)], fill=GOLD)
    return im

tmp = tempfile.mkdtemp(); n = 0
for c in clips:
    g = S[S["clip"] == c].sort_values("idx")
    fr = segment_frames(int(g.part.iloc[0]), g.episode.iloc[0], float(g.t_frame.iloc[0]), len(g),
                        Image.open(f"data_seq/frames/{g.frame.iloc[0]}.jpg").convert("RGB"))
    goal = Image.open(f"data_seq/goals/{c}.jpg").convert("RGB").resize((150, 150))
    for k, (_, r) in enumerate(g.iterrows()):
        kp = (int(r.idx) // STRIDE) * STRIDE
        fp = g[g.idx == kp].frame.iloc[0]
        im = Image.fromarray(fr[min(k, len(fr) - 1)]).convert("RGBA")
        im.alpha_composite(inset(PR[fp], GT[f"act__{fp}"]), (W - 276, 16))
        box = Image.new("RGBA", (166, 188), (20, 22, 26, 235)); box.paste(goal, (8, 30))
        ImageDraw.Draw(box).text((8, 6), "goal image (224x224)", font=SMALL, fill=WHITE); im.alpha_composite(box, (16, 16))
        d = ImageDraw.Draw(im)
        d.rectangle([0, H - 84, W, H], fill=(20, 22, 26, 225))
        d.text((16, H - 80), "OmniVLA-7B int4 (this repo): image goal, one prediction per 1.05 s (its Jetson latency)", font=FONT, fill=WHITE)
        d.text((16, H - 54), f"blue: predicted path   white: path the driver took   open loop, {WEIGHTS_NOTE}", font=SMALL, fill=(200, 200, 205))
        d.text((16, H - 30), "Video: FrodoBots-2K (FrodoBots Lab), CC BY-SA 4.0. This rendering is CC BY-SA 4.0.",
               font=SMALL, fill=(200, 200, 205))
        im.convert("RGB").save(f"{tmp}/f{n:05d}.png"); n += 1
    print(f"[DEMO] {c}: {len(g)} frames", flush=True)
os.makedirs(OUT, exist_ok=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/f%05d.png", "-vf", "scale=960:-2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", f"{OUT}/demo.mp4"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/f%05d.png", "-vf",
                "fps=7,scale=448:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                f"{OUT}/demo.gif"], check=True)
for f in ("demo.mp4", "demo.gif"):
    print(f"[DEMO] {OUT}/{f}: {os.path.getsize(f'{OUT}/{f}') / 1e6:.2f} MB")
