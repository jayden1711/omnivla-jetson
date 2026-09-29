# Render the online-footage demo: OmniVLA-7B int4 predictions made ON THE JETSON (eval/jetson/online_demo.py) drawn over
# the clips of eval/demo/online_demo_plan.json (Wikimedia Commons). Each video frame shows the latest prediction that
# had finished by that moment of the replay (predictions start on the frame the video had reached and take their
# measured time), so the display lags the camera exactly as the robot would. The path is a top-down view in the
# model's action units (no camera calibration or scale exists for this footage). Credits at the end.
#   python eval/demo/render_online_demo.py FRAMES_DIR online_demo.npz   -> docs/media/demo_online.{mp4,gif}
import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FR, PRED = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("DEMO_OUT", os.path.join(HERE, "..", "..", "docs", "media"))
P = json.load(open(os.path.join(HERE, "online_demo_plan.json"))); FM = json.load(open(os.path.join(FR, "frames_meta.json")))
Z = np.load(PRED, allow_pickle=True)
FONT = next((ImageFont.truetype(p, 17) for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
             if os.path.exists(p)), ImageFont.load_default())
SMALL, BIG = FONT.font_variant(size=13), FONT.font_variant(size=20)
WHITE, BLUE, GOLD, GREY = (255, 255, 255, 255), (80, 170, 255, 255), (255, 200, 60, 255), (190, 190, 196, 255)
S, W, H, FPS_OUT = 480, 720, 540, 25
LABEL = "OmniVLA-7B int4 on Jetson Orin Nano 8GB, open-loop on online footage (not controlling a robot)"

def inset(pred):
    """top-down view, x forward (up), y left, in action units (0-10)"""
    s, M = 170, 10.0
    im = Image.new("RGBA", (s, s), (20, 22, 26, 230)); d = ImageDraw.Draw(im)
    px = lambda x, y: (s / 2 - y / M * (s - 20), s - 16 - x / M * (s - 34))
    for u in (2, 4, 6, 8, 10):
        yy = px(u, 0)[1]; d.line([(8, yy), (s - 8, yy)], fill=(70, 72, 78), width=1); d.text((10, yy - 15), f"{u}", font=SMALL, fill=GREY)
    d.text((s - 8 - d.textlength("top view (units)", font=SMALL), 4), "top view (units)", font=SMALL, fill=GREY)
    if pred is not None:
        pts = [px(0, 0)] + [px(float(x), float(y)) for x, y in pred[:, :2]]
        d.line(pts, fill=BLUE, width=4, joint="curve")
        for q in pts[1:]:
            d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=BLUE)
    x0, y0 = px(0, 0); d.polygon([(x0, y0 - 9), (x0 - 7, y0 + 6), (x0 + 7, y0 + 6)], fill=GOLD)
    return im

def wrap(d, text, font, width):
    words, lines, cur = [], [], ""
    for w_ in text.split():                                   # break words longer than a line (URLs)
        while d.textlength(w_, font=font) > width:
            k = max(i for i in range(1, len(w_)) if d.textlength(w_[:i], font=font) <= width)
            words.append(w_[:k]); w_ = w_[k:]
        words.append(w_)
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=font) > width and cur:
            lines.append(cur); cur = w_
        else:
            cur = t
    return lines + [cur]

tmp = tempfile.mkdtemp(); n = 0
def emit(im):
    global n
    im.convert("RGB").save(f"{tmp}/f{n:05d}.png"); n += 1
for s in P["segments"]:
    sid, fps = s["id"], FM[s["id"]]["fps"]; nf = FM[s["id"]]["n_frames"]; clip = P["clips"][s["clip"]]
    rows = Z[f"rows__{sid}"]                                     # frame index, call s, forward s, v, w
    starts = np.concatenate([[0.0], np.cumsum(rows[:, 1])[:-1]]); done = starts + rows[:, 1]
    goal = Image.open(os.path.join(FR, "goal", sid + ".jpg")).convert("RGB").resize((110, 110)) if s["mode"] == 6 else None
    for j in range(int(round(nf / fps * FPS_OUT))):
        t = j / FPS_OUT; fi = min(int(t * fps), nf - 1)
        k = int(np.searchsorted(done, t, side="right")) - 1          # latest prediction finished by time t
        im = Image.new("RGBA", (W, H), (14, 15, 18, 255))
        im.paste(Image.open(os.path.join(FR, "disp", sid, f"{fi:05d}.jpg")).convert("RGB"), (0, 30))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 29], fill=(14, 15, 18, 255)); d.text((10, 7), LABEL, font=SMALL, fill=WHITE)
        pred = np.asarray(Z[f"act__{sid}__{k}"]).reshape(8, 4) if k >= 0 else None
        im.alpha_composite(inset(pred), (S - 176, 30 + S - 176))    # over the frame, bottom right
        x = S + 12; y = 42
        d.text((x, y), "goal" if s["mode"] == 6 else "instruction", font=SMALL, fill=GREY); y += 20
        if s["mode"] == 7:
            for line in wrap(d, f"\"{s['lang']}\"", BIG, W - x - 10):
                d.text((x, y), line, font=BIG, fill=WHITE); y += 26
            y += 8
        else:
            d.text((x, y), f"goal image: the clip's frame at {int(s['goal_time']) // 60}:{int(s['goal_time']) % 60:02d}", font=SMALL, fill=WHITE); y += 18
            im.paste(goal, (x, y)); y += 120
        if k >= 0:
            r = rows[k]
            d.text((x, y), "latency on the Jetson", font=SMALL, fill=GREY); y += 18
            d.text((x, y), f"{r[1] * 1000:.0f} ms", font=BIG, fill=GOLD); y += 30
            d.text((x, y), f"prediction {k + 1} of {len(rows)}", font=SMALL, fill=GREY); y += 18
            d.text((x, y), f"from frame {int(r[0])} ({(t - r[0] / fps):.2f} s old)", font=SMALL, fill=GREY); y += 26
            d.text((x, y), "command (OmniVLA controller)", font=SMALL, fill=GREY); y += 18
            d.text((x, y), f"v {r[3]:.2f} m/s  w {r[4]:+.2f} rad/s", font=FONT, fill=WHITE); y += 30
        else:
            d.text((x, y), "first prediction running...", font=SMALL, fill=GREY); y += 26
        d.text((x, H - 96), f"{'image goal' if s['mode'] == 6 else 'language'} mode", font=SMALL, fill=GREY)
        for i_, line in enumerate(wrap(d, f"Video: {clip['author']}, {clip['license']}, Wikimedia Commons (cropped, overlays added)", SMALL, W - x - 10)):
            d.text((x, H - 76 + 16 * i_), line, font=SMALL, fill=GREY)
        emit(im)
    print(f"[RENDER] {sid}: {len(rows)} predictions, {j + 1} output frames", flush=True)
n_video = n
card = Image.new("RGBA", (W, H), (14, 15, 18, 255)); d = ImageDraw.Draw(card); y = 30
d.text((30, y), "Credits", font=BIG, fill=WHITE); y += 40
for c in P["clips"].values():
    for line in [c["file"][:-5], f"by {c['author']}, {c['license']} ({c['license_url']})", c["url"]]:
        for l2 in wrap(d, line, SMALL, W - 60):
            d.text((30, y), l2, font=SMALL, fill=WHITE if line == c["file"][:-5] else GREY); y += 17
    y += 12
for line in ["Changes: excerpts, center-cropped to a square, resized, predictions and text drawn over them.",
             "This video is licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).",
             "Predictions: OmniVLA-7B (Hirose et al., 2025) as GPTQ int4, run on a Jetson Orin Nano 8 GB by github.com/jayden1711/omnivla-jetson."]:
    for l2 in wrap(d, line, SMALL, W - 60):
        d.text((30, y), l2, font=SMALL, fill=GREY); y += 17
for _ in range(FPS_OUT * 6):
    emit(card)
os.makedirs(OUT, exist_ok=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS_OUT), "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "30", "-movflags", "+faststart", f"{OUT}/demo_online.mp4"], check=True)
# GIF: every segment's first 5 s at 5 fps, 432 px wide, then the credits for 3 s (size budget < 5 MB)
sel, j0 = [], 0
for s in P["segments"]:
    m_ = int(round(FM[s["id"]]["n_frames"] / FM[s["id"]]["fps"] * FPS_OUT))
    sel += list(range(j0, j0 + min(m_, 5 * FPS_OUT), FPS_OUT // 5)); j0 += m_
sel += list(range(n_video, n_video + 3 * FPS_OUT, FPS_OUT // 5))
gdir = tempfile.mkdtemp()
for i_, f in enumerate(sel):
    os.link(f"{tmp}/f{f:05d}.png", f"{gdir}/g{i_:05d}.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "5", "-i", f"{gdir}/g%05d.png", "-vf",
                "scale=432:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                f"{OUT}/demo_online.gif"], check=True)
for f in ("demo_online.mp4", "demo_online.gif"):
    print(f"[RENDER] {OUT}/{f}: {os.path.getsize(f'{OUT}/{f}') / 1e6:.2f} MB", flush=True)
