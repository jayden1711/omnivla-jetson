# Frames for the online-footage demo: for each segment of eval/demo/online_demo_plan.json, every frame of the clip
# (native frame rate), center-cropped to a square, as the model input (224x224, JPEG q95: model/<seg>/NNNNN.jpg) and for
# the rendered video (480x480: disp/<seg>/NNNNN.jpg), plus the image goal (goal/<seg>.jpg). Needs ffmpeg.
#   python eval/demo/online_frames.py CLIPS_DIR OUT_DIR
import json, os, subprocess, sys

CLIPS, OUT = sys.argv[1], sys.argv[2]
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_demo_plan.json")))
meta = {}
for s in P["segments"]:
    src = os.path.join(CLIPS, P["clips"][s["clip"]]["file"])
    fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src],
                         capture_output=True, text=True, check=True).stdout.strip()
    num, den = (int(x) for x in fps.split("/"))
    for sub, size in (("model", 224), ("disp", 480)):
        d = os.path.join(OUT, sub, s["id"]); os.makedirs(d, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(s["start"]), "-to", str(s["end"]), "-i", src,
                        "-vf", f"crop='min(iw,ih)':'min(iw,ih)',scale={size}:{size}:flags=area", "-q:v", "2" if sub == "model" else "3",
                        "-start_number", "0", os.path.join(d, "%05d.jpg")], check=True)
    if s.get("goal_time") is not None:
        os.makedirs(os.path.join(OUT, "goal"), exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(s["goal_time"]), "-i", src, "-frames:v", "1",
                        "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=224:224:flags=area", "-q:v", "2",
                        os.path.join(OUT, "goal", s["id"] + ".jpg")], check=True)
    n = len(os.listdir(os.path.join(OUT, "model", s["id"])))
    meta[s["id"]] = dict(fps=num / den, n_frames=n)
    print(f"[FRAMES] {s['id']}: {n} frames at {num / den:.3f} fps", flush=True)
json.dump(meta, open(os.path.join(OUT, "frames_meta.json"), "w"), indent=1)
