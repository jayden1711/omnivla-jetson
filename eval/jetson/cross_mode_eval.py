# Cross-mode test of a weights folder with the deployed runtime on the Jetson: image goal (img3m, 100 frames, mode 6),
# 5 s pose goal (pose5, mode 4) and object goal (LeLaN, 210 frames, mode 7), each with the blank / shuffled image
# controls, plus the other object's prompt for LeLaN. Runtime defaults (75% pruning in modes 4/6, none in 7).
# Output keys follow build/kaggle_compress.py ("<tag>[-<control>]__<test>__<frame>", (1, 8, 4)), so the Mac analyses
# (eval/analysis/tests_analysis.py, lelan_analysis.py, cross_mode_analysis.py) read them unchanged.
#   OMNIVLA_WEIGHTS=weights_cast ./deploy/launch.sh eval/jetson/cross_mode_eval.py TAG DATA_DIR   -> results/cross_<TAG>.npz
# DATA_DIR: gt_tests.csv, gt_frodobots.npz, shuffle_map.csv, lelan_lang.npz, goal_img3m/ (frames: $OMNIVLA_ROOT/frames)
import csv, json, os, sys, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "deploy"))
from omnivla_deploy import OmniVLADeploy

TAG, DD = sys.argv[1], sys.argv[2]
FR = os.path.join(ROOT, "frames")
rows = [r for r in csv.DictReader(open(f"{DD}/gt_tests.csv")) if r["reliable_path"] == "True"]
gz = np.load(f"{DD}/gt_frodobots.npz")
smap = {r["frame"]: r["shuffled_from"] for r in csv.DictReader(open(f"{DD}/shuffle_map.csv"))}
LZ = np.load(f"{DD}/lelan_lang.npz"); LM = json.loads(str(LZ["meta"]))
BLACK = Image.new("RGB", (224, 224), (0, 0, 0))
ld = lambda f: Image.open(f"{FR}/{f}.jpg").convert("RGB")
TESTS = {
    "img3m": (6, sorted(r["frame"] for r in rows if r["img_ok"] == "True")),
    "pose5": (4, sorted(r["frame"] for r in rows if f"goal__{r['frame']}" in gz.files)),
    "lelan": (7, sorted(LM)),
}
only = os.environ.get("TESTS_ONLY")
m = OmniVLADeploy(os.path.join(ROOT, "deploy"))
m.warmup(modes=(4, 6, 7))
out, lat, t0 = {}, {}, time.time()
for test, (mode, frames) in TESTS.items():
    if only and test not in only.split(","):
        continue
    for variant in (None, "blank", "shuffled") + (("lang_shuffled",) if test == "lelan" else ()):
        for f in frames:
            if test == "lelan":
                img = BLACK if variant == "blank" else Image.fromarray(LZ[f"img__{LM[f]['img_shuffle'] if variant == 'shuffled' else f}"])
                obj = LM[f]["distractor" if variant == "lang_shuffled" else "target"]
                o = m.predict(img, mode=7, lang="move toward " + obj)
            else:
                img = BLACK if variant == "blank" else ld(smap[f] if variant == "shuffled" else f)
                kw = dict(goal_image=Image.open(f"{DD}/goal_img3m/{f}.jpg").convert("RGB")) if mode == 6 else dict(goal_pose=gz[f"goal__{f}"].astype(np.float64))
                o = m.predict(img, mode=mode, **kw)
            out[f"{TAG}{'' if variant is None else '-' + variant}__{test}__{f}"] = o["actions"][None]
            if variant is None:
                lat.setdefault(test, []).append(o["t_fwd"])
        print(f"[XMODE] {TAG} {test} {variant or 'real'}: {len(frames)} frames | {time.time() - t0:.0f} s", flush=True)
meta = dict(tag=TAG, weights=os.environ.get("OMNIVLA_WEIGHTS", "weights"), vision=m.vision_kernel, head_step=m.head_step,
            latency_ms_median={k: float(np.median(v)) * 1000 for k, v in lat.items()})
out[f"meta__{TAG}"] = np.array(json.dumps(meta))
np.savez(os.path.join(ROOT, "results", f"cross_{TAG}.npz"), **out)
print(f"[XMODE] DONE {json.dumps(meta)}", flush=True)
