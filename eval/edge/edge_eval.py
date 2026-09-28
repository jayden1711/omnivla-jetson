# OmniVLA-edge on the same driving tests as the 7B model: img3m (image goal), pose5 / pose20 (pose goal) and lang (CAST
# language goal), with the same blind controls. Preprocessing follows OmniVLA's inference/run_omnivla_edge.py: 5 past
# frames + the current one at 96x96, the current frame at 224x224 for CLIP, a black dummy satellite image, no mask.
# FrodoBots tests get the real past frames (eval/data/history_extract.py, 0.3 s apart as in training); "nohist" repeats
# the current frame instead (what run_omnivla_edge.py does). CAST samples start an episode, so the current frame is
# repeated, as OmniVLA's CAST loader does. Controls: blank = all frames black; shuffled = another sample's frames;
# lang_shuffled = another sample's instruction.
#   OMNIVLA_SRC=<OmniVLA clone> python eval/edge/edge_eval.py <omnivla-edge.pth> <cast_lang.npz> [<lelan_lang.npz>]
# (from the repo root). EDGE_TESTS=frodobots,lang,lelan selects tests (default: all given). LeLaN object goal: prompt =
# the object phrase, as run_omnivla_edge.py's sample; current frame repeated as history; controls as the 7B run
# (lang_shuffled = the other object's phrase on the same image).
# Output: results/edge_tests.npz (updated in place), keys "<variant>__<test>__<frame>" -> (1, 8, 4) actions.
# Accuracy only: latency and memory on the Jetson are in results/edge_jetson.md.
import json, os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

SRC = os.environ["OMNIVLA_SRC"]
sys.path.insert(0, os.path.join(SRC, "inference")); sys.path.insert(0, SRC)
import clip
import inspect, textwrap
import model_omnivla_edge as ME
from utils_policy import load_model, transform_images_map, transform_images_PIL_mask

# CPU shim: OmniVLA_edge.forward moves masks with .to(obs_img.get_device()), which is -1 on CPU. Same code with .device.
_src = textwrap.dedent(inspect.getsource(ME.OmniVLA_edge.forward))
assert _src.count("obs_img.get_device()") == 1
exec(compile(_src.replace("obs_img.get_device()", "obs_img.device"), ME.__file__, "exec"), ME.__dict__, _ns := {})
ME.OmniVLA_edge.forward = _ns["forward"]

CKPT, CAST = sys.argv[1], sys.argv[2]
LELAN = sys.argv[3] if len(sys.argv) > 3 else None
OUT = "results/edge_tests.npz"
KNOWN = ("frodobots", "lang", "lelan")                      # frodobots = img3m + pose5 + pose20; lang = CAST; lelan = LeLaN
RUN = {t.strip() for t in os.environ.get("EDGE_TESTS", "frodobots,lang" + (",lelan" if LELAN else "")).split(",") if t.strip()}
if RUN - set(KNOWN):
    sys.exit(f"[EDGE] unknown test(s) in EDGE_TESTS: {sorted(RUN - set(KNOWN))}; known: {KNOWN}")
if "lelan" in RUN and not LELAN:
    sys.exit("[EDGE] EDGE_TESTS includes lelan but no lelan_lang.npz was given (third argument)")

def run_test(name, keys, one):
    """runs one(k) for every sample; fails loudly on 0 samples or if any sample is missing from the results"""
    keys = list(keys)
    if not keys:
        sys.exit(f"[EDGE] {name}: 0 samples - refusing to report results for an empty test")
    print(f"[EDGE] {name}: running {len(keys)} samples", flush=True)
    for k in keys:
        one(k)
    done = sum(f"edge__{name}__{k}" in res for k in keys)
    if done != len(keys):
        sys.exit(f"[EDGE] {name}: only {done}/{len(keys)} samples have results")
    print(f"[EDGE] {name}: done, {done} samples", flush=True)
torch.manual_seed(0)
dev = torch.device("cpu")
params = dict(model_type="omnivla-edge", len_traj_pred=8, learn_angle=True, context_size=5, obs_encoder="efficientnet-b0",
              encoding_size=256, obs_encoding_size=1024, goal_encoding_size=1024, late_fusion=False, mha_num_attention_heads=4,
              mha_num_attention_layers=4, mha_ff_dim_factor=4, clip_type="ViT-B/32")        # run_omnivla_edge.py
model, text_encoder, _ = load_model(CKPT, params, dev)
model, text_encoder = model.to(dev).eval(), text_encoder.to(dev).eval()
M96, M224 = np.ones((96, 96, 3), np.float32), np.ones((224, 224, 3), np.float32)
SAT = transform_images_map(Image.new("RGB", (352, 352), (0, 0, 0)))
BLACK = Image.new("RGB", (224, 224), (0, 0, 0))

@torch.no_grad()
def predict(frames, mode, goal_pose=None, goal_img=None, lang="xxxx"):
    """frames: 6 PIL images, oldest first, the last one current. mode: 4 pose, 6 image, 7 language."""
    cur = frames[-1]
    obs = transform_images_PIL_mask([f.resize((96, 96)) for f in frames], M96)
    obs_cur = torch.split(obs, 3, dim=1)[-1]
    big = transform_images_PIL_mask(cur.resize((224, 224)), M224)
    maps = torch.cat((SAT, SAT, obs_cur), axis=1)
    gimg = transform_images_PIL_mask((goal_img or cur).resize((96, 96)), M96)
    gp = torch.as_tensor(np.asarray(goal_pose if goal_pose is not None else [1.0 / 0.1, -10.0 / 0.1, 0.0, -1.0]),
                         dtype=torch.float32)[None]                                           # sample goal, masked out unless mode 4
    feat = text_encoder.encode_text(clip.tokenize(lang, truncate=True))
    act, _, _ = model(obs, gp, maps, gimg, torch.tensor([mode]), feat, big)
    return act.float().numpy()

# ---- FrodoBots tests (same frames and ground truth as eval/analysis/tests_analysis.py) ----
gz, tn = np.load("results/gt_frodobots.npz"), np.load("results/gt_tests.npz")
tdf = pd.read_csv("results/gt_tests.csv"); rel = tdf[tdf.reliable_path]
smap = dict(pd.read_csv("results/shuffle_map.csv").values)
TESTS = {"img3m": (6, sorted(rel[rel.img_ok].frame)),
         "pose5": (4, sorted(f for f in rel.frame if f"goal__{f}" in gz.files)),
         "pose20": (4, sorted(f for f in rel[rel.far_ok].frame if f"goal20__{f}" in tn.files))}
img = lambda f: Image.open(f"data_frames/frames/{f}.jpg").convert("RGB")
hist = lambda f: [Image.open(f"data_frames/history/{f}_h{k}.jpg").convert("RGB") for k in range(5, 0, -1)] + [img(f)]
res = dict(np.load(OUT)) if os.path.exists(OUT) else {}
def frodo(test, mode, f):
    goal = {"pose5": gz.get(f"goal__{f}"), "pose20": tn.get(f"goal20__{f}"), "img3m": None}[test]
    gimg = Image.open(f"data_frames/goal_img3m/{f}.jpg").convert("RGB") if mode == 6 else None
    kw = dict(goal_pose=goal, goal_img=gimg)
    res[f"edge__{test}__{f}"] = predict(hist(f), mode, **kw)
    res[f"edge-nohist__{test}__{f}"] = predict([img(f)] * 6, mode, **kw)
    res[f"edge-blank__{test}__{f}"] = predict([BLACK] * 6, mode, **kw)
    res[f"edge-shuffled__{test}__{f}"] = predict(hist(smap[f]), mode, **kw)
if "frodobots" in RUN:
    for test, (mode, frames) in TESTS.items():
        run_test(test, frames, lambda f, test=test, mode=mode: frodo(test, mode, f))
else:
    print("[EDGE] frodobots (img3m, pose5, pose20): skipped (not in EDGE_TESTS)", flush=True)

# ---- CAST language test ----
cz = np.load(CAST); cm = json.loads(str(cz["meta"]))
cimg = lambda k: Image.fromarray(cz[f"img__{k}"]).resize((224, 224))                         # as OmniVLA's CAST loader
def cast(k):
    lang = cm[k]["instruction"]
    res[f"edge__lang__{k}"] = predict([cimg(k)] * 6, 7, lang=lang)
    res[f"edge-blank__lang__{k}"] = predict([BLACK] * 6, 7, lang=lang)
    res[f"edge-shuffled__lang__{k}"] = predict([cimg(cm[k]["img_shuffle"])] * 6, 7, lang=lang)
    res[f"edge-lang_shuffled__lang__{k}"] = predict([cimg(k)] * 6, 7, lang=cm[cm[k]["lang_shuffle"]]["instruction"])
if "lang" in RUN:
    run_test("lang", sorted(cm), cast)
else:
    print("[EDGE] lang (CAST): skipped (not in EDGE_TESTS)", flush=True)

# ---- LeLaN object-goal test ----
if "lelan" in RUN:
    lz = np.load(LELAN); lm = json.loads(str(lz["meta"]))
    limg = lambda k: Image.fromarray(lz[f"img__{k}"])
    def lelan(k):
        res[f"edge__lelan__{k}"] = predict([limg(k)] * 6, 7, lang=lm[k]["target"])
        res[f"edge-blank__lelan__{k}"] = predict([BLACK] * 6, 7, lang=lm[k]["target"])
        res[f"edge-shuffled__lelan__{k}"] = predict([limg(lm[k]["img_shuffle"])] * 6, 7, lang=lm[k]["target"])
        res[f"edge-lang_shuffled__lelan__{k}"] = predict([limg(k)] * 6, 7, lang=lm[k]["distractor"])
    run_test("lelan", sorted(lm), lelan)
else:
    print("[EDGE] lelan (LeLaN): skipped (not in EDGE_TESTS)", flush=True)
res["meta__edge"] = np.array(json.dumps(dict(name="edge", checkpoint=os.path.basename(CKPT), device=str(dev))))
np.savez(OUT, **res)
print(f"[EDGE] saved {OUT} ({len(res)} keys)")
