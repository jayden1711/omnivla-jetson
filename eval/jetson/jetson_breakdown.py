# Latency breakdown of the deployed runtime (vision / projector / LLM) and LLM token counts.
# Run in the OmniVLA clone (needs results/gt_tests.csv): $OMNIVLA_ROOT/deploy/launch.sh jetson_breakdown.py
import csv, os, time, sys
import numpy as np
import torch
from PIL import Image
ROOT = os.environ.get("OMNIVLA_ROOT", "/mnt/nvme/omnivla")
sys.path.insert(0, f"{ROOT}/deploy")
from omnivla_deploy import OmniVLADeploy

rows = [r for r in csv.DictReader(open("gt_tests.csv")) if r["reliable_path"] == "True" and r["img_ok"] == "True"]
names = sorted(r["frame"] for r in rows)[:12]
m = OmniVLADeploy(f"{ROOT}/deploy"); vla = m.vla
T, t0, info = {}, {}, {}
def pre(name):
    def f(mod, *a, **k):
        torch.cuda.synchronize(); t0[name] = time.time()
    return f
def post(name):
    def f(mod, a, *rest, **k):
        torch.cuda.synchronize(); T.setdefault(name, []).append(time.time() - t0[name])
    return f
parts = {"vision.dino": vla.vision_backbone.featurizer, "vision.siglip": vla.vision_backbone.fused_featurizer,
         "projector": vla.projector, "llm": vla.language_model.model}
for n, mod in parts.items():
    mod.register_forward_pre_hook(pre(n)); mod.register_forward_hook(post(n))
vla.language_model.model.register_forward_pre_hook(lambda mod, a, k: info.__setitem__("T", (k.get("inputs_embeds") if k.get("inputs_embeds") is not None else a[0]).shape[1]), with_kwargs=True)
m.warmup(modes=(4, 6), n=2)
for mode in (4, 6):
    T.clear(); tot = []
    for f in names:
        img = Image.open(f"{ROOT}/frames/{f}.jpg").convert("RGB")
        gimg = Image.open(f"goal_img3m/{f}.jpg").convert("RGB")
        o = m.predict(img, mode=mode, goal_pose=np.array([20.0, 0.0, 1.0, 0.0]), goal_image=gimg)
        tot.append(o["t_fwd"])
    s = " | ".join(f"{k} {np.median(v[2:]) * 1000:.0f} ms (x{len(v) // len(names)})" for k, v in T.items())
    print(f"[BD] mode {mode}: total fwd {np.median(tot[2:]) * 1000:.0f} ms | LLM tokens {info['T']} | {s}", flush=True)
