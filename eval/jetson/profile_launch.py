# Where the time goes inside one prediction, per mode, and how much of it is CPU launch overhead (what CUDA graphs could
# remove). torch.profiler over 10 predictions per mode after warm-up: per stage (DINOv2, SigLIP, projector, LLM, action
# head) wall time, summed GPU kernel time, kernel count; GPU idle = wall - kernel time (an upper bound on what CUDA
# graphs can save). Run on the Jetson from the repo root:
#   ./deploy/launch.sh eval/jetson/profile_launch.py            -> results/latency_profile_jetson.json
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "deploy"); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

REF = os.path.join(D, "tests", "reference"); R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
if not all(os.path.exists(os.path.join(REF, s, f + ".jpg")) for s in ("frames", "goals") for f in frames):
    sys.exit("[PROF] reference images missing: run ./deploy/launch.sh deploy/tools/reference_check.py first")
IMG = {f: Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB") for f in frames}
GOAL = {f: Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB") for f in frames}
m = OmniVLADeploy(D, verbose=False, **json.loads(os.environ.get("RUNTIME_KW", "{}")))
m.warmup(modes=(4, 6, 7))
vla = m.vla
STAGES = {"dino": vla.vision_backbone.featurizer, "siglip": vla.vision_backbone.fused_featurizer,
          "projector": vla.projector, "llm": vla.language_model.model}
for name, mod in STAGES.items():                              # label each stage in the trace
    mod.register_forward_pre_hook(lambda mod, a, n=name: mod.__dict__.__setitem__("_rf", torch.profiler.record_function(n).__enter__()))
    mod.register_forward_hook(lambda mod, a, o: mod.__dict__.pop("_rf").__exit__(None, None, None))
_pa = m.action_head.predict_action                            # called directly, not through forward
def _pa_prof(*a, **k):
    with torch.profiler.record_function("action_head"):
        return _pa(*a, **k)
m.action_head.predict_action = _pa_prof
STAGES["action_head"] = None
CASES = {"4": lambda f: m.predict(IMG[f], mode=4, goal_pose=R[f"goal__{f}"]),
         "6": lambda f: m.predict(IMG[f], mode=6, goal_image=GOAL[f]),
         "7": lambda f: m.predict(IMG[f], mode=7, lang="move toward the door")}
out = {}
for name, run in CASES.items():
    for f in frames[:2]:
        run(f)
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        t = time.time()
        for f in frames:
            run(f)
        torch.cuda.synchronize(); wall = (time.time() - t) / len(frames)
    ev = prof.key_averages()
    st = {}
    for s in STAGES:
        e = [x for x in ev if x.key == s]
        if e:
            st[s] = dict(wall_ms=e[0].cpu_time_total / 1000 / len(frames), gpu_ms=e[0].device_time_total / 1000 / len(frames))
    kern = [x for x in ev if x.device_type == torch.autograd.DeviceType.CUDA]
    gpu = sum(x.device_time_total for x in kern) / 1000 / len(frames)
    n_k = sum(x.count for x in kern) / len(frames)
    out[name] = dict(wall_ms=wall * 1000, gpu_kernel_ms=gpu, gpu_idle_ms=wall * 1000 - gpu, kernels_per_prediction=n_k, stages=st)
    print(f"[PROF] mode {name}: wall {wall * 1000:.0f} ms | GPU kernels {gpu:.0f} ms ({n_k:.0f} launches) | GPU idle "
          f"{wall * 1000 - gpu:.0f} ms | " + " | ".join(f"{k} {v['wall_ms']:.0f}/{v['gpu_ms']:.0f}" for k, v in st.items()), flush=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "latency_profile_jetson.json"), "w"), indent=1)
print("[PROF] stage numbers: wall/GPU ms per prediction; saved results/latency_profile_jetson.json")
