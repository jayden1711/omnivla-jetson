# Latency estimate for Marlin per-channel int4 in the vision encoders, measured inside the deployed runtime (CUDA graphs on):
# every quantized vision linear (GemLite and HQQ portable) is replaced by a Marlin layer of the same shape with RANDOM
# weights (zero-padded to Marlin's k % 128 / n % 256), so outputs are meaningless and only the timing is real.
#   ./deploy/launch.sh eval/jetson/vision_marlin_estimate.py [all|siglip_mlp]   -> results/vision_marlin_estimate_<set>.json
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

SET = sys.argv[1] if len(sys.argv) > 1 else "all"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "deploy"); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy
import marlin

REF = os.path.join(D, "tests", "reference"); R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
IMG = {f: Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB") for f in frames}
GOAL = {f: Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB") for f in frames}
m = OmniVLADeploy(D, verbose=False)
vla, vb = m.vla, m.vla.vision_backbone
mem0 = torch.cuda.memory_allocated()


class MarlinPad(torch.nn.Module):
    def __init__(self, k, n, bias, dev):
        super().__init__()
        self.k, self.n = k, n
        self.kp, self.np_ = -(-k // 128) * 128, -(-n // 256) * 256
        self.L = marlin.Layer(self.kp, self.np_, groupsize=-1).to(dev)
        with torch.no_grad():
            self.L.B.random_(0, 2 ** 31 - 1); self.L.s.fill_(1e-4)
        self.bias = None if bias is None else bias.detach().clone().half()

    def forward(self, x):
        sh = x.shape[:-1]; x = x.reshape(-1, self.k).half()
        if self.kp != self.k:
            x = torch.nn.functional.pad(x, (0, self.kp - self.k))
        y = self.L(x)[:, : self.n]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*sh, self.n)


n_rep = 0
for enc in (vb.featurizer, vb.fused_featurizer):
    for nm, mod in list(enc.named_modules()):
        tn = type(mod).__name__
        if tn not in ("GLWrap", "HQQLinear"):
            continue
        if SET == "siglip_mlp" and not (enc is vb.fused_featurizer and tn == "HQQLinear"):
            continue
        inner = mod.m if tn == "GLWrap" else mod
        if hasattr(inner, "meta"):
            n_out, k_in = inner.meta["shape"]
        else:
            n_out, k_in = int(inner.out_features), int(inner.in_features)
        bias = getattr(inner, "bias", None)
        parent = enc.get_submodule(nm.rsplit(".", 1)[0]) if "." in nm else enc
        setattr(parent, nm.rsplit(".", 1)[1], MarlinPad(int(k_in), int(n_out), bias if torch.is_tensor(bias) else None, m.dev))
        n_rep += 1
import gc; gc.collect(); torch.cuda.empty_cache()
mem1 = torch.cuda.memory_allocated()
print(f"[VMEST] set {SET}: replaced {n_rep} vision linears; torch allocated {mem0/2**20:.0f} -> {mem1/2**20:.0f} MB "
      "(random Marlin weights: same size as real ones)", flush=True)
m.warmup(modes=(4, 6))
T = {}
def hook(name, mod):
    st = {}
    def a(*_):                                                 # hooks must return None (a pre-hook's return replaces the inputs)
        torch.cuda.synchronize(); st["t"] = time.perf_counter()
    def b(*_):
        torch.cuda.synchronize(); T.setdefault(name, []).append(time.perf_counter() - st["t"])
    mod.register_forward_pre_hook(a); mod.register_forward_hook(b)
res = {"set": SET, "n_replaced": n_rep, "torch_alloc_mb_before": mem0 / 2**20, "torch_alloc_mb_after": mem1 / 2**20}
# unhooked end to end first
for case in ("4", "6_new_goal"):
    ts = []
    for rep in range(3):
        for f in frames:
            o = m.predict(IMG[f], mode=4, goal_pose=R[f"goal__{f}"]) if case == "4" else m.predict(IMG[f], mode=6, goal_image=GOAL[f])
            if rep:
                ts.append(o["t_fwd"])
    res[f"fwd_ms_{case}"] = round(float(np.median(ts)) * 1000, 2)
for name, mod in (("dino", vb.featurizer), ("siglip", vb.fused_featurizer), ("vision_total", vb)):
    hook(name, mod)
for f in frames * 2:
    m.predict(IMG[f], mode=4, goal_pose=R[f"goal__{f}"])
res.update({f"{k}_ms_mode4": round(float(np.median(v)) * 1000, 2) for k, v in T.items()})
print(f"[VMEST] {res}", flush=True)
json.dump(res, open(os.path.join(ROOT, "results", f"vision_marlin_estimate_{SET}.json"), "w"), indent=1)
print("[VMEST] DONE", flush=True)
