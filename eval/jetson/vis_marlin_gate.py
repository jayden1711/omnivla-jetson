# Gate for the Marlin per-channel int4 vision layers on the Orin (sm_87). Run: ./deploy/launch.sh eval/jetson/vis_marlin_gate.py
#   WEIGHTS_DIR CHECK.pt   (CHECK.pt = prequant_vis_<tag>_check.pt from the Kaggle export). Exit 0 = PASS, 3 = FAIL.
# 1) weights: for all vision layers, Marlin(I) (sliced to the real shape) must hash to the export's W^T (bit-exact).
# 2) correctness + determinism: the runtime module (VisMarlinPC, padding included) vs dequantize-then-matmul in fp32, at
#    the real token counts (one image and two images per call): rel mean error < 1e-3 and 50 repeats bit-identical.
# 3) padding: DINOv2 layers need no padding; repack two of them zero-padded (k + 128, n + 256) and compare with the
#    unpadded layer on the same inputs (bit-identical expected unless Marlin's work split changes the fp16 reduction).
import hashlib, json, os, sys
import torch
import marlin

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "deploy"))
from omnivla_deploy import VisMarlinPC

W, CHK = sys.argv[1], sys.argv[2]
dev = torch.device("cuda:0")
SH = json.load(open(os.path.join(W, "SHARDS.json")))["vis_marpc"]
MAN = json.load(open(os.path.join(W, "VIS_MARPC_MANIFEST.json")))["deq_sha256_WT_fp16"]
ref = torch.load(CHK, map_location="cpu", weights_only=False)
print(f"[VGATE] torch {torch.__version__} | {torch.cuda.get_device_name(0)} | {len(SH['names'])} vision layers", flush=True)
fail = False

def wt(mod):                                                   # Marlin(I) = W^T exactly (single products)
    k = mod.kp; out = torch.empty(mod.kp, mod.np_, dtype=torch.half, device=dev)
    for i in range(0, k, 1024):
        j = min(i + 1024, k)
        I = torch.zeros(j - i, k, dtype=torch.half, device=dev); I[torch.arange(j - i), torch.arange(i, j)] = 1
        out[i:j] = mod.L(I)
    return out

# 1) weights
ok = bad = 0; mods = {}
for s in SH["shards"]:
    part = torch.load(os.path.join(W, "vis_marpc", s), map_location="cpu", weights_only=False)
    for n, st in part.items():
        mod = VisMarlinPC({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in st.items()}, dev)
        with torch.no_grad():
            full = wt(mod)
        h = hashlib.sha256(full[: mod.k, : mod.n].contiguous().cpu().numpy().tobytes()).hexdigest()
        pad_zero = bool((full[mod.k:] == 0).all() and (full[:, mod.n:] == 0).all())
        good = h == MAN[n] and pad_zero
        ok += good; bad += not good
        if not good:
            print(f"[VGATE] WEIGHT MISMATCH {n} (hash {'ok' if h == MAN[n] else 'differs'}, padding zero {pad_zero})", flush=True)
        if n in ref:
            mods[n] = mod
        del full
    del part
print(f"[VGATE] weights: {ok}/{ok + bad} layers bit-exact vs the Kaggle export (padded weights exactly zero)", flush=True)
fail |= bad > 0

# 2) correctness + determinism at the runtime token counts
for n, mod in sorted(mods.items()):
    Wr = ref[n].to(dev).float(); b = ref[n + ".bias"]
    T1 = 261 if ".featurizer." in n else 256
    for T in (T1, 2 * T1):
        torch.manual_seed(T)
        x = torch.randn(T, mod.k, dtype=torch.half, device=dev)
        with torch.no_grad():
            y0 = mod(x); torch.cuda.synchronize()
            r = x.float() @ Wr.T + (0 if b is None else b.to(dev).float())
            f16 = (x @ Wr.half().T + (0 if b is None else b.to(dev))).float()
            same = 0
            for _ in range(50):
                y = mod(x); torch.cuda.synchronize(); same += int(torch.equal(y, y0))
        e = (y0.float() - r).abs(); ef = (f16 - r).abs()
        rel, relf = float(e.mean() / r.abs().mean()), float(ef.mean() / r.abs().mean())
        good = rel < 1e-3 and same == 50
        fail |= not good
        print(f"[VGATE] {n} ({mod.k}->{mod.n}, padded {mod.kp}->{mod.np_}) T={T}: rel mean {rel:.2e} (fp16 cuBLAS {relf:.2e}) "
              f"max {float(e.max()):.3g} | bit-identical {same}/50 -> {'PASS' if good else 'FAIL'}", flush=True)

# 3) padding test on unpadded DINOv2 layers
for n in [x for x in sorted(mods) if ".featurizer.blocks.0." in x and x.endswith(("attn.qkv", "mlp.fc2"))]:
    mod = mods[n]; Wq = ref[n]; sc = ref[n + ".scale"]
    o, i = Wq.shape; kp, np_ = i + 128, o + 256
    Wp = torch.zeros(np_, kp, dtype=torch.half); Wp[:o, :i] = Wq
    sp = torch.ones(np_, 1, dtype=torch.half); sp[:o] = sc
    fq = torch.nn.Linear(kp, np_, bias=False, dtype=torch.half); fq.weight.data = Wp
    L = marlin.Layer(kp, np_, groupsize=-1); L.pack(fq, sp)
    st = {"marlin_B": L.B.to(dev), "marlin_s": L.s.to(dev), "k": i, "n": o, "kp": kp, "np": np_, "bias": ref[n + ".bias"]}
    padded = VisMarlinPC(st, dev)
    for T in (261, 522):
        torch.manual_seed(T + 1)
        x = torch.randn(T, i, dtype=torch.half, device=dev)
        with torch.no_grad():
            a, p = mod(x), padded(x)
        d = float((a.float() - p.float()).abs().max())
        print(f"[VGATE] padding {n} ({i}->{o} vs padded {kp}->{np_}) T={T}: bit-identical {torch.equal(a, p)} | max |diff| {d:.3g}", flush=True)
print(f"[VGATE] {'FAIL' if fail else 'ALL PASS'}", flush=True)
sys.exit(3 if fail else 0)
