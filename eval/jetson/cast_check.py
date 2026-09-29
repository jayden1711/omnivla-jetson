# Jetson check of the omnivla-finetuned-cast weights folder (deploy/weights_cast, tools/build_cast_weights.py): language
# mode (7) on the held-out CAST episodes of build/cast_split.json through the deployed runtime, compared with the
# Kaggle-simulated outputs of the same int4 weights (castgptq_pc4) and with bf16 (cast_bf16); latency, memory, and 20
# repeated predictions per sample on 5 samples (must be bit-identical). Statistics: eval/analysis/cast_gptq_analysis.py
# with results/cast_jetson.npz (scipy is not in the deploy venv).
#   OMNIVLA_WEIGHTS=weights_cast ./deploy/launch.sh eval/jetson/cast_check.py cast_lang.npz cast_split.json compress_castgptq.npz
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "deploy"))
from omnivla_deploy import OmniVLADeploy

CZ = np.load(sys.argv[1]); CM = json.loads(str(CZ["meta"])); S = json.load(open(sys.argv[2])); K = dict(np.load(sys.argv[3], allow_pickle=True))
keys = [k for k in S["heldout"] if f"castgptq_pc4__lang__{k}" in K]
assert len(keys) == len(S["heldout"]), (len(keys), len(S["heldout"]))
img = lambda k: Image.fromarray(CZ[f"img__{k}"]).resize((224, 224))            # as the Kaggle run (and OmniVLA's CAST loader)
m = OmniVLADeploy(os.path.join(ROOT, "deploy"))
assert m.vision_kernel == "marlin" and m.head_step == 210000, (m.vision_kernel, m.head_step)
m.warmup(modes=(7,))
torch.cuda.reset_peak_memory_stats()
out, lat = {}, []
avail = lambda: int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024
for i, k in enumerate(keys):
    o = m.predict(img(k), mode=7, lang=CM[k]["instruction"]); out[k] = o["actions"]; lat.append(o["t_fwd"])
    if avail() < int(os.environ.get("CAST_MIN_AVAIL_MB", "700")):   # stop before the allocator fails (NVML assert)
        print(f"[CAST] STOP: MemAvailable {avail()} MB after {i + 1} predictions", flush=True); sys.exit(4)
    if i % 10 == 0 or os.environ.get("CAST_VERBOSE"):          # memory per prediction count (CUDA graphs are per token count)
        print(f"[CAST] {i + 1}/{len(keys)} | T {m._S['ids'].shape[1]} | {o['t_fwd'] * 1000:.0f} ms | torch reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB"
              f" | MemAvailable {avail()} MB", flush=True)
ade = lambda a, b: float(np.linalg.norm(np.asarray(a).reshape(8, 4)[:, :2] - np.asarray(b).reshape(8, 4)[:, :2], axis=1).mean())
xk = [ade(out[k], K[f"castgptq_pc4__lang__{k}"]) for k in keys]
rel = [k for k in keys if CM[k]["reliable"]]
ej = np.array([ade(out[k], CZ[f"gt__{k}"]) for k in rel]); eb = np.array([ade(K[f"cast_bf16__lang__{k}"], CZ[f"gt__{k}"]) for k in rel])
ek = np.array([ade(K[f"castgptq_pc4__lang__{k}"], CZ[f"gt__{k}"]) for k in rel])
same = 0
for k in keys[:5]:
    ref = m.predict(img(k), mode=7, lang=CM[k]["instruction"])["actions"]
    same += sum(np.array_equal(m.predict(img(k), mode=7, lang=CM[k]["instruction"])["actions"], ref) for _ in range(20))
res = dict(n=len(keys), n_reliable=len(rel), jetson_vs_kaggle_int4_mean=float(np.mean(xk)), jetson_vs_kaggle_int4_max=float(np.max(xk)),
           error_jetson=float(ej.mean()), error_kaggle_int4=float(ek.mean()), error_bf16=float(eb.mean()),
           latency_ms_median=float(np.median(lat)) * 1000,
           latency_ms_p95=float(np.percentile(lat, 95)) * 1000, torch_weights_gib=m.weights_gib,
           torch_peak_gib=torch.cuda.max_memory_allocated() / 2**30, repeats_bit_identical=f"{same}/100")
print(f"[CAST] {json.dumps(res)}", flush=True)
np.savez(os.path.join(ROOT, "results", "cast_jetson.npz"), **{f"jetson_int4__lang__{k}": v for k, v in out.items()})
json.dump(res, open(os.path.join(ROOT, "results", "cast_jetson.json"), "w"), indent=1)
print("[CAST] DONE", flush=True)
