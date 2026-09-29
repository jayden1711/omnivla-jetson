# Soak test: continuous predictions cycling pose goal (4), image goal (6, new goal every call) and language (7) on the
# 10 reference frames, for MINUTES. Every 30 s: median latency per mode, RAM used (tegrastats), torch allocated/reserved,
# junction temperature and GPU clock. With GRAPHS=1 the runtime uses CUDA graphs (all three graph sets stay in use).
#   GRAPHS=1 ./deploy/launch.sh eval/jetson/graph_soak.py 30     -> results/soak_graphs<0|1>.csv and .json
import csv, json, os, re, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "deploy"); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
G = os.environ.get("GRAPHS") == "1"
REF = os.path.join(D, "tests", "reference"); R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
IMG = {f: Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB") for f in frames}
GOAL = {f: Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB") for f in frames}
teg_log = f"/tmp/tegra_soak_g{int(G)}.log"
if os.path.exists(teg_log):
    os.remove(teg_log)
teg = subprocess.Popen(["tegrastats", "--interval", "1000", "--logfile", teg_log])
def teg_last():
    try:
        line = open(teg_log).read().splitlines()[-1]
    except (OSError, IndexError):
        return np.nan, np.nan
    ram = re.search(r"RAM (\d+)/", line); tj = re.search(r"tj@([\d.]+)C", line)
    return (int(ram[1]) if ram else np.nan), (float(tj[1]) if tj else np.nan)
def gpu_mhz():
    try:
        return int(open("/sys/class/devfreq/17000000.gpu/cur_freq").read()) / 1e6
    except (OSError, ValueError):
        return np.nan
m = OmniVLADeploy(D, verbose=False, cuda_graphs=G, **({"llm_kv_cache": os.environ["KVCACHE"] == "1"} if os.environ.get("KVCACHE") else {}))
m.warmup(modes=(4, 6, 7))
run = {4: lambda f: m.predict(IMG[f], mode=4, goal_pose=R[f"goal__{f}"]),
       6: lambda f: m.predict(IMG[f], mode=6, goal_image=GOAL[f]),
       7: lambda f: m.predict(IMG[f], mode=7, lang=str(R["lang__m7"]))}
out = os.path.join(ROOT, "results", f"soak_graphs{int(G)}.csv")
rows, t0, n, lat = [], time.time(), 0, {4: [], 6: [], 7: []}
next_log = t0 + 30
while time.time() - t0 < MIN * 60:
    for f in frames:
        for mode in (4, 6, 7):
            lat[mode].append(run[mode](f)["t_fwd"]); n += 1
        if time.time() >= next_log:
            ram, tj = teg_last()
            rows.append(dict(t_min=round((time.time() - t0) / 60, 2), n=n, ram_mb=ram, tj_c=tj, gpu_mhz=gpu_mhz(),
                             torch_alloc_gib=round(torch.cuda.memory_allocated() / 2**30, 3),
                             torch_reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 3),
                             **{f"lat{md}_ms": round(float(np.median(v)) * 1000, 1) for md, v in lat.items() if v}))
            print(f"[SOAK] {rows[-1]}", flush=True)
            lat = {4: [], 6: [], 7: []}; next_log += 30
teg.terminate()
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
late = [r for r in rows if r["t_min"] >= 10]
fit = lambda k: float(np.polyfit([r["t_min"] for r in late], [r[k] for r in late], 1)[0] * 10) if len(late) > 2 else float("nan")
summ = dict(graphs=G, minutes=MIN, predictions=n, ram_first_mb=rows[0]["ram_mb"], ram_last_mb=rows[-1]["ram_mb"],
            ram_max_mb=max(r["ram_mb"] for r in rows), ram_drift_mb_per_10min_after_10min=fit("ram_mb"),
            tj_max_c=max(r["tj_c"] for r in rows), gpu_mhz_min=min(r["gpu_mhz"] for r in rows),
            torch_reserved_gib_min_max=[min(r["torch_reserved_gib"] for r in rows), max(r["torch_reserved_gib"] for r in rows)],
            **{f"lat{md}_ms_first_last": [rows[0][f"lat{md}_ms"], rows[-1][f"lat{md}_ms"]] for md in (4, 6, 7)},
            **{f"lat{md}_drift_ms_per_10min_after_10min": fit(f"lat{md}_ms") for md in (4, 6, 7)})
json.dump(summ, open(out.replace(".csv", ".json"), "w"), indent=1)
print("[SOAK] summary " + json.dumps(summ), flush=True)
