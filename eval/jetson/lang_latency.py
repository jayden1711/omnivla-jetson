# Latency and memory of the language modes (7 language, 8 language + pose) on the Jetson, at a given image-token pruning
# for language modes (lang_prune_frac). One process per setting (like final_validate.py): 10 reference frames x 5 = 50
# predictions per mode after warm-up; fresh tegrastats log for the RAM peak.
#   ./deploy/launch.sh eval/jetson/lang_latency.py 0.0     (and 0.5)   -> results/lang_latency_p<pct>.json
# MODES=4,6,7,8 measures other modes too (image goal: a new goal every call); RUNTIME_KW='{"cuda_graphs": true}' passes
# runtime options; TAG names the output file.
import json, os, re, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "deploy"); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
REF = os.path.join(D, "tests", "reference"); R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
IMG = {f: Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB") for f in frames}
teg_log = f"/tmp/tegra_lang_{int(FRAC * 100)}.log"
if os.path.exists(teg_log):
    os.remove(teg_log)                                         # tegrastats --logfile appends: fresh log every run
teg = subprocess.Popen(["tegrastats", "--interval", "500", "--logfile", teg_log])
ram = lambda: max([int(x) for x in re.findall(r"RAM (\d+)/\d+MB", open(teg_log).read())] or [0]) if os.path.exists(teg_log) else 0
time.sleep(1.5); ram_idle = ram()
MODES = [int(x) for x in os.environ.get("MODES", "7,8").split(",")]
KW = json.loads(os.environ.get("RUNTIME_KW", "{}"))
if os.environ.get("GRAPHS") == "1":                               # shorthand for RUNTIME_KW='{"cuda_graphs": true}'
    KW["cuda_graphs"] = True
if os.environ.get("NOCACHE") == "1":                              # without the LLM key/value cache
    KW["llm_kv_cache"] = False
if os.environ.get("CACHE") == "1":                                # with it (the runtime default may differ)
    KW["llm_kv_cache"] = True
m = OmniVLADeploy(D, lang_prune_frac=FRAC, **KW)
t_w = time.time(); m.warmup(modes=tuple(MODES), n=2); t_w = time.time() - t_w
torch.cuda.reset_peak_memory_stats()
out = dict(lang_prune_frac=FRAC, runtime_kw=KW, warmup_s=t_w, n_keep=m.n_keep_lang, torch_weights_gib=m.weights_gib,
           ram_idle_mb=ram_idle, modes={})
GOAL = {f: Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB") for f in frames}
LANG = {7: "move toward the bench", 8: "stay on the path"}
for mode in MODES:
    fwd, e2e = [], []
    for rep in range(5):
        for f in frames:
            kw = dict(goal_pose=R[f"goal__{f}"]) if mode in (4, 8) else dict(goal_image=GOAL[f]) if mode == 6 else {}
            if mode in (7, 8):
                kw["lang"] = LANG[mode]
            t = time.time(); o = m.predict(IMG[f], mode=mode, **kw); e2e.append(time.time() - t); fwd.append(o["t_fwd"])
    out["modes"][mode] = dict(n=len(fwd), fwd_ms_median=float(np.median(fwd) * 1000), fwd_ms_p90=float(np.percentile(fwd, 90) * 1000),
                              e2e_ms_median=float(np.median(e2e) * 1000))
time.sleep(1.0)
out.update(torch_peak_gib=torch.cuda.max_memory_allocated() / 2**30, ram_peak_mb=ram(), ram_total_mb=7620)
out["ram_headroom_mb"] = out["ram_total_mb"] - out["ram_peak_mb"]
teg.terminate()
json.dump(out, open(os.path.join(ROOT, "results", f"lang_latency_p{int(FRAC * 100)}{os.environ.get('TAG', '')}.json"), "w"), indent=1)
print("[LANGLAT] " + json.dumps(out), flush=True)
