# Power and energy per inference of the deployed runtime at the current power mode (nvpmodel), from tegrastats.
# Board input power (VDD_IN) is sampled every 100 ms: 15 s idle with the model loaded, then back-to-back predictions
# on the 10 reference frames (deploy/tests/reference) in three cases: pose goal (4), image goal with a new goal every
# call (6, full cost), image goal with an unchanged goal (6c, cached goal vision) and language (7, unpruned). Energy = mean
# VDD_IN during the case x wall time per prediction; "above idle" subtracts the idle power.
# Run through launch.sh (it also runs jetson_clocks, which maxes clocks within the current power mode):
#   ./deploy/launch.sh eval/jetson/power_measure.py [N_PER_CASE]     -> results/power_<mode>.json
# Use eval/jetson/run_power_sweep.sh for 15W, 25W and MAXN_SUPER.
import json, os, re, subprocess, sys, threading, time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.environ.get("OMNIVLA_DEPLOY", os.path.join(ROOT, "deploy"))
sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
REF = os.path.join(D, "tests", "reference")
R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
if not all(os.path.exists(os.path.join(REF, s, f + ".jpg")) for s in ("frames", "goals") for f in frames):
    sys.exit("[POWER] reference images missing: run ./deploy/launch.sh deploy/tools/reference_check.py first")
IMG = {f: Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB") for f in frames}
GOAL = {f: Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB") for f in frames}
pm = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
mode_name = (re.findall(r"NV Power Mode:\s*(\S+)", pm) or ["unknown"])[0]

samples, stop = [], threading.Event()                  # (time, VDD_IN mW, VDD_CPU_GPU_CV mW, VDD_SOC mW, GPU MHz, tj C)
def gpu_mhz():                                         # GPU clock from devfreq (tegrastats shows no clocks without root)
    try:
        return int(open("/sys/class/devfreq/17000000.gpu/cur_freq").read()) / 1e6
    except (OSError, ValueError):
        return np.nan
def reader():
    p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        t = time.time()
        v = {k: int(x) for k, x in re.findall(r"(VDD_\w+) (\d+)mW/\d+mW", line)}
        tj = re.search(r"tj@([\d.]+)C", line)
        if "VDD_IN" in v:
            samples.append((t, v["VDD_IN"], v.get("VDD_CPU_GPU_CV", np.nan), v.get("VDD_SOC", np.nan),
                            gpu_mhz(), float(tj[1]) if tj else np.nan))
        if stop.is_set():
            p.terminate(); break
threading.Thread(target=reader, daemon=True).start()

def window(t0, t1):
    s = np.array([x[1:] for x in samples if t0 <= x[0] <= t1], dtype=float)
    return s if len(s) else np.full((1, 5), np.nan)

m = OmniVLADeploy(D, verbose=False)
m.warmup(modes=(4, 6, 7))
time.sleep(3); t0 = time.time(); time.sleep(15); idle = window(t0, time.time())
out = dict(power_mode=mode_name, n_per_case=N, idle_mw=float(np.nanmean(idle[:, 0])), cases={})
cases = {"4": lambda i, f: m.predict(IMG[f], mode=4, goal_pose=R[f"goal__{f}"]),
         "6": lambda i, f: m.predict(IMG[f], mode=6, goal_image=GOAL[f]),                   # goal changes every call
         "6c": lambda i, f: m.predict(IMG[f], mode=6, goal_image=GOAL[frames[0]], goal_id="fixed"),
         "7": lambda i, f: m.predict(IMG[f], mode=7, lang="move toward the bench")}         # language, no pruning
for name, run in cases.items():
    fwd = []; t0 = time.time()
    for i in range(N):
        fwd.append(run(i, frames[i % len(frames)])["t_fwd"])
    t1 = time.time(); w = window(t0, t1); per = (t1 - t0) / N
    p_in = float(np.nanmean(w[:, 0]))
    out["cases"][name] = dict(wall_s_per_inference=per, fwd_ms_median=float(np.median(fwd) * 1000), vdd_in_mw=p_in,
                              vdd_in_peak_mw=float(np.nanmax(w[:, 0])), vdd_cpu_gpu_cv_mw=float(np.nanmean(w[:, 1])),
                              vdd_soc_mw=float(np.nanmean(w[:, 2])), n_samples=int(len(w)),
                              gpu_mhz_median=float(np.nanmedian(w[:, 3])), tj_max_c=float(np.nanmax(w[:, 4])),
                              energy_j_per_inference=p_in / 1000 * per,
                              energy_above_idle_j=(p_in - out["idle_mw"]) / 1000 * per)
    c = out["cases"][name]
    print(f"[POWER] {mode_name} case {name}: {c['fwd_ms_median']:.0f} ms fwd, {per * 1000:.0f} ms wall, VDD_IN {p_in / 1000:.2f} W "
          f"(idle {out['idle_mw'] / 1000:.2f} W) -> {c['energy_j_per_inference']:.2f} J/inference "
          f"({c['energy_above_idle_j']:.2f} J above idle) | GPU {c['gpu_mhz_median']:.0f} MHz, tj max {c['tj_max_c']:.1f} C", flush=True)
stop.set()
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
path = os.path.join(ROOT, "results", f"power_{mode_name}.json")
json.dump(out, open(path, "w"), indent=1)
print(f"[POWER] saved {path}")
