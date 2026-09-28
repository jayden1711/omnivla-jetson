# Validation of the deployed runtime. MODE 6: image-goal test (img3m, 100 frames); MODE 4: pose goal 5.1 s ahead.
# Output: mf_results/e2e_deploy{VALIDATE_TAG}@MODE/<frame>.npz + _summary.npz. --soak MINUTES: continuous inference,
# logging latency, memory, temperatures and clocks (tegrastats) every 30 s to results_soak.csv.
# Run in the OmniVLA clone: $OMNIVLA_ROOT/deploy/launch.sh final_validate.py MODE [--limit N] | --soak MINUTES
import csv, os, re, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get("OMNIVLA_ROOT", "/mnt/nvme/omnivla")
sys.path.insert(0, f"{ROOT}/deploy")
from omnivla_deploy import OmniVLADeploy

SOAK = "--soak" in sys.argv
SOAK_MODE = int(os.environ.get("SOAK_MODE", "4"))              # 6: image goal with the goal cache, sequential clips
MODE = SOAK_MODE if SOAK else int(sys.argv[1])
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
rows = [r for r in csv.DictReader(open("gt_tests.csv")) if r["reliable_path"] == "True" and r["img_ok"] == "True"]
names = sorted(r["frame"] for r in rows)[:LIMIT]
gz = np.load("gt_frodobots.npz")
TAG = os.environ.get("VALIDATE_TAG", "")                        # e.g. _gptq (with OMNIVLA_WEIGHTS=weights_gptq)
OUT = f"mf_results/e2e_deploy{TAG}@{MODE}" + ("_soak" if SOAK else ""); os.makedirs(OUT, exist_ok=True)

teg_log = f"/tmp/tegra_deploy_{MODE}{'_soak' if SOAK else ''}.log"
if os.path.exists(teg_log):
    os.remove(teg_log)                                   # tegrastats --logfile appends: fresh log every run
teg = subprocess.Popen(["tegrastats", "--interval", "500", "--logfile", teg_log])
def teg_lines():
    try:
        return open(teg_log).read().splitlines()
    except FileNotFoundError:
        return []
def ram(lines):
    v = [int(m) for l in lines for m in re.findall(r"RAM (\d+)/\d+MB", l)]
    return max(v) if v else np.nan
time.sleep(1.5)
ram_idle = ram(teg_lines())

m = OmniVLADeploy(f"{ROOT}/deploy", **({"goal_refresh": int(os.environ["GOAL_REFRESH"])} if os.environ.get("GOAL_REFRESH") else {}))
def inputs(f):
    img = Image.open(f"{ROOT}/frames/{f}.jpg").convert("RGB")
    if MODE == 6:
        return dict(image=img, mode=6, goal_image=Image.open(f"goal_img3m/{f}.jpg").convert("RGB"))
    return dict(image=img, mode=4, goal_pose=gz[f"goal__{f}"].astype(np.float64))
m.predict(**inputs(names[0]))                            # warm-up
torch.cuda.reset_peak_memory_stats()

if not SOAK:
    for i, f in enumerate(names):
        te = time.time(); o = m.predict(**inputs(f)); te = time.time() - te
        np.savez(f"{OUT}/{f}.npz", act=o["actions"][None], t_fwd=o["t_fwd"], t_e2e=te)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[FINAL] deploy@{MODE} {i+1}/{len(names)} fwd {o['t_fwd']*1000:.0f} ms", flush=True)
    teg.terminate()
    s = dict(config="deploy", mode=MODE, torch_weights_gib=m.weights_gib, torch_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
             torch_reserved_peak_gib=torch.cuda.max_memory_reserved() / 2**30, ram_idle_mb=ram_idle,
             ram_peak_mb=ram(teg_lines()), ram_total_mb=7620)
    np.savez(f"{OUT}/_summary.npz", **{k: np.array(v) for k, v in s.items()})
    print(f"[FINAL] SUMMARY {s}", flush=True)
    sys.exit(0)

# ---- soak ----
minutes = float(sys.argv[sys.argv.index("--soak") + 1])
t_end, t_next, i, win, calls, trims = time.time() + 60 * minutes, time.time() + 30, 0, [], [], [0.0]
rss = lambda: [int(l.split()[1]) // 1024 for l in open("/proc/self/status") if l.startswith("RssAnon")][0]
log = open("results_soak.csv" if MODE == 4 else f"results_soak{MODE}{TAG}.csv", "w"); w = csv.writer(log)
if MODE == 6:                                              # 24 sequential clips at the image-goal cadence (1.05 s = 21 frames)
    seq = list(csv.DictReader(open("seq_frames.csv")))
    SEQ = [(r["frame"], r["clip"]) for r in sorted(seq, key=lambda r: (r["clip"], int(r["idx"]))) if int(r["idx"]) % 21 == 0]
    GIMG = {c: Image.open(f"seq/goals/{c}.jpg").convert("RGB") for c in sorted({c for _, c in SEQ})}
    def inputs_soak(i):
        f, c = SEQ[i % len(SEQ)]
        return dict(image=Image.open(f"seq/{f}.jpg").convert("RGB"), mode=6, goal_image=GIMG[c])
else:
    inputs_soak = lambda i: inputs(names[i % len(names)])
reuse = []
w.writerow(["t_s", "n", "fwd_ms_mean", "fwd_ms_max", "call_ms_max", "trim_ms_max", "rss_anon_mb", "torch_alloc_gib", "torch_reserved_gib", "ram_mb", "gpu_c", "cpu_c", "gr3d_mhz", "kv_reuse_frac"])
t0 = time.time()
while time.time() < t_end:
    tc = time.time(); o = m.predict(**inputs_soak(i)); calls.append(time.time() - tc); i += 1
    reuse.append(float(o.get("cache", {}).get("kv_reused", False)))
    win.append(o["t_fwd"]); trims.append(o.get("t_trim", 0.0))
    if time.time() >= t_next:
        L = teg_lines()[-4:]
        last = L[-1] if L else ""
        gpu = re.search(r"gpu@([\d.]+)C", last, re.I); cpu = re.search(r"cpu@([\d.]+)C", last, re.I)
        clk = re.search(r"GR3D_FREQ \d+%@\[?(\d+)", last)
        row = [round(time.time() - t0), i, round(1000 * np.mean(win), 1), round(1000 * max(win), 1),
               round(1000 * max(calls), 1), round(1000 * max(trims), 1), rss(),
               round(torch.cuda.memory_allocated() / 2**30, 3), round(torch.cuda.memory_reserved() / 2**30, 3), ram(L),
               gpu.group(1) if gpu else "", cpu.group(1) if cpu else "", clk.group(1) if clk else "", round(float(np.mean(reuse)), 2)]
        w.writerow(row); log.flush(); win, calls, trims, reuse = [], [], [0.0], []; t_next += 30
        print(f"[SOAK] {row}", flush=True)
teg.terminate()
print(f"[SOAK] DONE {i} inferences in {(time.time()-t0)/60:.1f} min; peak torch {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
