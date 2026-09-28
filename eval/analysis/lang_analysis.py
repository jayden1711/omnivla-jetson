# Language-goal test (CAST, eval/data/cast_extract.py): does the model use the image and the instruction, and how do
# the deployed int4 config and OmniVLA-edge compare with bf16. Rules fixed before the runs:
#  - samples: reliable ones (ground-truth displacement >= 1 action unit after 8 steps)
#  - error = mean distance between predicted and driven waypoints over the 8 steps (action units, the episode's
#    normalization factor as in OmniVLA's CAST loader); also steps 1-5 and the final step
#  - a control is "clearly worse" if paired Wilcoxon p < 0.01 and its mean error is >= 15% higher (as tests_analysis.py)
#  - the test checks perception if both image controls are clearly worse; it checks instruction use if the shuffled
#    instruction is clearly worse
#  - turn direction: for samples whose driven path ends >= 1 unit to the side, share of predictions ending on that side
# Usage (from a folder with the files): python eval/analysis/lang_analysis.py cast_lang.npz compress_lang.npz \
#   compress_gptqx_lang.npz edge_tests.npz          -> results/lang_summary.md, results/lang_per_sample.csv
import json, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

cast, *files = sys.argv[1:]
CZ = np.load(cast); CM = json.loads(str(CZ["meta"]))
D = {}
for p in files:
    D.update(dict(np.load(p, allow_pickle=True)))
K = sorted(k for k in CM if CM[k]["reliable"])
if not K:
    sys.exit("[LANG] 0 reliable samples - refusing to write a summary")
GT = {k: CZ[f"gt__{k}"] for k in K}
rng = np.random.default_rng(0)

def robot(k):
    b = CM[k]["path"].rstrip("/").split("/")[-1]
    for pat, name in ((r"^tartan", "TartanDrive"), (r"^uw_", "Seattle"), (r"^\d{4}-\d{2}-\d{2}-", "GoStanford-style (dated)"),
                      (r"^jackal", "Jackal (SCAND/RECON)"), (r"^spot", "Spot (SCAND)"), (r"^cory", "Cory Hall"),
                      (r"^(sim|no)\d", "sim*/no* scenes"), (r"^[A-Z][a-z]{2}-\d{2}-\d{4}-", "SACSoN-style indoor (bww/soda)")):
        if re.search(pat, b):
            return name
    return "other"

def pred(cfg, k):
    a = D.get(f"{cfg}__lang__{k}")
    return None if a is None else np.asarray(a).reshape(-1, 8, 4)[0]
def err(p, g, s=slice(0, 8)):
    return float(np.linalg.norm(p[s, :2] - g[s, :2], axis=1).mean())
def errs(cfg, s=slice(0, 8)):
    return np.array([err(pred(cfg, k), GT[k], s) for k in K])
def boot(x):
    i = rng.integers(0, len(x), (5000, len(x))); return np.percentile(x[i].mean(1), [2.5, 97.5])
def worse(ctrl, real):
    p = wilcoxon(ctrl, real).pvalue; rel = ctrl.mean() / real.mean() - 1
    return p, rel, bool(p < 0.01 and rel >= 0.15)
def turn_acc(cfg):
    ks = [k for k in K if abs(GT[k][-1, 1]) >= 1.0]
    return np.mean([np.sign(pred(cfg, k)[-1, 1]) == np.sign(GT[k][-1, 1]) for k in ks]), len(ks)

L = []
say = L.append
for c in ("bf16", "fp16", "gptqx_pc4", "edge"):             # every sample or none: a partial run must not be summarized
    n = sum(pred(c, k) is not None for k in K)
    if 0 < n < len(K):
        sys.exit(f"[LANG] {c}: predictions for only {n}/{len(K)} samples")
if pred("bf16", K[0]) is None:
    sys.exit("[LANG] no bf16 predictions (the reference) in the given files")
say("# Language-goal test on CAST (behavioral instructions the deployed checkpoint was not trained on)\n")
say("**What this test is:** CAST's instructions describe behaviors (\"move along the corridor\", \"follow the dirt path, "
    "making a gentle left turn\"). The deployed checkpoint, omnivla-original, was not trained on CAST or on instructions "
    "of this kind: its language training is LeLaN-style object goals (\"move toward <object>\"). The OmniVLA authors "
    "released a separate checkpoint, omnivla-finetuned-cast, for CAST-style instructions; it is not evaluated here. So "
    "this is a test of the base checkpoint outside its training, not of language mode as trained. The in-distribution "
    "object-goal test is results/lelan_summary.md.\n")
say(f"{len(K)} of {len(CM)} samples reliable (moving >= 1 action unit in 8 steps), one per CAST episode, from "
    f"{len(set(robot(k) for k in K))} robot groups. Instructions are CAST's (the deployed checkpoint was not trained on CAST); "
    "the images come from GNM robots, and GNM is in OmniVLA's training mix. 128x128 images upscaled to 224x224. "
    "Open-loop single predictions. Error = mean distance to the driven waypoints over 8 steps, in action units.\n")
say("int4 = the deployed weights (GPTQ int4 LLM, identical tensor for tensor to the Jetson weights; HQQ4 vision; 75% token "
    "pruning), run on a Kaggle T4 with fp16 dequantized weights: Kaggle-simulated, not the Marlin/GemLite kernels, not "
    "bit-exact with the Jetson (on the image-goal test the two differ by 0.003 action units on average). Language mode has not been run on the Jetson.\n")

# ---- perception and instruction checks (fp16 7B, as for the other tests; edge and int4 for reference) ----
say("## Does the model use the image and the instruction?\n")
say("| Model | Real | Blank image | Shuffled image | Shuffled instruction | Checks perception | Uses instruction |")
say("|---|---|---|---|---|---|---|")
checks = {}
for name, cfg in (("7B fp16", "fp16"), ("7B int4 (Kaggle-simulated)", "gptqx_pc4"), ("OmniVLA-edge", "edge")):
    if pred(cfg, K[0]) is None:
        continue
    real = errs(cfg); cells, v = [], {}
    for c in ("blank", "shuffled", "lang_shuffled"):
        if pred(f"{cfg}-{c}", K[0]) is None:
            cells.append("-"); continue
        e = errs(f"{cfg}-{c}"); p, rel, w = worse(e, real); v[c] = w
        cells.append(f"{e.mean():.3f} ({100 * rel:+.0f}%, p={p:.1g})")
    perc = v.get("blank") and v.get("shuffled"); instr = v.get("lang_shuffled")
    checks[cfg] = (perc, instr)
    say(f"| {name} | {real.mean():.3f} | " + " | ".join(cells) + f" | {'yes' if perc else 'no'} | {'yes' if instr else 'no'} |")
say("")

if not any(pc and ins for pc, ins in checks.values()):
    say("**Result: this test does not show that any of these models follows the held-out instructions.** " +
        ("No model passes the instruction check (a shuffled instruction is not clearly worse). " if not any(i for _, i in checks.values()) else "") +
        ("The 7B model also fails the perception check here, so its error on this test cannot rank configs. " if not checks.get("fp16", (1, 1))[0] else "") +
        "Treat the accuracy table below as descriptive only.\n")

# ---- post hoc (added after seeing the error-based checks): turn direction under the controls ----
say("Turn direction under the same controls (added after the checks above; samples whose driven path ends >= 1 unit to "
    "the side; chance = 50%):\n")
say("| Model | Real | Blank image | Shuffled image | Shuffled instruction |")
say("|---|---|---|---|---|")
from scipy.stats import binomtest
TK = [k for k in K if abs(GT[k][-1, 1]) >= 1.0]
for name, cfg in (("7B fp16", "fp16"), ("7B int4 (Kaggle-simulated)", "gptqx_pc4"), ("OmniVLA-edge", "edge")):
    cells = []
    for c in ("", "-blank", "-shuffled", "-lang_shuffled"):
        if pred(cfg + c, TK[0]) is None:
            cells.append("-"); continue
        n = sum(np.sign(pred(cfg + c, k)[-1, 1]) == np.sign(GT[k][-1, 1]) for k in TK)
        cells.append(f"{100 * n / len(TK):.0f}% (p={binomtest(int(n), len(TK)).pvalue:.1g})")
    say(f"| {name} | " + " | ".join(cells) + " |")
say(f"\n{len(TK)} samples; p: two-sided binomial test against 50%.\n")

# ---- configs ----
say("## Accuracy\n")
say("| Model | Error, 8 steps [95% CI] | Steps 1-5 | Final step | Turn direction correct | vs bf16 (95% CI), p | Fidelity to bf16 |")
say("|---|---|---|---|---|---|---|")
ref = errs("bf16"); rows = {}
for name, cfg in (("7B bf16", "bf16"), ("7B fp16", "fp16"), ("7B int4 (Kaggle-simulated)", "gptqx_pc4"), ("OmniVLA-edge", "edge")):
    if pred(cfg, K[0]) is None:
        continue
    e = errs(cfg); d = e - ref; lo, hi = boot(e); dlo, dhi = boot(d); ta, nt = turn_acc(cfg)
    p = wilcoxon(e, ref).pvalue if np.any(d != 0) else np.nan
    fid = np.mean([err(pred(cfg, k), pred("bf16", k)) for k in K]) if cfg.startswith(("gptq", "fp16", "bf16")) else np.nan
    rows[cfg] = e
    say(f"| {name} | {e.mean():.3f} [{lo:.3f}, {hi:.3f}] | {errs(cfg, slice(0, 5)).mean():.3f} | {errs(cfg, slice(7, 8)).mean():.3f} | "
        f"{100 * ta:.0f}% of {nt} | {d.mean():+.3f} ({dlo:+.3f}, {dhi:+.3f}), p={p:.2g} | " + ("-" if np.isnan(fid) else f"{fid:.3f}") + " |")
if "edge" in rows and "gptqx_pc4" in rows:
    d = rows["edge"] - rows["gptqx_pc4"]; lo, hi = boot(d)
    say(f"\nOmniVLA-edge minus 7B int4: {d.mean():+.3f} action units (95% CI {lo:+.3f}, {hi:+.3f}), p={wilcoxon(rows['edge'], rows['gptqx_pc4']).pvalue:.2g}.")
g = np.array([np.linalg.norm(GT[k][:, :2], axis=1).mean() for k in K])
v = np.median([np.linalg.norm(np.diff(np.vstack([[0, 0], GT[k][:, :2]]), axis=0), axis=1).mean() for k in K])
straight = np.array([np.linalg.norm(np.stack([v * np.arange(1, 9), np.zeros(8)], 1) - GT[k][:, :2], axis=1).mean() for k in K])
say(f"\nModel-free references: standing still {g.mean():.3f}; straight ahead at the test set's median speed "
    f"({v:.2f} units per step, uses the ground truth of all samples) {straight.mean():.3f}.\n")

# ---- per robot group ----
say("## By robot group (error, 8 steps)\n")
grp = pd.Series({k: robot(k) for k in K})
cols = [c for c in ("bf16", "gptqx_pc4", "edge") if c in rows]
say("| Robot group | n | " + " | ".join({"bf16": "7B bf16", "gptqx_pc4": "7B int4", "edge": "edge"}[c] for c in cols) + " |")
say("|---|---|" + "---|" * len(cols))
for gname, idx in grp.groupby(grp).groups.items():
    ii = [K.index(k) for k in idx]
    say(f"| {gname} | {len(ii)} | " + " | ".join(f"{rows[c][ii].mean():.3f}" for c in cols) + " |")
os.makedirs("results", exist_ok=True)
pd.DataFrame({"sample": K, "robot": [robot(k) for k in K], "instruction": [CM[k]["instruction"] for k in K],
              **{f"err_{c}": rows[c] for c in rows}}).round(4).to_csv("results/lang_per_sample.csv", index=False)
open("results/lang_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
