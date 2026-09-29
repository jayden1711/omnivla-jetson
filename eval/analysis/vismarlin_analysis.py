# Vision linears as Marlin per-channel int4: accuracy vs the deployed HQQ4 vision (same Kaggle run, same int4 LLM).
# Rules fixed before the run (2026-09-28):
#  - image goal (img3m, 100 frames, 75% pruning): error vs the actual driven path (mean over 8 waypoints). A variant
#    FAILS if its mean error is higher than the HQQ4 baseline with paired Wilcoxon p < 0.05.
#  - object goal (LeLaN, 210 frames, no pruning): picks-named-object rate. A variant FAILS if it is lower than the
#    baseline with paired exact McNemar p < 0.05, or more than 5 points lower regardless of p.
#  - adoption: among the all-linears variants (RTN, GPTQ) that pass both tests, the one with the lower image-goal
#    fidelity error to bf16 (same frames); if neither passes, the SigLIP-MLP-only variant if it passes; else none.
# Usage: python eval/analysis/vismarlin_analysis.py compress_vismarlin.npz lelan_lang.npz [compress_tests.npz] -> results/vismarlin_summary.md
import json, math, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

vz, lz = sys.argv[1], sys.argv[2]
D = dict(np.load(vz, allow_pickle=True))
if len(sys.argv) > 3:
    D.update({k: v for k, v in np.load(sys.argv[3], allow_pickle=True).items() if k.startswith("bf16__img3m__")})
LZ = np.load(lz); M = json.loads(str(LZ["meta"])); K = sorted(M)
gz = np.load("results/gt_frodobots.npz")
tdf = pd.read_csv("results/gt_tests.csv").set_index("frame"); rel = tdf[tdf.reliable_path]
F = sorted(rel[rel.img_ok].index)
BASE = "vm_base_hqq4"
VARS = [("RTN, all 204 vision linears", "vm_rtn_all"), ("GPTQ, all 204 vision linears", "vm_gptq_all"),
        ("RTN, SigLIP MLP only (54)", "vm_rtn_mlp")]

def a(cfg, test, f):
    v = D.get(f"{cfg}__{test}__{f}")
    return None if v is None else np.asarray(v).reshape(-1, 8, 4)[0]
for c in [BASE] + [c for _, c in VARS]:
    for test, keys in (("img3m", F), ("lelan", K)):
        n = sum(a(c, test, k) is not None for k in keys)
        if n != len(keys):
            sys.exit(f"[VM] {c} {test}: {n}/{len(keys)} predictions - refusing a partial summary")
ade = lambda p, g: float(np.linalg.norm(p[:, :2] - g[:, :2], axis=1).mean())
bear = lambda xy: math.degrees(math.atan2(xy[1], xy[0]))
adiff = lambda x, y: abs((x - y + 180) % 360 - 180)
def picks(c):
    out = []
    for k in K:
        b = bear(a(c, "lelan", k)[-1, :2])
        out.append(adiff(b, bear(M[k]["target_xy"])) < adiff(b, bear(M[k]["distractor_xy"])))
    return np.array(out)
def mcnemar(x, y):
    n01, n10 = int(np.sum(x & ~y)), int(np.sum(~x & y))
    return binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
drive = {c: np.array([ade(a(c, "img3m", f), gz[f"act__{f}"]) for f in F]) for c in [BASE] + [c for _, c in VARS]}
pk = {c: picks(c) for c in drive}
has_bf = a("bf16", "img3m", F[0]) is not None
fid = {c: np.array([ade(a(c, "img3m", f), a("bf16", "img3m", f)) for f in F]) for c in drive} if has_bf else {}
L = ["# Vision linears as Marlin per-channel int4: accuracy (Kaggle T4, fp16 fake-quant)\n",
     "Same run and same deployed int4 LLM (GPTQ, hash-checked against the deployment) for every row; only the vision "
     "linears change. Baseline = the deployed HQQ4 vision (group 64). Marlin rows are per-channel int4 (groupsize -1; "
     "grouped Marlin is wrong on sm_87), simulated with fp16 dequantized weights. Image goal: img3m, 100 frames, 75% "
     "token pruning (as deployed). Object goal: LeLaN, 210 frames, no pruning (as deployed in language modes).\n",
     "Rules fixed before the run: a variant fails if image-goal driving error is higher than the baseline with Wilcoxon "
     "p < 0.05, or if object-goal accuracy is lower with McNemar p < 0.05 or by more than 5 points.\n",
     "| Vision weights | Image goal: error vs driven path | vs baseline (p) | Fidelity to bf16 | vs baseline outputs | Object goal: picks named object | vs baseline (p) | Pass |",
     "|---|---|---|---|---|---|---|---|"]
passed = {}
for name, c in [("HQQ4 group 64 (deployed baseline)", BASE)] + VARS:
    d = drive[c]; dd = d - drive[BASE]
    pw = wilcoxon(d, drive[BASE]).pvalue if c != BASE and np.any(dd != 0) else float("nan")
    pm = mcnemar(pk[c], pk[BASE]) if c != BASE else float("nan")
    dpts = 100 * (pk[c].mean() - pk[BASE].mean())
    ok = c == BASE or (not (dd.mean() > 0 and pw < 0.05) and not (dpts < 0 and pm < 0.05) and dpts >= -5)
    passed[c] = ok
    xb = np.mean([ade(a(c, "img3m", f), a(BASE, "img3m", f)) for f in F])
    L.append(f"| {name} | {d.mean():.3f} | {'-' if c == BASE else f'{dd.mean():+.3f} (p={pw:.2g})'} | "
             f"{fid[c].mean():.3f} | {xb:.3f} | {100 * pk[c].mean():.0f}% | "
             f"{'-' if c == BASE else f'{dpts:+.1f} pts (p={pm:.2g})'} | {'-' if c == BASE else ('yes' if ok else 'NO')} |"
             if has_bf else
             f"| {name} | {d.mean():.3f} | {'-' if c == BASE else f'{dd.mean():+.3f} (p={pw:.2g})'} | - | {xb:.3f} | "
             f"{100 * pk[c].mean():.0f}% | {'-' if c == BASE else f'{dpts:+.1f} pts (p={pm:.2g})'} | {'-' if c == BASE else ('yes' if ok else 'NO')} |")
cand = [c for c in ("vm_rtn_all", "vm_gptq_all") if passed[c]]
if cand:
    pick = min(cand, key=lambda c: fid[c].mean()) if has_bf else ("vm_gptq_all" if "vm_gptq_all" in cand else cand[0])
elif passed["vm_rtn_mlp"]:
    pick = "vm_rtn_mlp"
else:
    pick = None
label = ("adopt " + {c: n for n, c in VARS}[pick]) if pick else "no variant passes: keep HQQ4 vision"
L.append(f"\n**Decision (rule above): {label}.**\n")
open("results/vismarlin_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print(f"[VM] ADOPT={pick}")
