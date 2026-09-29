# GPTQ int4 of omnivla-finetuned-cast on held-out CAST episodes (language mode, no pruning). Rules fixed before the run
# (2026-09-28):
#  - samples: the held-out side of build/cast_split.json (split by recording, no overlap with calibration), reliable only
#  - error = mean distance to the driven waypoints over 8 steps (action units), as eval/analysis/lang_analysis.py
#  - checks on bf16 and int4 (as lang_analysis.py): a control is "clearly worse" if paired Wilcoxon p < 0.01 and its mean
#    error is >= 15% higher; perception = blank and shuffled image both clearly worse; instruction use = shuffled
#    instruction clearly worse
#  - int4 FAILS if (a) its error is higher than bf16 with paired Wilcoxon p < 0.05, or (b) bf16 passes the instruction
#    check and int4 does not, or (c) turn direction (samples ending >= 1 unit to the side) is correct in > 5 points fewer
#    samples than bf16 with McNemar p < 0.05
# Usage: python eval/analysis/cast_gptq_analysis.py cast_lang.npz build/cast_split.json compress_castgptq.npz
#   -> results/cast_gptq_summary.md
import json, sys
import numpy as np
from scipy.stats import binomtest, wilcoxon

cz, sp, *files = sys.argv[1:]
CZ = np.load(cz); CM = json.loads(str(CZ["meta"])); S = json.load(open(sp))
D = {}
for p in files:
    D.update(dict(np.load(p, allow_pickle=True)))
K = [k for k in S["heldout"] if CM[k]["reliable"]]
assert not set(K) & set(S["calibration"])
GT = {k: CZ[f"gt__{k}"] for k in K}
def pred(c, k):
    a = D.get(f"{c}__lang__{k}")
    return None if a is None else np.asarray(a).reshape(-1, 8, 4)[0]
for c in ("cast_bf16", "castgptq_pc4"):
    for v in ("", "-blank", "-shuffled", "-lang_shuffled"):
        n = sum(pred(c + v, k) is not None for k in K)
        if n != len(K):
            sys.exit(f"[CASTQ] {c}{v}: {n}/{len(K)} predictions - refusing a partial summary")
err = lambda c: np.array([np.linalg.norm(pred(c, k)[:, :2] - GT[k][:, :2], axis=1).mean() for k in K])
side = [k for k in K if abs(GT[k][-1, 1]) >= 1]
turn = lambda c: np.array([np.sign(pred(c, k)[-1, 1]) == np.sign(GT[k][-1, 1]) for k in side])
def clearly_worse(c, v):
    a, b = err(c), err(f"{c}-{v}")
    p = wilcoxon(b, a).pvalue
    return p < 0.01 and b.mean() >= 1.15 * a.mean(), b.mean() / a.mean() - 1, p
def mcnemar(x, y):
    n01, n10 = int(np.sum(x & ~y)), int(np.sum(~x & y))
    return binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
L = ["# omnivla-finetuned-cast: GPTQ int4 on held-out CAST episodes (Kaggle T4, fp16 fake-quant)\n",
     f"{len(K)} reliable held-out samples ({S['n_recordings_heldout']} recordings); GPTQ calibrated on {len(S['calibration'])} "
     f"samples from {S['n_recordings_cal']} other recordings (split by recording: {S['rule']}). Language mode, no token "
     "pruning. int4 = GPTQ per-channel int4 (Marlin format) for the LLM and the vision linears; simulated with fp16 "
     "dequantized weights, not the Jetson's kernels. In-distribution: omnivla-finetuned-cast was trained on CAST.\n",
     "| Model | Error (8 steps) | Blank image | Shuffled image | Shuffled instruction | Checks perception | Uses instruction | Turn direction correct |",
     "|---|---|---|---|---|---|---|---|"]
chk = {}
for name, c in (("bf16", "cast_bf16"), ("GPTQ int4", "castgptq_pc4")):
    cells, v = [], {}
    for x in ("blank", "shuffled", "lang_shuffled"):
        ok, d, p = clearly_worse(c, x); v[x] = ok
        cells.append(f"{d * 100:+.0f}% (p={p:.2g})")
    chk[c] = v
    L.append(f"| {name} | {err(c).mean():.3f} | " + " | ".join(cells) + f" | {'yes' if v['blank'] and v['shuffled'] else 'no'} | "
             f"{'yes' if v['lang_shuffled'] else 'no'} | {100 * turn(c).mean():.0f}% of {len(side)} |")
eb, eq = err("cast_bf16"), err("castgptq_pc4"); pw = wilcoxon(eq, eb).pvalue
tb, tq = turn("cast_bf16"), turn("castgptq_pc4"); pt = mcnemar(tq, tb); dt = 100 * (tq.mean() - tb.mean())
fid = np.mean([np.linalg.norm(pred("castgptq_pc4", k)[:, :2] - pred("cast_bf16", k)[:, :2], axis=1).mean() for k in K])
fa = eq.mean() > eb.mean() and pw < 0.05
fb = chk["cast_bf16"]["lang_shuffled"] and not chk["castgptq_pc4"]["lang_shuffled"]
fc = dt < -5 and pt < 0.05
L += ["", f"int4 vs bf16: error {eq.mean() - eb.mean():+.3f} action units (Wilcoxon p={pw:.2g}); fidelity to bf16 {fid:.3f}; "
      f"turn direction {dt:+.1f} pts (McNemar p={pt:.2g}).", "",
      f"Rules: (a) error not significantly higher: {'FAIL' if fa else 'pass'}; (b) instruction use kept: "
      f"{'FAIL' if fb else 'pass'}{' (bf16 does not pass the instruction check: rule not applicable)' if not chk['cast_bf16']['lang_shuffled'] else ''}; "
      f"(c) turn direction kept: {'FAIL' if fc else 'pass'}.", "",
      f"**Result: {'int4 FAILS' if fa or fb or fc else 'no significant accuracy loss from GPTQ int4'}.**"]
open("results/cast_gptq_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
