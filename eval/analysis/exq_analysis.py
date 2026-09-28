# Execution-weighted quantization results on the image-goal test (img3m). Fidelity = distance to bf16, drive = distance
# to the driven path (xy, action units), over all 8 steps and weighted by execution overlap at latency L.
# Paired per-frame Wilcoxon + bootstrap 95% CI. Usage: python eval/analysis/exq_analysis.py TAG file.npz [...] [--pairs ext:base,...]
import json, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

TAG = sys.argv[1]
files = [a for a in sys.argv[2:] if a.endswith(".npz")]
pairs = [p.split(":") for p in sys.argv[sys.argv.index("--pairs") + 1].split(",")] if "--pairs" in sys.argv else []
D = {}
for f in files:
    D.update(dict(np.load(f, allow_pickle=True)))
ref = np.load("results/compress_tests.npz"); gz = np.load("results/gt_frodobots.npz")
tdf = pd.read_csv("results/gt_tests.csv"); FR = sorted(tdf[tdf.reliable_path & tdf.img_ok].frame)
def exec_w(L):
    return np.array([max(0.0, min(2 * L, 0.3 * (k + 1)) - max(L, 0.3 * k)) / 0.3 for k in range(8)])
LATS = {"all8": np.ones(8), "L0.43": exec_w(0.43), "L1.05": exec_w(1.05), "L1.43": exec_w(1.43), "L2.2": exec_w(2.2)}
cfgs = sorted({k.split("__")[0] for k in D if "__img3m__" in k})
per = {}                                                      # cfg -> metric -> (n_frames,)
for c in cfgs:
    if not all(f"{c}__img3m__{f}" in D for f in FR):
        print(f"[EXQ] {c}: incomplete ({sum(f'{c}__img3m__{f}' in D for f in FR)} frames), skipped"); continue
    P = np.stack([D[f"{c}__img3m__{f}"][0][:, :2] for f in FR])
    B = np.stack([ref[f"bf16__img3m__{f}"][0][:, :2] for f in FR]); G = np.stack([gz[f"act__{f}"][:, :2] for f in FR])
    ef, ed = np.linalg.norm(P - B, axis=2), np.linalg.norm(P - G, axis=2)          # (n, 8)
    per[c] = {}
    for ln, w in LATS.items():
        w = w / w.sum()
        per[c][f"fid_{ln}"] = ef @ w; per[c][f"drive_{ln}"] = ed @ w
    per[c]["step_fid"] = ef.mean(0); per[c]["step_drive"] = ed.mean(0)
rows = []
for c, v in per.items():
    rows.append(dict(config=c, **{k: float(np.mean(x)) for k, x in v.items() if not k.startswith("step")},
                     **{f"fid_s{k + 1}": float(v["step_fid"][k]) for k in range(8)}))
T = pd.DataFrame(rows).set_index("config").round(4); T.to_csv(f"results/{TAG}_summary.csv")
rng = np.random.default_rng(0)
lines = [f"# {TAG}: img3m, {len(FR)} frames", "", "| config | fid all8 | fid L0.43 | fid L1.05 | fid L1.43 | fid L2.2 | drive all8 | drive L1.05 | drive L2.2 |",
         "|---|---|---|---|---|---|---|---|---|"]
for c, r in T.iterrows():
    lines.append(f"| {c} | {r.fid_all8:.3f} | {r['fid_L0.43']:.3f} | {r['fid_L1.05']:.3f} | {r['fid_L1.43']:.3f} | {r['fid_L2.2']:.3f} | "
                 f"{r.drive_all8:.3f} | {r['drive_L1.05']:.3f} | {r['drive_L2.2']:.3f} |")
prow = []
if pairs:
    lines += ["", "## Paired comparisons (extension - baseline; negative = extension better)", "",
              "| extension | baseline | metric | diff | 95% CI | p |", "|---|---|---|---|---|---|"]
    for e, b in pairs:
        if e not in per or b not in per:
            continue
        for met in ("fid_all8", "fid_L0.43", "fid_L1.05", "fid_L1.43", "fid_L2.2", "drive_all8", "drive_L1.05", "drive_L2.2"):
            d = per[e][met] - per[b][met]
            boot = [rng.choice(d, len(d)).mean() for _ in range(3000)]
            p = wilcoxon(per[e][met], per[b][met]).pvalue if np.any(d != 0) else np.nan
            lo, hi = np.percentile(boot, [2.5, 97.5])
            lines.append(f"| {e} | {b} | {met} | {d.mean():+.4f} | ({lo:+.4f}, {hi:+.4f}) | {p:.2g} |")
            prow.append(dict(ext=e, base=b, metric=met, diff=d.mean(), lo=lo, hi=hi, p=p))
    pd.DataFrame(prow).to_csv(f"results/{TAG}_pairs.csv", index=False)
open(f"results/{TAG}_summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
