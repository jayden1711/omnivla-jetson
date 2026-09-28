# Temporal-reuse results on the sequential set: reads mf_results/cache_*/ (jetson_cache.py) and compares each method with
# 'full' on the same frames: latency, speedup, reused tokens, error vs full, driving error vs the actual path
# (paired Wilcoxon + bootstrap CI). Run from the repo root.
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sq = np.load("results/seq.npz"); fr = pd.read_csv("results/seq_frames.csv").set_index("frame")
def exec_w(L):
    w = np.array([max(0.0, min(2 * L, 0.3 * (k + 1)) - max(L, 0.3 * k)) / 0.3 for k in range(8)]); return w / w.sum()
def load(d):
    out = {}
    for f in glob.glob(f"{d}/*.npz"):
        z = np.load(f); out[os.path.basename(f)[:-4]] = {k: z[k] for k in z.files}
    return out
rng = np.random.default_rng(0); rows = []; lines = ["# Study 8: temporal reuse (Jetson, deployed config)", ""]
dirs = sorted(glob.glob("results/mf_results/cache_*"))
groups = {}
for d in dirs:
    m = re.match(r"results/mf_results/cache_(.+)@(\d)_s(\d+)$", d)
    if m:
        groups.setdefault((int(m[2]), int(m[3])), {})[m[1]] = load(d)
for (mode, stride), G in sorted(groups.items()):
    if "full" not in G:
        continue
    F = G["full"]; w = exec_w(1.05 if mode == 6 else 0.43)
    lines += [f"## mode {mode}, stride {stride} ({stride / 20:.2f} s between processed frames)", "",
              "| method | frames | latency median / mean ms | speedup (median) | LLM tokens | reused cur / goal | goal vision skipped | err vs full all8 (reuse frames) | err vs full exec | drive all8 | drive vs full [95% CI] | p |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tf = np.array([F[k]["t_fwd"] + F[k].get("t_sel", 0) for k in F]); base_med = np.median(tf)
    for meth, R in sorted(G.items(), key=lambda x: (x[0] != "full", x[0])):
        keys = sorted(set(R) & set(F))
        if not keys:
            continue
        t = np.array([R[k]["t_fwd"] + R[k].get("t_sel", 0) for k in keys])
        reuse_fr = [k for k in keys if not bool(R[k]["refresh"])]
        e_all = np.array([np.linalg.norm(R[k]["act"][:, :2] - F[k]["act"][:, :2], axis=1) for k in keys])
        e_re = np.array([np.linalg.norm(R[k]["act"][:, :2] - F[k]["act"][:, :2], axis=1) for k in reuse_fr]) if reuse_fr else np.zeros((1, 8))
        gt = [k for k in keys if fr.loc[k, "reliable"] and f"act__{k}" in sq.files]
        dm = np.array([np.linalg.norm(R[k]["act"][:, :2] - sq[f"act__{k}"][:, :2], axis=1).mean() for k in gt])
        df_ = np.array([np.linalg.norm(F[k]["act"][:, :2] - sq[f"act__{k}"][:, :2], axis=1).mean() for k in gt])
        d = dm - df_
        if meth != "full" and np.any(d != 0):
            boot = [rng.choice(d, len(d)).mean() for _ in range(3000)]; lo, hi = np.percentile(boot, [2.5, 97.5]); p = wilcoxon(dm, df_).pvalue
        else:
            lo = hi = p = np.nan
        r = dict(mode=mode, stride=stride, method=meth, n=len(keys), n_reuse_frames=len(reuse_fr), t_med=np.median(t) * 1000,
                 t_mean=t.mean() * 1000, speedup=base_med / np.median(t), n_llm=np.mean([R[k]["n_llm"] for k in keys]),
                 reuse_cur=np.mean([R[k]["n_reuse_cur"] for k in keys]), reuse_goal=np.mean([R[k]["n_reuse_goal"] for k in keys]),
                 goal_vis_skipped=np.mean([bool(R[k]["vis_goal_skipped"]) for k in keys]),
                 err_full_all8=e_re.mean(), err_full_exec=(e_re @ w).mean(), err_full_all8_allframes=e_all.mean(),
                 max_abs_diff=max(float(np.abs(R[k]["act"] - F[k]["act"]).max()) for k in keys),
                 drive=dm.mean(), drive_full=df_.mean(), drive_diff=d.mean(), lo=lo, hi=hi, p=p, n_gt=len(gt))
        rows.append(r)
        lines.append(f"| {meth} | {len(keys)} ({len(reuse_fr)} reuse) | {r['t_med']:.0f} / {r['t_mean']:.0f} | {r['speedup']:.2f}x | "
                     f"{r['n_llm']:.0f} | {r['reuse_cur']:.1f} / {r['reuse_goal']:.0f} | {r['goal_vis_skipped']:.2f} | {r['err_full_all8']:.3f} | "
                     f"{r['err_full_exec']:.3f} | {r['drive']:.3f} | {r['drive_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.2g} |")
    lines.append("")
pd.DataFrame(rows).round(4).to_csv("results/cache_summary.csv", index=False)
open("results/cache_summary.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
