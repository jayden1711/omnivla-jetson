# Choose the goal K/V refresh interval from the deploy_refresh_sweep.py runs (mf_results/dcache_g_r{R}_s{S}/).
# Reference = r1 (no K/V reuse). K/V reuse is approximate because the LLM attention is bidirectional.
# Decision rule, fixed before the sweep: the largest N such that every cache age up to (N-1) x 1.05 s has, per 1 s age
# bin, mean error vs the uncached model <= 0.20 action units and driving error not significantly worse (Wilcoxon p >= 0.05
# or mean diff <= 0). Bins pool the no-refresh runs at strides 5, 10 and 21.
import glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sq = np.load("results/seq.npz"); fr = pd.read_csv("results/seq_frames.csv").set_index("frame")
def load(r, s):
    d = f"results/mf_results/dcache_g_r{r}_s{s}"
    return {os.path.basename(p)[:-4]: dict(np.load(p)) for p in glob.glob(f"{d}/*.npz")}
def ew(L):
    w = np.array([max(0.0, min(2 * L, 0.3 * (k + 1)) - max(L, 0.3 * k)) / 0.3 for k in range(8)]); return w / w.sum()
W = ew(1.05)
xy = lambda a: np.asarray(a).reshape(8, 4)[:, :2]
rng = np.random.default_rng(0)
def ci(x):
    b = [rng.choice(x, len(x)).mean() for _ in range(3000)]; return np.percentile(b, [2.5, 97.5])
lines = ["# Goal-cache refresh study (deploy runtime, GPTQ default weights; 24 sequential clips)", "",
         "Reference: the same runtime with goal_refresh=1 (goal vision features cached = exact; no K/V reuse).", ""]
rows = []
for s in (5, 10, 21):
    ref = load(1, s)
    for r in (999,) if s != 21 else (2, 3, 4, 999):
        R = load(r, s)
        for k in R:
            if k not in ref or not bool(R[k]["kv_reused"]):
                continue
            e = np.linalg.norm(xy(R[k]["act"]) - xy(ref[k]["act"]), axis=1)
            gt = f"act__{k}"
            ok = bool(fr.loc[k, "reliable"]) and gt in sq.files
            dm = np.linalg.norm(xy(R[k]["act"]) - sq[gt][:, :2], axis=1).mean() if ok else np.nan
            dr = np.linalg.norm(xy(ref[k]["act"]) - sq[gt][:, :2], axis=1).mean() if ok else np.nan
            rows.append(dict(stride=s, refresh=r, frame=k, age_s=float(R[k]["age_s"]), err=e.mean(), err_exec=e @ W, drive=dm, drive_ref=dr))
D = pd.DataFrame(rows)
# ---- error vs cache age (no-refresh runs) ----
A = D[D.refresh == 999].copy(); A["bin"] = np.ceil(A.age_s - 1e-9).clip(1, 5).astype(int)
lines += ["## Error vs cache age (no refresh within a clip; strides 5/10/21 pooled)", "",
          "| age bin (s) | frames | err vs uncached (all 8) | err (exec @1.05 s) | drive diff vs uncached [95% CI] | p | rule ok |", "|---|---|---|---|---|---|---|"]
okbins = {}
for b, g in A.groupby("bin"):
    g2 = g.dropna(subset=["drive"]); d = (g2.drive - g2.drive_ref).values
    p = wilcoxon(g2.drive, g2.drive_ref).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
    lo, hi = ci(d)
    ok = g.err.mean() <= 0.20 and (d.mean() <= 0 or (p >= 0.05))
    okbins[b] = ok
    lines.append(f"| ({b - 1}, {b}] | {len(g)} | {g.err.mean():.3f} | {g.err_exec.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.2g} | {'yes' if ok else 'NO'} |")
max_age = 0
for b in sorted(okbins):
    if okbins[b]:
        max_age = b
    else:
        break
N = int(np.floor(max_age / 1.05)) + 1
# ---- per refresh interval at the deployed cadence ----
lines += ["", "## Refresh interval at the deployed image-goal cadence (one prediction per 1.05 s, 120 frames)", "",
          "| goal_refresh N | K/V reused | latency median / mean ms | err vs uncached (reuse frames) | drive diff vs uncached (all frames) [95% CI] | p |",
          "|---|---|---|---|---|---|"]
ref21 = load(1, 21)
for r in (1, 2, 3, 4, 999):
    R = load(r, 21); ks = sorted(set(R) & set(ref21))
    t = np.array([R[k]["t_fwd"] for k in ks]) * 1000
    reu = np.mean([bool(R[k]["kv_reused"]) for k in ks])
    e = D[(D.stride == 21) & (D.refresh == r)].err.mean() if r > 1 else 0.0
    gk = [k for k in ks if bool(fr.loc[k, "reliable"]) and f"act__{k}" in sq.files]
    dm = np.array([np.linalg.norm(xy(R[k]["act"]) - sq[f"act__{k}"][:, :2], axis=1).mean() for k in gk])
    dr = np.array([np.linalg.norm(xy(ref21[k]["act"]) - sq[f"act__{k}"][:, :2], axis=1).mean() for k in gk])
    d = dm - dr; lo, hi = ci(d) if np.any(d != 0) else (0.0, 0.0)
    p = wilcoxon(dm, dr).pvalue if np.any(d != 0) else np.nan
    lines.append(f"| {r if r < 999 else 'goal change only (>=5 here)'} | {reu * 100:.0f}% | {np.median(t):.0f} / {t.mean():.0f} | {e:.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.2g} |")
lines += ["", f"**Decision:** largest age with every bin passing = {max_age} s -> goal_refresh N = floor({max_age} / 1.05) + 1 = **{N}** "
          f"(max cache age at 1.05 s cadence = {(N - 1) * 1.05:.2f} s). Clips are 5 s, so ages beyond 5 s are untested."]
D.to_csv("results/refresh_frames.csv", index=False)
open("results/refresh_summary.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
