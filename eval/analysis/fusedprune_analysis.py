# Fused low-bit kernels + token pruning: pose-mode latency and image-goal accuracy (fidelity vs bf16, driving error),
# paired against the unpruned model with the same kernel and against NF4. Also imported by final_analysis / gptq_validation.
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "results"
T = dict(np.load(f"{R}/compress_tests.npz", allow_pickle=True))
gz = np.load(f"{R}/gt_frodobots.npz"); tdf = pd.read_csv(f"{R}/gt_tests.csv")
FR = sorted(tdf[tdf.reliable_path & tdf.img_ok].frame)
ade = lambda a, b: np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean()
def load(name):
    for d in (f"{R}/mf_results/e2e_{name}", f"{R}/mf_results/e2e_{name}_mem"):
        m = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in glob.glob(f"{d}/*.npz") if not p.endswith(("_summary.npz", ".tmp.npz"))}
        if m:
            s = dict(np.load(f"{d}/_summary.npz")) if os.path.exists(f"{d}/_summary.npz") else {}
            return m, {k: v.item() for k, v in s.items()}
    return {}, {}
def boot(x, n=10000):
    i = np.random.default_rng(0).integers(0, len(x), (n, len(x))); return np.percentile(x[i].mean(1), [2.5, 97.5]).round(3).tolist()
def drive(m, fr):
    return np.array([ade(m[f]["act"][0], gz[f"act__{f}"]) for f in fr])
nf, _ = load("nf4@6")
rows = []
for K in ("pq_marpc", "gl_pq_q2lora"):
    # gl_pq_q2lora@6 has 10 frames only and is bit-identical there to gl_q2lora@6 (100 frames), used as the baseline
    base6, _ = load("gl_q2lora@6" if K == "gl_pq_q2lora" else f"{K}@6")
    for P in ("", "attn50", "attn75", "attn88", "hybrid88", "spatial75"):
        tag = f"@{P}" if P else ""
        m6, s6 = load(f"{K}@6{tag}" if (P or K != "gl_pq_q2lora") else "gl_q2lora@6"); m4, s4 = load(f"{K}@4{tag}")
        if not (m6 or m4):
            continue
        r = dict(kernel=K, prune=P or "none",
                 pose_fwd_ms=np.median([v["t_fwd"] for v in m4.values()]) * 1000 if m4 else np.nan, pose_n=len(m4),
                 pose_ram_peak_mb=s4.get("ram_peak_mb"))
        if m6 and base6:
            fr = [f for f in FR if f in m6 and f in base6 and f in nf]
            dr, db, dn = drive(m6, fr), drive(base6, fr), drive(nf, fr)
            fid = np.array([ade(m6[f]["act"][0], T[f"bf16__img3m__{f}"][0]) for f in fr])
            r.update(img_n=len(fr), img_fwd_ms=np.median([v["t_fwd"] for v in m6.values()]) * 1000,
                     fid_vs_bf16=fid.mean(), drive=dr.mean(),
                     vs_unpruned=(dr - db).mean(), vs_unpruned_ci=boot(dr - db),
                     p_unpruned=wilcoxon(dr, db).pvalue if np.any(dr != db) else np.nan,
                     vs_nf4=(dr - dn).mean(), vs_nf4_ci=boot(dr - dn), p_nf4=wilcoxon(dr, dn).pvalue)
        rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(f"{R}/fusedprune_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("[FPRUNE]\n" + D.round(3).to_string(index=False))
