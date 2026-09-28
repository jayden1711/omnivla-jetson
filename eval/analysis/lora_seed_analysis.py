# 2-bit HQQ + rank-16 correction: execution-weighted vs uniform loss, 3 seeds each. Seed-level Welch t-test and exact
# permutation test, plus a frame-level paired Wilcoxon on the per-frame mean over seeds.
import itertools, numpy as np, pandas as pd
from scipy.stats import ttest_ind, wilcoxon
T = pd.read_csv("results/lora_seeds_summary.csv").set_index("config")
D = {}
for f in ("results/compress_lora_uni.npz", "results/compress_lora_exec.npz", "results/compress_lora_seedsa.npz", "results/compress_lora_seedsb.npz"):
    D.update(dict(np.load(f, allow_pickle=True)))
ref = np.load("results/compress_tests.npz"); gz = np.load("results/gt_frodobots.npz")
tdf = pd.read_csv("results/gt_tests.csv"); FR = sorted(tdf[tdf.reliable_path & tdf.img_ok].frame)
def ew(L):
    w = np.array([max(0.0, min(2 * L, 0.3 * (k + 1)) - max(L, 0.3 * k)) / 0.3 for k in range(8)]); return w / w.sum()
W = {"all8": np.ones(8) / 8, "L0.43": ew(0.43), "L1.05": ew(1.05), "L1.43": ew(1.43), "L2.2": ew(2.2)}
arms = {a: [f"hqq2_vis4_lora_shared_{a}_s{s}_spatial75" for s in (0, 1, 2)] for a in ("uniform", "exec_deploy")}
lines = ["# Study 6 (1a): 2-bit + rank-16 correction, uniform vs execution-weighted loss (3 seeds each)", "",
         "| metric | uniform seeds 0/1/2 | exec_deploy seeds 0/1/2 | mean diff (exec - uni) | Welch p | permutation p | frame-level p (seed-mean) |",
         "|---|---|---|---|---|---|---|"]
rows = []
for kind in ("fid", "drive"):
    for ln, w in W.items():
        per = {}
        for a, cfgs in arms.items():
            M = []
            for c in cfgs:
                P = np.stack([D[f"{c}__img3m__{f}"][0][:, :2] for f in FR])
                R = np.stack([ref[f"bf16__img3m__{f}"][0][:, :2] for f in FR]) if kind == "fid" else np.stack([gz[f"act__{f}"][:, :2] for f in FR])
                M.append(np.linalg.norm(P - R, axis=2) @ w)
            per[a] = np.array(M)                                    # (3 seeds, 100 frames)
        u, e = per["uniform"].mean(1), per["exec_deploy"].mean(1)
        allm = np.concatenate([u, e]); obs = e.mean() - u.mean()
        perm = [np.mean(allm[list(i)]) - np.mean(allm[[j for j in range(6) if j not in i]]) for i in itertools.combinations(range(6), 3)]
        pp = np.mean([abs(x) >= abs(obs) - 1e-12 for x in perm])
        pw = wilcoxon(per["exec_deploy"].mean(0), per["uniform"].mean(0)).pvalue
        lines.append(f"| {kind} {ln} | {' / '.join(f'{x:.3f}' for x in u)} | {' / '.join(f'{x:.3f}' for x in e)} | {obs:+.3f} | "
                     f"{ttest_ind(e, u, equal_var=False).pvalue:.2g} | {pp:.2f} | {pw:.2g} |")
        rows.append(dict(metric=f"{kind}_{ln}", **{f"uni_s{i}": u[i] for i in range(3)}, **{f"exec_s{i}": e[i] for i in range(3)}, diff=obs, perm_p=pp, frame_p=pw))
pd.DataFrame(rows).round(4).to_csv("results/lora_seeds_test.csv", index=False)
open("results/lora_seeds_test.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
