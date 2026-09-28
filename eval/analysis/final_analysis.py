# Summary of the deployed config's validation run (final_validate.py outputs). Run from the repo root.
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from fusedprune_analysis import load, boot, T, FR, gz, ade   # noqa

m6, s6 = load("deploy@6"); m4, s4 = load("deploy@4"); nf, _ = load("nf4@6"); ref, _ = load("pq_marpc@6@spatial75")
nf4_4, nf4s4 = load("nf4@4_mem"); nf4_6, nf4s6 = load("nf4@6_mem")
fr = [f for f in FR if f in m6 and f in nf]
A = lambda m: np.array([m[f]["act"][0] for f in fr])
a, an, bf = A(m6), A(nf), np.array([T[f"bf16__img3m__{f}"][0] for f in fr]); g = np.array([gz[f"act__{f}"] for f in fr])
d = lambda x, y: np.array([ade(p, q) for p, q in zip(x, y)])
fid, dr, dn, db = d(a, bf), d(a, g), d(an, g), d(bf, g)
same = max(np.abs(m6[f]["act"] - ref[f]["act"]).max() for f in fr)
med = lambda m, k="t_fwd": np.median([v[k] for v in m.values()]) * 1000
R = {
 "frames (img3m)": len(fr), "max |d| vs research pipeline pq_marpc@6@spatial75": same,
 "pose latency median fwd / e2e ms": f"{med(m4):.0f} / {med(m4, 't_e2e'):.0f}  (NF4 {med(nf4_4):.0f})",
 "image latency median fwd / e2e ms": f"{med(m6):.0f} / {med(m6, 't_e2e'):.0f}  (NF4 {med(nf4_6):.0f})",
 "weights GiB": round(s6["torch_weights_gib"], 2),
 "torch peak GiB pose / image": f"{s4['torch_peak_gib']:.2f} / {s6['torch_peak_gib']:.2f}",
 "RAM peak MB pose / image": f"{s4['ram_peak_mb']} / {s6['ram_peak_mb']}",
 "RAM headroom MB pose / image (+-0.2 GB)": f"{7620 - s4['ram_peak_mb']} / {7620 - s6['ram_peak_mb']}  (NF4 {7620 - nf4s4['ram_peak_mb']} / {7620 - nf4s6['ram_peak_mb']})",
 "fidelity vs bf16 [95% CI]": f"{fid.mean():.3f} {boot(fid)}  (NF4 {d(an, bf).mean():.3f})",
 "driving error (all 8 steps)": f"{dr.mean():.3f}  (NF4 {dn.mean():.3f}, bf16 {db.mean():.3f})",
 "driving vs NF4 [95% CI], p": f"{(dr-dn).mean():+.3f} {boot(dr-dn)}, p={wilcoxon(dr, dn).pvalue:.2f}",
 "driving vs bf16 [95% CI], p": f"{(dr-db).mean():+.3f} {boot(dr-db)}, p={wilcoxon(dr, db).pvalue:.2f}",
}
with open("results/final_validation.md", "w") as f:
    f.write("# Final validation of the deployed config (2026-09-26)\n")
    f.write("Marlin per-channel int4 LLM + GemLite HQQ4 vision + elision + uniform-grid pruning of 75% of current-image tokens,\n"
            "pre-packed weights, deploy/omnivla_deploy.py on the Jetson Orin Nano. img3m image-goal test (100 reliable frames) and\n"
            "pose mode on the same frames. Single-prediction, in-distribution (FrodoBots) evaluation. Fresh tegrastats log per run.\n\n")
    for k, v in R.items():
        f.write(f"- {k}: {v}\n")
print("[FINAL]"); [print(f"  {k}: {v}") for k, v in R.items()]
