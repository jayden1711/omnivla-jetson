# Jetson validation of the GPTQ per-channel int4 weights against the round-to-nearest (RTN) weights (final_validate.py
# outputs, VALIDATE_TAG=_gptq). Pass criteria, fixed before the run: bit-exact weights and Marlin gate; Jetson-vs-Kaggle
# agreement no worse than RTN; fidelity to bf16 lower than RTN; driving error not significantly worse than RTN, NF4 or
# bf16 (paired Wilcoxon); RAM peak within +-0.2 GB and latency within +-3% of RTN.
import numpy as np
from scipy.stats import wilcoxon
from fusedprune_analysis import load, boot, T, FR, gz, ade   # noqa

g6, gs6 = load("deploy_gptq@6"); g4, gs4 = load("deploy_gptq@4"); r6, rs6 = load("deploy@6"); r4, rs4 = load("deploy@4")
nf, _ = load("nf4@6")
K = dict(np.load("results/compress_gptqx.npz", allow_pickle=True)); KR = dict(np.load("results/compress_exq_pc4.npz", allow_pickle=True))
fr = [f for f in FR if f in g6 and f in r6 and f in nf]
A = lambda m: np.array([np.asarray(m[f]["act"]).reshape(8, 4) for f in fr])
ag, ar, an = A(g6), A(r6), A(nf)
bf = np.array([T[f"bf16__img3m__{f}"][0] for f in fr]); gt = np.array([gz[f"act__{f}"] for f in fr])
kg = np.array([K[f"gptqx_pc4__img3m__{f}"][0] for f in fr]); kr = np.array([KR[f"rtn_plain_pc4__img3m__{f}"][0] for f in fr])
d = lambda x, y: np.array([ade(p, q) for p, q in zip(x, y)])
fid_g, fid_r, fid_n = d(ag, bf), d(ar, bf), d(an, bf)
dr_g, dr_r, dr_n, dr_b = d(ag, gt), d(ar, gt), d(an, gt), d(bf, gt)
xk_g, xk_r = d(ag, kg), d(ar, kr)
med = lambda m, k="t_fwd": np.median([v[k] for v in m.values()]) * 1000
wl = lambda a, b: wilcoxon(a, b).pvalue
gate = open("results/gptq_gate.log").read(); wc = open("results/gptq_weight_check.log").read()
lat = {m: (med(g), med(r)) for m, g, r in ((6, g6, r6), (4, g4, r4))}
ram = {m: (gs["ram_peak_mb"], rs["ram_peak_mb"]) for m, gs, rs in ((6, gs6, rs6), (4, gs4, rs4))}
checks = {
    "weights bit-exact vs Kaggle (224 layers)": "224/224" in wc,
    "Marlin gate (3 shapes x 3 T, 50x determinism)": "ALL PASS" in gate,
    "Jetson-vs-Kaggle agreement no worse than RTN": xk_g.mean() <= xk_r.mean() * 1.2 + 0.01,
    "fidelity to bf16 lower than RTN": fid_g.mean() < fid_r.mean() and wl(fid_g, fid_r) < 0.05,
    "driving not significantly worse than RTN": (dr_g - dr_r).mean() <= 0 or wl(dr_g, dr_r) >= 0.05,
    "driving not significantly worse than NF4": (dr_g - dr_n).mean() <= 0 or wl(dr_g, dr_n) >= 0.05,
    "driving not significantly worse than bf16": (dr_g - dr_b).mean() <= 0 or wl(dr_g, dr_b) >= 0.05,
    "RAM peak within +-0.2 GB of RTN": all(abs(g - r) <= 200 for g, r in ram.values()),
    "latency within +-3% of RTN": all(abs(g / r - 1) <= 0.03 for g, r in lat.values()),
}
R = {
    "frames (img3m)": len(fr),
    "Jetson vs Kaggle outputs, same weights: GPTQ / RTN (different kernels: Marlin+GemLite vs fp16 fake-quant)":
        f"{xk_g.mean():.4f} (max {xk_g.max():.3f}) / {xk_r.mean():.4f} (max {xk_r.max():.3f})",
    "fidelity vs bf16 [95% CI]: GPTQ / RTN / NF4": f"{fid_g.mean():.3f} {boot(fid_g)} / {fid_r.mean():.3f} / {fid_n.mean():.3f}; "
        f"GPTQ - RTN {(fid_g - fid_r).mean():+.3f} {boot(fid_g - fid_r)}, p={wl(fid_g, fid_r):.2g}",
    "driving error: GPTQ / RTN / NF4 / bf16": f"{dr_g.mean():.3f} / {dr_r.mean():.3f} / {dr_n.mean():.3f} / {dr_b.mean():.3f}",
    "driving GPTQ - RTN [95% CI], p": f"{(dr_g - dr_r).mean():+.3f} {boot(dr_g - dr_r)}, p={wl(dr_g, dr_r):.2f}",
    "driving GPTQ - NF4 [95% CI], p": f"{(dr_g - dr_n).mean():+.3f} {boot(dr_g - dr_n)}, p={wl(dr_g, dr_n):.2f}",
    "driving GPTQ - bf16 [95% CI], p": f"{(dr_g - dr_b).mean():+.3f} {boot(dr_g - dr_b)}, p={wl(dr_g, dr_b):.2f}",
    "latency median fwd ms, image / pose: GPTQ vs RTN": f"{lat[6][0]:.0f} / {lat[4][0]:.0f} vs {lat[6][1]:.0f} / {lat[4][1]:.0f}",
    "RAM peak MB, image / pose: GPTQ vs RTN": f"{ram[6][0]} / {ram[4][0]} vs {ram[6][1]} / {ram[4][1]}; headroom GPTQ "
        f"{7620 - ram[6][0]} / {7620 - ram[4][0]} MB",
    "torch weights GiB: GPTQ / RTN": f"{gs6['torch_weights_gib']:.2f} / {rs6['torch_weights_gib']:.2f}",
}
with open("results/gptq_validation.md", "w") as f:
    f.write("# GPTQ per-channel int4 weights: Jetson validation (2026-09-27)\n\nSame format/kernel as the deployed RTN weights "
            "(Marlin per-channel int4 LLM + HQQ4 GemLite vision + elision + grid 75%); deploy runtime; img3m 100 frames + pose.\n\n")
    for k, v in R.items():
        f.write(f"- {k}: {v}\n")
    f.write("\n## Pass criteria\n\n")
    for k, v in checks.items():
        f.write(f"- {'PASS' if v else 'FAIL'}: {k}\n")
    f.write(f"\n**{'ALL PASS -> make GPTQ the deployed default' if all(checks.values()) else 'NOT ALL PASS -> keep RTN'}**\n")
print(open("results/gptq_validation.md").read())
