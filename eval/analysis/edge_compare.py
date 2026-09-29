# OmniVLA-7B (deployed int4 config) vs OmniVLA-edge: accuracy on every driving test, Jetson latency and memory.
# Accuracy: error vs the driven path (8 steps, action units), same frames and metric as tests_analysis.py; the language
# test as lang_analysis.py. 7B int4 = Jetson outputs of the deployed runtime where they exist (img3m, pose5), otherwise
# the Kaggle-simulated deployed weights (pose20, lang; identical int4 tensors, fp16 fake-quant kernels). Blind controls
# decide whether a test checks perception (both image controls clearly worse for the 7B fp16 model: Wilcoxon p < 0.01
# and >= 15% worse). A model is called better on a test only if the paired Wilcoxon test gives p < 0.05 AND the 95%
# bootstrap CI of the mean difference excludes 0; if only one of the two holds, the verdict is "mixed".
# Usage, from a folder with results/ (gt files, compress_tests.npz, compress_gptqx_lang.npz, compress_lang.npz,
# edge_tests.npz, mf_results/e2e_deploy_gptq@{4,6}) and cast_lang.npz:
#   python eval/analysis/edge_compare.py cast_lang.npz      -> results/edge_vs_7b.md, results/edge_vs_7b.csv
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "results"
D = {}
for f in ("compress_tests.npz", "compress_gptqx_lang.npz", "compress_lang.npz", "edge_tests.npz"):
    D.update(dict(np.load(f"{R}/{f}", allow_pickle=True)))
gz, tn = np.load(f"{R}/gt_frodobots.npz"), np.load(f"{R}/gt_tests.npz")
tdf = pd.read_csv(f"{R}/gt_tests.csv"); rel = tdf[tdf.reliable_path]
CZ = np.load(sys.argv[1]); CM = json.loads(str(CZ["meta"]))
def jetson(name):
    return {os.path.basename(p)[:-4]: np.load(p)["act"].reshape(8, 4) for p in glob.glob(f"{R}/mf_results/e2e_{name}/*.npz")
            if not p.endswith("_summary.npz")}
J = {"img3m": jetson("deploy_gptq@6"), "pose5": jetson("deploy_gptq@4")}
FR = {"img3m": sorted(rel[rel.img_ok].frame), "pose5": sorted(f for f in rel.frame if f"goal__{f}" in gz.files),
      "pose20": sorted(f for f in rel[rel.far_ok].frame if f"goal20__{f}" in tn.files),
      "lang": sorted(k for k in CM if CM[k]["reliable"])}
GT = lambda t, f: CZ[f"gt__{f}"] if t == "lang" else gz[f"act__{f}"]
rng = np.random.default_rng(0)
def get(cfg, t, f):
    if cfg == "int4" and t in J:
        return J[t][f]
    a = D.get(f"{'gptqx_pc4' if cfg == 'int4' else cfg}__{t}__{f}")
    return None if a is None else np.asarray(a).reshape(-1, 8, 4)[0]
def errs(cfg, t):
    return np.array([np.linalg.norm(get(cfg, t, f)[:, :2] - GT(t, f)[:, :2], axis=1).mean() for f in FR[t]])
def boot(x):
    i = rng.integers(0, len(x), (5000, len(x))); return np.percentile(x[i].mean(1), [2.5, 97.5])
for t, fr in FR.items():                                   # an empty test or missing outputs must not become a table row
    if not fr:
        sys.exit(f"[CMP] {t}: 0 frames - refusing to write a summary")
    for c in ("bf16", "int4", "edge", "fp16"):
        n = sum(get(c, t, f) is not None for f in fr) if c != "int4" or t not in J else sum(f in J[t] for f in fr)
        if n != len(fr):
            sys.exit(f"[CMP] {t}: {c} has outputs for {n}/{len(fr)} frames")

rows, L = [], []
say = L.append
NAMES = {"img3m": "Image goal (img3m)", "pose5": "Pose goal 5 s (pose5)", "pose20": "Pose goal 20 s (pose20)", "lang": "Language (CAST)"}
for t, fr in FR.items():
    real = errs("fp16", t)
    ctl = [(errs(f"fp16-{c}", t), c) for c in ("blank", "shuffled")]
    perc = all(wilcoxon(e, real).pvalue < 0.01 and e.mean() / real.mean() - 1 >= 0.15 for e, _ in ctl)
    b, q, e_ = errs("bf16", t), errs("int4", t), errs("edge", t)
    d = e_ - q; lo, hi = boot(d); p = wilcoxon(e_, q).pvalue
    sig_p, sig_ci = p < 0.05, lo > 0 or hi < 0
    verdict = ("7B better" if d.mean() > 0 else "edge better") if sig_p and sig_ci else \
              ("mixed (rank test and CI disagree)" if sig_p or sig_ci else "no significant difference")
    agree = None
    if t in J:                                             # Kaggle-simulated int4 vs the Jetson, same weights
        agree = np.mean([np.linalg.norm(J[t][f][:, :2] - np.asarray(D[f"gptqx_pc4__{t}__{f}"]).reshape(8, 4)[:, :2], axis=1).mean() for f in fr])
    r = dict(test=t, n=len(fr), checks_perception=perc, bf16=b.mean(), int4=q.mean(),
             int4_source="Jetson" if t in J else "Kaggle-simulated", edge=e_.mean(), edge_minus_int4=d.mean(), ci_lo=lo, ci_hi=hi,
             p=p, verdict=verdict, kaggle_vs_jetson=agree)
    if D.get(f"edge-nohist__{t}__{fr[0]}") is not None:
        r["edge_no_history"] = errs("edge-nohist", t).mean()
    rows.append(r)
T = pd.DataFrame(rows)
T.round(4).to_csv(f"{R}/edge_vs_7b.csv", index=False)

say("# OmniVLA-7B (this repo's int4 deployment) vs OmniVLA-edge\n")
say("## Accuracy (error vs the driven path, 8 steps, action units; lower is better)\n")
say("| Test | n | Test checks perception | 7B bf16 | 7B int4 (deployed) | OmniVLA-edge | edge - 7B int4 [95% CI], p | Verdict |")
say("|---|---|---|---|---|---|---|---|")
for r in rows:
    src = "" if r["int4_source"] == "Jetson" else " (Kaggle-sim.)"
    say(f"| {NAMES[r['test']]} | {r['n']} | {'yes' if r['checks_perception'] else 'no'} | {r['bf16']:.3f} | {r['int4']:.3f}{src} | "
        f"{r['edge']:.3f} | {r['edge_minus_int4']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], p={r['p']:.2g} | {r['verdict']} |")
say("\n7B int4 on img3m and pose5: outputs of the deployed runtime on the Jetson. On pose20 and language: the deployed int4 "
    "weights run on a Kaggle T4 with fp16 dequantized kernels (not Marlin/GemLite, not bit-exact). OmniVLA-edge: fp32, "
    "run off the Jetson with OmniVLA's edge preprocessing and the 5 real past frames (eval/edge/edge_eval.py); "
    "language samples have no past frames, so the current frame is repeated, as OmniVLA's CAST loader does. "
    "Kaggle-simulated vs Jetson outputs of the same int4 weights: "
    + ", ".join(f"{r['test']} {r['kaggle_vs_jetson']:.4f}" for r in rows if r["kaggle_vs_jetson"] is not None)
    + " action units on average.")
nh = [r for r in rows if "edge_no_history" in r]
if nh:
    say("Edge without past frames (current frame repeated, as run_omnivla_edge.py does): "
        + ", ".join(f"{r['test']} {r['edge_no_history']:.3f}" for r in nh) + ".")
say("\n## On the Jetson Orin Nano 8 GB\n")
say("| | 7B int4 (this repo) | OmniVLA-edge |")
say("|---|---|---|")
say("| Latency, pose goal | 375-395 ms with CUDA graphs (default; differs between boots), 424 ms without (MAXN_SUPER) | 112.8 ms (MAXN_SUPER + "
    "jetson_clocks); 131.7-151.5 ms at 25W |")
say("| Latency, image goal | 980-1012 ms with CUDA graphs, 1056 ms without (MAXN_SUPER) | same model call as pose |")
say("| Latency, language goal | 650-735 ms with CUDA graphs, 700-770 ms without (no pruning in language modes; "
    "results/jetson_validation_2026-09-28.md) | "
    "same model call as pose |")
say("| Memory | 4.14 GiB weights on the GPU; RAM peak 6565 / 6769 MB of 7620 (pose / image) | 1.10 GB peak GPU memory |")
say("| Weights on disk | 4.1 GB | 0.43 GB + CLIP ViT-B/32 for the text encoder (0.35 GB) |")
say("| Power, energy per inference (MAXN_SUPER) | 20.2 W, 8.8 J pose; 22.7 W, 23.8 J image goal; 21.5 W, 16.8 J "
    "language (results/power_summary.md) | not measured |")
say("\n7B: results/gptq_validation.md (deployed runtime, 100 frames). Edge: results/edge_jetson.md (measured earlier with "
    "OmniVLA's run_omnivla_edge.py and a timing wrapper around the model call, not with this repo's package; the edge "
    "latency covers the model call only, the 7B latency the forward pass of the runtime).")
open(f"{R}/edge_vs_7b.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
