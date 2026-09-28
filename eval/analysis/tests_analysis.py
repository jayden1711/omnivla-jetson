# Real-driving tests (img3m, pose20, pose5): perception-dependence check and per-config results.
# A test counts as primary only if both blind controls (blank / shuffled current image) are worse than the real model:
# paired Wilcoxon p < 0.01 and >= 15% larger mean error vs the actual path.
# Usage: python eval/analysis/tests_analysis.py results/compress_tests.npz [more npz] [--tag tests] [--tp]
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "tests"
files = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--tag"]
D = {}
for p in files:
    D.update(dict(np.load(p, allow_pickle=True)))
META = {k[6:]: json.loads(str(D[k])) for k in D if k.startswith("meta__")}
gz = np.load("results/gt_frodobots.npz"); tn = np.load("results/gt_tests.npz")
tdf = pd.read_csv("results/gt_tests.csv").set_index("frame"); rel = tdf[tdf.reliable_path]
FR = {"img3m": sorted(rel[rel.img_ok].index), "pose20": sorted(f for f in rel[rel.far_ok].index if f"goal20__{f}" in tn.files),
      "pose5": sorted(f for f in rel.index if f"goal__{f}" in gz.files)}
rng = np.random.default_rng(0)

def ade(p, g, n=8):
    return np.linalg.norm(p[:n, :2] - g[:n, :2], axis=1).mean()
def hermite(goal, frac):
    gx, gy, c, s = goal; L = np.hypot(gx, gy) + 1e-6; t = frac
    h10, h01, h11 = t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
    return np.stack([h10 * L + h01 * gx + h11 * L * c, h01 * gy + h11 * L * s], 1)
def baselines(test, f):
    i = np.arange(1, 9)
    v = tdf.loc[f, "speed_now_mps"]
    out = {"straight ahead at current speed": np.stack([v * 0.3 * i / 0.25, 0 * i], 1)}
    if test == "pose5":
        g, frac = gz[f"goal__{f}"], i / 17
    elif test == "pose20":
        g, frac = tn[f"goal20__{f}"], i / 67
    else:
        g, frac = tn[f"imggoalpose__{f}"], np.minimum(3 * i / max(tdf.loc[f, "img_motion_s"] * 10, 1), 1)
    lab = " (oracle goal pose)" if test == "img3m" else ""
    out["straight line to goal" + lab] = np.stack([g[0] * frac, g[1] * frac], 1)
    out["goal interpolation" + lab] = hermite(g, frac)
    return out

def get(cfg, test, f):
    a = D.get(f"{cfg}__{test}__{f}")
    return None if a is None else a[0]

lines, rows = [], []
def say(s=""):
    print(s); lines.append(s)
say(f"Frames: " + ", ".join(f"{t} {len(v)}" for t, v in FR.items()) + " (reliable 2.4 s ground-truth windows).")
for test, frames in FR.items():
    say(f"\n## {test}")
    ref = "bf16"
    real = np.array([ade(get("fp16", test, f), gz[f"act__{f}"]) for f in frames])
    # perception-dependence check
    say("| Control / baseline | Error vs actual path (all 8) | Steps 1-5 | vs real fp16 model | p |")
    say("|---|---|---|---|---|")
    say(f"| real model (fp16) | {real.mean():.3f} | {np.mean([ade(get('fp16', test, f), gz[f'act__{f}'], 5) for f in frames]):.3f} | - | - |")
    verdict = []
    for v in ("blank", "shuffled"):
        b = np.array([ade(get(f"fp16-{v}", test, f), gz[f"act__{f}"]) for f in frames])
        b5 = np.mean([ade(get(f"fp16-{v}", test, f), gz[f"act__{f}"], 5) for f in frames])
        p = wilcoxon(b, real).pvalue
        rel_worse = b.mean() / real.mean() - 1
        verdict.append(p < 0.01 and rel_worse >= 0.15)
        say(f"| blind: {v} current image | {b.mean():.3f} | {b5:.3f} | {100 * rel_worse:+.0f}% | {p:.2g} |")
    bl = {}
    for f in frames:
        for k, w in baselines(test, f).items():
            bl.setdefault(k, []).append((ade(w, gz[f"act__{f}"]), ade(w, gz[f"act__{f}"], 5)))
    for k, v in bl.items():
        v = np.array(v); p = wilcoxon(v[:, 0], real).pvalue
        say(f"| model-free: {k} | {v[:, 0].mean():.3f} | {v[:, 1].mean():.3f} | {100 * (v[:, 0].mean() / real.mean() - 1):+.0f}% | {p:.2g} |")
    primary = all(verdict)
    say(f"**{test}: {'PRIMARY' if primary else 'NOT primary'}** (blind controls clearly worse: {verdict}).")
    # per-config results
    say(f"\n| Config | Fidelity vs bf16 all 8 | Fid. 1-5 | Drive all 8 | Drive 1-5 | Drive vs bf16 (95% CI) | p |")
    say("|---|---|---|---|---|---|---|")
    refd = np.array([ade(get(ref, test, f), gz[f"act__{f}"]) for f in frames])
    for c in [c for c in META if not c.endswith(("-blank", "-shuffled"))]:
        if get(c, test, frames[0]) is None:
            continue
        fid = np.array([ade(get(c, test, f), get(ref, test, f)) for f in frames])
        fid5 = np.array([ade(get(c, test, f), get(ref, test, f), 5) for f in frames])
        dr = np.array([ade(get(c, test, f), gz[f"act__{f}"]) for f in frames])
        dr5 = np.array([ade(get(c, test, f), gz[f"act__{f}"], 5) for f in frames])
        d = dr - refd
        p = wilcoxon(dr, refd).pvalue if np.any(d != 0) else np.nan
        boot = [rng.choice(d, len(d)).mean() for _ in range(3000)]
        say(f"| {c} | {fid.mean():.3f} | {fid5.mean():.3f} | {dr.mean():.3f} | {dr5.mean():.3f} | {d.mean():+.3f} "
            f"({np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}) | {p:.2g} |")
        rows.append(dict(test=test, primary=primary, config=c, fid_all=fid.mean(), fid_1_5=fid5.mean(), drive_all=dr.mean(),
                         drive_1_5=dr5.mean(), drive_minus_bf16=d.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                         p_vs_bf16=p, t4_ms=META[c]["t4_latency_s"] * 1000, weights_gib=META[c]["weight_bytes_gpu"] / 2**30))
T = pd.DataFrame(rows)
T.round(4).to_csv(f"results/{TAG}_summary.csv", index=False)
open(f"results/{TAG}_summary.md", "w").write("\n".join(lines) + "\n")

# figure: fidelity vs drive error per primary test
prim = T[T.primary]
if len(prim):
    tests = prim.test.unique()
    SURF = "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, len(tests), figsize=(max(7.5, 5.2 * len(tests)), 4.8), facecolor=SURF, squeeze=False)
    for ax, t in zip(axes[0], tests):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.6)
        s = prim[prim.test == t]
        b = s[s.config == "bf16"].drive_all.item()
        ax.axhline(b, color="#8a8984", lw=1, ls="--")
        placed = []
        for _, r in s.sort_values("fid_all").iterrows():
            vis_full = "visfp16" in r.config or r.config in ("bf16", "fp16") or "nf4llm" in r.config
            col, mk = ("#eb6834", "s") if vis_full else ("#2a78d6", "o")
            ax.errorbar(max(r.fid_all, 1e-3), r.drive_all, yerr=[[r.drive_minus_bf16 - r.ci_lo], [r.ci_hi - r.drive_minus_bf16]],
                        fmt=mk, color=col, ms=6, capsize=2, mec=SURF)
            dy = 4 + 9 * sum(1 for (fx, fy) in placed if abs(np.log10(max(fx, 1e-3)) - np.log10(max(r.fid_all, 1e-3))) < 0.12 and abs(fy - r.drive_all) < 0.06)
            placed.append((r.fid_all, r.drive_all))
            ax.annotate(r.config.replace("_", " "), (max(r.fid_all, 1e-3), r.drive_all), xytext=(4, dy), textcoords="offset points", fontsize=6.5, color="#52514e")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_title(f"{t}", loc="left", fontsize=10)
        ax.set_xlabel("fidelity: error vs bf16 (action units, symlog)")
    axes[0][0].set_ylabel("error vs actual driven path (action units)\nbars: 95% CI of difference from bf16")
    handles = [plt.Line2D([], [], color="#2a78d6", marker="o", ls=""), plt.Line2D([], [], color="#eb6834", marker="s", ls="")]
    axes[0][0].legend(handles, ["vision 4-bit", "vision full precision"], frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Compression limit, primary test (image goal ~3 m); dashed line: bf16", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"results/{TAG}_fid_vs_drive.png", dpi=200, facecolor=SURF)
print(f"[TA] saved results/{TAG}_summary.md/.csv")
