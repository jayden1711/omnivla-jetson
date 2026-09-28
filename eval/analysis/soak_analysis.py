# 30-minute soak summaries from results_soak*.csv (every 30 s) and tegra_soak*.log (tegrastats, 0.5 s).
# Default: pose mode with and without malloc_trim; --image: image-goal soaks with goal_refresh 1 and 3.
import os, re
import numpy as np
import pandas as pd

import sys
if "--image" in sys.argv:
    RUNS = [("image goal, goal_refresh=1 (default: goal vision cache only)", "results/results_soak6_r1.csv", "results/tegra_soak6_r1.log"),
            ("image goal, goal_refresh=3 (opt-in goal K/V reuse)", "results/results_soak6_r3.csv", "results/tegra_soak6_r3.log")]
    OUT, TITLE = "results/soak6_summary.md", "# 30-min soaks, image-goal mode, GPTQ default weights + goal cache (2026-09-27)\n" \
        "24 sequential clips cycled (goal changes every 5 predictions), one prediction per clip frame at the 1.05 s cadence.\n"
else:
    RUNS = [("without malloc_trim", "results/results_soak_notrim.csv", "results/tegra_soak_notrim.log"),
            ("with malloc_trim every 20 predictions (deployed)", "results/results_soak.csv", "results/tegra_soak.log")]
    OUT, TITLE = "results/soak_summary.md", "# 30-min soaks, deployed config, pose mode (continuous inference, 2026-09-26)\n"

slope = lambda x, y: np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0]
out = open(OUT, "w")
out.write(TITLE + "Orin thermal throttling starts near 99-105 C junction; jetson_clocks holds clocks at max. "
          "System RAM from tegrastats includes the model load in the first minute.\n")
for label, csvp, logp in RUNS:
    if not os.path.exists(csvp):
        continue
    S = pd.read_csv(csvp)
    L = open(logp).read().splitlines()
    ram = np.array([int(re.search(r"RAM (\d+)/", l).group(1)) for l in L])
    tj = np.array([float(re.search(r"tj@([\d.]+)C", l).group(1)) for l in L])
    cpu = [list(map(int, re.findall(r"\d+%@(\d+)", re.search(r"CPU \[([^\]]*)\]", l).group(1)))) for l in L]
    gr3d = np.array([int(re.search(r"GR3D_FREQ (\d+)%", l).group(1)) for l in L])
    pin = np.array([int(re.search(r"VDD_IN (\d+)mW", l).group(1)) for l in L])
    t = np.arange(len(L)) * 0.5 / 60
    late = t > 10
    R = {
        "duration min / inferences": f"{S.t_s.iloc[-1]/60:.1f} / {int(S.n.iloc[-1])}",
        "latency ms: first min / last min / max 30-s mean / worst single": f"{S.fwd_ms_mean.iloc[:2].mean():.1f} / {S.fwd_ms_mean.iloc[-2:].mean():.1f} / {S.fwd_ms_mean.max():.1f} / {S.fwd_ms_max.max():.1f}",
        "latency drift ms per 10 min": f"{10 * slope(S.t_s / 60, S.fwd_ms_mean):+.2f}",
        "torch allocated / reserved GiB (min=max?)": f"{S.torch_alloc_gib.min():.3f}-{S.torch_alloc_gib.max():.3f} / {S.torch_reserved_gib.min():.3f}-{S.torch_reserved_gib.max():.3f}",
        "system RAM MB at 10 min / end / max": f"{ram[np.argmin(abs(t - 10))]} / {ram[-20:].mean():.0f} / {ram.max()}",
        "system RAM drift after 10 min, MB per 10 min": f"{10 * slope(t[late], ram[late]):+.1f}",
        "junction temp C max / last 5 min": f"{tj.max():.1f} / {tj[t > t[-1] - 5].mean():.1f}",
        "CPU clock MHz min / max (all cores, all samples)": f"{min(min(c) for c in cpu)} / {max(max(c) for c in cpu)}",
        "GPU load % median / board power W mean": f"{np.median(gr3d):.0f} / {pin.mean()/1000:.1f}",
    }
    if "rss_anon_mb" in S:
        m = S.t_s > 600
        R["process RssAnon MB start / end; drift after 10 min per 10 min"] = f"{S.rss_anon_mb.iloc[0]} / {S.rss_anon_mb.iloc[-1]}; {10 * slope(S.t_s[m] / 60, S.rss_anon_mb[m]):+.1f}"
        R["worst call incl. preprocessing ms / worst malloc_trim ms"] = f"{S.call_ms_max.max():.1f} / {S.trim_ms_max.max():.1f}"
    if "kv_reuse_frac" in S:
        R["goal K/V reused on (fraction of predictions) / latency 30-s mean ms range"] = f"{S.kv_reuse_frac.mean():.2f} / {S.fwd_ms_mean.min():.0f}-{S.fwd_ms_mean.max():.0f}"
    out.write(f"\n## {label}\n"); [out.write(f"- {k}: {v}\n") for k, v in R.items()]
    print(f"[SOAK] {label}"); [print(f"  {k}: {v}") for k, v in R.items()]
out.close()
