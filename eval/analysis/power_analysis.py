# Summary of eval/jetson/power_measure.py runs (results/power_<mode>.json) -> results/power_summary.md
import glob, json, os

rows = [json.load(open(p)) for p in sorted(glob.glob("results/power_*.json"))]
order = {"15W": 0, "25W": 1, "MAXN_SUPER": 2}
rows.sort(key=lambda r: order.get(r["power_mode"], 9))
L = ["# Power and energy per inference on the Jetson Orin Nano 8 GB", "",
     "Board input power (tegrastats VDD_IN, 100 ms samples), deployed runtime through launch.sh (jetson_clocks on),",
     "10 reference frames, back-to-back predictions. Energy = mean VDD_IN x wall time per prediction. Cases: 4 pose goal,",
     "6 image goal with a new goal every call, 6c image goal with an unchanged goal, 7 language (no pruning).",
     "Script: eval/jetson/power_measure.py. Measured 2026-09-28 without CUDA graphs (the runtime default since then),",
     "with the LLM key/value cache on; with graphs each prediction is 3-10% shorter.", ""]
if not rows:
    L += ["**Pending: not run on the Jetson yet.** Run `./eval/jetson/run_power_sweep.sh` on the Jetson."]
else:
    L += ["| Power mode | Case | Latency (fwd, median) | Wall time / inference | VDD_IN | Idle | Energy / inference | Above idle | GPU clock | tj max |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for c, v in r["cases"].items():
            L.append(f"| {r['power_mode']} | {c} | {v['fwd_ms_median']:.0f} ms | {v['wall_s_per_inference'] * 1000:.0f} ms | "
                     f"{v['vdd_in_mw'] / 1000:.2f} W | {r['idle_mw'] / 1000:.2f} W | {v['energy_j_per_inference']:.2f} J | "
                     f"{v['energy_above_idle_j']:.2f} J | " + (f"{v['gpu_mhz_median']:.0f} MHz | {v['tj_max_c']:.0f} C |" if "gpu_mhz_median" in v else "- | - |"))
if any("gpu_mhz_median" not in v for r in rows for v in r["cases"].values()):
    L += ["", "GPU clock and tj: - = not logged (that run was made before the logging was added)."]
missing = [m for m in order if m not in {r["power_mode"] for r in rows}]
if rows and missing:
    L += ["", f"**Pending:** {', '.join(missing)} not measured yet."]
open("results/power_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
