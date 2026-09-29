# Does the CAST checkpoint lose anything outside language mode? omnivla-finetuned-cast vs omnivla-original, both as the
# deployed int4 runtime on the Jetson (GPTQ LLM + Marlin vision; eval/jetson/cross_mode_eval.py), on the validated
# image-goal test (img3m), the 5 s pose test (pose5) and the object-goal test (LeLaN), with the same controls; bf16
# references where they exist (original: compress_tests.npz / compress_lelan.npz; CAST: compress_castxm.npz).
# Rules (as tests_analysis.py / lelan_analysis.py): a control is "clearly worse" if paired Wilcoxon p < 0.01 and its
# error is >= 15% higher (LeLaN: >= 15 points lower picks rate, McNemar p < 0.01). CAST vs original: paired Wilcoxon
# (error) / McNemar (picks); "loses" = worse with p < 0.05.
# Usage (from the research folder with results/): python eval/analysis/cross_mode_analysis.py cross_jet_cast.npz
#   cross_jet_orig.npz [compress_castxm.npz] -> results/cross_mode_summary.md
import json, math, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

D = {}
for p in sys.argv[1:]:
    D.update({k: v for k, v in np.load(p, allow_pickle=True).items()})
for p in ("results/compress_tests.npz", "results/compress_lelan.npz"):
    D.update({k: v for k, v in np.load(p, allow_pickle=True).items() if k.startswith("bf16__")})
gz = np.load("results/gt_frodobots.npz"); LZ = np.load("results/lelan_lang.npz"); LM = json.loads(str(LZ["meta"]))
tdf = pd.read_csv("results/gt_tests.csv").set_index("frame"); rel = tdf[tdf.reliable_path]
FR = {"img3m": sorted(rel[rel.img_ok].index), "pose5": sorted(f for f in rel.index if f"goal__{f}" in gz.files), "lelan": sorted(LM)}
a = lambda c, t, f: None if f"{c}__{t}__{f}" not in D else np.asarray(D[f"{c}__{t}__{f}"]).reshape(-1, 8, 4)[0]
ade = lambda p, g: float(np.linalg.norm(p[:, :2] - g[:, :2], axis=1).mean())
bear = lambda xy: math.degrees(math.atan2(xy[1], xy[0])); adiff = lambda x, y: abs((x - y + 180) % 360 - 180)
def err(c, t):
    if any(a(c, t, f) is None for f in FR[t]):
        return None
    return np.array([ade(a(c, t, f), gz[f"act__{f}"]) for f in FR[t]])
def picks(c):
    if any(a(c, "lelan", k) is None for k in FR["lelan"]):
        return None
    return np.array([adiff(bear(a(c, "lelan", k)[-1, :2]), bear(LM[k]["target_xy"])) < adiff(bear(a(c, "lelan", k)[-1, :2]), bear(LM[k]["distractor_xy"]))
                     for k in FR["lelan"]])
def mcn(x, y):
    n01, n10 = int(np.sum(x & ~y)), int(np.sum(~x & y)); return binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
L = ["# CAST checkpoint vs original checkpoint outside language mode\n",
     "Both as the deployed int4 runtime on the Jetson (GPTQ int4 LLM + GPTQ int4 Marlin vision; 75% token pruning in pose and "
     "image-goal modes, none in language mode), same frames and controls. omnivla-finetuned-cast was fine-tuned from the "
     "original on CAST; these tests use FrodoBots (image goal, 5 s pose goal) and LeLaN (object goal), which are in the "
     "original's training mix. Errors in action units vs the human's driven path; lower is better.\n"]
verdict = []
for t, title in (("img3m", "Image goal (img3m, 100 frames)"), ("pose5", "Pose goal 5 s ahead (pose5)")):
    L += [f"## {title}\n", "| Model | Error | Blank image | Shuffled image | Checks perception |", "|---|---|---|---|---|"]
    for name, c in (("original int4 (Jetson)", "jet_orig"), ("CAST int4 (Jetson)", "jet_cast"), ("original bf16 (GPU)", "bf16"), ("CAST bf16 (GPU)", "cast_bf16")):
        e = err(c, t)
        if e is None:
            continue
        cells, ok = [], []
        for v in ("blank", "shuffled"):
            ev = err(f"{c}-{v}", t)
            if ev is None:
                cells.append("-"); continue
            p = wilcoxon(ev, e).pvalue; d = ev.mean() / e.mean() - 1
            ok.append(p < 0.01 and d >= 0.15); cells.append(f"{100 * d:+.0f}% (p={p:.2g})")
        L.append(f"| {name} | {e.mean():.3f} | " + " | ".join(cells) + f" | {'-' if len(ok) < 2 else ('yes' if all(ok) else 'no')} |")
    eo, ec = err("jet_orig", t), err("jet_cast", t); p = wilcoxon(ec, eo).pvalue
    lose = ec.mean() > eo.mean() and p < 0.05
    verdict.append((title, ec.mean() - eo.mean(), p, lose))
    L.append(f"\nCAST vs original (Jetson int4, paired): {ec.mean() - eo.mean():+.3f} (Wilcoxon p={p:.2g}) -> "
             f"{'CAST is WORSE' if lose else ('CAST is better' if ec.mean() < eo.mean() and p < 0.05 else 'no significant difference')}.")
    bo, bc = err("bf16", t), err("cast_bf16", t)
    if bo is not None and bc is not None:                    # the same comparison without quantization
        pb = wilcoxon(bc, bo).pvalue
        L.append(f"CAST vs original in bf16 (GPU, paired): {bc.mean() - bo.mean():+.3f} (Wilcoxon p={pb:.2g}); int4 vs bf16 of the "
                 f"same checkpoint: original {eo.mean() - bo.mean():+.3f} (p={wilcoxon(eo, bo).pvalue:.2g}), CAST {ec.mean() - bc.mean():+.3f} "
                 f"(p={wilcoxon(ec, bc).pvalue:.2g}).")
    L.append("")
L += ["## Object goal (LeLaN, 210 frames, \"move toward <object>\")\n",
      "| Model | Picks named object | Blank image | Shuffled image | Other object's prompt | Uses instruction |", "|---|---|---|---|---|---|"]
for name, c in (("original int4 (Jetson)", "jet_orig"), ("CAST int4 (Jetson)", "jet_cast"), ("original bf16 (GPU)", "bf16"), ("CAST bf16 (GPU)", "cast_bf16")):
    pk = picks(c)
    if pk is None:
        continue
    cells, ins = [], "-"
    for v in ("blank", "shuffled", "lang_shuffled"):
        pv = picks(f"{c}-{v}")
        if pv is None:
            cells.append("-"); continue
        p = mcn(pk, pv); d = 100 * (pv.mean() - pk.mean()); cells.append(f"{100 * pv.mean():.0f}% ({d:+.0f} pts, p={p:.1g})")
        if v == "lang_shuffled":
            ins = "yes" if p < 0.01 and d <= -15 else "no"
    L.append(f"| {name} | {100 * pk.mean():.0f}% | " + " | ".join(cells) + f" | {ins} |")
po, pc = picks("jet_orig"), picks("jet_cast"); p = mcn(pc, po); d = 100 * (pc.mean() - po.mean())
lose = d < 0 and p < 0.05
verdict.append(("Object goal (LeLaN)", d, p, lose))
L.append(f"\nCAST vs original (Jetson int4, paired): {d:+.1f} points (McNemar p={p:.2g}) -> "
         f"{'CAST is WORSE' if lose else ('CAST is better' if d > 0 and p < 0.05 else 'no significant difference')}.\n")
lost = [v[0] for v in verdict if v[3]]
L.append("## Verdict\n")
L.append(f"**By the rule (Jetson int4 on both): the CAST checkpoint {'loses accuracy on: ' + ', '.join(lost) if lost else 'does not lose measurable accuracy outside language mode'}.** "
         "Single open-loop predictions on in-distribution test frames. Read it together with the bf16 lines above: they "
         "show whether a gap comes from the checkpoint itself or from how each checkpoint reacts to quantization.")
open("results/cross_mode_summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
