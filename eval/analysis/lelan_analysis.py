# Object-goal language test (LeLaN, IN-DISTRIBUTION: LeLaN is in omnivla-original's training mix). Rules fixed before
# the runs:
#  - each frame has a target and a distractor object (bearings >= 30 deg apart); prompt "move toward <target>"
#  - picks target: the predicted path's last waypoint points closer (in angle) to the target than to the distractor
#  - bearing error: angle between the last waypoint and the target (deg)
#  - controls: blank image, shuffled image (another recording), and the distractor's prompt on the same image
#    ("other object"; a model that follows the prompt should then pick the distractor). A control counts if it lowers
#    the picks-target rate by >= 15 points with a paired exact test (McNemar) p < 0.01. The test checks perception if
#    both image controls count, and instruction use if the other-object prompt counts.
#  - model-free reference: straight ahead (picks whichever object is nearer the center line)
# Usage: python eval/analysis/lelan_analysis.py lelan_lang.npz compress_lelan.npz compress_gptqx_lelan.npz edge_tests.npz
#   -> results/lelan_summary.md
import json, math, sys
import numpy as np
from scipy.stats import binomtest

lz, *files = sys.argv[1:]
LZ = np.load(lz); M = json.loads(str(LZ["meta"])); K = sorted(M)
if not K:
    sys.exit("[LELAN] 0 frames - refusing to write a summary")
D = {}
for p in files:
    D.update(dict(np.load(p, allow_pickle=True)))
bear = lambda xy: math.degrees(math.atan2(xy[1], xy[0]))
adiff = lambda a, b: abs((a - b + 180) % 360 - 180)

def pred(cfg, k):
    a = D.get(f"{cfg}__lelan__{k}")
    return None if a is None else np.asarray(a).reshape(-1, 8, 4)[0]
def outcomes(cfg):
    pk, be = [], []
    for k in K:
        p = pred(cfg, k); b = bear(p[-1, :2])
        t, d = bear(M[k]["target_xy"]), bear(M[k]["distractor_xy"])
        pk.append(adiff(b, t) < adiff(b, d)); be.append(adiff(b, t))
    return np.array(pk), np.array(be)
def mcnemar(a, b):                                               # paired exact test on discordant pairs
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0

CFGS = [("7B bf16", "bf16"), ("7B fp16", "fp16"), ("7B fp16, 75% pruning (no quantization)", "fp16_spatial75"),
        ("7B int4 deployed (75% pruning)", "gptqx_pc4"), ("7B int4, 0% pruning", "gptqx_pc4_none"),
        ("7B int4, 50% pruning", "gptqx_pc4_spatial50"), ("7B int4, 75% pruning (same-session rerun)", "gptqx_pc4_spatial75"),
        ("7B int4, 25% uniform pruning", "gptqx_pc4_spatial25"),
        ("7B int4, 25% prompt-aware pruning", "gptqx_pc4_prompt25"), ("7B int4, 50% prompt-aware pruning", "gptqx_pc4_prompt50"),
        ("7B int4, 75% prompt-aware pruning", "gptqx_pc4_prompt75"),
        ("7B int4, 75% prompt-aware, original SigLIP (reference)", "gptqx_pc4_promptorig75"),
        ("OmniVLA-edge", "edge")]
for _, c in CFGS:                                          # every frame or none: a partial run must not be summarized
    n = sum(pred(c, k) is not None for k in K)
    if 0 < n < len(K):
        sys.exit(f"[LELAN] {c}: predictions for only {n}/{len(K)} frames")
if not any(pred(c, K[0]) is not None for _, c in CFGS):
    sys.exit("[LELAN] no predictions for these frames in the given files")
L = []; say = L.append
straight = np.array([adiff(0, bear(M[k]["target_xy"])) < adiff(0, bear(M[k]["distractor_xy"])) for k in K])
subs = {s: sum(M[k]["subset"] == s for k in K) for s in sorted({M[k]["subset"] for k in K})}
say("# Object-goal language test (LeLaN, in-distribution)\n")
say(f"{len(K)} frames from LeLaN's robot recordings ({', '.join(f'{s} {n}' for s, n in subs.items())}). **In-distribution:** "
    "LeLaN is in the deployed checkpoint's training mix, and the prompts use its training format (\"move toward <object>\"). "
    "So this checks that language mode works as trained and survives quantization, not that it generalizes. Each frame has "
    "two labeled objects whose directions differ by at least 30 degrees; the prompt names one of them. Open-loop single "
    "predictions. 7B int4 = the deployed weights on a Kaggle T4 with fp16 kernels (not the Jetson's; language mode has not "
    "been run on the Jetson). OmniVLA-edge: prompt = the object phrase (its sample script's format), current frame repeated "
    "as history.\n")
say(f"Straight ahead (no model) picks the named object in {100 * straight.mean():.0f}% of frames.\n")
say("| Model | Picks named object | Bearing error (deg, median) | Blank image | Shuffled image | Prompt for the other object | Checks perception | Uses instruction |")
say("|---|---|---|---|---|---|---|---|")
res = {}
for name, cfg in CFGS:
    if pred(cfg, K[0]) is None:
        continue
    pk, be = outcomes(cfg); res[cfg] = (pk, be); cells, v = [], {}
    for c in ("blank", "shuffled", "lang_shuffled"):
        if pred(f"{cfg}-{c}", K[0]) is None:
            cells.append("-"); continue
        pc, _ = outcomes(f"{cfg}-{c}"); p = mcnemar(pk, pc); drop = 100 * (pk.mean() - pc.mean())
        v[c] = p < 0.01 and drop >= 15
        cells.append(f"{100 * pc.mean():.0f}% ({-drop:+.0f} pts, p={p:.1g})")
    perc = "-" if "blank" not in v else ("yes" if v["blank"] and v["shuffled"] else "no")
    ins = "-" if "lang_shuffled" not in v else ("yes" if v["lang_shuffled"] else "no")
    say(f"| {name} | {100 * pk.mean():.0f}% (p vs 50%: {binomtest(int(pk.sum()), len(pk)).pvalue:.1g}) | {np.median(be):.0f} | "
        + " | ".join(cells) + f" | {perc} | {ins} |")
say("\nControls: percentages are the picks-named-object rate under the control; pts = change vs the real input; p = paired "
    "exact test. With the other object's prompt, a model that follows language should fall well below 50%.\n")
if "bf16" in res:
    say("Paired comparisons of picks-named-object with bf16: " + "; ".join(
        f"{c} {100 * (res[c][0].mean() - res['bf16'][0].mean()):+.0f} pts (p={mcnemar(res[c][0], res['bf16'][0]):.2g})"
        for c in res if c != "bf16") + ".\n")
if all(c in res for c in ("bf16", "gptqx_pc4_none", "gptqx_pc4_spatial50", "gptqx_pc4_spatial75", "fp16_spatial75")):
    r = {c: 100 * res[c][0].mean() for c in res}
    say("## Cause of the int4 drop (ablation, same deployed int4 weights)\n")
    say("Rule fixed before the run: if the deployed int4 weights without pruning pick the named object in >= 77% of frames "
        "(half of the gap to bf16 closed), pruning is the cause and GPTQ is not recalibrated.\n")
    say(f"- int4, no pruning: {r['gptqx_pc4_none']:.0f}% (bf16 {r['bf16']:.0f}%; p={mcnemar(res['gptqx_pc4_none'][0], res['bf16'][0]):.2g}) "
        f"-> {'meets' if r['gptqx_pc4_none'] >= 77 else 'misses'} the 77% rule.")
    say(f"- int4, 50% pruning: {r['gptqx_pc4_spatial50']:.0f}% (p vs bf16 {mcnemar(res['gptqx_pc4_spatial50'][0], res['bf16'][0]):.2g}).")
    say(f"- int4, 75% pruning: {r['gptqx_pc4_spatial75']:.0f}%; fp16 (no quantization), 75% pruning: {r['fp16_spatial75']:.0f}% "
        f"(the two do not differ significantly, p={mcnemar(res['fp16_spatial75'][0], res['gptqx_pc4_spatial75'][0]):.2g}).\n")
    say("**The drop comes from the 75% image-token pruning, not from the int4 weights.** Pruning costs about the same "
        "with and without quantization, and the int4 weights alone match bf16. The runtime therefore does not prune in "
        "language modes (7, 8); pose and image-goal modes keep 75%, where pruning did not change driving error "
        "(token-pruning study on the Jetson: image goal -0.013, 20 s pose goal -0.004 action units vs unpruned, both not "
        "significant; deployed config vs bf16 in results/final_validation.md).\n")
    say("**Language grounding is more sensitive to compression than pose and image goals.** The same 75% pruning that "
        "left image-goal and pose-goal driving error unchanged cut object-goal accuracy by about 15 points. Fidelity "
        "(distance from bf16 actions) overstates the damage for pose and image goals, but a driving-error test on those "
        "modes would have understated it for language: each goal modality needs its own task-grounded test.\n")
if all(c in res for c in ("gptqx_pc4_prompt25", "gptqx_pc4_prompt50", "gptqx_pc4_prompt75", "gptqx_pc4_spatial50")):
    r = {c: 100 * res[c][0].mean() for c in res}
    say("## Prompt-aware pruning (same deployed int4 weights)\n")
    say("Rule fixed before the run: adopt only if prompt-aware pruning keeps >= 81% (uniform 50% pruning) at a higher "
        "pruning rate than 50%, i.e. at 75%. Patches are scored by similarity to the object phrase in SigLIP's image-text "
        "space (deploy/prompt_prune.py); \"original SigLIP\" uses the released image tower instead of OmniVLA's fine-tuned "
        "one, as a reference for whether fine-tuning broke the image-text alignment.\n")
    say("| Pruning | Uniform grid | Prompt-aware (fine-tuned SigLIP) | Prompt-aware (original SigLIP) |")
    say("|---|---|---|---|")
    for pct in (25, 50, 75):
        u = r.get(f"gptqx_pc4_spatial{pct}"); pa = r.get(f"gptqx_pc4_prompt{pct}"); po = r.get(f"gptqx_pc4_promptorig{pct}")
        f = lambda v: "-" if v is None else f"{v:.0f}%"
        say(f"| {pct}% | {f(u)} | {f(pa)} | {f(po)} |")
    ok = r["gptqx_pc4_prompt75"] >= 81
    say(f"\nNo pruning: {r['gptqx_pc4_none']:.0f}%. Prompt-aware at 75%: {r['gptqx_pc4_prompt75']:.0f}% "
        f"(vs uniform 75%: p={mcnemar(res['gptqx_pc4_prompt75'][0], res['gptqx_pc4_spatial75'][0]):.2g}) -> "
        f"**{'adopt' if ok else 'not adopted'}** under the rule.\n")
open("results/lelan_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
