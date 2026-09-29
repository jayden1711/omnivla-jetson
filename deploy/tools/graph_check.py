# graph_check.py - CUDA graphs (OmniVLADeploy(..., cuda_graphs=True)) must give bit-identical actions to eager mode.
# Two processes (two 7B copies do not fit in 8 GB):
#   ./launch.sh tools/graph_check.py eager     records eager outputs + latency (10 reference frames, modes 4, 6, 7)
#   ./launch.sh tools/graph_check.py graphs    runs with CUDA graphs, compares bit for bit, prints the latency change
#   ./launch.sh tools/graph_check.py nocache   same comparison for eager mode without the LLM key/value cache
# Modes 4 and 6 are also compared with the validated deployment's recorded outputs (tests/reference). Exit 0 = PASS.
import json, os, sys
import numpy as np
from PIL import Image

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

WHICH = sys.argv[1] if len(sys.argv) > 1 else ""
assert WHICH in ("eager", "graphs", "nocache"), "usage: graph_check.py eager|graphs|nocache"
REF = os.path.join(D, "tests", "reference"); R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
if not all(os.path.exists(os.path.join(REF, s, f + ".jpg")) for s in ("frames", "goals") for f in frames):
    sys.exit("[GRAPH] reference images missing: run ./launch.sh tools/reference_check.py first")
OUT = os.path.join(D, "tests", "graph_eager.npz")
m = OmniVLADeploy(D, cuda_graphs=(WHICH == "graphs"), llm_kv_cache=(WHICH != "nocache"))
m.warmup(modes=(4, 6, 7), n=3)                                  # also captures the graphs (first use per shape)
res, lat = {}, {4: [], 6: [], 7: []}
for rep in range(2):                                            # second pass: every graph already captured
    for f in frames:
        img = Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB")
        for mode, kw in ((4, dict(goal_pose=R[f"goal__{f}"])), (6, dict(goal_image=Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB"))),
                         (7, dict(lang="move toward the door"))):
            o = m.predict(img, mode=mode, **kw)
            res[f"act_m{mode}__{f}"] = o["actions"]
            if rep == 1:
                lat[mode].append(o["t_fwd"])
med = {k: float(np.median(v) * 1000) for k, v in lat.items()}
if WHICH == "eager":
    np.savez(OUT, **res, lat=np.array(json.dumps(med)))
    print(f"[GRAPH] eager outputs saved to {OUT}; median fwd ms {med}"); sys.exit(0)
E = np.load(OUT)
same = {k: bool(np.array_equal(v, E[k])) for k, v in res.items()}
ref_ok = all(np.array_equal(res[f"act_m{m_}__{f}"], R[f"act_m{m_}__{f}"]) for m_ in (4, 6) for f in frames)
em = json.loads(str(E["lat"]))
print(f"[GRAPH] identical to eager: {sum(same.values())}/{len(same)}; modes 4/6 identical to the validated deployment: {ref_ok}")
for mode in (4, 6, 7):
    print(f"[GRAPH] mode {mode}: eager {em[str(mode)]:.0f} ms -> {WHICH} {med[mode]:.0f} ms ({med[mode] / em[str(mode)]:.3f}x)")
ok = all(same.values()) and ref_ok
print(f"[GRAPH] {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
