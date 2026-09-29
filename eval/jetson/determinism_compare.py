# Compare the determinism.py runs: every output must be bit-identical across runs and equal to tests/reference.
import glob, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = np.load(os.path.join(ROOT, "deploy", "tests", "reference", "reference.npz"))
runs = [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", "determinism", "run_*.npz")))]
if not runs:
    raise SystemExit("[DET] no runs found")
bad = 0
for k in sorted(runs[0]):
    vals = [r[k] for r in runs]
    n_same = sum(np.array_equal(v, vals[0]) for v in vals)
    ref_k = k.replace("miss__", "act_m6__")
    ref_ok = np.array_equal(vals[0], R[ref_k])
    spread = max(float(np.abs(v - vals[0]).max()) for v in vals)
    bad += n_same != len(vals) or not ref_ok
    print(f"[DET] {k}: {n_same}/{len(vals)} runs identical to run 1 (max spread {spread:.3g}); equal to reference: {ref_ok}")
print(f"[DET] {len(runs)} runs; {'PASS: deterministic' if bad == 0 else f'FAIL: {bad} outputs vary or differ'}")
