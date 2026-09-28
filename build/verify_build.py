# Compare a built weights/ folder with the validated deployment, tensor by tensor (runs locally, no Kaggle).
#   python build/verify_build.py [WEIGHTS_DIR]            (default build/out/weights; uses its TENSOR_SHA256.json)
#   python build/verify_build.py WEIGHTS_DIR --rehash     (recompute the hashes from the shards and base.safetensors; needs torch)
# Exit 0 = every layer and every base tensor matches build/reference_tensor_hashes.json.
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
W = args[0] if args else os.path.join(HERE, "out", "weights")
R = json.load(open(os.path.join(HERE, "reference_tensor_hashes.json")))
if "--rehash" in sys.argv:
    sys.path.insert(0, HERE)
    from safetensors import safe_open
    from tensor_hashes import layers, th
    B = {n: {k: th(v) for k, v in sorted(st.items())} for n, st in layers(os.path.join(W, "marpc"))}
    with safe_open(os.path.join(W, "base.safetensors"), "pt", device="cpu") as f:
        B["__base__"] = {k: th(f.get_tensor(k)) for k in sorted(f.keys())}
else:
    B = json.load(open(os.path.join(W, "TENSOR_SHA256.json")))
same = [n for n in R if B.get(n) == R[n]]
diff = [n for n in R if n in B and B[n] != R[n]]
missing, extra = sorted(set(R) - set(B)), sorted(set(B) - set(R))
print(f"[VERIFY] reference entries {len(R)} (224 LLM layers, 204 vision layers, base tensors); identical {len(same)}; "
      f"differ {len(diff)}; missing {len(missing)}; extra {len(extra)}")
for n in diff[:10]:
    ks = [k for k in R[n] if B[n].get(k) != R[n][k]]
    print(f"[VERIFY] differs: {n} ({len(ks)} tensors, e.g. {ks[:3]})")
ok = len(same) == len(R) and not extra
print("[VERIFY]", "BUILD MATCHES THE VALIDATED DEPLOYMENT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
