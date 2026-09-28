# Correctness gate for per-channel Marlin on sm_87 with offline-packed weights: 3 LLM layer shapes x T in (75, 300, 555),
# output vs dequantize-then-matmul (rel mean < 1e-3) and 50 repeated calls bitwise identical. Exit 0 = PASS, 3 = FAIL.
import os, sys
import torch
import marlin

ROOT = os.environ.get("OMNIVLA_ROOT", "/mnt/nvme/omnivla")
PQ = f"{ROOT}/prequant"
dev = torch.device("cuda:0")
CFG = sys.argv[1] if len(sys.argv) > 1 else "marpc"     # marpc (rounding) or marpcg (GPTQ)
q = torch.load(f"{PQ}/prequant_{CFG}.pt", mmap=True, weights_only=False)
ref = torch.load(f"{PQ}/prequant_{CFG}_check.pt", weights_only=False)
print(f"[GATE] torch {torch.__version__} | {torch.cuda.get_device_name(0)} | marlin {marlin.__file__}", flush=True)
ok_all = True
for name, Wref in ref.items():
    st = q[name]
    k, n = int(st["k"]), int(st["n"])
    L = marlin.Layer(k, n, groupsize=-1).to(dev)
    with torch.no_grad():
        L.B.copy_(st["marlin_B"].to(dev)); L.s.copy_(st["marlin_s"].to(dev))
    W = Wref.to(dev).float()
    for T in (75, 300, 555):
        torch.manual_seed(T)
        x = torch.randn(T, k, dtype=torch.half, device=dev)
        y0 = L(x); torch.cuda.synchronize()
        r = x.float() @ W.T
        e = (y0.float() - r).abs()
        rel_mean, rel_max = float(e.mean() / r.abs().mean()), float(e.max() / r.abs().max())
        same = 0
        for _ in range(50):
            y = L(x); torch.cuda.synchronize()
            same += int(torch.equal(y, y0))
        ok = rel_mean < 1e-3 and same == 50
        ok_all &= ok
        print(f"[GATE] {name} ({k}->{n}) T={T}: rel mean {rel_mean:.2e} max {rel_max:.2e} | bit-identical {same}/50 -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            print("[GATE] FAIL -> drop per-channel Marlin", flush=True); sys.exit(3)
print(f"[GATE] {'ALL PASS' if ok_all else 'FAIL'}", flush=True)
