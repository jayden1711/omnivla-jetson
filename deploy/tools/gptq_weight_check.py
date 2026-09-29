# gptq_weight_check.py WEIGHTS_DIR PREQUANT_DIR [MANIFEST] - bit-exact check of the packed GPTQ Marlin layers against the export.
# Marlin(I) returns W^T exactly, so its fp16 bytes must hash to the manifest's deq_sha256_WT_fp16 for all 224 LLM layers.
import hashlib, json, sys
import torch, marlin

W, PQ = sys.argv[1], sys.argv[2]
MAN = sys.argv[3] if len(sys.argv) > 3 else "prequant_marpcg_manifest.json"   # weights_cast: prequant_marpcg_cast_manifest.json
man = json.load(open(f"{PQ}/{MAN}"))["deq_sha256_WT_fp16"]
idx = json.load(open(f"{W}/SHARDS.json"))["marpc"]
ok = bad = 0
for s in idx["shards"]:
    part = torch.load(f"{W}/marpc/{s}", weights_only=False)
    for n, st in part.items():
        if "marlin_B" not in st:
            continue
        k, nn = int(st["k"]), int(st["n"])
        L = marlin.Layer(k, nn, groupsize=-1).cuda()
        with torch.no_grad():
            L.B.copy_(st["marlin_B"].cuda()); L.s.copy_(st["marlin_s"].cuda())
            WT = torch.empty(k, nn, dtype=torch.half, device="cuda")
            for i in range(0, k, 1024):
                j = min(i + 1024, k)
                I = torch.zeros(j - i, k, dtype=torch.half, device="cuda"); I[torch.arange(j - i), torch.arange(i, j)] = 1
                WT[i:j] = L(I)
        h = hashlib.sha256(WT.cpu().numpy().tobytes()).hexdigest()
        ok += h == man[n]; bad += h != man[n]
        if h != man[n]:
            print(f"[GWC] MISMATCH {n}", flush=True)
        del L, WT; torch.cuda.empty_cache()
    del part
print(f"[GWC] {ok}/{ok + bad} GPTQ layers bit-exact vs Kaggle (Marlin(I) == exported W^T)", flush=True)
