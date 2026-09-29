# add_vision_marlin.py PREQUANT_VIS.pt MANIFEST.json WEIGHTS_DIR - add Marlin per-channel int4 vision layers to a deploy
# weights folder (CPU only). Splits the Kaggle export (build/kaggle_compress.py STUDY=vismarlin) into ~256 MB shards in
# WEIGHTS_DIR/vis_marpc/, reloads every tensor and compares it bitwise with the source, then adds "vis_marpc" to
# SHARDS.json and the shard hashes to SHA256SUMS (both backed up as *.bak_vis first) and copies the manifest as
# VIS_MARPC_MANIFEST.json. The HQQ4 vision layers in marpc/ stay (runtime fallback: vision_kernel="hqq4").
#   CUDA_VISIBLE_DEVICES= python tools/add_vision_marlin.py prequant_vis_gptq.pt prequant_vis_gptq_manifest.json weights
import hashlib, json, os, shutil, sys
import torch

src, man, W = sys.argv[1:4]
M = json.load(open(man))
fh = hashlib.sha256(open(src, "rb").read()).hexdigest()
assert fh == M["files"][os.path.basename(src)], f"{src}: sha256 differs from the manifest"
D = torch.load(src, map_location="cpu", weights_only=False)
assert set(D) == set(M["deq_sha256_WT_fp16"]), "layer names differ from the manifest"
out = os.path.join(W, "vis_marpc")
if os.path.exists(out):
    sys.exit(f"[VISM] {out} exists: remove it first")
os.makedirs(out)
nbytes = lambda st: sum(v.numel() * v.element_size() for v in st.values() if torch.is_tensor(v))
shards, cur, size = [], {}, 0
for n in sorted(D):
    cur[n] = D[n]; size += nbytes(D[n])
    if size >= 256 * 2**20:
        shards.append(cur); cur, size = {}, 0
if cur:
    shards.append(cur)
files = []
for i, part in enumerate(shards):
    f = f"shard_{i:03d}.pt"; torch.save(part, os.path.join(out, f)); files.append(f)
    back = torch.load(os.path.join(out, f), map_location="cpu", weights_only=False)
    for n, st in part.items():
        for k, v in st.items():
            b = back[n][k]
            same = torch.equal(v, b) and v.dtype == b.dtype if torch.is_tensor(v) else v == b
            assert same, f"reload mismatch {f} {n}.{k}"
print(f"[VISM] {len(D)} layers -> {len(files)} shards, every tensor reloaded bit-identical", flush=True)
for f in ("SHARDS.json", "SHA256SUMS"):
    shutil.copy(os.path.join(W, f), os.path.join(W, f + ".bak_vis"))
SH = json.load(open(os.path.join(W, "SHARDS.json")))
SH["vis_marpc"] = {"shards": files, "names": sorted(D)}
json.dump(SH, open(os.path.join(W, "SHARDS.json"), "w"), indent=0)
shutil.copy(man, os.path.join(W, "VIS_MARPC_MANIFEST.json"))
lines = [l for l in open(os.path.join(W, "SHA256SUMS")).read().splitlines()
         if not l.split()[-1].startswith(("vis_marpc/", "SHARDS.json", "VIS_MARPC_MANIFEST.json"))]
for rel in [f"vis_marpc/{f}" for f in files] + ["SHARDS.json", "VIS_MARPC_MANIFEST.json"]:
    lines.append(f"{hashlib.sha256(open(os.path.join(W, rel), 'rb').read()).hexdigest()}  {rel}")
open(os.path.join(W, "SHA256SUMS"), "w").write("\n".join(lines) + "\n")
print(f"[VISM] SHARDS.json + SHA256SUMS updated (backups *.bak_vis); padded layers: {len(M['padding'])}", flush=True)
