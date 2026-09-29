# build_cast_weights.py CKPT_DIR PREQUANT_DIR OUT - deploy weights folder for NHirose/omnivla-finetuned-cast (CPU only).
# CKPT_DIR: the checkpoint (safetensors shards, configs, action_head/proprio_projector --210000); PREQUANT_DIR: the
# castgptq Kaggle export (prequant_marpcg_cast.pt + manifest: GPTQ int4 LLM for Marlin; prequant_vis_cast_gptq.pt +
# manifest: GPTQ int4 vision for Marlin, zero-padded SigLIP shapes). Output like build/kaggle_build.py's weights/:
# marpc/ (LLM) and vis_marpc/ (vision) 256 MB shards, SHARDS.json, base.safetensors (all other tensors, fp16; no lm_head),
# model/ (configs, tokenizer, heads), TENSOR_SHA256.json, BUILD_MANIFEST.json, VIS_MARPC_MANIFEST.json, SHA256SUMS.
# No HQQ4 vision: the runtime uses Marlin vision for this folder (OMNIVLA_WEIGHTS=weights_cast).
#   CUDA_VISIBLE_DEVICES= python tools/build_cast_weights.py /mnt/nvme/omnivla/ckpt/omnivla-finetuned-cast ../prequant/cast weights_cast
import glob, hashlib, json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CK, PQ, OUT = sys.argv[1:4]
t0 = time.time()
if os.path.exists(OUT):
    sys.exit(f"[CASTW] {OUT} exists: remove it first")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
ML = json.load(open(f"{PQ}/prequant_marpcg_cast_manifest.json")); MV = json.load(open(f"{PQ}/prequant_vis_cast_gptq_manifest.json"))
for f, h in list(ML["files"].items()) + list(MV["files"].items()):
    assert sha(f"{PQ}/{f}") == h, f"{f}: sha256 differs from the Kaggle manifest"
assert "finetuned-cast" in ML.get("model", ""), ML.get("model")
os.makedirs(f"{OUT}/model")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/build")

def shard(entries, sub):
    os.makedirs(f"{OUT}/{sub}")
    nb = lambda st: sum(v.numel() * v.element_size() for v in st.values() if torch.is_tensor(v))
    files, cur, size = [], {}, 0
    def write(part):
        p = f"{OUT}/{sub}/shard_{len(files):03d}.pt"; torch.save(part, p); back = torch.load(p, weights_only=False)
        for n, st in part.items():
            for k, v in st.items():
                assert (torch.equal(back[n][k], v) and back[n][k].dtype == v.dtype) if torch.is_tensor(v) else back[n][k] == v, (n, k)
        files.append(os.path.basename(p))
    for n in sorted(entries):
        st = {k: (v.detach().clone().contiguous() if torch.is_tensor(v) else v) for k, v in entries[n].items()}
        if cur and size + nb(st) > 256 << 20:
            write(cur); cur, size = {}, 0
        cur[n] = st; size += nb(st)
    write(cur)
    return files
llm = torch.load(f"{PQ}/prequant_marpcg_cast.pt", map_location="cpu", mmap=True, weights_only=False)   # 3.4 GB: file-backed
assert len(llm) == 224 and set(llm) == set(ML["deq_sha256_WT_fp16"])
f_llm = shard(llm, "marpc"); del llm
vis = torch.load(f"{PQ}/prequant_vis_cast_gptq.pt", map_location="cpu", weights_only=False)
assert len(vis) == 204 and set(vis) == set(MV["deq_sha256_WT_fp16"])
names_llm, names_vis = sorted(ML["deq_sha256_WT_fp16"]), sorted(vis)
f_vis = shard(vis, "vis_marpc"); del vis
json.dump({"marpc": {"shards": f_llm, "names": names_llm}, "vis_marpc": {"shards": f_vis, "names": names_vis}},
          open(f"{OUT}/SHARDS.json", "w"), indent=0)
shutil.copy(f"{PQ}/prequant_vis_cast_gptq_manifest.json", f"{OUT}/VIS_MARPC_MANIFEST.json")
shutil.copy(f"{PQ}/prequant_marpcg_cast_manifest.json", f"{OUT}/LLM_MARPC_MANIFEST.json")
quant = set(names_llm) | set(names_vis)
base = {}
for s in sorted(glob.glob(f"{CK}/model-*.safetensors")):
    with safe_open(s, "pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("language_model.lm_head") or k.rsplit(".", 1)[0] in quant:
                continue
            base[k] = f.get_tensor(k).to(torch.float16).contiguous()
save_file(base, f"{OUT}/base.safetensors")
with safe_open(f"{OUT}/base.safetensors", "pt", device="cpu") as f:
    assert set(f.keys()) == set(base) and all(torch.equal(f.get_tensor(k), base[k]) for k in base)
n_base = len(base); del base
heads = sorted(os.path.basename(p) for p in glob.glob(f"{CK}/action_head--*_checkpoint.pt") + glob.glob(f"{CK}/proprio_projector--*_checkpoint.pt"))
assert heads == ["action_head--210000_checkpoint.pt", "proprio_projector--210000_checkpoint.pt"], heads
for fn in ["config.json", "configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py",
           "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
           "special_tokens_map.json", "added_tokens.json", "generation_config.json"] + heads:
    shutil.copy2(f"{CK}/{fn}", f"{OUT}/model/{fn}")
from tensor_hashes import layers, th
TH = {n: {k: th(v) for k, v in sorted(st.items())} for sub in ("marpc", "vis_marpc") for n, st in layers(f"{OUT}/{sub}")}
with safe_open(f"{OUT}/base.safetensors", "pt", device="cpu") as f:
    TH["__base__"] = {k: th(f.get_tensor(k)) for k in sorted(f.keys())}
json.dump(TH, open(f"{OUT}/TENSOR_SHA256.json", "w"), indent=0, sort_keys=True)
rev = json.load(open(f"{CK}/.cast_revision.json"))["sha"] if os.path.exists(f"{CK}/.cast_revision.json") else ML.get("model_rev")
json.dump({"checkpoint": f"NHirose/omnivla-finetuned-cast@{rev}", "head_step": 210000, "build_s": round(time.time() - t0),
           "llm": {k: ML[k] for k in ("n_calib", "prune", "calib_keys_first", "model_rev")}, "vision": "GPTQ per-channel int4 (Marlin, padded SigLIP)",
           "n_layers": {"llm_marlin": len(names_llm), "vision_marlin": len(names_vis), "base_tensors": n_base},
           "calibration": "CAST calibration episodes (build/cast_split.json), language mode, no pruning"},
          open(f"{OUT}/BUILD_MANIFEST.json", "w"), indent=1)
with open(f"{OUT}/SHA256SUMS", "w") as out:
    for root, _, fs in sorted(os.walk(OUT)):
        for fn in sorted(fs):
            if fn != "SHA256SUMS":
                fp = os.path.join(root, fn); out.write(f"{sha(fp)}  {os.path.relpath(fp, OUT)}\n")
print(f"[CASTW] {OUT}: LLM {len(names_llm)} + vision {len(names_vis)} Marlin layers in {len(f_llm)} + {len(f_vis)} shards, "
      f"base.safetensors {n_base} tensors, heads step 210000 | {time.time() - t0:.0f} s", flush=True)
