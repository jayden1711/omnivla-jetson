# kaggle_build.py - build the deployable weights/ folder from NHirose/omnivla-original on a Kaggle T4 (run by build_model.sh).
# LLM: GPTQ per-channel int4 packed for Marlin (kaggle_compress.py STUDY=gptqx). Vision: HQQ 4-bit g64 (repacked for
# GemLite at load time). Output: 256 MB shards, base.safetensors (other tensors, fp16), model/, SHA256SUMS,
# TENSOR_SHA256.json and BUILD_MANIFEST.json. torch.save embeds a random serialization id, so builds are compared
# with per-tensor hashes, not file hashes.
import glob, hashlib, json, os, shutil, subprocess, sys, time
import torch

W = "/kaggle/working"; OUT = f"{W}/weights"; MODEL_DIR = "/tmp/omnivla-original"
REV = os.environ.get("OMNIVLA_REV", "main")
t0 = time.time()
os.makedirs(f"{OUT}/marpc", exist_ok=True); os.makedirs(f"{OUT}/model", exist_ok=True)
from huggingface_hub import snapshot_download, HfApi
rev_sha = HfApi().model_info("NHirose/omnivla-original", revision=REV).sha
snapshot_download("NHirose/omnivla-original", revision=rev_sha, local_dir=MODEL_DIR,
                  ignore_patterns=["dist_head*", "lora_adapter/*", "modeling_prismatic_____.py"])
print(f"[BUILD] checkpoint NHirose/omnivla-original @ {rev_sha}", flush=True)

# ---- GPTQ LLM (kaggle_compress.py STUDY=gptqx) ----
if not os.path.exists(f"{W}/prequant_marpcg.pt"):
    env = dict(os.environ, STUDY="gptqx", N_CALIB="64", TESTS_ONLY="img3m", PRUNE="spatial75")
    r = subprocess.run([sys.executable, f"{W}/kaggle_compress.py"], env=env, cwd=W)
    assert r.returncode == 0, "GPTQ step failed"
man_g = json.load(open(f"{W}/prequant_marpcg_manifest.json"))
g = torch.load(f"{W}/prequant_marpcg.pt", map_location="cpu", weights_only=False)
print(f"[BUILD] GPTQ LLM layers: {len(g)}", flush=True)

# ---- HQQ 4-bit vision ----
import re
from safetensors import safe_open
from safetensors.torch import save_file
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
HQQLinear.set_backend(HQQBackend.PYTORCH)
VIS_LIN = re.compile(r"^vision_backbone\.(featurizer|fused_featurizer)\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))$")
tensors = {}
for shard in sorted(glob.glob(f"{MODEL_DIR}/model-*.safetensors")):
    with safe_open(shard, "pt", device="cpu") as f:
        for k in f.keys():
            mod, p = k.rsplit(".", 1)
            if VIS_LIN.match(mod) and p in ("weight", "bias"):
                tensors.setdefault(mod, {})[p] = (shard, k)
vis = {}
for mod in sorted(tensors):
    w = {}
    for p, (shard, k) in tensors[mod].items():
        with safe_open(shard, "pt", device="cpu") as f:
            w[p] = f.get_tensor(k).to(torch.float16)
    o, i = w["weight"].shape
    lin = torch.nn.Linear(i, o, bias="bias" in w, dtype=torch.float16)
    with torch.no_grad():
        lin.weight.copy_(w["weight"])
        if "bias" in w:
            lin.bias.copy_(w["bias"])
    L = HQQLinear(lin, BaseQuantizeConfig(nbits=4, group_size=64), compute_dtype=torch.float16, device="cuda", del_orig=True)
    vis[mod] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in L.state_dict().items()}
    del L
print(f"[BUILD] HQQ4 vision layers: {len(vis)}", flush=True)

# ---- assemble weights/ (sorted names, 256 MB shards) ----
entries = dict(g); entries.update(vis)
LIM = 256 << 20
nb = lambda st: sum(v.numel() * v.element_size() for v in st.values() if torch.is_tensor(v))
files, cur, size = [], {}, 0
def write(part):
    p = f"{OUT}/marpc/shard_{len(files):03d}.pt"
    torch.save(part, p); back = torch.load(p, weights_only=False)
    for n, st in part.items():
        for k, v in st.items():
            assert (torch.equal(back[n][k], v) and back[n][k].dtype == v.dtype) if torch.is_tensor(v) else back[n][k] == v, (n, k)
    files.append(os.path.basename(p))
for n in sorted(entries):
    st = {k: (v.detach().clone().contiguous() if torch.is_tensor(v) else v) for k, v in entries[n].items()}
    if cur and size + nb(st) > LIM:
        write(cur); cur, size = {}, 0
    cur[n] = st; size += nb(st)
write(cur)
json.dump({"marpc": {"shards": files, "names": sorted(entries)}}, open(f"{OUT}/SHARDS.json", "w"), indent=0)
base = {}
for shard in sorted(glob.glob(f"{MODEL_DIR}/model-*.safetensors")):
    with safe_open(shard, "pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("language_model.lm_head") or k.rsplit(".", 1)[0] in entries:
                continue
            base[k] = f.get_tensor(k).to(torch.float16).contiguous()
save_file(base, f"{OUT}/base.safetensors")
with safe_open(f"{OUT}/base.safetensors", "pt", device="cpu") as f:
    assert set(f.keys()) == set(base) and all(torch.equal(f.get_tensor(k), base[k]) for k in base)
keep = ["config.json", "configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py",
        "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "generation_config.json",
        "action_head--120000_checkpoint.pt", "proprio_projector--120000_checkpoint.pt"]
for fn in keep:
    shutil.copy2(f"{MODEL_DIR}/{fn}", f"{OUT}/model/{fn}")
print(f"[BUILD] weights/: {len(entries)} pre-packed layers in {len(files)} shards, base.safetensors {len(base)} tensors", flush=True)
del base

# ---- per-tensor hashes, checksums, manifest ----
sys.path.insert(0, W)
from tensor_hashes import layers, th
TH = {n: {k: th(v) for k, v in sorted(st.items())} for n, st in layers(f"{OUT}/marpc")}
with safe_open(f"{OUT}/base.safetensors", "pt", device="cpu") as f:
    TH["__base__"] = {k: th(f.get_tensor(k)) for k in sorted(f.keys())}
json.dump(TH, open(f"{OUT}/TENSOR_SHA256.json", "w"), indent=0, sort_keys=True)
import transformers, hqq, marlin
man = {"checkpoint": f"NHirose/omnivla-original@{rev_sha}", "build_s": round(time.time() - t0),
       "versions": {"torch": torch.__version__, "transformers": transformers.__version__ + " (openvla-oft fork bc339d9)",
                    "hqq": hqq.__version__, "marlin": "IST-DASLab/marlin@1f25790", "gpu": torch.cuda.get_device_name(0)},
       "gptq": {k: man_g[k] for k in ("n_calib", "prune", "calib_keys_first")}, "n_layers": len(entries), "shards": files}
# ---- optional: compare with reference_tensor_hashes.json of the validated deployment ----
ref = [p for p in [f"{W}/reference_tensor_hashes.json"] if os.path.exists(p)] + glob.glob("/kaggle/input/**/reference_tensor_hashes.json", recursive=True)
if ref:
    R = json.load(open(ref[0]))
    same = sum(1 for n in R if n in TH and TH[n] == R[n]); diff = [n for n in R if n in TH and TH[n] != R[n]]
    miss = [n for n in R if n not in TH]
    print(f"[BUILD] vs validated deployment: {same}/{len(R)} layers identical (tensor level); {len(diff)} differ; {len(miss)} missing"
          + (f"; first differing: {diff[:3]}" if diff else ""), flush=True)
    man["reference_match"] = dict(identical=same, total=len(R), differ=diff[:20], missing=miss[:20])
json.dump(man, open(f"{OUT}/BUILD_MANIFEST.json", "w"), indent=1)
with open(f"{OUT}/SHA256SUMS", "w") as out:
    for root, _, fs in sorted(os.walk(OUT)):
        for fn in sorted(fs):
            if fn == "SHA256SUMS":
                continue
            fp = os.path.join(root, fn); h = hashlib.sha256()
            with open(fp, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            out.write(f"{h.hexdigest()}  {os.path.relpath(fp, OUT)}\n")
for f in ("prequant_marpcg.pt", "prequant_marpcg_check.pt"):          # intermediates: keep the kernel output to weights/
    if os.path.exists(f"{W}/{f}"):
        os.remove(f"{W}/{f}")
print(f"[BUILD] DONE in {time.time() - t0:.0f} s -> {OUT}", flush=True)
