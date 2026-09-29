# fetch_ckpt_subset.py REPO REVISION OUT_DIR MANIFEST.json [MANIFEST.json ...] - download only what a weights build needs
# from a Hugging Face checkpoint: the small files (configs, tokenizer, action head / proprio projector) and, from each
# model-*.safetensors shard, only the tensors that are NOT in the quantized manifests (they arrive pre-packed from the
# Kaggle export) and not the unused lm_head. Tensors are fetched with HTTP range requests at their offsets in the
# safetensors header and written unchanged (same dtype and bytes) into slim shards with the same names, so
# build_cast_weights.py reads them as if they were the full files. Standard library only.
#   python tools/fetch_ckpt_subset.py NHirose/omnivla-finetuned-cast 7d3744a... ckpt_cast prequant_marpcg_cast_manifest.json prequant_vis_cast_gptq_manifest.json
import json, os, struct, sys, urllib.request

repo, rev, out, *mans = sys.argv[1:]
quant = set()
for mf in mans:
    quant |= set(json.load(open(mf))["deq_sha256_WT_fp16"])
base = f"https://huggingface.co/{repo}/resolve/{rev}/"
def _get(path, rng):
    for attempt in range(5):                                   # retry dropped connections
        try:
            req = urllib.request.Request(base + path, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if rng:
                assert len(data) == rng[1] - rng[0] + 1, (path, rng, len(data))
            return data
        except Exception as e:
            print(f"[FETCH] retry {attempt + 1} {path} {rng}: {e!r}", flush=True)
    raise SystemExit(f"[FETCH] failed: {path} {rng}")
def get(path, rng=None, chunk=32 << 20):                       # large ranges in 32 MB pieces
    if rng is None or rng[1] - rng[0] + 1 <= chunk:
        return _get(path, rng)
    return b"".join(_get(path, (a, min(a + chunk - 1, rng[1]))) for a in range(rng[0], rng[1] + 1, chunk))
os.makedirs(out, exist_ok=True)
small = ["config.json", "configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py", "preprocessor_config.json",
         "processor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
         "added_tokens.json", "generation_config.json", "model.safetensors.index.json",
         "action_head--210000_checkpoint.pt", "proprio_projector--210000_checkpoint.pt"]
for f in small:
    open(os.path.join(out, f), "wb").write(get(f)); print(f"[FETCH] {f}", flush=True)
idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
kept = skipped = nbytes = 0
for shard in sorted(set(idx.values())):
    n = struct.unpack("<Q", get(shard, (0, 7)))[0]
    hdr = json.loads(get(shard, (8, 8 + n - 1)))
    start = 8 + n
    keep = {k: v for k, v in hdr.items() if k != "__metadata__" and not k.startswith("language_model.lm_head")
            and k.rsplit(".", 1)[0] not in quant}
    skipped += sum(1 for k in hdr if k != "__metadata__") - len(keep)
    new_hdr, blobs, off = {}, [], 0
    for k in sorted(keep, key=lambda k: keep[k]["data_offsets"][0]):
        a, b = keep[k]["data_offsets"]
        blob = get(shard, (start + a, start + b - 1)) if b > a else b""
        new_hdr[k] = {"dtype": keep[k]["dtype"], "shape": keep[k]["shape"], "data_offsets": [off, off + len(blob)]}
        blobs.append(blob); off += len(blob)
    if "__metadata__" in hdr:
        new_hdr["__metadata__"] = hdr["__metadata__"]
    h = json.dumps(new_hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(os.path.join(out, shard), "wb") as fh:
        fh.write(struct.pack("<Q", len(h))); fh.write(h)
        for bl in blobs:
            fh.write(bl)
    kept += len(keep); nbytes += off
    print(f"[FETCH] {shard}: kept {len(keep)} tensors ({off / 2**20:.0f} MB)", flush=True)
json.dump({"repo": repo, "sha": rev, "subset": "non-quantized tensors only (fetch_ckpt_subset.py)"},
          open(os.path.join(out, ".cast_revision.json"), "w"))
print(f"[FETCH] done: {kept} tensors kept ({nbytes / 2**20:.0f} MB), {skipped} skipped (quantized or lm_head)", flush=True)
