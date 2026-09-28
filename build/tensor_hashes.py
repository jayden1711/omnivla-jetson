# tensor_hashes.py - per-layer SHA-256 of every pre-packed tensor (sha256 of dtype|shape|bytes), independent of
# torch.save's container bytes (which embed a random serialization id), so two builds can be compared exactly.
# Usage: python tensor_hashes.py OUT.json FILE.pt|SHARD_DIR [prefix-filter ...]
import hashlib, json, os, sys, glob
import torch

def th(v):
    if torch.is_tensor(v):
        t = v.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""    # reshape: 0-dim tensors cannot be viewed as bytes
        return hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode() + raw).hexdigest()
    return repr(v)

def layers(src):
    files = sorted(glob.glob(os.path.join(src, "*.pt"))) if os.path.isdir(src) else [src]
    for f in files:
        d = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        for n in sorted(d):
            yield n, d[n]

if __name__ == "__main__":
    out, src, prefixes = sys.argv[1], sys.argv[2], sys.argv[3:]
    H = json.load(open(out)) if os.path.exists(out) else {}
    for n, st in layers(src):
        if prefixes and not n.startswith(tuple(prefixes)):
            continue
        H[n] = {k: th(v) for k, v in sorted(st.items())}
    json.dump(H, open(out, "w"), indent=0, sort_keys=True)
    print(f"[TH] {len(H)} layers in {out}")
