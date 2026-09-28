# Temporal reuse between frames on the deployed runtime, over the sequential clips (seq_extract.py), every STRIDE-th
# frame (21 = image-goal cadence, 9 = pose cadence, 1 = native 20 fps). METHOD (combine with '+'):
#   full (reference) | goalvis: goal vision features once per goal (exact) | goalkv<N>: goal tokens' LLM K/V reused,
#   full refresh every N frames (approximate: bidirectional attention) | vlac<tau>: VLA-Cache-style token reuse |
#   vittrunc: skip the unused last ViT block (exact) | vit<tau>: partial ViT recompute | freq<tau>: FreqCache-style reuse
# Output: mf_results/cache_<METHOD>@<MODE>_s<STRIDE>/<frame>.npz.
# Run in the OmniVLA clone: $OMNIVLA_ROOT/deploy/launch.sh jetson_cache.py METHOD STRIDE [MODE] [--clips N] [--frames N]
import csv, math, os, sys, time
import numpy as np
import torch
from PIL import Image
ROOT = os.environ.get("OMNIVLA_ROOT", "/mnt/nvme/omnivla")
sys.path.insert(0, f"{ROOT}/deploy")
from omnivla_deploy import OmniVLADeploy, N_IMG, _grid_keep
import transformers.models.llama.modeling_llama as ML

METHOD, STRIDE = sys.argv[1], int(sys.argv[2])
MODE = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 6
NCLIP = int(sys.argv[sys.argv.index("--clips") + 1]) if "--clips" in sys.argv else 999
NFR = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 999
OUT = f"mf_results/cache_{METHOD}@{MODE}_s{STRIDE}"; os.makedirs(OUT, exist_ok=True)
parts = METHOD.split("+")
GOALVIS = any(p.startswith(("goalvis", "goalkv")) for p in parts)
GOALKV = next((int(p[6:]) for p in parts if p.startswith("goalkv")), None)
VLAC = next((float(p[4:]) for p in parts if p.startswith("vlac")), None)
FREQ = next((float(p[4:]) for p in parts if p.startswith("freq")), None)
VIT = next((float(p[3:]) for p in parts if p.startswith("vit") and p != "vittrunc"), None)
VITTRUNC = "vittrunc" in parts or VIT is not None      # skip the last ViT block (its output is never used)
assert MODE == 6 or not GOALVIS, "goal caching needs the image-goal mode"

m = OmniVLADeploy(f"{ROOT}/deploy"); vla = m.vla; S = m._S
KEEP = _grid_keep(m.n_keep)                                  # kept current-image patch indices (0..255)
C = dict(mode="off", kv={}, pos=None, reuse_pos=None, score=None, goal_key=None, mode_next="store", last_img=None)
REUSE = GOALKV is not None or VLAC is not None or FREQ is not None

# ---------------- attention with K/V capture and injection (same math as the fork's LlamaSdpaAttention) ----------------
def attn_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False,
                 use_cache=False, cache_position=None):
    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(v, position_ids)
    q, k = ML.apply_rotary_pos_emb(q, k, cos, sin)
    L = self.layer_idx
    if C["mode"] == "store":                                   # full frame: keep post-RoPE K/V of every token
        C["kv"][L] = [k.detach().clone(), v.detach().clone()]
        if L == 2:                                             # action -> image attention (VLA-Cache relevance veto)
            qa = q[0, :, -(8 * 4 + 1):].float(); att = torch.softmax(qa @ k[0].float().transpose(1, 2) / self.head_dim ** 0.5, -1)
            C["score"] = att.mean((0, 1))
    if C["mode"] == "reuse":
        Kc, Vc = C["kv"][L]
        src = C["reuse_src"]
        k2 = torch.cat([k, Kc[:, :, src]], 2); v2 = torch.cat([v, Vc[:, :, src]], 2)
        Kc[:, :, C["comp_dst"]] = k; Vc[:, :, C["comp_dst"]] = v          # refresh the cache for recomputed tokens
        out = torch.nn.functional.scaled_dot_product_attention(q, k2, v2, attn_mask=None, is_causal=False)
    else:
        mask = attention_mask
        if mask is not None:
            mask = mask[:, :, :, : k.shape[-2]]
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            D = mask.shape[-1]; mask = mask[:, :, -1, :].clone().unsqueeze(2).expand(-1, -1, D, -1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    return self.o_proj(out), None, past_key_value
for layer in vla.language_model.model.layers:
    layer.self_attn.forward = attn_forward.__get__(layer.self_attn)

# ---------------- token selection: drop reused tokens after the deployed elision + pruning ----------------
ob = vla._build_multimodal_attention_MMN                       # the deployed build (elision + grid pruning)
def build(inp, patches, am, aml, mid):
    emb, mask = ob(inp, patches, am, aml, mid)
    ids = S["ids"][0]
    if C["mode"] == "reuse":
        pos = ids.tolist(); R = set(C["reuse_pos"])
        keep = torch.tensor([p not in R for p in pos], device=emb.device)
        store_index = {p: i for i, p in enumerate(C["pos"])}
        C["reuse_src"] = torch.tensor([store_index[p] for p in pos if p in R], device=emb.device)
        C["comp_dst"] = torch.tensor([store_index[p] for p in pos if p not in R], device=emb.device)
        S["ids"] = ids[keep].unsqueeze(0)
        return emb[:, keep], mask[:, keep]
    C["pos"] = ids.tolist()
    return emb, mask
vla._build_multimodal_attention_MMN = build

# ---------------- goal-image vision cache (exact) ----------------
vb = vla.vision_backbone; vb_deploy = vb.forward
GV = {"key": None, "feat": None, "skipped": False}
def vb_fwd(pixel_values, *a, **kw):
    GV["skipped"] = False
    if GOALVIS and not S["skip_img"] and C.get("goal_key") is not None:
        if GV["key"] == C["goal_key"]:                          # goal unchanged: encode only the current image
            S["skip_img"] = True
            try:
                f = vb_deploy(pixel_values, *a, **kw)           # deployed path: [current | zeros]
            finally:
                S["skip_img"] = False
            GV["skipped"] = True
            return torch.cat([f[:, : GV["feat"].shape[1]], GV["feat"]], dim=1)
        f = vb_deploy(pixel_values, *a, **kw)
        n = f.shape[1] // 2; GV["key"], GV["feat"] = C["goal_key"], f[:, n:].clone()
        return f
    return vb_deploy(pixel_values, *a, **kw)
vb.forward = vb_fwd

# ---------------- part (a): partial ViT recompute for unchanged patches (DINOv2 + SigLIP) ----------------
VT = dict(call=0, reuse_patches=None, cache={})
def vit_block(blk, x, kv=None):
    """timm Block.forward, with extra cached keys/values appended (same ops as the fused-attention path)"""
    a = blk.attn; B, N, Cd = x.shape
    qkv = a.qkv(blk.norm1(x)).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0); q, k = a.q_norm(q), a.k_norm(k)
    K, V = (k, v) if kv is None else (torch.cat([k, kv[0]], 2), torch.cat([v, kv[1]], 2))
    o = torch.nn.functional.scaled_dot_product_attention(q, K, V)
    o = a.proj_drop(a.proj(o.transpose(1, 2).reshape(B, N, Cd)))
    x = x + blk.drop_path1(blk.ls1(o))
    x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return x, k, v
def make_vit_forward(model, name):
    take, npre = len(model.blocks) - 2, model.num_prefix_tokens
    def fwd(img):
        ci = VT["call"]; VT["call"] += 1
        x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(img))))
        cache = VT["cache"].get(name)
        if VIT is not None and ci == 0 and VT["reuse_patches"] is not None and cache is not None and len(VT["reuse_patches"]):
            R = torch.as_tensor(np.asarray(VT["reuse_patches"]) + npre, device=x.device)
            keep = torch.ones(x.shape[1], dtype=torch.bool, device=x.device); keep[R] = False
            Cidx = torch.nonzero(keep).squeeze(1); xc = x[:, Cidx]
            for i, blk in enumerate(model.blocks[: take + 1]):
                Kc, Vc = cache["kv"][i]
                xc, k, v = vit_block(blk, xc, (Kc[:, :, R], Vc[:, :, R]))
                Kc[:, :, Cidx] = k; Vc[:, :, Cidx] = v
            out = cache["out"]; out[:, Cidx] = xc
            return out[:, npre:].clone()
        if VIT is not None and ci == 0:                             # full compute of the current image: fill the cache
            kv = []
            for blk in model.blocks[: take + 1]:
                x, k, v = vit_block(blk, x); kv.append([k.clone(), v.clone()])
            VT["cache"][name] = dict(kv=kv, out=x.clone())
            return x[:, npre:]
        for blk in model.blocks[: take + 1]:                      # vittrunc: the last block's output is never used
            x = blk(x)
        return x[:, npre:]
    return fwd
if VITTRUNC:
    vla.vision_backbone.featurizer.forward = make_vit_forward(vla.vision_backbone.featurizer, "dino")
    vla.vision_backbone.fused_featurizer.forward = make_vit_forward(vla.vision_backbone.fused_featurizer, "siglip")

# ---------------- similarity / FreqCache helpers (pixels of the 224x224 processor input) ----------------
P = 14
def patches(a):
    return a.reshape(16, P, 16, P, 3).transpose(0, 2, 1, 3, 4).reshape(256, -1)
def rawcos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)
def phase_shift(a, b):
    F = np.fft.fft2(a.mean(-1)) * np.conj(np.fft.fft2(b.mean(-1))); r = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    return (dy - 224 if dy > 112 else dy), (dx - 224 if dx > 112 else dx)
_k = np.arange(P); DCT = np.sqrt(2 / P) * np.cos(np.pi * (2 * _k[None, :] + 1) * _k[:, None] / (2 * P)); DCT[0] /= np.sqrt(2)
def edge_energy(pt):                                           # orthonormal 2-D DCT-II high-pass energy per patch (u,v >= P/4)
    g = pt.reshape(-1, P, P, 3).mean(-1); D = DCT @ g @ DCT.T; D[:, : P // 4, : P // 4] = 0
    return (D ** 2).sum((1, 2))
def spectral_entropy(a):
    Pw = np.abs(np.fft.fft2(a.mean(-1))) ** 2; p = Pw.ravel() / Pw.sum()
    return float(-(p * np.log(p + 1e-20)).sum() / np.log(p.size))

VREF = {}                                                      # ViT: per patch reference
REF = {}                                                       # per kept token: reference patch (when its K/V were computed)
def select_reuse(img_np, fi):
    """-> list of LLM positions to reuse (current-image tokens: position = patch index + 1; goal tokens)"""
    if C["mode_next"] != "reuse":
        return None
    reuse = []
    if GOALKV is not None:
        reuse += [p for p in C["pos"] if p > N_IMG and p <= 2 * N_IMG]          # goal-image tokens (positions 257..512)
    if VLAC is not None or FREQ is not None:
        pc = patches(img_np)
        cand = []
        if VLAC is not None:
            sim = rawcos(pc[KEEP], np.concatenate([REF[int(t)] for t in KEEP]))
            veto = set()
            if C["score"] is not None:
                sc = C["score"].cpu().numpy(); cur = np.array([sc[C["pos"].index(t + 1)] for t in KEEP])
                veto = set(KEEP[np.argsort(-cur)[: len(KEEP) // 4]])
            cand = [t for t, s in zip(KEEP, sim) if s >= VLAC and t not in veto]
        else:
            dy, dx = phase_shift(img_np, C["last_img"]); sy, sx = int(round(dy / P)), int(round(dx / P))
            e = edge_energy(pc[KEEP]); hi = e > e.mean() + 0.25 * e.std()
            alpha = 0.08 + 0.42 * math.exp(-spectral_entropy(img_np))
            ok = []
            for t, h, ee in zip(KEEP, hi, e):
                r, c = divmod(int(t), 16); src = (r - sy) * 16 + (c - sx)
                if h or not (0 <= r - sy < 16 and 0 <= c - sx < 16) or src != t:   # aligned source must be this cached token
                    continue
                if rawcos(pc[t:t + 1], REF[t])[0] >= FREQ:
                    ok.append((ee, t))
            cand = [t for _, t in sorted(ok)[: int(alpha * len(KEEP))]]
        reuse += [int(t) + 1 for t in cand]
        for t in KEEP:
            if t not in set(cand):
                REF[t] = pc[t:t + 1]                           # recomputed -> new reference
    return reuse

# ---------------- run the clips ----------------
rows = list(csv.DictReader(open("seq_frames.csv")))
clips = sorted({r["clip"] for r in rows})[:NCLIP]
gp = np.array([20.0, 0.0, 1.0, 0.0])
m.warmup(modes=(MODE,), n=2)
tstats = []
for clip in clips:
    fr = sorted((int(r["idx"]), r["frame"]) for r in rows if r["clip"] == clip)
    fr = [f for i, f in fr if i % STRIDE == 0][:NFR]
    gimg = Image.open(f"seq/goals/{clip}.jpg").convert("RGB") if MODE == 6 else None
    C.update(kv={}, pos=None, score=None, goal_key=clip, mode_next="store", last_img=None); REF.clear(); GV["key"] = None
    VT["cache"].clear(); VREF.clear()
    for j, f in enumerate(fr):
        path = f"{OUT}/{f}.npz"
        img = Image.open(f"seq/{f}.jpg").convert("RGB"); a = np.asarray(img.resize((224, 224)), np.float32) / 255.0
        refresh = C["pos"] is None or (GOALKV is not None and GOALKV > 0 and j % GOALKV == 0) or \
            (GOALKV is None and VLAC is None and FREQ is None and VIT is None)
        C["mode_next"] = "store" if refresh else "reuse"
        t_sel = 0.0
        if refresh:
            REF.update({int(t): patches(a)[t:t + 1] for t in KEEP})
            reuse = None
        else:
            ts = time.time(); reuse = select_reuse(a, j); t_sel = time.time() - ts
        C["mode"], C["reuse_pos"] = ("reuse", reuse) if reuse else (("store" if REUSE else "off"), None)
        VT["call"] = 0; VT["reuse_patches"] = None; nvit = 0; tv = time.time()
        if VIT is not None and not refresh and VT["cache"]:
            pc = patches(a); sim = rawcos(pc, np.concatenate([VREF[t] for t in range(256)]))
            VT["reuse_patches"] = [t for t in range(256) if sim[t] >= VIT]; nvit = len(VT["reuse_patches"])
            for t in range(256):
                if sim[t] < VIT:
                    VREF[t] = pc[t:t + 1]
        if VIT is not None and (refresh or not VT["cache"]):
            pc = patches(a); VREF.update({t: pc[t:t + 1] for t in range(256)})
        t_sel += time.time() - tv
        o = m.predict(img, mode=MODE, goal_pose=gp, goal_image=gimg)
        n_llm = int(S["ids"].shape[1]); ncur = sum(1 for p in (reuse or []) if p <= N_IMG); ngoal = len(reuse or []) - ncur
        np.savez(path, act=o["actions"], t_fwd=o["t_fwd"], t_sel=t_sel, n_llm=n_llm, n_reuse_cur=ncur, n_reuse_goal=ngoal,
                 vis_goal_skipped=GV["skipped"], refresh=refresh, n_vit_reuse=nvit)
        C["last_img"] = a; C["mode"] = "off"
        tstats.append((o["t_fwd"] + t_sel, n_llm, ncur, ngoal, GV["skipped"], nvit))
T = np.array(tstats, dtype=float)
print(f"[CACHE] {METHOD} mode {MODE} stride {STRIDE}: {len(T)} frames | fwd+select median {np.median(T[:, 0]) * 1000:.0f} ms "
      f"mean {T[:, 0].mean() * 1000:.0f} | LLM tokens mean {T[:, 1].mean():.0f} | reused cur {T[:, 2].mean():.1f} goal "
      f"{T[:, 3].mean():.0f} | goal vision skipped {T[:, 4].mean():.2f} | ViT patches reused {T[:, 5].mean():.0f}/256", flush=True)
