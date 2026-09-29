# kaggle_compress.py - OmniVLA 7B compression studies on a Kaggle T4 (needs the openvla-oft transformers fork).
# STUDY: limit (bit-width sweep) | tests (real-driving tests, incl. blind controls) | sens (per-component 2-bit
# sensitivity) | teacher (bf16 labels on held-out samples) | lora (low-bit + rank-16 correction) | wanda (2:4 pruning) |
# exq (GPTQ/ActQuant calibration and step weighting) | a8 (simulated activation quantization) | gptqx (GPTQ int4 export
# for Marlin, used by build/kaggle_build.py) | lang (CAST language-goal test, eval/data/cast_extract.py).
# Output: /kaggle/working/compress_<STUDY>.npz, keys "<cfg>__m<mode>__<frame>".
import gc, glob, json, math, os, shutil, sys, tarfile, time
import numpy as np
import torch

STUDY = os.environ.get("STUDY", "limit")
OUT = f"/kaggle/working/compress_{STUDY}{os.environ.get('OUT_SUFFIX', '')}.npz"
WORK, MODEL_DIR = "/tmp/omni", "/tmp/omnivla-original"
MODES = {7: dict(lan_prompt=True), 4: dict(pose_goal=True), 8: dict(lan_prompt=True, pose_goal=True), 6: dict(image_goal=True)}

# ---------------- setup ----------------
if not os.path.exists(f"{WORK}/prismatic"):
    tars = glob.glob("/kaggle/input/**/omnivla_bundle.tar.gz", recursive=True)
    dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/prismatic/__init__.py", recursive=True)]
    if dirs:
        shutil.copytree(os.path.dirname(dirs[0]), WORK, ignore=shutil.ignore_patterns("._*"), dirs_exist_ok=True)
    else:
        os.makedirs(WORK, exist_ok=True)
        with tarfile.open(tars[0]) as t:
            t.extractall(WORK)
os.chdir(WORK); sys.path.insert(0, WORK)
from huggingface_hub import snapshot_download
snapshot_download("NHirose/omnivla-original", local_dir=MODEL_DIR, ignore_patterns=["dist_head*", "lora_adapter/*", "modeling_prismatic_____.py"])
import utm
from PIL import Image
import inference.run_omnivla as R
import inspect, transformers.models.llama.modeling_llama as _ml
assert "is_causal=False" in inspect.getsource(_ml.LlamaSdpaAttention.forward), "install the openvla-oft transformers fork"
print("[GUARD] transformers-openvla-oft fork OK", flush=True)
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, POSE_DIM
AutoConfig.register("openvla", R.OpenVLAConfig)
AutoImageProcessor.register(R.OpenVLAConfig, R.PrismaticImageProcessor)
AutoProcessor.register(R.OpenVLAConfig, R.PrismaticProcessor)
AutoModelForVision2Seq.register(R.OpenVLAConfig, R.OpenVLAForActionPrediction_MMNv1)
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
action_tokenizer = R.ActionTokenizer(processor.tokenizer)
HSRC = open("harness.py").read()
ELIDE_SRC = HSRC[HSRC.index("# ---------- Idea 1"):HSRC.index("# ---------- modes")]
assert "full[:, ids[0]] = h" in ELIDE_SRC
ELIDE_SRC = ELIDE_SRC.replace("full[:, ids[0]] = h", "full[:, ids[0].to(h.device)] = h")   # same values; lets backward cross GPUs

gz = np.load(glob.glob("/kaggle/input/**/gt_frodobots.npz", recursive=True)[0])
GOALS = {k[6:]: gz[k].astype(np.float64) for k in gz.files if k.startswith("goal__")}
POSE_FRAMES = [(n, f"./frames/{n}.jpg") for n in sorted(GOALS)]
VER_FRAMES = [(n, f"./frames/{n}.jpg") for n in [l.strip() for l in open("verify_frames.txt") if l.strip()]]
TESTS = {}
tz = glob.glob("/kaggle/input/**/gt_tests.npz", recursive=True)
if STUDY == "tests" and not tz:
    print("[C] input listing:", sorted(glob.glob("/kaggle/input/**/*.np*", recursive=True))[:20], flush=True)
    raise SystemExit("gt_tests.npz missing from the attached dataset version")
if tz:
    import pandas as pd
    tn = np.load(tz[0]); tdf = pd.read_csv(glob.glob("/kaggle/input/**/gt_tests.csv", recursive=True)[0])
    smap = dict(pd.read_csv(glob.glob("/kaggle/input/**/shuffle_map.csv", recursive=True)[0]).values)
    rel = tdf[tdf.reliable_path]
    GOAL20 = {k[8:]: tn[k].astype(np.float64) for k in tn.files if k.startswith("goal20__")}
    gdir = [d for d in glob.glob("/kaggle/input/**/goal_img3m", recursive=True) if os.path.isdir(d)] + glob.glob(f"{WORK}/goal_img3m")
    TESTS = {
        "pose5": dict(mode=4, frames=[f for f in rel.frame if f in GOALS], goal=lambda f: GOALS[f], gimg=None),
        "pose20": dict(mode=4, frames=[f for f in rel[rel.far_ok].frame if f in GOAL20], goal=lambda f: GOAL20[f], gimg=None),
        "img3m": dict(mode=6, frames=[f for f in rel[rel.img_ok].frame], goal=None, gimg=lambda f: f"{gdir[0]}/{f}.jpg"),
    }
    cz = glob.glob("/kaggle/input/**/cast_lang.npz", recursive=True)
    if cz:                                             # language goal (CAST); image resized as OmniVLA's CAST loader does
        CZ = np.load(cz[0]); CM = json.loads(str(CZ["meta"]))
        TESTS["lang"] = dict(mode=7, frames=sorted(CM), goal=None, gimg=None,
                             img=lambda k: Image.fromarray(CZ[f"img__{k}"]).resize((224, 224)),
                             shuf={k: v["img_shuffle"] for k, v in CM.items()}, lang=lambda k: CM[k]["instruction"],
                             lshuf={k: v["lang_shuffle"] for k, v in CM.items()})
    lz = glob.glob("/kaggle/input/**/lelan_lang.npz", recursive=True)
    if lz:                                             # object goal (LeLaN); lang_shuffled = the other object's prompt
        LZ = np.load(lz[0]); LM = json.loads(str(LZ["meta"]))
        TESTS["lelan"] = dict(mode=7, frames=sorted(LM), goal=None, gimg=None, img=lambda k: Image.fromarray(LZ[f"img__{k}"]),
                              shuf={k: v["img_shuffle"] for k, v in LM.items()}, lang=lambda k: "move toward " + LM[k]["target"],
                              lang_alt=lambda k: "move toward " + LM[k]["distractor"])
    sz = glob.glob("/kaggle/input/**/seq_demo.npz", recursive=True)
    if sz:                                             # image goal on the sequential demo clips (every 21st frame)
        SZ = np.load(sz[0]); SM = json.loads(str(SZ["meta"]))
        TESTS["seq"] = dict(mode=6, frames=sorted(SM), goal=None, gimg=None, img=lambda k: Image.fromarray(SZ[f"img__{k}"]),
                            gimg_arr=lambda k: Image.fromarray(SZ[f"goal__{SM[k]}"]))
    if os.environ.get("TESTS_ONLY"):
        want = os.environ["TESTS_ONLY"].split(",")
        missing = [t for t in want if not TESTS.get(t, {}).get("frames")]
        if missing:                                    # a missing input file must not turn into an empty, "passing" test
            raise SystemExit(f"[C] TESTS_ONLY: no frames for {missing} (available: "
                             f"{ {k: len(v['frames']) for k, v in TESTS.items()} }); is the input dataset attached?")
        TESTS = {k: v for k, v in TESTS.items() if k in want}
    print("[C] tests:", {k: len(v["frames"]) for k, v in TESTS.items()}, flush=True)
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:                                              # quick integration test: 3 pose frames, 2 verify frames
    POSE_FRAMES, VER_FRAMES, OUT = POSE_FRAMES[:3], VER_FRAMES[:2], OUT.replace(".npz", "_smoke.npz")
res = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}

# ---------------- model building ----------------
def split_map():
    lay = {i: (0 if i < 14 else 1) for i in range(32)}
    dm = {"vision_backbone": 0, "projector": 0, "language_model.model.embed_tokens": 0,
          "language_model.model.norm": 1, "language_model.lm_head": 1}
    dm.update({f"language_model.model.layers.{i}": d for i, d in lay.items()})
    return dm

def module_bits(name, cfg):
    """bits for an nn.Linear given the config (None = keep full precision)"""
    import re
    for pattern, b in cfg.get("override", {}).items():         # regex rules, first match wins (listed order)
        if re.search(pattern, name):
            return b
    if name.startswith("language_model.model.layers."):
        return cfg.get("llm")
    if name.startswith("vision_backbone.") and ".blocks." in name:
        return cfg.get("vis")
    return None

def build(cfg):
    base = cfg["base"]
    dt = torch.bfloat16 if base in ("bf16", "bnb_nf4_all") else torch.float16
    if base == "bnb_nf4_all":                                   # same config as the NF4 baseline
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["projector", "lm_head"])
        vla = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, torch_dtype=dt, quantization_config=q, device_map=split_map(), low_cpu_mem_usage=True)
    elif base == "bnb_nf4_llm":                                 # LLM NF4, vision full precision
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["projector", "lm_head", "vision_backbone"])
        vla = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, torch_dtype=dt, quantization_config=q, device_map=split_map(), low_cpu_mem_usage=True)
    else:
        vla = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, torch_dtype=dt, device_map=split_map(), low_cpu_mem_usage=True)
    n_q = {} if cfg.get("wanda") else quantize_hqq(vla, cfg, dt)
    return vla, dt, n_q

def quantize_hqq(vla, cfg, dt):
    n_q = {}
    if cfg.get("llm") or cfg.get("vis") or cfg.get("override"):
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        targets = [(n, m) for n, m in vla.named_modules() if isinstance(m, torch.nn.Linear) and module_bits(n, cfg)]
        for n, m in targets:
            b = module_bits(n, cfg)
            gs = cfg.get("group", {}).get(str(b), 64) if isinstance(cfg.get("group"), dict) else 64
            ql = HQQLinear(m, quant_config=BaseQuantizeConfig(nbits=b, group_size=gs), compute_dtype=dt,
                           device=str(m.weight.device), del_orig=True)
            parent = vla.get_submodule(n.rsplit(".", 1)[0]); setattr(parent, n.rsplit(".", 1)[1], ql)
            n_q[b] = n_q.get(b, 0) + 1
        gc.collect(); torch.cuda.empty_cache()
    return n_q

# ---------------- held-out data (studies 3 and 5; never the test rides) ----------------
HO = {}
_hz = glob.glob("/kaggle/input/**/heldout.npz", recursive=True)
if _hz:
    _h = np.load(_hz[0])
    _hdir = os.path.dirname([p for p in glob.glob("/kaggle/input/**/heldout_frames/*.jpg", recursive=True)][0])
    _gdir = os.path.dirname([p for p in glob.glob("/kaggle/input/**/heldout_goals_img/*.jpg", recursive=True)][0])
    HO = {k[6:]: dict(goal=_h[k].astype(np.float64), img=f"{_hdir}/{k[6:]}.jpg", gimg=f"{_gdir}/{k[6:]}.jpg")
          for k in _h.files if k.startswith("goal__")}
    print(f"[C] held-out samples: {len(HO)}", flush=True)
if STUDY in ("teacher", "lora", "wanda", "exq", "a8", "gptqx") and not HO:
    print("[C] input listing:", sorted(glob.glob("/kaggle/input/**/*.np*", recursive=True))[:20], flush=True)
    raise SystemExit("held-out data missing from the attached dataset version")

TRAIN, SEL = {"on": False}, {"set": 0}
class LoRA(torch.nn.Module):
    """y = Q(W) x + B_k A_k x ; k = shared (0) or the modality's set (study 3)"""
    def __init__(self, base, r, n_sets, in_f, out_f, dev):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(n_sets, r, in_f, device=dev) / in_f ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(n_sets, out_f, r, device=dev))
    def forward(self, x):
        y = self.base(x)
        k = SEL["set"]
        return y + ((x.float() @ self.A[k].T) @ self.B[k].T).to(y.dtype)

def attach_lora(vla, per_modality, rank_total=16):
    n_sets = 2 if per_modality else 1
    r = rank_total // n_sets                                       # equal total size: 1 x r16 vs 2 x r8
    params, n = [], 0
    for p_ in vla.parameters():
        p_.requires_grad_(False)
    targets = [(nm, m) for nm, m in vla.named_modules() if nm.startswith("language_model.model.layers.")
               and type(m).__name__ in ("HQQLinear", "Linear")]
    for nm, m in targets:
        if hasattr(m, "meta") and "shape" in m.meta:
            out_f, in_f = m.meta["shape"]
        else:
            out_f, in_f = m.out_features, m.in_features
        dev = next((b.device for b in m.buffers()), None) or next(m.parameters()).device
        w = LoRA(m, r, n_sets, in_f, out_f, dev)
        parent = vla.get_submodule(nm.rsplit(".", 1)[0]); setattr(parent, nm.rsplit(".", 1)[1], w)
        params += [w.A, w.B]; n += w.A.numel() + w.B.numel()
    print(f"[L] LoRA on {len(targets)} LLM linears: {n_sets} set(s) x rank {r} = {n/1e6:.1f} M params", flush=True)
    return params

def train_lora(inf, CUR, MODE, E, lcfg):
    teach = np.load(glob.glob("/kaggle/input/**/compress_teacher.npz", recursive=True)[0])
    torch.manual_seed(lcfg.get("seed", 0))
    params = attach_lora(R.vla, lcfg["per_modality"])
    opt = torch.optim.AdamW(params, lr=lcfg.get("lr", 2e-4), weight_decay=0.0)
    TRAIN.update(on=True, scaler=torch.cuda.amp.GradScaler())
    samples = [(k, m) for k in sorted(HO) for m in (4, 6) if f"teach__m{m}__{k}" in teach.files]
    samples = samples[: int(os.environ.get("LORA_MAX", 10 ** 9))]
    print(f"[L] training samples: {len(samples)} (teacher labels {len(teach.files)})", flush=True)
    rng_ = np.random.default_rng(lcfg.get("seed", 0))
    LW = {m: step_weights(lcfg.get("loss_w", "uniform"), m) for m in (4, 6)}
    print(f"[L] loss weights {lcfg.get('loss_w', 'uniform')}: " + " | ".join(f"m{m} {np.round(v, 3).tolist()}" for m, v in LW.items()), flush=True)
    step, t0, losses = 0, time.time(), []
    for ep in range(lcfg.get("epochs", 2)):
        rng_.shuffle(samples)
        for k, m in samples:
            R.satellite = R.lan_prompt = R.pose_goal = R.image_goal = False
            for k2, v2 in MODES[m].items():
                setattr(R, k2, v2)
            MODE["id"] = m; E["on"] = True; E["skip_img"] = m != 6
            SEL["set"] = {4: 0, 6: 1}[m] if lcfg["per_modality"] else 0
            CUR["img"] = Image.open(HO[k]["img"]).convert("RGB")
            CUR["goal"] = HO[k]["goal"] if m == 4 else None
            CUR["gimg"] = Image.open(HO[k]["gimg"]).convert("RGB") if m == 6 else None
            TRAIN["target"] = torch.from_numpy(teach[f"teach__m{m}__{k}"]).float()
            TRAIN["w"] = torch.from_numpy(LW[m]).float()
            opt.zero_grad(set_to_none=True)
            inf.run_omnivla()
            TRAIN["scaler"].step(opt); TRAIN["scaler"].update()
            losses.append(CAP_REF["loss"]); step += 1
            if step % 50 == 0:
                print(f"[L] ep {ep} step {step}/{len(samples) * lcfg.get('epochs', 2)} L1 {np.mean(losses[-50:]):.4f} "
                      f"| {(time.time()-t0)/step:.2f} s/step", flush=True)
    TRAIN["on"] = False
    torch.save({f"{i}": p_.detach().cpu() for i, p_ in enumerate(params)}, f"/kaggle/working/lora_{lcfg['name']}.pt")
    return losses

WANDA_E = None
def wanda_2_4(vla, inf, CUR, MODE, n_calib=64):
    """Wanda (|W| * ||x||) 2:4 pruning of all LLM linears, calibrated on held-out frames (modes 4 and 6)."""
    stats, hooks = {}, []
    lin = [(n, m) for n, m in vla.named_modules() if n.startswith("language_model.model.layers.") and isinstance(m, torch.nn.Linear)]
    for n, m in lin:
        def h(mod, a, n=n):
            x = a[0].detach().float().reshape(-1, a[0].shape[-1])
            stats[n] = stats.get(n, 0) + (x ** 2).sum(0)
        hooks.append(m.register_forward_pre_hook(h))
    keys = sorted(HO)[:n_calib]
    for i, k in enumerate(keys):
        m = 4 if i % 2 == 0 else 6
        R.satellite = R.lan_prompt = R.pose_goal = R.image_goal = False
        for k2, v2 in MODES[m].items():
            setattr(R, k2, v2)
        MODE["id"] = m
        E_ = WANDA_E; E_["on"] = True; E_["skip_img"] = m != 6
        CUR["img"] = Image.open(HO[k]["img"]).convert("RGB"); CUR["goal"] = HO[k]["goal"] if m == 4 else None
        CUR["gimg"] = Image.open(HO[k]["gimg"]).convert("RGB") if m == 6 else None
        inf.run_omnivla()
    for h in hooks:
        h.remove()
    with torch.no_grad():
        for n, m in lin:
            W = m.weight.data
            score = W.abs().float() * stats[n].to(W.device).sqrt()[None, :]
            g = score.reshape(W.shape[0], -1, 4)
            idx = g.argsort(dim=-1)[..., :2]                       # 2 smallest of every 4 input weights
            mask = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False).reshape(W.shape)
            W.mul_(mask.to(W.dtype))
    print(f"[C] Wanda 2:4 applied to {len(lin)} LLM linears (calibration {len(keys)} held-out samples)", flush=True)

def weight_bytes():
    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))

# ---------------- study 6: execution weights, deployed pruning, action-aware calibration ----------------
STEP_S = 0.3
LAT = {"deploy": {4: 0.43, 6: 1.05}, "nf4": {4: 1.43, 6: 2.2}}     # measured medians (final_validation.md, deploy_table.md)
def exec_weights(L):
    """ChunkExecutor (deploy/ros2/rover_protocol.py): a chunk captured at t is entered at its own age when it arrives
    (t+L) and replaced by the next one (t+2L). Step k (0-based) covers [0.3k, 0.3(k+1)] s after t; its weight is the
    fraction of that interval executed. Latency jitter is < 10 ms at the deployed config, so no smoothing."""
    return np.array([max(0.0, min(2 * L, STEP_S * (k + 1)) - max(L, STEP_S * k)) / STEP_S for k in range(8)])
def step_weights(kind, mode):
    """per-step loss weights, normalized to mean 1 (same total weight as uniform)"""
    if kind == "uniform":
        return np.ones(8)
    if kind.startswith("exec_"):
        w = exec_weights(LAT[kind[5:]][mode])
    elif kind.startswith("hard_"):
        w = (exec_weights(LAT[kind[5:]][mode]) > 0).astype(float)
    else:
        raise ValueError(kind)
    return w / w.mean()

PRUNE = os.environ.get("PRUNE", "")
PP = dict(active=None, frac=0.0, drop=None)          # prompt-aware pruning state (deploy/prompt_prune.py)
def apply_prune(vla, ns, spec=None):
    """deployed uniform-grid pruning of current-image tokens on top of elision. spec: "spatial<pct>", "none" (back to
    elision only; used to change the pruning after GPTQ, EVAL_PRUNES), "prompt<pct>" / "promptorig<pct>" (drop the
    patches least similar to the instruction, scored with OmniVLA's fine-tuned / the original SigLIP image tower; needs
    install_prompt_scoring) or None (= the PRUNE setting)"""
    spec = PRUNE if spec is None else spec
    PP.update(active=None, drop=None)
    if spec == "none":
        vla._build_multimodal_attention_MMN = ns["build_elide"]; vla.language_model.forward = ns["lm_fwd"]
        print("[C] pruning: off (elision only)", flush=True)
        return
    if not spec:
        return
    dyn = spec.startswith("prompt")
    if dyn:
        kind = "orig" if spec.startswith("promptorig") else "ft"
        frac, N_IMG = int(spec[len("promptorig" if kind == "orig" else "prompt"):]) / 100.0, 256
        PP.update(active=kind, frac=frac)
    else:
        assert spec.startswith("spatial"), spec
        frac, N_IMG = int(spec[7:]) / 100.0, 256
    n_keep = N_IMG - int(round(N_IMG * frac))
    drop = np.setdiff1d(np.arange(N_IMG), np.unique(np.round(np.linspace(0, N_IMG - 1, n_keep)).astype(int))) + 1
    ST = {}
    _ob, _ol = ns["_orig_build"], ns["_orig_lm"]
    def build_p(inp, patches, am, aml, mid):
        emb, mask = _ob(inp, patches, am, aml, mid)
        keep = mask[0].bool().clone(); full_len = mask.shape[1]
        if dyn:
            assert PP["drop"] is not None, "prompt scoring did not run for this prediction"
            keep[PP["drop"].to(keep.device)] = False; PP["drop"] = None
        else:
            keep[torch.as_tensor(drop, device=keep.device)] = False
        ST.update(ids=torch.arange(full_len, device=mask.device)[keep].unsqueeze(0), full=full_len)
        return emb[:, keep], mask[:, keep]
    vla._build_multimodal_attention_MMN = build_p
    def lm_fwd_p(*a, **kw):
        kw["labels"] = None; kw["position_ids"] = ST["ids"]
        out = _ol(*a, **kw)
        h = out.hidden_states[-1]
        full = h.new_zeros(h.shape[0], ST["full"], h.shape[2]); full[:, ST["ids"][0].to(h.device)] = h
        out.hidden_states = tuple(out.hidden_states[:-1]) + (full,)
        return out
    vla.language_model.forward = lm_fwd_p
    print(f"[C] pruning: {'prompt-aware (' + PP['active'] + ')' if dyn else 'spatial'} {int(frac * 100)}% of current-image "
          f"tokens (keep {n_keep})", flush=True)

def install_prompt_scoring(vla, inf, phrases):
    """score current-image patches against the instruction for prompt<pct> / promptorig<pct> (deploy/prompt_prune.py)"""
    sys.path.insert(0, "/kaggle/working")
    import prompt_prune as P
    te = P.TextEncoder(device="cpu")                                     # text on the CPU in fp32, once
    TXT = te.encode(sorted(set(phrases)))
    orig = te.image_tower().half().to("cuda:0").eval()                   # original SigLIP image trunk (reference variant)
    del te; gc.collect()
    ff = vla.vision_backbone.fused_featurizer
    def scores(emb):
        t = TXT[P.object_phrase(inf.lan_inst_prompt)].to(emb.device)
        PP["drop"] = P.drop_positions(emb, t, PP["frac"])
    def pre(mod, args):
        if PP["active"] == "orig" and PP["drop"] is None:
            x = args[0].to("cuda:0", torch.float16)
            with torch.no_grad():
                scores(P.patch_embeddings(orig, P.penultimate(orig, x)))
    def post(mod, args, out):
        if PP["active"] == "ft" and PP["drop"] is None:
            x = out[0] if isinstance(out, (tuple, list)) else out
            with torch.no_grad():
                scores(P.patch_embeddings(ff, x))
    ff.register_forward_pre_hook(pre); ff.register_forward_hook(post)
    print(f"[C] prompt scoring ready: {len(TXT)} phrases, SigLIP text tower timm/ViT-SO400M-14-SigLIP", flush=True)

def llm_linears(vla):
    return [(n, m) for n, m in vla.named_modules() if n.startswith("language_model.model.layers.") and isinstance(m, torch.nn.Linear)]

def set_sample(k, m, CUR, MODE, E):
    R.satellite = R.lan_prompt = R.pose_goal = R.image_goal = False
    for k2, v2 in MODES[m].items():
        setattr(R, k2, v2)
    MODE["id"] = m; E["on"] = True; E["skip_img"] = m != 6
    CUR["img"] = Image.open(HO[k]["img"]).convert("RGB"); CUR["goal"] = HO[k]["goal"] if m == 4 else None
    CUR["gimg"] = Image.open(HO[k]["gimg"]).convert("RGB") if m == 6 else None

def importance(vla, inf, CUR, MODE, E, calib):
    """Per-step action Fisher per token for every LLM linear (Hutchinson over the 4 action dims, exact over steps):
    S[n][i] = (8, T): ||d(z . a_k)/d y_t||^2 for output y of linear n, sample i, step k.
    B[n][mode] = (8, in): sum over samples/tokens of S * x^2 (x = linear input); B0[n] = sum of x^2 (plain)."""
    lin = llm_linears(vla)
    S, B, B0, cur = {n: [] for n, _ in lin}, {}, {}, {}
    hooks = [m.register_forward_hook(lambda mod, a, out, n=n: cur.__setitem__(n, (a[0].detach(), out))) for n, m in lin]
    gam = vla.language_model.model.layers[0].input_layernorm.weight
    gam.requires_grad_(True)
    TRAIN.update(on=True, imp=True)
    gen = torch.Generator(device="cpu").manual_seed(0)
    t0 = time.time()
    for i, (k, m) in enumerate(calib):
        set_sample(k, m, CUR, MODE, E); cur.clear()
        inf.run_omnivla()
        pred = CAP_REF["pred_graph"]
        outs = [cur[n][1] for n, _ in lin]
        per = {n: [] for n, _ in lin}
        for step in range(8):
            z = torch.randn(4, generator=gen).to(pred.device)
            L = (pred[0, step].float() * z).sum() * 1024.0
            grads = torch.autograd.grad(L, outs, retain_graph=step < 7, allow_unused=True)
            for (n, _), g in zip(lin, grads):
                per[n].append(torch.nan_to_num(g[0].float() ** 2, posinf=0.0).sum(-1) if g is not None else None)
        for n, _ in lin:
            s = torch.stack(per[n]).float()                             # (8, T)
            x2 = cur[n][0][0].float() ** 2                              # (T, in)
            B.setdefault(n, {}).setdefault(m, 0)
            B[n][m] = B[n][m] + s @ x2
            B0[n] = B0.get(n, 0) + x2.sum(0)
            S[n].append(s.cpu())                                       # fp32: sums of squared grads overflow fp16
        del outs, per; cur.clear(); CAP_REF.pop("pred_graph", None)
        if i % 16 == 0:
            print(f"[X] importance {i + 1}/{len(calib)} | {(time.time() - t0) / (i + 1):.1f} s/sample", flush=True)
    for h in hooks:
        h.remove()
    gam.requires_grad_(False); TRAIN.update(on=False, imp=False)
    return S, B, B0

def token_w(S_ni, kind, mode):
    """per-token weight (T,) for one sample: plain -> None (unweighted); else step weights . S"""
    if kind == "plain":
        return None
    w = torch.from_numpy(step_weights("uniform" if kind == "act" else kind, mode)).float()
    tw = torch.nan_to_num(w @ S_ni.float(), nan=0.0, posinf=0.0)
    return tw / tw.mean().clamp(min=1e-20)                       # per-sample mean 1: weights shape tokens within a sample

def col_w(B_n, B0_n, kind):
    if kind == "plain":
        return B0_n
    return sum(torch.from_numpy(step_weights("uniform" if kind == "act" else kind, m)).float().to(B_n[m].device) @ B_n[m] for m in B_n)

def q_pc4(w, s):                               # Marlin per-channel convention: q in 0..15, zero 8
    return (torch.clamp(torch.round(w / s) + 8, 0, 15) - 8) * s

def gptq(W, H, fmt, blocksize=128, percdamp=0.01, gs=32):
    """GPTQ (Frantar et al. 2023), column-wise error feedback with the (optionally token-weighted) Hessian H."""
    W = W.float().clone(); H = H.clone(); n = W.shape[1]
    assert torch.isfinite(H).all(), "non-finite Hessian"
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(n, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    if fmt == "pc4":
        s = (W.abs().amax(1, keepdim=True) * 2 / 15).clamp(min=1e-8).half().float()
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, d = W1[:, i], Hinv1[i, i]
            if fmt == "g2":
                if (i1 + i) % gs == 0:
                    g = W1[:, i:i + gs]
                    xmin = torch.minimum(g.min(1)[0], torch.zeros(1, device=g.device)); xmax = torch.maximum(g.max(1)[0], torch.zeros(1, device=g.device))
                    sc = ((xmax - xmin) / 3).clamp(min=1e-8).half().float(); zr = torch.round(-xmin / sc)
                q = (torch.clamp(torch.round(w / sc) + zr, 0, 3) - zr) * sc
            else:
                q = q_pc4(w[:, None], s)[:, 0]
            Q1[:, i] = q; err = (w - q) / d
            W1[:, i:] -= err[:, None] @ Hinv1[i, i:][None, :]
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q

def actq(W, b, fmt, iters=8, gs=32):
    """ActQuant-style intra-tensor scale optimization (Akbari et al. 2026): weighted least squares for the scales with
    omega_ij = F_ij * sqrt(sigma^2 + w_ij^2). F is approximated as separable (row factor x column factor b_j), so
    within a row/group only the column factor b_j = sum_t s_t x_tj^2 matters (s_t = action-Fisher token weight)."""
    W = W.float(); b = (b.float() / b.float().mean()).clamp(min=1e-6)
    if fmt == "pc4":
        om = b[None, :] * torch.sqrt(W.var(1, keepdim=True) + W ** 2)
        s0 = (W.abs().amax(1, keepdim=True) * 2 / 15).clamp(min=1e-8)
        err = lambda s: (om * (W - q_pc4(W, s)) ** 2).sum(1, keepdim=True)
        best_s, best_e = s0.half().float(), err(s0.half().float())
        for r in torch.linspace(0.5, 1.0, 21).tolist():
            s = (s0 * r).half().float(); e = err(s); better = e < best_e
            best_s, best_e = torch.where(better, s, best_s), torch.where(better, e, best_e)
        s = best_s
        for _ in range(iters):
            q = torch.clamp(torch.round(W / s) + 8, 0, 15) - 8
            s_new = ((om * W * q).sum(1, keepdim=True) / (om * q * q).sum(1, keepdim=True).clamp(min=1e-12)).clamp(min=1e-8).half().float()
            e = err(s_new); better = e < best_e
            best_s, best_e = torch.where(better, s_new, best_s), torch.where(better, e, best_e); s = best_s
        return q_pc4(W, best_s)
    # g2: asymmetric 2-bit, group gs along input, float zero (HQQ / GemLite format): W ~= s*q - m
    o, i = W.shape
    G = W.reshape(o, i // gs, gs); bb = b.reshape(1, i // gs, gs).expand_as(G)
    om = bb * torch.sqrt(G.var(-1, keepdim=True) + G ** 2)
    mn, mx = G.min(-1, keepdim=True)[0], G.max(-1, keepdim=True)[0]
    def deq(s, m):
        return s * torch.clamp(torch.round((G + m) / s), 0, 3) - m
    err = lambda s, m: (om * (G - deq(s, m)) ** 2).sum(-1, keepdim=True)
    best = None
    for r in torch.linspace(0.6, 1.0, 9).tolist():
        c = (mn + mx) / 2; half = (mx - mn) / 2 * r
        s = ((2 * half) / 3).clamp(min=1e-8).half().float(); m = (-(c - half)).half().float()
        e = err(s, m)
        if best is None:
            best = [s, m, e]
        else:
            bt = e < best[2]; best = [torch.where(bt, s, best[0]), torch.where(bt, m, best[1]), torch.where(bt, e, best[2])]
    s, m = best[0], best[1]
    for _ in range(iters):                                     # alternate: rounding, then 2x2 weighted LSQ for (s, m)
        q = torch.clamp(torch.round((G + m) / s), 0, 3)
        sw, sq, sqq = om.sum(-1, keepdim=True), (om * q).sum(-1, keepdim=True), (om * q * q).sum(-1, keepdim=True)
        sy, syq = (om * G).sum(-1, keepdim=True), (om * G * q).sum(-1, keepdim=True)
        det = (sqq * sw - sq * sq)
        ok = det.abs() > 1e-12
        s_new = torch.where(ok, (syq * sw - sq * sy) / det.where(ok, torch.ones_like(det)), s).clamp(min=1e-8).half().float()
        m_new = (s_new * sq - sy) / sw.clamp(min=1e-12)
        m_new = m_new.half().float()
        e = err(s_new, m_new); bt = e < best[2]
        best = [torch.where(bt, s_new, best[0]), torch.where(bt, m_new, best[1]), torch.where(bt, e, best[2])]
        s, m = best[0], best[1]
    return deq(best[0], best[1]).reshape(o, i)

class _Captured(Exception):
    pass

def capture_layer0(vla, inf, CUR, MODE, E, calib):
    """inputs (hidden states + kwargs) of decoder layer 0 for every calibration sample (quantized vision, fp16 LLM)"""
    L0 = vla.language_model.model.layers[0]; store = []
    def pre(mod, args, kw):
        hs = args[0] if args else kw["hidden_states"]
        kw2 = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in kw.items() if k != "hidden_states"}
        kw2["past_key_value"] = None; kw2["use_cache"] = False       # replaying with the model's DynamicCache grows it every call (OOM)
        store.append((hs.detach().cpu(), kw2)); raise _Captured
    h = L0.register_forward_pre_hook(pre, with_kwargs=True)
    for k, m in calib:
        set_sample(k, m, CUR, MODE, E)
        try:
            inf.run_omnivla()
        except _Captured:
            pass
    h.remove()
    return store

def gptq_model(vla, inps, S, calib, kind, fmt, DT):
    """sequential GPTQ over the 32 decoder layers; H_n = sum_i sum_t w_t x_t x_t^T with w_t = token_w(kind)"""
    layers = vla.language_model.model.layers
    t0 = time.time()
    for li, layer in enumerate(layers):
        dev = next(layer.parameters()).device
        lin = [(n, m) for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)]
        pref = f"language_model.model.layers.{li}."
        H, cur_i = {n: None for n, _ in lin}, {"i": 0}
        def acc(mod, a, out, n=None):
            x = a[0][0].float()                                         # (T, in)
            tw = token_w(S[pref + n][cur_i["i"]], kind, calib[cur_i["i"]][1]) if kind != "plain" else None
            xw = x if tw is None else x * tw.to(x.device)[:, None]
            with torch.autocast("cuda", enabled=False):                 # fp32 Hessian (the layer replay runs under autocast)
                Hn = xw.T @ x
            H[n] = Hn if H[n] is None else H[n] + Hn
        hooks = [m.register_forward_hook(lambda mod, a, out, n=n: acc(mod, a, out, n)) for n, m in lin]
        with torch.no_grad(), torch.autocast("cuda", dtype=DT):
            for i, (hs, kw) in enumerate(inps):
                cur_i["i"] = i
                layer(hs.to(dev), **{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kw.items()})
        for h_ in hooks:
            h_.remove()
        for n, m in lin:
            m.weight.data = gptq(m.weight.data, H[n], fmt).to(m.weight.dtype)
        del H
        with torch.no_grad(), torch.autocast("cuda", dtype=DT):              # quantized outputs -> next layer inputs
            new = []
            for hs, kw in inps:
                out = layer(hs.to(dev), **{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kw.items()})
                new.append(((out[0] if isinstance(out, tuple) else out).detach().cpu(), kw))
            inps = new
        torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"[X] gptq {kind}/{fmt} layer {li} | {time.time() - t0:.0f} s", flush=True)

def actq_model(vla, B, B0, kind, fmt):
    for n, m in llm_linears(vla):
        m.weight.data = actq(m.weight.data, col_w(B[n], B0[n], kind).to(m.weight.device), fmt).to(m.weight.dtype)

def rtn_model(vla, fmt):
    for n, m in llm_linears(vla):
        W = m.weight.data.float()
        if fmt == "pc4":
            m.weight.data = q_pc4(W, (W.abs().amax(1, keepdim=True) * 2 / 15).clamp(min=1e-8).half().float()).to(m.weight.dtype)
        else:
            m.weight.data = _rtn_g2(W).to(m.weight.dtype)

def hqq_model(vla, DT):
    """HQQ 2-bit g32 (the study 1/3 quantizer), dequantized back into the fp16 Linear"""
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
    HQQLinear.set_backend(HQQBackend.PYTORCH)
    for n, m in llm_linears(vla):
        ql = HQQLinear(m, quant_config=BaseQuantizeConfig(nbits=2, group_size=32), compute_dtype=DT, device=str(m.weight.device), del_orig=False)
        m.weight.data = ql.dequantize().reshape(m.weight.shape).to(m.weight.dtype); del ql

def _rtn_g2(W, gs=32):
    o, i = W.shape; G = W.reshape(o, i // gs, gs)
    mn, mx = G.min(-1, keepdim=True)[0], G.max(-1, keepdim=True)[0]
    s = ((mx - mn) / 3).clamp(min=1e-8).half().float(); z = torch.round(-mn / s)
    return ((torch.clamp(torch.round(G / s) + z, 0, 3) - z) * s).reshape(o, i)


# ---------------- study 7: activation quantization (simulated) ----------------
def act_stats(vla, inf, CUR, MODE, E, calib):
    """per LLM linear: max |x_j| per input channel (SmoothQuant) and per-token max|x| (spike ratio r = max/median)"""
    lin = llm_linears(vla); cmax, ratio, hooks = {}, {}, []
    def h(mod, a, n=None):
        x = a[0].detach().float()[0]                                      # (T, in)
        cm = x.abs().amax(0); cmax[n] = cm if n not in cmax else torch.maximum(cmax[n], cm)
        tm = x.abs().amax(1); ratio.setdefault(n, []).append(float(tm.max() / tm.median().clamp(min=1e-6)))
    for n, m in lin:
        hooks.append(m.register_forward_pre_hook(lambda mod, a, n=n: h(mod, a, n)))
    for k, m in calib:
        set_sample(k, m, CUR, MODE, E)
        inf.run_omnivla()
    for hk in hooks:
        hk.remove()
    return cmax, {n: float(np.max(v)) for n, v in ratio.items()}

def _hadamard(n, dev, seed):
    """randomized orthogonal rotation: Sylvester-Hadamard x random signs for powers of two, else random orthogonal (QR)"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if n & (n - 1) == 0:
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = H / n ** 0.5 * torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)[None, :]
    else:
        H, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return H.to(dev)

def qa8(x):                                                                # per-token dynamic symmetric int8 (fake)
    s = (x.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-8)
    return torch.clamp(torch.round(x / s), -127, 127) * s

class A8Linear(torch.nn.Module):
    """y = Q_A(x T) @ Q_W(W T)^T  with T = diag(1/s) (smoothing) or an orthogonal rotation; Q_A optional"""
    def __init__(self, base, pre=None, a8=True):
        super().__init__()
        self.base, self.pre, self.a8 = base, pre, a8                      # base.weight holds Q_W(W T) (quantized in place)
    def forward(self, x):
        with torch.autocast("cuda", enabled=False):            # fp32: under fp16 autocast the 1e-8 scale floor
            xt = x.float()                                     # underflows -> 0/0 = NaN
            if self.pre is not None:
                xt = self.pre(xt)
            if self.a8:
                xt = qa8(xt)
            return (xt.half() @ self.base.weight.T).to(x.dtype)

def a8_model(vla, sub, cmax, ratio, DT):
    """install A8Linear wrappers per the sub-config; returns the list of (parent, attr, base) to undo"""
    kind, wb = sub["kind"], sub.get("wbits", 4)
    excl = set()
    if sub.get("excl", "").startswith("top"):                               # QFeM-style: the top fraction of modules by spike ratio keep A16
        fr = float(sub["excl"][3:]); rs = sorted(ratio.values()); th = rs[int(len(rs) * (1 - fr))]
        excl = {n for n, r in ratio.items() if r >= th}
    elif sub.get("excl") == "L0-3":
        excl = {n for n, _ in llm_linears(vla) if int(n.split(".")[3]) < 4}
    elif sub.get("excl") == "L1":
        excl = {n for n, _ in llm_linears(vla) if int(n.split(".")[3]) == 1}
    elif sub.get("excl") == "mlpL0-3":
        excl = {n for n, _ in llm_linears(vla) if int(n.split(".")[3]) < 4 and ".mlp." in n}
    elif sub.get("excl", "").startswith("r>"):                              # QFeM-style: spike modules keep A16
        th = float(sub["excl"][2:]); excl = {n for n, r in ratio.items() if r > th}
    undo, rots = [], {}
    wmax_in = {n: m.weight.data.float().abs().amax(0) for n, m in llm_linears(vla)}   # before anything is replaced
    for n, m in llm_linears(vla):
        W = m.weight.data.float(); dev = W.device; pre = None
        short = n.rsplit(".", 1)[1]
        if kind in ("smooth", "qoq") and short in (("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj") if kind == "smooth" else ("o_proj", "down_proj")):
            alpha = sub.get("alpha", 0.85)
            grp = [x for x in (("q_proj", "k_proj", "v_proj") if short in ("q_proj", "k_proj", "v_proj") else ("gate_proj", "up_proj") if short in ("gate_proj", "up_proj") else (short,))]
            wmax = torch.stack([wmax_in[n.rsplit(".", 1)[0] + "." + g] for g in grp]).amax(0)
            s = (cmax[n].to(dev).clamp(min=1e-5) ** alpha / wmax.clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)
            W = W * s[None, :]; pre = (lambda s: (lambda x: x / s))(s)
        if kind in ("rot", "qoq") and (kind == "rot" or short in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            key = (W.shape[1], str(dev))
            if key not in rots:
                rots[key] = _hadamard(W.shape[1], dev, 0)
            H = rots[key]; W = W @ H; pre = (lambda H: (lambda x: x @ H))(H)
        if n in excl and sub.get("rot_excl_off"):                              # excluded modules: no transform at all
            W = m.weight.data.float(); pre = None
        if wb == 4:
            Wq = q_pc4(W, (W.abs().amax(1, keepdim=True) * 2 / 15).clamp(min=1e-8).half().float())
        elif wb == 8:
            sw = (W.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8); Wq = torch.clamp(torch.round(W / sw), -127, 127) * sw
        else:
            Wq = W
        m.weight.data = Wq.to(m.weight.dtype)
        w = A8Linear(m, pre, a8=(n not in excl) and kind != "none")
        parent = vla.get_submodule(n.rsplit(".", 1)[0]); setattr(parent, n.rsplit(".", 1)[1], w); undo.append((parent, n.rsplit(".", 1)[1], m))
    return undo, len(excl)

# ---------------- evaluation ----------------
def run(cfg):
    name = cfg["name"]
    if f"meta__{name}" in res:
        print(f"[C] {name}: done, skipping", flush=True); return
    t0 = time.time()
    vla, DT, n_q = build(cfg)
    vla.vision_backbone.set_num_images_in_input(2)
    VOCAB = vla.language_model.config.vocab_size
    class _NoLMHead(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((), device=x.device, dtype=torch.float32).expand(*x.shape[:-1], VOCAB)
    vla.language_model.lm_head = _NoLMHead()
    gc.collect(); torch.cuda.empty_cache()
    wb = weight_bytes()
    ns = {"vla": vla, "torch": torch}; exec(ELIDE_SRC, ns); E = ns["E"]
    apply_prune(vla, ns)
    dev = torch.device("cuda:0")
    rc = R.InferenceConfig(); rc.vla_path = MODEL_DIR
    pp = R.init_module(R.ProprioProjector, "pose_projector", rc, dev, {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM})
    ah = R.init_module(R.L1RegressionActionHead_idcat, "action_head", rc, dev,
                       {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM}, to_bf16=True).to(DT)
    NP = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + 1
    for k, v in dict(vla=vla, action_head=ah, pose_projector=pp, device_id=dev, NUM_PATCHES=NP,
                     action_tokenizer=action_tokenizer, processor=processor).items():
        setattr(R, k, v)
    MODE, CAP = {"id": 4}, {}
    global CAP_REF
    CAP_REF = CAP
    def fwd(vla, action_head, noisy_action_projector, pose_projector, batch, action_tokenizer, device_id,
            use_l1_regression, use_diffusion, use_film, num_patches, **kw):
        modality_id = torch.as_tensor([MODE["id"]], dtype=torch.float32)
        torch.cuda.synchronize(); t = time.time()
        grad = TRAIN.get("on", False)
        with (torch.enable_grad() if grad else torch.no_grad()), torch.autocast("cuda", dtype=DT):
            out = vla(input_ids=batch["input_ids"].to(device_id), attention_mask=batch["attention_mask"].to(device_id),
                      pixel_values=batch["pixel_values"].to(DT).to(device_id), modality_id=modality_id.to(DT).to(device_id),
                      labels=batch["labels"].to(device_id), output_hidden_states=True,
                      proprio=batch["goal_pose"].to(DT).to(device_id), proprio_projector=pose_projector,
                      noisy_actions=None, noisy_action_projector=None, diffusion_timestep_embeddings=None, use_film=use_film)
        gt = batch["labels"][:, 1:].to(device_id)
        mask = get_current_action_mask(gt) | get_next_actions_mask(gt)
        h = out.hidden_states[-1].to(device_id)[:, num_patches:-1]
        ahs = h[mask].reshape(batch["input_ids"].shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(DT)
        with (torch.enable_grad() if grad else torch.no_grad()), torch.autocast("cuda", dtype=DT):
            pred = action_head.predict_action(ahs, modality_id.to(DT).to(device_id))
        if grad and TRAIN.get("imp"):                          # exq: keep the graph for per-step Fisher importances
            CAP["pred_graph"] = pred
        elif grad:                                             # L1 to the bf16 teacher's actions (study 3); per-step weights (study 6)
            w = TRAIN.get("w")
            err = (pred.float() - TRAIN["target"].to(pred.device)).abs()
            loss = err.mean() if w is None else (err * w.to(pred.device)[None, :, None]).mean()
            TRAIN["scaler"].scale(loss).backward()
            CAP["loss"] = float(loss.detach())
        torch.cuda.synchronize(); CAP["t"] = time.time() - t
        CAP["act"] = pred.detach().float().cpu().numpy()
        return pred.detach(), modality_id
    inf = R.Inference(save_dir="./inference", lan_inst_prompt="move toward blue trash bin",
                      goal_utm=utm.from_latlon(37.8738930785863, -122.26746181032362), goal_compass=0.0,
                      goal_image_PIL=Image.open("./inference/goal_img.jpg").convert("RGB"),
                      action_tokenizer=action_tokenizer, processor=processor)
    inf.run_forward_pass = fwd
    inf.save_robot_behavior = lambda *a, **k: None
    CUR = {"img": None, "goal": None, "gimg": None}
    _dt = inf.data_transformer_omnivla
    def _wrap(cur, lan, gimg, gpose, *a, **k):
        return _dt(CUR["img"], lan, CUR["gimg"] if CUR["gimg"] is not None else gimg,
                   CUR["goal"] if CUR["goal"] is not None else gpose, *a, **k)
    inf.data_transformer_omnivla = _wrap
    if cfg.get("wanda"):
        E["on"] = True; E["skip_img"] = False
        global WANDA_E
        WANDA_E = E
        wanda_2_4(vla, inf, CUR, MODE)
        n_q = quantize_hqq(vla, cfg, DT)
        gc.collect(); torch.cuda.empty_cache()
        wb = weight_bytes()
    if cfg.get("lora"):
        losses = train_lora(inf, CUR, MODE, E, dict(cfg["lora"], name=name))
        res[f"loss__{name}"] = np.array(losses, dtype=np.float32)
    if cfg.get("teacher"):                                       # bf16 labels for held-out samples (study 3)
        for k in sorted(HO):
            for m in (4, 6):
                R.satellite = R.lan_prompt = R.pose_goal = R.image_goal = False
                for k2, v2 in MODES[m].items():
                    setattr(R, k2, v2)
                MODE["id"] = m; E["on"] = True; E["skip_img"] = m != 6
                CUR["img"] = Image.open(HO[k]["img"]).convert("RGB"); CUR["goal"] = HO[k]["goal"] if m == 4 else None
                CUR["gimg"] = Image.open(HO[k]["gimg"]).convert("RGB") if m == 6 else None
                inf.run_omnivla()
                res[f"teach__m{m}__{k}"] = CAP["act"][0]
        print(f"[C] teacher labels: {sum(1 for x in res if x.startswith('teach__'))}", flush=True)
    lat = []
    def eval_tests(nm):
        BLACK = Image.new("RGB", (224, 224), (0, 0, 0))
        for tname, T in TESTS.items():
            m = T["mode"]
            R.satellite = R.lan_prompt = R.pose_goal = R.image_goal = False
            for k2, v2 in MODES[m].items():
                setattr(R, k2, v2)
            MODE["id"] = m; E["on"] = True; E["skip_img"] = m != 6      # mode 6 keeps the goal image (elision drops only the pose token)
            SEL["set"] = {4: 0, 6: 1}.get(m, 0) if cfg.get("lora", {}).get("per_modality") else 0
            for variant in cfg.get("blind", [None]):
                if variant is not None and (variant == "lang_shuffled" and "lang" not in T or
                                            os.environ.get("BLIND_TESTS") and tname not in os.environ["BLIND_TESTS"].split(",")):
                    continue
                for i, f in enumerate(T["frames"][: 3 if SMOKE else None]):
                    src = f if variant != "shuffled" else (T["shuf"][f] if "shuf" in T else smap[f])
                    CUR["img"] = BLACK if variant == "blank" else (T["img"](src) if "img" in T else Image.open(f"./frames/{src}.jpg").convert("RGB"))
                    CUR["goal"] = T["goal"](f) if T["goal"] else None
                    CUR["gimg"] = Image.open(T["gimg"](f)).convert("RGB") if T["gimg"] else (T["gimg_arr"](f) if "gimg_arr" in T else None)
                    if "lang" in T:                                           # instruction (another sample's for lang_shuffled)
                        if variant == "lang_shuffled" and "lang_alt" in T:
                            inf.lan_inst_prompt = T["lang_alt"](f)            # same image, the other object's prompt
                        else:
                            inf.lan_inst_prompt = T["lang"](T["lshuf"][f] if variant == "lang_shuffled" else f)
                    if i == 0:
                        inf.run_omnivla()                                     # warm-up (shape change)
                    inf.run_omnivla()
                    tag = nm if variant is None else f"{nm}-{variant}"
                    res[f"{tag}__{tname}__{f}"] = CAP["act"]
                    if variant is None:
                        lat.append(CAP["t"])

    if cfg.get("exq"):                                           # study 6: calibration methods x weightings x formats
        keys = sorted(HO)[: int(os.environ.get("N_CALIB", "64"))]
        calib = [(k, m) for k in keys for m in (4, 6)]
        snap = {n: m.weight.data.cpu().clone() for n, m in llm_linears(vla)}
        t1 = time.time()
        S, B, B0 = importance(vla, inf, CUR, MODE, E, calib)
        print(f"[X] importance done: {len(calib)} samples in {time.time() - t1:.0f} s", flush=True)
        for li in range(32):                                     # diagnostic: mean per-step importance per layer and mode
            for m_ in (4, 6):
                v = [S[f"language_model.model.layers.{li}.mlp.down_proj"][j].float().mean(1).numpy() for j, (_, mm) in enumerate(calib) if mm == m_]
                res[f"imp__L{li}__m{m_}"] = np.mean(v, 0)
        inps0 = None
        for sub in cfg["exq"]:
            sn = f"{sub['method']}_{sub['kind']}_{sub['fmt']}"
            if f"meta__{sn}" in res:
                continue
            t1 = time.time()
            if sub["method"] == "gptq":
                if inps0 is None:
                    inps0 = capture_layer0(vla, inf, CUR, MODE, E, calib)
                gptq_model(vla, inps0, S, calib, sub["kind"], sub["fmt"], DT)
            elif sub["method"] == "actq":
                actq_model(vla, B, B0, sub["kind"], sub["fmt"])
            elif sub["method"] == "rtn":
                rtn_model(vla, sub["fmt"])
            elif sub["method"] == "hqq":
                hqq_model(vla, DT)
            tq = time.time() - t1
            lat.clear(); eval_tests(sn)
            res[f"meta__{sn}"] = np.array(json.dumps(dict(name=sn, sub=sub, quant_s=tq, t4_latency_s=float(np.mean(lat)),
                                                         weight_bytes_gpu=wb, prune=PRUNE, n_calib=len(calib))))
            np.savez(OUT, **res)
            print(f"[X] {sn}: quant {tq:.0f} s | T4 {np.mean(lat) * 1000:.0f} ms/inf", flush=True)
            with torch.no_grad():
                for n, m in llm_linears(vla):
                    m.weight.data = snap[n].to(m.weight.device)
        frames_loop = ()
    elif cfg.get("gptqx"):                                        # GPTQ per-channel int4 export for deployment (Marlin format)
        import hashlib, marlin
        keys = sorted(HO)[: int(os.environ.get("N_CALIB", "64"))]
        calib = [(k, m) for k in keys for m in (4, 6)]
        snap = {n: m.weight.data.cpu().clone() for n, m in llm_linears(vla)}
        t1 = time.time()
        inps0 = capture_layer0(vla, inf, CUR, MODE, E, calib)
        gptq_model(vla, inps0, None, calib, "plain", "pc4", DT)
        print(f"[G] GPTQ done in {time.time() - t1:.0f} s", flush=True)
        out, sums, chk = {}, {}, {}
        CHECKL = ["language_model.model.layers.0.self_attn.q_proj", "language_model.model.layers.15.mlp.up_proj",
                  "language_model.model.layers.31.mlp.down_proj"]
        for n, m in llm_linears(vla):
            Wo = snap[n].float()
            sc = (Wo.abs().amax(1, keepdim=True) * 2 / 15).clamp(min=1e-8).half()          # the scale GPTQ used (static)
            Wq = m.weight.data.detach().cpu().half()                                        # fp16 dequantized GPTQ weights
            q = torch.round(Wq.float() / sc.float()) + 8
            assert q.min() >= 0 and q.max() <= 15, n
            ref = ((q - 8) * sc.float()).half()
            assert torch.equal(ref, Wq), f"GPTQ weights not on the Marlin grid: {n}"
            o, i = Wq.shape
            fq = torch.nn.Linear(i, o, bias=False, dtype=torch.float16); fq.weight.data = ref
            L = marlin.Layer(i, o, groupsize=-1); L.pack(fq, sc)
            out[n] = {"marlin_B": L.B.clone(), "marlin_s": L.s.clone(), "k": i, "n": o}
            sums[n] = hashlib.sha256(ref.T.contiguous().numpy().tobytes()).hexdigest()   # W^T fp16, as Marlin(I) returns it
            if n in CHECKL:
                chk[n] = ref
        torch.save(out, "/kaggle/working/prequant_marpcg.pt"); torch.save(chk, "/kaggle/working/prequant_marpcg_check.pt")
        man = {"deq_sha256_WT_fp16": sums, "files": {f: hashlib.sha256(open(f"/kaggle/working/{f}", "rb").read()).hexdigest()
               for f in ("prequant_marpcg.pt", "prequant_marpcg_check.pt")}, "n_calib": len(calib), "prune": PRUNE,
               "calib_keys_first": keys[:3]}
        json.dump(man, open("/kaggle/working/prequant_marpcg_manifest.json", "w"), indent=1)
        print(f"[G] exported {len(out)} Marlin layers | files {man['files']}", flush=True)
        del out
        if os.environ.get("EVAL_PRUNES"):                     # same weights, several pruning levels (ablation)
            if any(x.startswith("prompt") for x in os.environ["EVAL_PRUNES"].split(",")):
                LMm = json.loads(str(np.load(glob.glob("/kaggle/input/**/lelan_lang.npz", recursive=True)[0])["meta"]))
                install_prompt_scoring(vla, inf, [LMm[k]["target"] for k in LMm] + [LMm[k]["distractor"] for k in LMm])
            for spec in os.environ["EVAL_PRUNES"].split(","):
                apply_prune(vla, ns, spec); tag = f"gptqx_pc4_{spec}"
                lat.clear(); eval_tests(tag)
                res[f"meta__{tag}"] = np.array(json.dumps(dict(name=tag, t4_latency_s=float(np.mean(lat)), prune=spec, calib_prune=PRUNE)))
                np.savez(OUT, **res)
        else:
            lat.clear(); eval_tests("gptqx_pc4")
            res["meta__gptqx_pc4"] = np.array(json.dumps(dict(name="gptqx_pc4", t4_latency_s=float(np.mean(lat)), prune=PRUNE)))
            np.savez(OUT, **res)
        frames_loop = ()
    elif cfg.get("a8"):                                           # study 7: simulated activation quantization
        keys = sorted(HO)[: int(os.environ.get("N_CALIB", "64"))]
        calib = [(k, m) for k in keys for m in (4, 6)]
        snap = {n: m.weight.data.cpu().clone() for n, m in llm_linears(vla)}
        cmax, ratio = act_stats(vla, inf, CUR, MODE, E, calib)
        res["a8ratio__names"] = np.array(list(ratio)); res["a8ratio__vals"] = np.array(list(ratio.values()))
        top = sorted(ratio.items(), key=lambda x: -x[1])[:12]
        print("[A] top spike ratios (max/median token |x|): " + ", ".join(f"{n.replace('language_model.model.layers.', 'L')} {r:.0f}" for n, r in top), flush=True)
        for sub in cfg["a8"]:
            sn = f"a8_{sub['kind']}_w{sub.get('wbits', 4)}" + (f"_excl{sub['excl']}" if sub.get("excl") else "") + (f"_a{sub['alpha']}" if "alpha" in sub else "") + ("_notrans" if sub.get("rot_excl_off") else "")
            if f"meta__{sn}" in res:
                continue
            undo, n_ex = a8_model(vla, sub, cmax, ratio, DT)
            if os.environ.get("A8_TRACE"):                              # report the first layer with non-finite values
                tr, th = {}, []
                for li, Lr in enumerate(vla.language_model.model.layers):
                    def _hk(mod, a, o, li=li):                          # must return None (a return value replaces the output)
                        tr.setdefault(li, (float((o[0] if isinstance(o, tuple) else o).float().abs().nan_to_num(nan=float('inf')).max()),))
                    th.append(Lr.register_forward_hook(_hk))
                set_sample(sorted(HO)[0], 6, CUR, MODE, E); inf.run_omnivla()
                for h_ in th:
                    h_.remove()
                bad = [li for li in sorted(tr) if not np.isfinite(tr[li][0])]
                print(f"[A] trace {sn}: max|h| per layer " + " ".join(f"{li}:{tr[li][0]:.0f}" for li in sorted(tr)) + f" | first non-finite layer {bad[0] if bad else None}", flush=True)
            lat.clear(); eval_tests(sn)
            res[f"meta__{sn}"] = np.array(json.dumps(dict(name=sn, sub=sub, n_excluded=n_ex, t4_latency_s=float(np.mean(lat)),
                                                         weight_bytes_gpu=wb, prune=PRUNE)))
            np.savez(OUT, **res)
            print(f"[A] {sn}: excluded {n_ex} modules | T4 {np.mean(lat) * 1000:.0f} ms/inf", flush=True)
            with torch.no_grad():
                for parent, attr, m in undo:
                    setattr(parent, attr, m)
                for n, m in llm_linears(vla):
                    m.weight.data = snap[n].to(m.weight.device)
            torch.cuda.empty_cache()
        frames_loop = ()
    elif STUDY in ("tests", "sens", "wanda", "lora") or cfg.get("tests"):
        eval_tests(name)
        frames_loop = ()
    elif cfg.get("teacher"):
        frames_loop = ()
    else:
        frames_loop = ((4, POSE_FRAMES, True), (7, VER_FRAMES, False), (8, VER_FRAMES, False))
    for m, frames, use_goal in frames_loop:
        R.satellite = R.lan_prompt = R.pose_goal = R.image_goal = False
        for k2, v2 in MODES[m].items():
            setattr(R, k2, v2)
        MODE["id"] = m; E["on"] = True; E["skip_img"] = True
        CUR["img"] = Image.open(frames[0][1]).convert("RGB"); CUR["goal"] = GOALS.get(frames[0][0]) if use_goal else None
        inf.run_omnivla()                                                     # warm-up
        for key, path in frames:
            CUR["img"] = Image.open(path).convert("RGB")
            CUR["goal"] = GOALS[key] if use_goal else None
            inf.run_omnivla()
            res[f"{name}__m{m}__{key}"] = CAP["act"]
            if m == 4:
                lat.append(CAP["t"])
    meta = dict(name=name, cfg=cfg, n_hqq_layers=n_q, weight_bytes_gpu=wb, t4_latency_s=float(np.mean(lat)),
                load_quant_s=None, total_s=time.time() - t0)
    res[f"meta__{name}"] = np.array(json.dumps(meta))
    np.savez(OUT, **res)
    print(f"[C] {name}: weights {wb/2**30:.2f} GiB | T4 {np.mean(lat)*1000:.0f} ms/inf | hqq layers {n_q} | {time.time()-t0:.0f} s", flush=True)
    del vla, ah, pp, inf
    for k in ("vla", "action_head", "pose_projector"):
        setattr(R, k, None)
    gc.collect(); torch.cuda.empty_cache()

G2 = {"2": 32}                     # 2-bit uses group 32 (HQQ's recommended range); 8/4/3-bit use group 64
if STUDY == "limit":
    CONFIGS = [dict(name="bf16", base="bf16"), dict(name="fp16", base="fp16"),
               dict(name="nf4_all", base="bnb_nf4_all"), dict(name="nf4llm_visfp16", base="bnb_nf4_llm")]
    for b in (8, 4, 3, 2):
        CONFIGS += [dict(name=f"hqq{b}_vis4", base="fp16", llm=b, vis=4, group=G2),
                    dict(name=f"hqq{b}_visfp16", base="fp16", llm=b, vis=None, group=G2)]
elif STUDY == "tests":
    CONFIGS = [dict(name="bf16", base="bf16"), dict(name="fp16", base="fp16", blind=[None, "blank", "shuffled"]),
               dict(name="nf4_all", base="bnb_nf4_all"), dict(name="nf4llm_visfp16", base="bnb_nf4_llm")]
    for b in (8, 4, 3, 2):
        CONFIGS += [dict(name=f"hqq{b}_vis4", base="fp16", llm=b, vis=4, group=G2),
                    dict(name=f"hqq{b}_visfp16", base="fp16", llm=b, vis=None, group=G2)]
elif STUDY == "teacher":
    CONFIGS = [dict(name="teacher_bf16", base="bf16", teacher=True)]
    TESTS = {}
elif STUDY == "lora":
    B_ = int(os.environ.get("BITS", "3")); V_ = os.environ.get("VIS", "4")
    V_ = None if V_ == "fp16" else int(V_)
    LW_, SEED_ = os.environ.get("LOSS_W", "uniform"), int(os.environ.get("SEED", "0"))
    SUF_ = (f"_{LW_}_s{SEED_}" if (LW_ != "uniform" or SEED_ or PRUNE) else "") + (f"_{PRUNE}" if PRUNE else "")
    CONFIGS = [dict(name=f"hqq{B_}_vis{V_ or 'fp16'}_lora_{m}{SUF_}", base="fp16", llm=B_, vis=V_, group=G2,
                    lora=dict(per_modality=(m == "permod"), epochs=int(os.environ.get("EPOCHS", "2")), lr=2e-4,
                              loss_w=LW_, seed=SEED_))
               for m in os.environ.get("LORA_MODES", "shared,permod").split(",")]
elif STUDY == "exq":                                  # study 6: LLM fp16 fake-quant in place, vision HQQ4
    SUBS = [dict(method="none", kind="plain", fmt="fp16")]
    for fmt, base in (("pc4", "rtn"), ("g2", "hqq")):
        SUBS += [dict(method=base, kind="plain", fmt=fmt), dict(method="gptq", kind="plain", fmt=fmt),
                 dict(method="gptq", kind="act", fmt=fmt), dict(method="gptq", kind="exec_deploy", fmt=fmt),
                 dict(method="actq", kind="act", fmt=fmt), dict(method="actq", kind="exec_deploy", fmt=fmt)]
    for fmt in ("pc4", "g2"):
        SUBS += [dict(method="gptq", kind="exec_nf4", fmt=fmt), dict(method="actq", kind="exec_nf4", fmt=fmt)]
    if os.environ.get("SUBS_ONLY"):
        SUBS = [x for x in SUBS if f"{x['method']}_{x['kind']}_{x['fmt']}" in os.environ["SUBS_ONLY"].split(",")]
    CONFIGS = [dict(name="exq", base="fp16", llm=None, vis=4, group=G2, exq=SUBS)]
elif STUDY == "gptqx":                                # GPTQ int4 export (deployment); BLIND=none,blank,... adds controls
    CONFIGS = [dict(name="gptqx", base="fp16", llm=None, vis=4, group=G2, gptqx=True)]
    if os.environ.get("BLIND"):
        CONFIGS[0]["blind"] = [None if b == "none" else b for b in os.environ["BLIND"].split(",")]
elif STUDY == "lelan_p75":                            # ablation: fp16 (no quantization) with PRUNE, e.g. spatial75
    CONFIGS = [dict(name=f"fp16_{PRUNE or 'none'}", base="fp16", tests=True, blind=[None, "blank", "shuffled", "lang_shuffled"])]
elif STUDY in ("lang", "lelan"):                      # language tests: fp16 with blind controls, bf16 reference
    CONFIGS = [dict(name="fp16", base="fp16", tests=True, blind=[None, "blank", "shuffled", "lang_shuffled"]),
               dict(name="bf16", base="bf16", tests=True)]
elif STUDY == "a8":                                   # study 7: W4A8 / W8A8, per-token dynamic int8 activations
    A8 = [dict(kind="none"), dict(kind="plain"), dict(kind="smooth", alpha=0.85), dict(kind="rot"), dict(kind="qoq", alpha=0.5),
          dict(kind="plain", excl="L0-3"), dict(kind="plain", excl="top0.13"), dict(kind="rot", excl="L0-3"),
          dict(kind="qoq", alpha=0.5, excl="top0.13"), dict(kind="plain", wbits=8), dict(kind="smooth", alpha=0.85, wbits=8)]
    if os.environ.get("A8_SET") == "c":                  # rotation / QoQ configs
        A8 = [dict(kind="rot", wbits=16), dict(kind="rot"), dict(kind="rot", excl="L0-3"), dict(kind="rot", excl="L1"),
              dict(kind="qoq", alpha=0.5), dict(kind="qoq", alpha=0.5, excl="L0-3"), dict(kind="rot", wbits=8)]
    if os.environ.get("A8_SET") == "b":                  # controls and per-layer exclusions
        A8 = [dict(kind="plain", wbits=16), dict(kind="rot", wbits=16), dict(kind="rot"), dict(kind="rot", excl="L1"),
              dict(kind="rot", excl="mlpL0-3"), dict(kind="qoq", alpha=0.5, excl="L0-3"), dict(kind="smooth", alpha=0.5),
              dict(kind="smooth", alpha=0.5, excl="L0-3"), dict(kind="rot", excl="L0-3", rot_excl_off=True)]
    CONFIGS = [dict(name="a8", base="fp16", llm=None, vis=4, group=G2, a8=A8)]
elif STUDY == "wanda":
    CONFIGS = [dict(name="fp16", base="fp16"), dict(name="wanda24_fp16", base="fp16", wanda=True),
               dict(name="wanda24_hqq4_vis4", base="fp16", wanda=True, llm=4, vis=4, group=G2),
               dict(name="hqq4_vis4", base="fp16", llm=4, vis=4, group=G2), dict(name="nf4_all", base="bnb_nf4_all")]
elif STUDY == "sens":
    B_ = int(os.environ.get("BITS", "2"))
    L = r"language_model\.model\.layers\."
    CONFIGS = [dict(name=f"sens_llmg{g}_{B_}bit", base="fp16", llm=4, vis=4, group=G2,
                    override={L + "(" + "|".join(str(4 * g + i) for i in range(4)) + r")\.": B_}) for g in range(8)]
    CONFIGS += [dict(name=f"sens_attn_{B_}bit", base="fp16", llm=4, vis=4, group=G2, override={r"language_model.*\.self_attn\.": B_}),
                dict(name=f"sens_mlp_{B_}bit", base="fp16", llm=4, vis=4, group=G2, override={r"language_model.*\.mlp\.": B_}),
                dict(name=f"sens_dino_{B_}bit", base="fp16", llm=4, vis=4, group=G2, override={r"vision_backbone\.featurizer\.": B_}),
                dict(name=f"sens_siglip_{B_}bit", base="fp16", llm=4, vis=4, group=G2, override={r"vision_backbone\.fused_featurizer\.": B_})]
    if os.environ.get("CONFIGS_JSON"):
        CONFIGS = json.loads(os.environ["CONFIGS_JSON"])
else:
    CONFIGS = json.loads(os.environ["CONFIGS_JSON"])
if SMOKE and STUDY in ("limit", "tests"):
    CONFIGS = [c for c in CONFIGS if c["name"] in ("nf4_all", "hqq3_vis4", "hqq2_visfp16", "fp16")]
for c in CONFIGS:
    run(c)
print(f"[C] saved {OUT} ({len(res)} keys)", flush=True)
