# Per-stage latency of one prediction, per mode, and the numbers needed to size three latency options (GPU preprocessing,
# Marlin for the vision linears, a faster attention kernel). Run on the Jetson from the repo root, both phases in one boot:
#   ./deploy/launch.sh eval/jetson/stage_profile.py stages    deployed runtime (CUDA graphs on): wall time per stage
#   ./deploy/launch.sh eval/jetson/stage_profile.py kernels   CUDA graphs off: per-linear times in the vision encoders,
#                                                             attention kernels, Marlin/SDPA microbenchmarks, which SDPA calls the runtime makes
# -> results/stage_profile_{stages,kernels}.json. Stage times use forward hooks with torch.cuda.synchronize(), so their
# sum is slightly above the unhooked end-to-end time, which is measured first without hooks.
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

PHASE = sys.argv[1] if len(sys.argv) > 1 else "stages"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "deploy"); sys.path.insert(0, D)
from omnivla_deploy import OmniVLADeploy

REF = os.path.join(D, "tests", "reference"); R = np.load(os.path.join(REF, "reference.npz"))
frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
IMG = {f: Image.open(os.path.join(REF, "frames", f + ".jpg")).convert("RGB") for f in frames}
GOAL = {f: Image.open(os.path.join(REF, "goals", f + ".jpg")).convert("RGB") for f in frames}
BIG = {f: im.resize((640, 480), Image.BICUBIC) for f, im in IMG.items()}   # rover camera size (JPEG 640x480 decoded)
LANG = {7: "move toward the bench", 8: "stay on the path"}
m = OmniVLADeploy(D, verbose=False, cuda_graphs=(PHASE == "stages"))
m.warmup(modes=(4, 6, 7, 8))
vla = m.vla
sync, now = torch.cuda.synchronize, time.perf_counter


def call(mode, f, big=False, goal=None):
    img = (BIG if big else IMG)[f]
    kw = dict(goal_pose=R[f"goal__{f}"])
    if mode == 6:
        kw = dict(goal_image=goal if goal is not None else GOAL[frames[0]])
    if mode in (7, 8):
        kw["lang"] = LANG[mode]
    return m.predict(img, mode=mode, **kw)


def ms(v):
    return round(float(np.median(v)) * 1000, 2)


out = dict(phase=PHASE, frames=len(frames))
if PHASE == "stages":
    CASES = {"4": dict(mode=4), "6_cached_goal": dict(mode=6), "6_new_goal": dict(mode=6, new_goal=True),
             "7": dict(mode=7), "8": dict(mode=8)}

    def run_case(c, big=False):
        res = []
        for i, f in enumerate(frames):
            g = GOAL[f] if c.get("new_goal") else GOAL[frames[0]]
            t = now(); o = call(c["mode"], f, big, g); t_call = now() - t
            res.append((t_call, o["t_fwd"]))
        return res
    # 1) end to end, no hooks (call = preprocessing + forward + action head + copy to CPU)
    e2e = {}
    for name, c in CASES.items():
        run_case(c)                                            # settle (goal cache, graphs for this shape)
        r = run_case(c) + run_case(c)
        e2e[name] = dict(call_ms=ms([a for a, _ in r]), fwd_ms=ms([b for _, b in r]))
        print(f"[STAGE] {name}: call {e2e[name]['call_ms']} ms | fwd {e2e[name]['fwd_ms']} ms (no hooks)", flush=True)
    out["e2e"] = e2e
    # 2) preprocessing split (CPU): image transform per image, prompt + tokenizer + collator; 224x224 and 640x480 input
    ip = m.processor.image_processor; orig_tf = ip.apply_transform; TT = []

    def tf(img):
        t = now(); r = orig_tf(img); TT.append(now() - t); return r
    ip.apply_transform = tf
    pre = {}
    for big in (False, True):
        for mode in (4, 6, 7):
            tot = []
            TT.clear()
            for f in frames * 2:
                img = (BIG if big else IMG)[f]
                gimg = GOAL[f] if mode == 6 else img
                t = now()
                m._inf.data_transformer_omnivla(img, LANG.get(mode, "xxxx"), gimg, R[f"goal__{f}"], prompt_builder=m.R.PurePromptBuilder,
                                                action_tokenizer=m.action_tokenizer, processor=m.processor)
                tot.append(now() - t)
            k = f"{'640x480' if big else '224x224'}_m{mode}"
            pre[k] = dict(total_ms=ms(tot), per_image_transform_ms=ms(TT), n_transforms_per_call=len(TT) // len(tot),
                          rest_ms=round(ms(tot) - ms(TT) * (len(TT) // len(tot)), 2))
            print(f"[STAGE] preprocessing {k}: {pre[k]}", flush=True)
    ip.apply_transform = orig_tf
    out["preprocessing"] = pre
    # 3) hooked stages
    T = {}
    def hook(name, mod):
        st = {}
        def a(*_a, **_k):
            sync(); st["t"] = now()
        def b(*_a, **_k):
            sync(); T.setdefault(name, []).append(now() - st["t"])
        mod.register_forward_pre_hook(a); mod.register_forward_hook(b)
    vb = vla.vision_backbone
    for name, mod in {"vla_forward": vla, "vision_total": vb, "dino": vb.featurizer, "siglip": vb.fused_featurizer,
                      "projector": vla.projector, "llm_wrapper": vla.language_model, "llm_layers": vla.language_model.model}.items():
        hook(name, mod)
    for name, obj, attr in (("preprocess", m._inf, "data_transformer_omnivla"), ("action_head", m.action_head, "predict_action")):
        fn = getattr(obj, attr)
        def w(*a, fn=fn, name=name, **k):
            sync(); t = now(); r = fn(*a, **k); sync(); T.setdefault(name, []).append(now() - t); return r
        setattr(obj, attr, w)
    stages = {}
    for name, c in CASES.items():
        T.clear(); tc = []
        for rep in range(2):
            for f in frames:
                g = GOAL[f] if c.get("new_goal") else GOAL[frames[0]]
                t = now(); o = call(c["mode"], f, False, g)
                t2 = now(); m.actions_to_cmd(o["actions"]); T.setdefault("controller", []).append(now() - t2)
                tc.append(t2 - t)
        n = len(tc)
        s = {k: dict(ms=round(float(np.sum(v)) / n * 1000, 2), calls_per_pred=len(v) // n) for k, v in T.items()}
        g = lambda k: s.get(k, {"ms": 0.0})["ms"]
        s["glue_in_vla_forward"] = dict(ms=round(g("vla_forward") - g("vision_total") - g("projector") - g("llm_wrapper"), 2),
                                        note="embedding, token selection masks, HF mask setup, concatenations")
        s["llm_wrapper_minus_layers"] = dict(ms=round(g("llm_wrapper") - g("llm_layers"), 2))
        s["vision_glue"] = dict(ms=round(g("vision_total") - g("dino") - g("siglip"), 2))
        s["predict_rest"] = dict(ms=round(ms(tc) - g("preprocess") - g("vla_forward") - g("action_head"), 2),
                                 note="host->device copies, action mask/reshape, copy to CPU, bookkeeping (median-based)")
        s["call_hooked_ms"] = ms(tc)
        stages[name] = s
        print(f"[STAGE] {name}: " + " | ".join(f"{k} {v['ms'] if isinstance(v, dict) else v}" for k, v in s.items()), flush=True)
    out["stages"] = stages
else:
    # ---- kernels phase (no CUDA graphs) ----
    import marlin
    vb = vla.vision_backbone
    # 1) per-linear time in the vision encoders (CUDA events around each module), pose mode (current image only)
    EV, SH = {}, {}
    def lin_hooks(prefix, enc):
        for nm, mod in enc.named_modules():
            tn = type(mod).__name__
            if tn not in ("GLWrap", "HQQLinear", "Linear") or (nm.endswith(".m") and type(enc.get_submodule(nm[:-2])).__name__ == "GLWrap"):
                continue                                       # a GLWrap's inner module is counted by the wrapper
            key = f"{prefix}.{nm}"
            def a(mod, inp, key=key):
                e = torch.cuda.Event(enable_timing=True); e.record(); EV.setdefault(key, []).append([e, None])
                SH[key] = (tuple(inp[0].shape), type(mod).__name__)
            def b(mod, inp, o, key=key):
                e = torch.cuda.Event(enable_timing=True); e.record(); EV[key][-1][1] = e
            mod.register_forward_pre_hook(a); mod.register_forward_hook(b)
    lin_hooks("dino", vb.featurizer); lin_hooks("siglip", vb.fused_featurizer)
    for f in frames:
        call(4, f)
    EV.clear()
    for f in frames:
        call(4, f)
    sync()
    per = {}
    for k, v in EV.items():
        t = [a.elapsed_time(b) for a, b in v]
        mod = vla.get_submodule("vision_backbone." + ("featurizer" if k.startswith("dino") else "fused_featurizer") + "." + k.split(".", 1)[1])
        inner = mod.m if type(mod).__name__ == "GLWrap" else mod
        if hasattr(inner, "meta"):
            shape = tuple(inner.meta["shape"])
        elif hasattr(inner, "weight") and torch.is_tensor(inner.weight):
            shape = tuple(inner.weight.shape)
        else:                                                  # GemLite: (out, in) from its attributes
            shape = (int(inner.out_features), int(inner.in_features))
        kind = k.rsplit(".", 1)[1]
        per[k] = dict(ms=float(np.median(t)) * len(t) / len(frames), calls=len(t) // len(frames), in_shape=SH[k][0],
                      out_in=shape, backend=type(mod).__name__ if type(mod).__name__ != "GLWrap" else "gemlite")
    groups = {}
    for k, v in per.items():
        g = f"{k.split('.')[0]}.{k.rsplit('.', 1)[1]}|{v['backend']}|{v['out_in']}|T={v['in_shape'][-2] if len(v['in_shape']) > 2 else v['in_shape'][0]}"
        G = groups.setdefault(g, dict(n=0, ms=0.0)); G["n"] += 1; G["ms"] += v["ms"]
    for g, v in sorted(groups.items()):
        print(f"[KERN] vision linear {g}: {v['n']} layers, {v['ms']:.2f} ms per prediction", flush=True)
    out["vision_linears"] = {g: dict(n=v["n"], ms=round(v["ms"], 3)) for g, v in groups.items()}
    # whole encoders (events, no hooks inside matter little: events are async)
    enc_t = {}
    pix = m._inf.data_transformer_omnivla(IMG[frames[0]], "xxxx", IMG[frames[0]], R[f"goal__{frames[0]}"], prompt_builder=m.R.PurePromptBuilder,
                                          action_tokenizer=m.action_tokenizer, processor=m.processor)["pixel_values"].to(m.dev, torch.float16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for name, enc, sl in (("dino", vb.featurizer, slice(0, 3)), ("siglip", vb.fused_featurizer, slice(3, 6))):
            x = pix[:, sl].contiguous()
            for _ in range(3):
                enc(x)
            ts = []
            for _ in range(20):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); enc(x); b.record(); sync(); ts.append(a.elapsed_time(b))
            enc_t[name] = round(float(np.median(ts)), 2)
    print(f"[KERN] encoder eager time per image (hooks on): {enc_t}", flush=True)
    out["encoder_eager_ms"] = enc_t
    # 2) Marlin per-channel at the vision linear shapes (random int4 weights; zero-padding k to 128 and n to 256)
    mb = {}
    for g in groups:
        enc_kind, backend, oi, Tt = g.split("|"); n_out, k_in = eval(oi); T_ = int(Tt[2:])
        kp, np_ = -(-k_in // 128) * 128, -(-n_out // 256) * 256
        L = marlin.Layer(kp, np_, groupsize=-1).to(m.dev)
        with torch.no_grad():                                  # kernel time does not depend on the weight values
            L.B.random_(0, 2 ** 31 - 1); L.s.fill_(0.001)
        x = torch.randn(T_, kp, dtype=torch.half, device=m.dev)
        ts = []
        for i in range(60):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); L(x); b.record(); sync()
            if i >= 10:
                ts.append(a.elapsed_time(b))
        # the current layer, same token count, standalone (for an apples-to-apples ratio)
        cur = next(vla.get_submodule("vision_backbone." + ("featurizer" if enc_kind.startswith("dino") else "fused_featurizer") + "." + k.split(".", 1)[1])
                   for k, v in per.items() if f"{k.split('.')[0]}.{k.rsplit('.', 1)[1]}" == enc_kind and v["out_in"] == (n_out, k_in))
        xc = torch.randn(1, T_, k_in, dtype=torch.half, device=m.dev); tc = []
        with torch.no_grad():
            for i in range(60):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); cur(xc); b.record(); sync()
                if i >= 10:
                    tc.append(a.elapsed_time(b))
        mb[g] = dict(marlin_ms=round(float(np.median(ts)), 4), current_ms=round(float(np.median(tc)), 4), padded=(np_, kp),
                     n_layers=groups[g]["n"])
        print(f"[KERN] marlin bench {g}: marlin {mb[g]['marlin_ms']:.3f} ms vs current {mb[g]['current_ms']:.3f} ms per layer "
              f"(padded to {np_}x{kp})", flush=True)
        del L
    sav = sum((v["current_ms"] - v["marlin_ms"]) * v["n_layers"] for v in mb.values())
    print(f"[KERN] Marlin for all vision linears: estimated saving {sav:.1f} ms per encoded image set (pose mode)", flush=True)
    out["marlin_bench"] = {g: v for g, v in mb.items()}; out["marlin_saving_ms_pose"] = round(sav, 2)
    # 3) attention: what the LLM passes to SDPA, and SDPA backends at the real shapes
    seen = {}
    import torch.nn.functional as F
    orig_sdpa = F.scaled_dot_product_attention
    def spy(q, k, v, attn_mask=None, **kw):
        seen.setdefault("calls", []).append(dict(q=tuple(q.shape), k=tuple(k.shape), mask=None if attn_mask is None else tuple(attn_mask.shape),
                                                 mask_all_zero=None if attn_mask is None else bool((attn_mask == 0).all()),
                                                 is_causal=kw.get("is_causal")))
        return orig_sdpa(q, k, v, attn_mask=attn_mask, **kw)
    F.scaled_dot_product_attention = spy
    for mode in (4, 6, 7):
        seen.clear(); call(mode, frames[0])
        c = seen.get("calls", [])
        uniq = sorted({json.dumps(x) for x in c})
        print(f"[KERN] SDPA calls mode {mode}: {len(c)} calls, distinct: {uniq}", flush=True)
        out[f"sdpa_calls_m{mode}"] = dict(n=len(c), distinct=[json.loads(u) for u in uniq])
    F.scaled_dot_product_attention = orig_sdpa
    from torch.nn.attention import sdpa_kernel, SDPBackend
    shapes = {}                                                # every distinct SDPA call seen in modes 4/6/7
    for mode in (4, 6, 7):
        for c in out[f"sdpa_calls_m{mode}"]["distinct"]:
            shapes.setdefault(f"H{c['q'][1]}_T{c['q'][2]}_D{c['q'][3]}", (c["q"], c["mask"] is not None, []))[2].append(mode)
    att = {}
    for nm, (qs, runtime_mask, modes_) in shapes.items():
        _, H, Tq, Dh = qs
        q, k, v = (torch.randn(1, H, Tq, Dh, dtype=torch.half, device=m.dev) for _ in range(3))
        zero = torch.zeros(1, 1, Tq, Tq, dtype=torch.half, device=m.dev)
        ref = orig_sdpa(q.float(), k.float(), v.float())
        def bench(masked):
            ts = []
            for i in range(40):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); o = orig_sdpa(q, k, v, attn_mask=zero if masked else None); b.record(); sync()
                if i >= 10:
                    ts.append(a.elapsed_time(b))
            return dict(ms=round(float(np.median(ts)), 4), max_err_vs_fp32=float((o.float() - ref).abs().max()))
        res = {"as_runtime": bench(runtime_mask)}              # default dispatcher, mask as the runtime passes it
        for bname, be in (("math", SDPBackend.MATH), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION),
                          ("cudnn", SDPBackend.CUDNN_ATTENTION)):
            for masked in (False, True):
                try:
                    with sdpa_kernel(be):
                        res[f"{bname}{'_mask' if masked else ''}"] = bench(masked)
                except Exception as e:
                    res[f"{bname}{'_mask' if masked else ''}"] = f"unavailable: {str(e).splitlines()[0][:120]}"
        att[nm] = dict(runtime_mask=runtime_mask, modes=modes_, **res)
        print(f"[KERN] SDPA {nm} (runtime mask {runtime_mask}, modes {modes_}): {res}", flush=True)
    out["sdpa_bench"] = att
    out["flash_available"] = bool(torch.backends.cuda.is_flash_attention_available())
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
out["vision_kernel"] = m.vision_kernel
json.dump(out, open(os.path.join(ROOT, "results", f"stage_profile_{PHASE}{os.environ.get('PROFILE_TAG', '')}.json"), "w"), indent=1, default=str)
print(f"[STAGE] DONE phase {PHASE}", flush=True)
