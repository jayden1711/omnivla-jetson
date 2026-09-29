"""OmniVLA-7B runtime for the Jetson Orin Nano 8 GB.

LLM: per-channel int4 on Marlin (groupsize=-1; grouped Marlin is wrong on sm_87). Vision: HQQ 4-bit on GemLite, SigLIP
MLP (width 4304) on HQQ's portable backend. Modality elision + uniform-grid pruning of 75% of the current-image tokens
in pose and image-goal modes (4, 6). Language modes (7, 8) run without pruning (lang_prune_frac=0): 75% pruning cut
object-goal accuracy from 84% to 69% (results/lelan_summary.md), while pose and image goals are unaffected.
Weights: pre-packed 256 MB shards from build/build_model.sh. Image-goal mode caches the goal's vision features (exact);
goal K/V reuse (goal_refresh > 1) is an approximation because OmniVLA's LLM attention is bidirectional.

  m = OmniVLADeploy("/mnt/nvme/omnivla/deploy")
  out = m.predict(pil_image, mode=4, goal_pose=np.array([x, y, cos, sin]))   # out["actions"]: (8, 4) action units
  v, w = OmniVLADeploy.actions_to_cmd(out["actions"])                         # m/s, rad/s (run_omnivla's controller)
Modes: 4 pose, 6 image, 7 language, 8 language + pose. Run through launch.sh.
"""
import ctypes, gc, hashlib, json, math, os, sys, time

import numpy as np
import torch

DEFAULT_REPO = os.environ.get("OMNIVLA_REPO", "/mnt/nvme/omnivla/OmniVLA")
N_IMG, CHUNK = 256, 8
GOAL_REFRESH = 1          # 1 = no goal K/V reuse (reuse costs +0.02 driving error, results/refresh_summary.md)


def _grid_keep(n):
    return np.unique(np.round(np.linspace(0, N_IMG - 1, n)).astype(int))


class OmniVLADeploy:
    def __init__(self, deploy_dir, repo=DEFAULT_REPO, prune_frac=0.75, verbose=True, trim_every=20, goal_refresh=GOAL_REFRESH,
                 vit_trunc=True, lang_prune_frac=0.0, cuda_graphs=True, llm_kv_cache=False):
        t0 = time.time()
        self.dir, self.log = deploy_dir, (print if verbose else (lambda *a, **k: None))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        cwd = os.getcwd(); os.chdir(repo)                      # run_omnivla uses repo-relative defaults
        try:
            import inference.run_omnivla as R
        finally:
            os.chdir(cwd)
        self.R = R
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from hqq.core.quantize import HQQLinear, HQQBackend
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        from hqq.backends.gemlite import patch_hqq_to_gemlite
        import marlin
        from prismatic.vla.constants import ACTION_DIM, POSE_DIM
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
        self._masks = (get_current_action_mask, get_next_actions_mask)
        self.dev = dev = torch.device("cuda:0"); self.DT = torch.float16
        W = os.path.join(deploy_dir, os.environ.get("OMNIVLA_WEIGHTS", "weights")); MD = os.path.join(W, "model")
        self.weights_dir = W

        AutoConfig.register("openvla", R.OpenVLAConfig)
        AutoImageProcessor.register(R.OpenVLAConfig, R.PrismaticImageProcessor)
        AutoProcessor.register(R.OpenVLAConfig, R.PrismaticProcessor)
        AutoModelForVision2Seq.register(R.OpenVLAConfig, R.OpenVLAForActionPrediction_MMNv1)
        self.processor = AutoProcessor.from_pretrained(MD, trust_remote_code=True)
        with init_empty_weights():
            vla = AutoModelForVision2Seq.from_config(AutoConfig.from_pretrained(MD), torch_dtype=torch.float16)
        VOCAB = vla.language_model.config.vocab_size

        class _NoLMHead(torch.nn.Module):                      # logits are never used: no 250 MB lm_head
            def forward(self, x):
                return torch.zeros((), device=x.device, dtype=torch.float32).expand(*x.shape[:-1], VOCAB)
        vla.language_model.lm_head = _NoLMHead()

        # 1) non-quantized tensors (fp16) from base.safetensors
        with safe_open(os.path.join(W, "base.safetensors"), "pt", device="cpu") as f:
            for k in f.keys():
                set_module_tensor_to_device(vla, k, dev, value=f.get_tensor(k))
        gc.collect()

        # 2) pre-packed quantized layers, one 256 MB shard in RAM at a time
        class MarlinPC(torch.nn.Module):
            def __init__(self, st):
                super().__init__()
                self.k, self.n = int(st["k"]), int(st["n"])
                self.L = marlin.Layer(self.k, self.n, groupsize=-1).to(dev)
                with torch.no_grad():
                    self.L.B.copy_(st["marlin_B"]); self.L.s.copy_(st["marlin_s"])
                self.in_features, self.out_features = self.k, self.n

            def forward(self, x):
                return self.L(x.reshape(-1, self.k).half()).reshape(*x.shape[:-1], self.n)

        idx = json.load(open(os.path.join(W, "SHARDS.json")))["marpc"]
        names, libc, built, counts = set(idx["names"]), ctypes.CDLL("libc.so.6"), set(), {}
        for s in idx["shards"]:
            part = torch.load(os.path.join(W, "marpc", s), weights_only=False)
            for nm in sorted(part):
                # .detach(): saved biases are Parameters; a non-detached copy's autograd node keeps the CPU tensor alive
                st = {k: (v.detach().to(dev) if torch.is_tensor(v) else v) for k, v in part.pop(nm).items()}
                if "marlin_B" in st:
                    ql, b = MarlinPC(st), "marlin_pc"
                else:
                    ql = HQQLinear(None, quant_config=None, compute_dtype=torch.float16, device="cuda", initialize=False)
                    ql.load_state_dict(st); b = "hqq4"
                parent = vla.get_submodule(nm.rsplit(".", 1)[0]); setattr(parent, nm.rsplit(".", 1)[1], ql)
                built.add(nm); counts[b] = counts.get(b, 0) + 1
            del part; gc.collect(); torch.cuda.empty_cache(); libc.malloc_trim(0)   # glibc keeps freed buffers otherwise
        assert built == names, f"missing quantized layers: {sorted(names - built)[:3]}"
        st = None; gc.collect(); libc.malloc_trim(0)

        # 3) buffers: rotary inv_freq is created on meta by the empty skeleton -> recompute; others to GPU
        for n, m in vla.named_modules():
            if hasattr(m, "inv_freq") and m.inv_freq.is_meta:
                inv = 1.0 / (m.base ** (torch.arange(0, m.dim, 2, dtype=torch.int64).float() / m.dim))
                m.register_buffer("inv_freq", inv.to(dev), persistent=False)
            for bn, buf in list(m._buffers.items()):
                if buf is not None and not buf.is_meta and buf.device != dev:
                    m._buffers[bn] = buf.to(dev)
        left = [n for n, p in list(vla.named_parameters()) + list(vla.named_buffers()) if p.is_meta]
        assert not left, f"tensors still on meta: {left[:5]}"
        vla.vision_backbone.set_num_images_in_input(2)

        # 4) vision HQQ4 -> GemLite (in/out and group must divide 32; SigLIP MLP 4304 stays on HQQ portable)
        class GLWrap(torch.nn.Module):                         # GemLite is not autocast-aware
            def __init__(self, m):
                super().__init__(); self.m = m

            def forward(self, x):
                return self.m(x.to(torch.float16))
        n_gl = 0
        for nm, m in [(nm, m) for nm, m in vla.named_modules() if type(m).__name__ == "HQQLinear"]:
            out_f, in_f = tuple(m.meta["shape"])
            if in_f % 32 or out_f % 32 or in_f % m.meta["group_size"]:
                continue
            parent = vla.get_submodule(nm.rsplit(".", 1)[0]); setattr(parent, nm.rsplit(".", 1)[1], GLWrap(patch_hqq_to_gemlite(m, None))); n_gl += 1
        gc.collect(); torch.cuda.empty_cache()

        # 5) heads
        rc = R.InferenceConfig(); rc.vla_path = MD
        self.pose_projector = R.init_module(R.ProprioProjector, "pose_projector", rc, dev, {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM}).eval()
        self.action_head = R.init_module(R.L1RegressionActionHead_idcat, "action_head", rc, dev,
                                         {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM}, to_bf16=True).to(self.DT).eval()
        self.num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + 1
        self.action_tokenizer = R.ActionTokenizer(self.processor.tokenizer)
        self.vla = vla.eval()
        self.goal_refresh = max(1, int(goal_refresh))
        # llm_kv_cache=False (default since 2026-09-28): the LLM does not build its key/value cache (use_cache=False). The
        # runtime never reads it (one forward pass per prediction, no generation); outputs are bit-identical on the Jetson.
        self._llm_kw = {} if llm_kv_cache else dict(use_cache=False)
        self._install_token_hooks(prune_frac, lang_prune_frac)
        if vit_trunc:
            self._install_vit_trunc()
        if cuda_graphs and self.goal_refresh > 1:             # goal K/V reuse branches on Python state between calls
            self.log("[DEPLOY] CUDA graphs off: not compatible with goal_refresh > 1", flush=True)
            cuda_graphs = False
        self.cuda_graphs = bool(cuda_graphs)
        if cuda_graphs:                                        # default since 2026-09-28: bit-exact, 3-10% faster on the Jetson
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import cuda_graphs as CG
            CG.install(self.vla, self.log)
        self._inf = R.Inference(save_dir="/tmp", lan_inst_prompt="", goal_utm=(0.0, 0.0), goal_compass=0.0,
                                goal_image_PIL=None, action_tokenizer=self.action_tokenizer, processor=self.processor)
        gc.collect(); torch.cuda.empty_cache(); libc.malloc_trim(0)
        # glibc retains freed per-frame buffers (~+118 MB / 10 min): trim every `trim_every` predictions (0 = never)
        self._libc, self._trim_every, self._n_pred = libc, int(trim_every), 0
        self.weights_gib = torch.cuda.memory_allocated() / 2**30
        self.log(f"[DEPLOY] ready in {time.time()-t0:.0f} s | layers {counts} (GemLite {n_gl}) | "
                 f"torch {self.weights_gib:.2f} GiB | pruning grid {int(prune_frac*100)}% (keep {self.n_keep}), "
                 f"language modes {int(lang_prune_frac*100)}% (keep {self.n_keep_lang})", flush=True)

    # ---- ViT truncation, modality elision, grid token pruning, goal cache ----
    def _install_vit_trunc(self):
        """timm 0.9.10 get_intermediate_layers(n={len-2}) runs every block and returns block len-2: stop there (exact)."""
        for mod in (self.vla.vision_backbone.featurizer, self.vla.vision_backbone.fused_featurizer):
            def fwd(img, mod=mod, take=len(mod.blocks) - 2, npre=mod.num_prefix_tokens):
                x = mod.norm_pre(mod.patch_drop(mod._pos_embed(mod.patch_embed(img))))
                for blk in mod.blocks[: take + 1]:
                    x = blk(x)
                return x[:, npre:]
            mod.forward = fwd

    def _install_token_hooks(self, frac, lang_frac=0.0):
        vla, S = self.vla, {"skip_img": False}
        self._S = S
        GC = self._GC = dict(mode="off", kv={}, goal_idx=None, key=None, vis_key=None, vis_feat=None, n=0, vis_skip=False)
        import transformers.models.llama.modeling_llama as ML

        def attn_forward(att, orig, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, cache_position=None):
            if GC["mode"] == "off":
                return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value,
                            output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position)
            b, n, _ = hidden_states.size()                     # same math as the fork's LlamaSdpaAttention (batch 1, no pads)
            q = att.q_proj(hidden_states).view(b, n, att.num_heads, att.head_dim).transpose(1, 2)
            k = att.k_proj(hidden_states).view(b, n, att.num_key_value_heads, att.head_dim).transpose(1, 2)
            v = att.v_proj(hidden_states).view(b, n, att.num_key_value_heads, att.head_dim).transpose(1, 2)
            cos, sin = att.rotary_emb(v, position_ids)
            q, k = ML.apply_rotary_pos_emb(q, k, cos, sin)
            if GC["mode"] == "store":                          # full pass: keep the goal tokens' post-RoPE keys/values
                GC["kv"][att.layer_idx] = (k[:, :, GC["goal_idx"]].clone(), v[:, :, GC["goal_idx"]].clone())
                mask = attention_mask
                if mask is not None:
                    mask = mask[:, :, :, : k.shape[-2]]
                    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
                    D = mask.shape[-1]; mask = mask[:, :, -1, :].clone().unsqueeze(2).expand(-1, -1, D, -1)
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
            else:                                              # reuse: goal tokens are not recomputed; their cached K/V join
                kg, vg = GC["kv"][att.layer_idx]
                o = torch.nn.functional.scaled_dot_product_attention(q, torch.cat([k, kg], 2), torch.cat([v, vg], 2),
                                                                     attn_mask=None, is_causal=False)
            o = o.transpose(1, 2).contiguous().view(b, n, att.hidden_size)
            return att.o_proj(o), None, past_key_value
        for layer in vla.language_model.model.layers:
            a = layer.self_attn
            a.forward = (lambda orig, att: (lambda *x, **kw: attn_forward(att, orig, *x, **kw)))(a.forward, a)
        def drop_for(f):
            n = N_IMG - int(round(N_IMG * f))
            return n, (torch.as_tensor(np.setdiff1d(np.arange(N_IMG), _grid_keep(n)) + 1, device=self.dev) if f > 0 else None)
        self.n_keep, self._drop = drop_for(frac)                # pose and image goal (4, 6)
        self.n_keep_lang, self._drop_lang = drop_for(lang_frac)  # language modes (7, 8)
        S["drop"] = self._drop
        ob = vla._build_multimodal_attention_MMN

        def build(inp, patches, am, aml, mid):
            emb, mask = ob(inp, patches, am, aml, mid)
            keep = mask[0].bool().clone()                      # elision: masked (unused-modality) tokens dropped
            drop = S["drop"]                                   # set per call by predict (mode-dependent)
            if drop is not None:
                keep[drop] = False                             # grid pruning of current-image tokens (positions 1..256)
            ids = torch.arange(mask.shape[1], device=mask.device)[keep]
            goal = (ids > N_IMG) & (ids <= 2 * N_IMG)           # goal-image token positions 257..512 (mode 6)
            if GC["mode"] == "store":
                GC["goal_idx"] = torch.nonzero(goal).squeeze(1)
            elif GC["mode"] == "reuse":                         # drop the goal tokens: their keys/values come from the cache
                keep = keep.clone(); keep[ids[goal]] = False; ids = ids[~goal]
            S.update(ids=ids.unsqueeze(0), full=mask.shape[1])
            return emb[:, keep], mask[:, keep]
        vla._build_multimodal_attention_MMN = build
        lm = vla.language_model; ol = lm.forward

        def lm_fwd(*a, **kw):
            kw["labels"] = None; kw["position_ids"] = S["ids"]  # original positions -> RoPE unchanged
            out = ol(*a, **kw)
            h = out.hidden_states[-1]
            full = h.new_zeros(h.shape[0], S["full"], h.shape[2]); full[:, S["ids"][0].to(h.device)] = h
            out.hidden_states = tuple(out.hidden_states[:-1]) + (full,)
            return out
        lm.forward = lm_fwd
        vb = vla.vision_backbone; ovb = vb.forward

        def vb_fwd(pixel_values, *a, **kw):                    # no goal image -> encode only the current image
            GC["vis_skip"] = False
            if not S["skip_img"] and GC["key"] is not None:    # image goal: reuse the goal's vision features (exact)
                if GC["vis_key"] == GC["key"]:
                    half = pixel_values.shape[1] // 2
                    vb.set_num_images_in_input(1)
                    try:
                        f = ovb(pixel_values[:, :half], *a, **kw)
                    finally:
                        vb.set_num_images_in_input(2)
                    GC["vis_skip"] = True
                    return torch.cat([f, GC["vis_feat"]], dim=1)
                f = ovb(pixel_values, *a, **kw)
                GC["vis_key"], GC["vis_feat"] = GC["key"], f[:, f.shape[1] // 2:].clone()
                return f
            if S["skip_img"]:
                half = pixel_values.shape[1] // 2
                vb.set_num_images_in_input(1)
                try:
                    f = ovb(pixel_values[:, :half], *a, **kw)
                finally:
                    vb.set_num_images_in_input(2)
                return torch.cat([f, torch.zeros_like(f)], dim=1)
            return ovb(pixel_values, *a, **kw)
        vb.forward = vb_fwd

    # ---- goal helpers ----
    @staticmethod
    def goal_pose_from_gps(cur_lat, cur_lon, cur_compass_deg, goal_lat, goal_lon, goal_compass_rad=0.0,
                           spacing=0.1, thres_dist=30.0):
        """run_omnivla's goal-pose normalization (controller convention: 0.1 m per unit, compass inverted)."""
        import utm
        cu, gu = utm.from_latlon(cur_lat, cur_lon), utm.from_latlon(goal_lat, goal_lon)
        cc = -float(cur_compass_deg) / 180.0 * math.pi
        dx, dy = gu[0] - cu[0], gu[1] - cu[1]
        rx = dx * math.cos(cc) + dy * math.sin(cc); ry = -dx * math.sin(cc) + dy * math.cos(cc)
        r = math.hypot(rx, ry)
        if r > thres_dist:
            rx, ry = rx * thres_dist / r, ry * thres_dist / r
        return np.array([ry / spacing, -rx / spacing, math.cos(goal_compass_rad - cc), math.sin(goal_compass_rad - cc)])

    # run_omnivla's hard-coded sample goal (used when a mode does not need a pose; masked out by the model then)
    _DEFAULT_GOAL = None

    def _default_goal(self):
        if OmniVLADeploy._DEFAULT_GOAL is None:
            OmniVLADeploy._DEFAULT_GOAL = self.goal_pose_from_gps(37.87371258374039, -122.26729417226024, 270.0,
                                                                  37.8738930785863, -122.26746181032362, 0.0)
        return OmniVLADeploy._DEFAULT_GOAL

    # ---- inference ----
    @torch.no_grad()
    def predict(self, image, mode=4, goal_pose=None, goal_image=None, lang=None, goal_id=None):
        """image: PIL RGB (any size; the processor resizes to 224). Returns dict(actions (8,4) float32, t_fwd s)."""
        R = self.R
        assert mode in (4, 6, 7, 8), "supported modes: 4 pose, 6 image, 7 language, 8 language+pose"
        if mode in (4, 8):
            assert goal_pose is not None, "pose modes need goal_pose = [x, y, cos, sin] (see goal_pose_from_gps)"
        if mode == 6:
            assert goal_image is not None, "mode 6 needs goal_image"
        if mode in (7, 8):
            assert lang, "language modes need lang"
        gpose = np.asarray(goal_pose, dtype=np.float64) if goal_pose is not None else self._default_goal()
        gimg = goal_image if goal_image is not None else image  # placeholder, not encoded (elided) unless mode 6
        lan = lang if mode in (7, 8) else "xxxx"
        batch = self._inf.data_transformer_omnivla(image, lan, gimg, gpose, prompt_builder=R.PurePromptBuilder,
                                                   action_tokenizer=self.action_tokenizer, processor=self.processor)
        self._S["skip_img"] = mode != 6
        self._S["drop"] = self._drop_lang if mode in (7, 8) else self._drop
        GC = self._GC
        if mode == 6:                                          # goal cache: refresh on goal change and every goal_refresh calls
            key = goal_id if goal_id is not None else hashlib.md5(np.asarray(goal_image).tobytes()).hexdigest()
            if key != GC["key"] or GC["n"] % self.goal_refresh == 0 or not GC["kv"]:
                GC.update(key=key, n=0, kv={}); GC["mode"] = "store" if self.goal_refresh > 1 else "off"
            else:
                GC["mode"] = "reuse"
            GC["n"] += 1
        else:
            GC["mode"] = "off"
        dev, DT = self.dev, self.DT
        mid = torch.as_tensor([mode], dtype=torch.float32)
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=DT):
            out = self.vla(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
                           pixel_values=batch["pixel_values"].to(DT).to(dev), modality_id=mid.to(DT).to(dev),
                           labels=batch["labels"].to(dev), output_hidden_states=True,
                           proprio=batch["goal_pose"].to(DT).to(dev), proprio_projector=self.pose_projector,
                           noisy_actions=None, noisy_action_projector=None, diffusion_timestep_embeddings=None, use_film=False,
                           **self._llm_kw)
        gt = batch["labels"][:, 1:].to(dev)
        mask = self._masks[0](gt) | self._masks[1](gt)
        h = out.hidden_states[-1][:, self.num_patches:-1]
        ahs = h[mask].reshape(1, CHUNK * 4, -1).to(DT)
        pred = self.action_head.predict_action(ahs, mid.to(DT).to(dev))
        torch.cuda.synchronize()
        out = dict(actions=pred.float().cpu().numpy()[0], t_fwd=time.time() - t)
        if mode == 6:
            out["cache"] = dict(kv_reused=GC["mode"] == "reuse", age=GC["n"] - 1, goal_vision_reused=GC["vis_skip"])
        GC["mode"] = "off"
        self._n_pred += 1
        if self._trim_every and self._n_pred % self._trim_every == 0:
            tt = time.time(); self._libc.malloc_trim(0); out["t_trim"] = time.time() - tt
        return out

    def warmup(self, modes=(4,), n=2):
        """The first calls JIT-compile the Triton/GemLite kernels (~14 s on the Orin); run before serving."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), (120, 120, 120)); t = time.time()
        for mode in modes:
            for _ in range(n):
                self.predict(img, mode=mode, goal_pose=self._default_goal(), goal_image=img, lang="move forward")
        self.log(f"[DEPLOY] warm-up modes {tuple(modes)} in {time.time()-t:.1f} s", flush=True)

    # ---- run_omnivla's controller (waypoint 4 of 8, PD to (v, w), then the 0.3 m/s / 0.3 rad/s limiter) ----
    @staticmethod
    def actions_to_cmd(actions, waypoint_select=4, spacing=0.1, dt=1 / 3, maxv=0.3, maxw=0.3):
        dx, dy, hx, hy = np.asarray(actions, dtype=np.float64)[waypoint_select].copy()
        dx, dy = dx * spacing, dy * spacing
        eps = 1e-8
        if abs(dx) < eps and abs(dy) < eps:
            a = math.atan2(hy, hx); a = (a + math.pi) % (2 * math.pi) - math.pi
            v, w = 0.0, a / dt
        elif abs(dx) < eps:
            v, w = 0.0, float(np.sign(dy)) * math.pi / (2 * dt)
        else:
            v, w = dx / dt, math.atan(dy / dx) / dt
        v, w = float(np.clip(v, 0, 0.5)), float(np.clip(w, -1.0, 1.0))
        if abs(v) <= maxv:
            if abs(w) <= maxw:
                return v, w
            rd = v / w
            return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)
        if abs(w) <= 0.001:
            return maxv * np.sign(v), 0.0
        rd = v / w
        if abs(rd) >= maxv / maxw:
            return maxv * np.sign(v), maxv * np.sign(w) / abs(rd)
        return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)


if __name__ == "__main__":                                    # smoke test: python omnivla_deploy.py DEPLOY_DIR IMAGE
    from PIL import Image
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    img = Image.open(sys.argv[2]).convert("RGB") if len(sys.argv) > 2 else Image.new("RGB", (224, 224), (120, 120, 120))
    m = OmniVLADeploy(d)
    g = np.array([20.0, 0.0, 1.0, 0.0])                       # 2 m straight ahead (controller units, 0.1 m)
    for i in range(3):
        o = m.predict(img, mode=4, goal_pose=g)
    v, w = m.actions_to_cmd(o["actions"])
    print(f"[DEPLOY] pose mode: fwd {o['t_fwd']*1000:.0f} ms | waypoint 4 {np.round(o['actions'][4], 3)} | cmd v {v:.3f} w {w:.3f}")
