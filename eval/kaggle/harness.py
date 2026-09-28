# harness.py - loads OmniVLA once and patches in modality elision (masked tokens dropped before the LLM, original
# positions kept). kaggle_compress.py reuses the elision block between the two marker comments below.
import math, time, numpy as np, torch, utm
from PIL import Image
import inference.run_omnivla as R

cfg = R.InferenceConfig()
vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = R.define_model(cfg)
for k, v in dict(vla=vla, action_head=action_head, pose_projector=pose_projector, device_id=device_id,
                 NUM_PATCHES=NUM_PATCHES, action_tokenizer=action_tokenizer, processor=processor).items():
    setattr(R, k, v)

# ---------- Idea 1: modality-elided inference ----------
E = {"on": False, "skip_img": False}
POS = {}
_orig_build = vla._build_multimodal_attention_MMN
def build_elide(inp, patches, am, aml, mid):
    emb, mask = _orig_build(inp, patches, am, aml, mid)
    POS.clear()
    if E["on"]:
        keep = mask[0].bool()                                   # batch size 1
        POS["ids"] = torch.arange(mask.shape[1], device=mask.device)[keep].unsqueeze(0)
        POS["full"] = mask.shape[1]
        POS["dropped"] = int((~keep).sum())
        emb, mask = emb[:, keep], mask[:, keep]
    return emb, mask
vla._build_multimodal_attention_MMN = build_elide

lm = vla.language_model
_orig_lm = lm.forward
def lm_fwd(*a, **kw):
    kw["labels"] = None   # loss is never used at inference; skipping it saves ~70 MB + compute
    ids = POS.get("ids")
    if ids is None:
        return _orig_lm(*a, **kw)
    kw["position_ids"] = ids        # keep ORIGINAL positions so RoPE is unchanged
    kw["labels"] = None             # loss not needed at inference
    out = _orig_lm(*a, **kw)
    h = out.hidden_states[-1]
    full = h.new_zeros(h.shape[0], POS["full"], h.shape[2])
    full[:, ids[0]] = h             # scatter back so downstream indexing is unchanged
    out.hidden_states = tuple(out.hidden_states[:-1]) + (full,)
    return out
lm.forward = lm_fwd

vb = vla.vision_backbone
_orig_vb = vb.forward
def vb_fwd(pixel_values, *a, **kw):
    if E["on"] and E["skip_img"]:
        half = pixel_values.shape[1] // 2                       # [current | goal] stacked on channels
        vb.set_num_images_in_input(1)
        try:
            f = _orig_vb(pixel_values[:, :half], *a, **kw)
        finally:
            vb.set_num_images_in_input(2)
        return torch.cat([f, torch.zeros_like(f)], dim=1)       # placeholder, dropped later anyway
    return _orig_vb(pixel_values, *a, **kw)
vb.forward = vb_fwd

# ---------- modes & inference object ----------
MODES = {6: dict(image_goal=True), 7: dict(lan_prompt=True), 4: dict(pose_goal=True), 8: dict(lan_prompt=True, pose_goal=True)}
def set_mode(m):
    R.satellite = R.lan_prompt = R.pose_goal = R.image_goal = False
    for k, v in MODES[m].items():
        setattr(R, k, v)

goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
inf = R.Inference(save_dir="./inference", lan_inst_prompt="move toward blue trash bin",
                  goal_utm=utm.from_latlon(goal_lat, goal_lon), goal_compass=-goal_compass / 180.0 * math.pi,
                  goal_image_PIL=Image.open("./inference/goal_img.jpg").convert("RGB"),
                  action_tokenizer=action_tokenizer, processor=processor)

CAP = {}
_orig_rfp = inf.run_forward_pass
def rfp(*a, **k):
    torch.cuda.synchronize(); t = time.time()
    out = _orig_rfp(*a, **k)
    torch.cuda.synchronize(); CAP["t"] = time.time() - t
    CAP["act"] = out[0].float().cpu().numpy()
    return out
inf.run_forward_pass = rfp

def trial(mode, elide, n=3):
    import gc; gc.collect(); torch.cuda.empty_cache()
    set_mode(mode)
    E["on"] = elide
    E["skip_img"] = elide and not MODES[mode].get("image_goal", False)
    inf.run_omnivla()                                           # warm-up (shapes change per mode)
    ts = []
    for _ in range(n):
        inf.run_omnivla(); ts.append(CAP["t"])
    return CAP["act"], sum(ts) / len(ts), POS.get("dropped", 0)


# ---------- current-image override ----------
CUR = {"img": None}          # None -> run_omnivla's default ./inference/current_img.jpg
def set_current_image(img):
    """img: file path, PIL image, or None to restore the default sample image."""
    CUR["img"] = Image.open(img).convert("RGB") if isinstance(img, str) else img
_orig_dt = inf.data_transformer_omnivla
def dt_cur(current_image_PIL, *a, **k):
    return _orig_dt(CUR["img"] if CUR["img"] is not None else current_image_PIL, *a, **k)
inf.data_transformer_omnivla = dt_cur
