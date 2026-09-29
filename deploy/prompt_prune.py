"""Prompt-aware pruning of current-image tokens for language modes: score each of the 256 image patches by its similarity
to the instruction in SigLIP's image-text space and keep the top ones (instead of a uniform grid).

Patch embeddings (MaskCLIP-style): the patch features go through the rest of SigLIP's image tower (last block, final
norm) and then through its attention-pooling head as if all attention fell on that patch (value projection, output
projection, MLP). The text embedding comes from the matching SigLIP text tower: timm/ViT-SO400M-14-SigLIP on Hugging Face
(open_clip, Apache-2.0), the WebLI checkpoint that timm's vit_so400m_patch14_siglip_224 (OmniVLA's second vision encoder)
was released with. Caveat: OmniVLA fine-tuned SigLIP's image blocks (weights ~27% different from the release) but not the
pooling head, so image-text alignment of the fine-tuned patch features is not guaranteed; that is what the object-goal
evaluation (eval/kaggle/lang_eval.sh, LANG_TEST=promptprune) measures. Not used by the runtime unless adopted.
"""
import numpy as np
import torch

N_IMG = 256


def penultimate(trunk, img):
    """SigLIP features after all but the last block (what the runtime's truncated forward returns)"""
    x = trunk.norm_pre(trunk.patch_drop(trunk._pos_embed(trunk.patch_embed(img))))
    for blk in trunk.blocks[:-1]:
        x = blk(x)
    return x[:, trunk.num_prefix_tokens:]


@torch.no_grad()
def patch_embeddings(trunk, x_pen):
    """(B, 256, C) penultimate features -> (B, 256, D) unit-norm patch embeddings in SigLIP's image-text space"""
    x = trunk.norm(trunk.blocks[-1](x_pen))
    ap = trunk.attn_pool
    B, N, C = x.shape
    if ap.pos_embed is not None:
        x = x + ap.pos_embed.unsqueeze(0).to(x.dtype)
    v = ap.kv(x).reshape(B, N, 2, ap.num_heads, ap.head_dim).permute(2, 0, 3, 1, 4)[1]   # (B, heads, N, hd)
    o = ap.proj(v.transpose(1, 2).reshape(B, N, C))
    o = o + ap.mlp(ap.norm(o))
    return torch.nn.functional.normalize(o.float(), dim=-1)


def object_phrase(instruction):
    """the part SigLIP was trained on (alt-text-like): "move toward the blue bin" -> "the blue bin" """
    s = instruction.strip()
    for p in ("move toward ", "move towards ", "go to ", "navigate to "):
        if s.lower().startswith(p):
            return s[len(p):]
    return s


def drop_positions(patch_emb, text_emb, frac):
    """LLM positions (patch index + 1) of the lowest-scoring round(256 * frac) patches"""
    n_drop = int(round(N_IMG * frac))
    s = (patch_emb[0] @ text_emb.float().reshape(-1)).float()                            # (256,)
    return torch.argsort(s)[:n_drop] + 1


class TextEncoder:
    """SigLIP text tower (open_clip). Load, encode the phrases you need, then free it: it is ~0.9 GB in fp16."""

    def __init__(self, device="cuda", name="hf-hub:timm/ViT-SO400M-14-SigLIP"):
        import open_clip
        self.model = open_clip.create_model(name, device=device).eval()
        if device != "cpu":
            self.model = self.model.half()
        self.tok = open_clip.get_tokenizer(name)
        self.device = device

    @torch.no_grad()
    def encode(self, phrases):
        e = self.model.encode_text(self.tok(list(phrases)).to(self.device))
        return {p: torch.nn.functional.normalize(v.float(), dim=-1) for p, v in zip(phrases, e)}

    def image_tower(self):
        """the original (not fine-tuned) SigLIP image trunk, for the reference variant"""
        return self.model.visual.trunk
