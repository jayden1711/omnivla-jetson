# Every change we make to the original OmniVLA

Upstream: https://github.com/NHirose/OmniVLA @ `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`, checkpoint
`NHirose/omnivla-original` (Hugging Face). The deployment uses a **fresh clone** of that commit plus one file patch
(section A). Everything else is done **at run time** by `omnivla_deploy.py` without editing OmniVLA's files (section B),
or **offline** when the weights are built (section C). `setup_jetson.sh` applies section A and installs section D.

## A. File changes to the OmniVLA repo (patches/omnivla.patch)

| File | Change | Why |
|---|---|---|
| `prismatic/vla/__init__.py` | `from .materialize import get_vla_dataset_and_collator` wrapped in `try/except ImportError` (sets it to `None`) | The import pulls in the training data pipeline (`dlimp`, TensorFlow), which is neither needed for inference nor installable on the Jetson (and must not be installed there). Inference never calls it. |

No other file of the repo is modified.

## B. Run-time changes made by omnivla_deploy.py (the repo's files stay untouched)

| # | Change (where) | Why | Effect on outputs |
|---|---|---|---|
| B1 | The model is built as an empty skeleton (`init_empty_weights`); non-quantized tensors come from `weights/base.safetensors`, quantized layers from pre-packed shards loaded one 256 MB shard at a time with `.detach()` and `malloc_trim(0)` after each (lines ~74-120) | `from_pretrained` + on-device quantization does not fit in 8 GB; each of whole-file mmap, non-detached Parameter copies and glibc buffer retention caused an OOM | none (same tensors) |
| B2 | Rotary `inv_freq` buffers recomputed after the empty-skeleton load (line ~124) | they are created on the meta device by the skeleton | none |
| B3 | `language_model.lm_head` replaced by a module returning zeros (line ~80) | OmniVLA's action head reads hidden states; the 250 MB vocabulary projection's logits are never used | none |
| B4 | LLM `labels` set to `None` (line ~237) | the training loss is not needed at inference | none |
| B5 | The 224 LLM linears replaced by Marlin per-channel int4 layers (`MarlinPC`, line ~92), GPTQ-calibrated weights (section C) | 7B in fp16 does not fit; Marlin is the fastest correct int4 kernel on sm_87 (grouped Marlin is wrong on sm_87 and is never used) | fidelity to bf16 0.48 (action units, image-goal test); driving error n.s. vs bf16 |
| B6 | The 204 vision linears replaced by HQQ 4-bit layers; 150 run on GemLite (Triton), the 54 SigLIP MLP layers (width 4304, not divisible by 32) on HQQ's portable backend; inputs cast to fp16 (`GLWrap`, line ~135) | memory and speed; GemLite is not autocast-aware | part of the 0.48 above |
| B7 | Modality elision: tokens masked out by the goal modality are dropped before the LLM, original position ids kept (`build`, `lm_fwd`, lines ~220-243); the goal image is not encoded when the mode does not use it (`vb_fwd`) | 1.54x faster; the masked tokens do not influence the actions except through bf16 rounding | max 1 bf16 ULP |
| B8 | Uniform-grid pruning of 75% of the current-image tokens before the LLM (keep 64 of 256; `build`), in pose and image-goal modes only; language modes (7, 8) are not pruned (`lang_prune_frac=0`) | fastest pruning level with no significant driving change | fidelity cost included in the deployed numbers; in language mode 75% pruning cut object-goal accuracy from 84% to 69% (results/lelan_summary.md), so it is off there |
| B9 | Each ViT stops after the block whose output OmniVLA uses (`_install_vit_trunc`, line ~172): timm 0.9.10's `get_intermediate_layers(n={len-2})` still runs the last block | saves 9-13 ms | none (bit-identical) |
| B10 | Image-goal mode: the goal image's vision features are computed once per goal and reused (`vb_fwd`, `GC`) | 1.22x faster while the goal is unchanged | none (bit-identical) |
| B11 | Optional (off by default, `goal_refresh > 1`): the goal tokens' keys/values are reused in the LLM via a patched attention forward (`attn_forward`, line ~188); with it off the original forward runs | 2.4x faster image-goal mode | APPROXIMATE: OmniVLA's LLM attention is bidirectional, so these keys/values depend on the current frame; measured +0.02 driving error (p < 0.01) |
| B12 | `predict()` calls OmniVLA's own preprocessing (`Inference.data_transformer_omnivla`, prompt builder, action tokenizer) and action head, but not `run_omnivla()`'s plotting/saving (`save_robot_behavior`) | the unclosed matplotlib figure per call leaked memory; no disk writes on the robot | none |
| B13 | `malloc_trim(0)` every 20 predictions | glibc heap retention (+118 MB / 10 min without it) | none |
| B14 | `actions_to_cmd` is a copy of `run_omnivla.py`'s controller (waypoint 4, velocity limits); the ROS node instead executes the whole chunk in time (`ros2/rover_protocol.py` ChunkExecutor) | the ROS node needs latency-compensated execution | n/a (control, not the model) |

## C. Offline weight changes (build/build_model.sh -> build/kaggle_build.py)

| Change | Why |
|---|---|
| LLM: GPTQ (Frantar et al. 2023), per-channel int4 in Marlin's convention (q in 0..15, zero 8, scale max\|w\|*2/15 per row), calibrated on 128 held-out FrodoBots samples (64 rides not used for testing, pose + image goals), through the deployed pipeline (elision + grid pruning); packed with Marlin @1f25790 | round-to-nearest in the same format was 2.3x further from bf16 (fidelity 1.09 vs 0.48) at the same speed |
| Vision: HQQ 4-bit, group 64 (data-free), hqq 0.2.8.post1 | quality/speed trade-off validated on the image-goal test |
| Everything else (embeddings, norms, projector, action head, proprio projector) stored in fp16 | small; kept at full precision |

## D. Environment requirements that change behavior (not code changes, but required)

| Requirement | Why |
|---|---|
| `transformers` = https://github.com/moojink/transformers-openvla-oft @ `bc339d9` | makes the LLM attention bidirectional (OpenVLA-OFT parallel decoding). Stock `transformers==4.40.1` silently runs causal attention and gives actions about 1 action unit off. `setup_jetson.sh` asserts the fork. |
| torch 2.8.0 / torchvision 0.23.0 Jetson wheels (jp6/cu126), numpy 1.26.4 (`constraints.txt`) | generic wheels have no CUDA on the Orin |

## E. Evaluation code (not used by the deployment)

The evaluation scripts in `eval/` patch the same OmniVLA objects at run time from their own files
(`eval/kaggle/harness.py`, `build/kaggle_compress.py`, `eval/jetson/jetson_cache.py`). They use a clone with the same
import guard as section A, disable `save_robot_behavior` (one open matplotlib figure per call), and apply the same
modality elision as B7. The deployment never imports any of them; `setup_jetson.sh` makes its own fresh clone.
