# Latency options: profile and estimates (Jetson, 2026-09-28, one boot, MAXN_SUPER)

Deployed runtime (GPTQ int4 LLM on Marlin, HQQ4 vision on GemLite + HQQ portable, CUDA graphs on). Reference frames
(10, 224x224), 3 passes per mode. Scripts: eval/jetson/stage_profile.py (stages, kernels),
eval/jetson/vision_marlin_estimate.py. Raw: stage_profile_{stages,kernels}.json, vision_marlin_estimate_{all,siglip_mlp}.json.

## Where the time goes (ms per prediction, median; stage times are synchronized forward hooks)

| stage | pose (4) | image, cached goal (6) | image, new goal (6) | language (7) | language + pose (8) |
|---|---|---|---|---|---|
| preprocessing (CPU) | 6.3 | 8.8 | 6.7 | 8.7 | 12.7 |
| DINOv2 | 46.9 | 49.3 | 91.4 (2 images) | 46.9 | 46.9 |
| SigLIP | 120.6 | 126.1 | 234.6 (2 images) | 120.3 | 119.9 |
| projector | 10.5 | 10.5 | 10.6 | 10.3 | 10.4 |
| LLM (32 layers) | 184.4 | 624.4 | 624.3 | 474.6 | 474.8 |
| other inside the model forward (embedding, token selection, masks) | 7.6 | 8.0 | 8.2 | 8.2 | 7.9 |
| action head | 3.9 | 3.9 | 4.0 | 4.0 | 3.9 |
| controller (actions -> v, w) | 0.15 | 0.17 | 0.16 | 0.17 | 0.16 |
| whole call without hooks | 382.0 | 821.3 | 988.2 | 665.8 | 668.3 |

Preprocessing split (CPU): 224x224 input 5.7-11.1 ms per call (two image transforms of 1.7-3.2 ms each; one of them is
the goal placeholder, which non-image modes do not use); 640x480 input 12.9-20.0 ms (7.2-8.8 ms per transform).

SigLIP costs 2.5x DINOv2 because its 52 MLP linears (width 4304, not divisible by 32) cannot use GemLite and run on
HQQ's portable backend, which dequantizes the weights on every call: 99 ms of SigLIP's time (eager measurement).

## Options

1. GPU preprocessing: NOT worth doing. The CPU stage is 6-13 ms at 224x224 and 13-20 ms at 640x480, so the realistic
   saving is < 10-15 ms (a GPU path still needs the host decode and copy). A GPU resize would also not be bit-exact
   with PIL's bicubic resize. (The unused goal-placeholder transform in non-image modes is a free CPU saving of 1.7 ms at
   224x224 or ~8 ms at 640x480. Not done here.)
2. Marlin per-channel int4 for the vision linears: DONE, now the default (results/vismarlin_summary.md): GPTQ
   per-channel int4, measured -103 ms pose / -206 ms image goal (new goal) / -107 ms language, accuracy unchanged.
   The estimate below was made before any weights existed. In-runtime estimate (random-weight Marlin
   layers of the real shapes swapped in, CUDA graphs on, zero-padded to Marlin's k%128 / n%256):

   | variant | pose fwd | image fwd, new goal | vision per image | torch memory |
   |---|---|---|---|---|
   | current | 373.7 | 980.5 | 168.1 | 4244 MB |
   | SigLIP MLP only (52 used + 2 in the unused attention-pool head) | 298.9 (-75) | 829.7 (-151) | 92.4 | -5 MB |
   | all vision linears (204) | 270.8 (-103) | 774.2 (-206) | 64.7 | -27 MB |

   Cached-goal image and language modes encode one image, so they save the same as pose mode (about -75 / -103 ms).
3. Faster attention: NOTHING TO GAIN. The runtime already calls SDPA without a mask (after elision the fork's mask is
   all zeros and is dropped; is_causal=False), and PyTorch 2.8 already dispatches FlashAttention on sm_87 (flash
   available; the dispatcher's timing and error match flash exactly at every shape). Per call: vision 0.14-0.15 ms,
   LLM 0.14 (pose, 108 tokens) / 0.30 (language, 309) / 0.34 ms (image, 363); total attention 7-18 ms per prediction.
   Memory-efficient, cuDNN and math backends are all slower or equal. Feasibility check took well under 1 h.
