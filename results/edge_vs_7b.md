# OmniVLA-7B (this repo's int4 deployment) vs OmniVLA-edge

## Accuracy (error vs the driven path, 8 steps, action units; lower is better)

| Test | n | Test checks perception | 7B bf16 | 7B int4 (deployed) | OmniVLA-edge | edge - 7B int4 [95% CI], p | Verdict |
|---|---|---|---|---|---|---|---|
| Image goal (img3m) | 100 | yes | 1.330 | 1.314 | 1.490 | +0.176 [+0.007, +0.337], p=0.012 | 7B better |
| Pose goal 5 s (pose5) | 100 | no | 1.298 | 1.220 | 1.387 | +0.167 [+0.018, +0.310], p=0.0096 | 7B better |
| Pose goal 20 s (pose20) | 93 | no | 1.610 | 1.515 (Kaggle-sim.) | 1.586 | +0.071 [-0.089, +0.232], p=0.033 | mixed (rank test and CI disagree) |
| Language (CAST) | 237 | no | 2.492 | 2.231 (Kaggle-sim.) | 2.188 | -0.043 [-0.218, +0.139], p=0.21 | no significant difference |

7B int4 on img3m and pose5: outputs of the deployed runtime on the Jetson. On pose20 and language: the deployed int4 weights run on a Kaggle T4 with fp16 dequantized kernels (not Marlin/GemLite, not bit-exact). OmniVLA-edge: fp32, run off the Jetson with OmniVLA's edge preprocessing and the 5 real past frames (eval/edge/edge_eval.py); language samples have no past frames, so the current frame is repeated, as OmniVLA's CAST loader does. Kaggle-simulated vs Jetson outputs of the same int4 weights: img3m 0.0030, pose5 0.0022 action units on average.
Edge without past frames (current frame repeated, as run_omnivla_edge.py does): img3m 1.451, pose5 1.399, pose20 1.573.

## On the Jetson Orin Nano 8 GB

| | 7B int4 (this repo) | OmniVLA-edge |
|---|---|---|
| Latency, pose goal | 375-395 ms with CUDA graphs (default; differs between boots), 424 ms without (MAXN_SUPER) | 112.8 ms (MAXN_SUPER + jetson_clocks); 131.7-151.5 ms at 25W |
| Latency, image goal | 980-1012 ms with CUDA graphs, 1056 ms without (MAXN_SUPER) | same model call as pose |
| Latency, language goal | 650-735 ms with CUDA graphs, 700-770 ms without (no pruning in language modes; results/jetson_validation_2026-09-28.md) | same model call as pose |
| Memory | 4.14 GiB weights on the GPU; RAM peak 6565 / 6769 MB of 7620 (pose / image) | 1.10 GB peak GPU memory |
| Weights on disk | 4.1 GB | 0.43 GB + CLIP ViT-B/32 for the text encoder (0.35 GB) |
| Power, energy per inference (MAXN_SUPER) | 20.2 W, 8.8 J pose; 22.7 W, 23.8 J image goal; 21.5 W, 16.8 J language (results/power_summary.md) | not measured |

7B: results/gptq_validation.md (deployed runtime, 100 frames). Edge: results/edge_jetson.md (measured earlier with OmniVLA's run_omnivla_edge.py and a timing wrapper around the model call, not with this repo's package; the edge latency covers the model call only, the 7B latency the forward pass of the runtime).
