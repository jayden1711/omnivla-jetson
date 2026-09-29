# OmniVLA-7B (this repo's int4 deployment) vs OmniVLA-edge

## Accuracy (error vs the driven path, 8 steps, action units; lower is better)

| Test | n | Test checks perception | 7B bf16 | 7B int4 (deployed) | OmniVLA-edge | edge - 7B int4 [95% CI], p | Verdict |
|---|---|---|---|---|---|---|---|
| Image goal (img3m) | 100 | yes | 1.330 | 1.301 | 1.490 | +0.189 [+0.020, +0.350], p=0.011 | 7B better |
| Pose goal 5 s (pose5) | 100 | no | 1.298 | 1.177 | 1.387 | +0.211 [+0.051, +0.363], p=0.0029 | 7B better |
| Pose goal 20 s (pose20) | 93 | no | 1.610 | 1.515 (Kaggle-sim.) | 1.586 | +0.071 [-0.089, +0.232], p=0.033 | mixed (rank test and CI disagree) |
| Language (CAST) | 237 | no | 2.492 | 2.231 (Kaggle-sim.) | 2.188 | -0.043 [-0.218, +0.139], p=0.21 | no significant difference |
| Object goal (LeLaN), picks the named object (higher is better) | 210 | yes | 84% | 82% | 70% | -12 points, McNemar p=2.2e-05 | 7B better |

7B int4 on img3m and pose5 and the object goal: outputs of the deployed runtime on the Jetson (results/cross_jet_orig.npz: GPTQ int4 LLM and GPTQ int4 Marlin vision, the current default; the Kaggle-simulated rows below still use the previous HQQ4 vision, which did not differ significantly from it: results/vismarlin_summary.md). On pose20 and language: the deployed int4 weights run on a Kaggle T4 with fp16 dequantized kernels (not Marlin/GemLite, not bit-exact). OmniVLA-edge: fp32, run off the Jetson with OmniVLA's edge preprocessing and the 5 real past frames (eval/edge/edge_eval.py); language samples have no past frames, so the current frame is repeated, as OmniVLA's CAST loader does. Kaggle-simulated vs Jetson outputs of the same Marlin-vision weights: img3m 0.0032 action units on average (results/vismarlin_summary.md).
Edge without past frames (current frame repeated, as run_omnivla_edge.py does): img3m 1.451, pose5 1.399, pose20 1.573.

## On the Jetson Orin Nano 8 GB

| | 7B int4 (this repo) | OmniVLA-edge |
|---|---|---|
| Latency, pose goal | 271 ms (Marlin vision, CUDA graphs, MAXN_SUPER; 374 ms with HQQ4 vision, 424 ms without graphs) | 112.8 ms (MAXN_SUPER + jetson_clocks); 131.7-151.5 ms at 25W |
| Latency, image goal | 775 ms, 711 ms when the goal is unchanged (981 ms with HQQ4 vision) | same model call as pose |
| Latency, language goal | 549-564 ms (no pruning in language modes; 663-672 ms with HQQ4 vision; results/vismarlin_summary.md) | same model call as pose |
| Memory | 4.13 GiB weights on the GPU; RAM peak 6272 / 6286 MB of 7620 (pose / image) | 1.10 GB peak GPU memory |
| Weights on disk | 4.1 GB | 0.43 GB + CLIP ViT-B/32 for the text encoder (0.35 GB) |
| Power, energy per inference (MAXN_SUPER) | 20.2 W, 8.8 J pose; 22.7 W, 23.8 J image goal; 21.5 W, 16.8 J language (results/power_summary.md) | not measured |

7B: results/vismarlin_summary.md and results/gptq_validation.md (deployed runtime, 100 frames). Edge: results/edge_jetson.md (measured earlier with OmniVLA's run_omnivla_edge.py and a timing wrapper around the model call, not with this repo's package; the edge latency covers the model call only, the 7B latency the forward pass of the runtime).
