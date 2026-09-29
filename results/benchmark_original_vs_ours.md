# Original OmniVLA-7B vs this repo vs OmniVLA-edge

## Accuracy (existing results; error vs the human's driven path, action units, lower is better)

| Test | Original 7B, bf16 (GPU) | This repo, int4 on the Jetson | This repo vs original | OmniVLA-edge |
|---|---|---|---|---|
| Image goal (img3m, 100 frames) | 1.330 | 1.301 | no significant difference (Wilcoxon p = 0.17) | 1.490 |
| Object goal (LeLaN, 210 frames), picks the named object | 84% | 82% | no significant difference (McNemar p = 0.22) | 70% |
| Pose goal 5 s (pose5, 100 frames) [1] | 1.298 | 1.177 | [1] | 1.387 |

[1] Not perception-validated: on this test a blank or shuffled camera image does not make the model clearly worse
(results/tests_summary.md, results/cross_mode_summary.md), so it mostly measures how the goal pose is followed. The
int4 model's lower error here (Wilcoxon p = 0.0006) is not evidence of better driving and is not reported as an
advantage.

This repo: results/cross_mode_summary.md (jet_orig, Marlin vision, 2026-09-29) and results/vismarlin_summary.md. Edge
and bf16: results/edge_vs_7b.md, results/lelan_summary.md. edge_vs_7b.md compares edge with the same Marlin-vision
Jetson outputs on the image-goal, 5 s pose and object-goal tests; its 20 s pose and CAST rows are Kaggle-simulated with
the previous HQQ4 vision.

## Memory

| | Original 7B, bf16 | This repo, int4 | OmniVLA-edge |
|---|---|---|---|
| Weights | 15.1 GB (checkpoint shards; ~15 GB of GPU memory in bf16/fp16) | 4.13 GiB on the GPU (4.1 GB on disk) | 0.43 GB (+ 0.35 GB CLIP text encoder) |
| Fits a Jetson Orin Nano 8 GB? | No: 7.6 GB of unified memory in total, ~5.8 GB available with the desktop off | Yes: 1.33-1.35 GB RAM left in the validation runs | Yes: 1.10 GB peak GPU memory |

## Latency (per prediction)

On the Jetson, where the models are meant to run:

| | Pose goal | Image goal | Language | Hardware |
|---|---|---|---|---|
| Original 7B | - | - | - | does not fit (15.1 GB of weights, 7.6 GB of memory) |
| This repo, int4 | 271 ms | 775 ms (711 ms, goal unchanged) | 549-564 ms | Jetson Orin Nano 8 GB, MAXN SUPER |
| OmniVLA-edge | 113 ms | same model call | same model call | Jetson Orin Nano 8 GB, MAXN SUPER |

Context on different hardware, not a speed comparison (the original cannot run on the Jetson, so it was timed where it
fits):

| | Pose goal | Image goal | Language | Hardware |
|---|---|---|---|---|
| Original 7B, fp16, as released | 511 ms | 520 ms | 537 ms | Kaggle, 2x Tesla T4 (the model is split across both GPUs) |
| Original 7B, fp16, with this repo's token elision | 317 ms | 544 ms | 327 ms | same |
| OmniVLA paper setup | not reported | not reported | not reported | desktop RTX 4090 PC, controlling the robot over the internet; action chunks of 8 at 3 Hz (2.4 s) |

T4: `LANG_TEST=xmode eval/kaggle/lang_eval.sh` (STUDY=t4lat), median of 20 predictions after 3 warm-up calls, whole
call including preprocessing (model forward + action head alone: 501 / 510 / 527 ms). "As released" = OmniVLA's own
inference path (all tokens). Elision drops the unused goal-modality tokens; in image-goal mode every token is used, so it
does not help there. The paper (arXiv 2509.19480) states: "Our local PC with an NVIDIA RTX 4090 receives front-camera
images and pose signals, and sends velocity commands from our policy to control the robot over the internet."
