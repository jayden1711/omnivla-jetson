# Results in detail

The headline numbers are in the [README](../README.md#results). This page keeps the comparisons behind them; the
tables the analysis scripts write are in [results/](../results/README.md). All accuracy numbers are single open-loop
predictions on in-distribution test frames, in action units (normalized waypoint spacing) against the path the human
driver took; lower is better. Latencies: Jetson Orin Nano 8 GB at MAXN SUPER, one boot unless a range is given
(latency differed by up to ~10% between boots).

## Deployed config vs bitsandbytes NF4 vs bf16

100 FrodoBots frames, image-goal driving test.

| | This repo | bitsandbytes NF4 | bf16 (cloud GPU) |
|---|---|---|---|
| Latency, pose goal | 271 ms | 1416 ms | - |
| Latency, image goal | 775 ms (711 ms when the goal is unchanged) | 2146 ms | - |
| RAM headroom | 1334-1348 MB in the validation runs (one mode per process) | 827 / 575 MB | does not fit |
| Distance from bf16 actions | 0.54 | 0.30 | 0 |
| Driving error vs the human's path | 1.301 | 1.317 | 1.330 |

The driving error of this repo's config is not significantly different from NF4 (p = 0.41) or bf16 (p = 0.17).
Distance from bf16 overstates the damage: NF4 is closer to bf16's actions and no better at driving. CUDA graphs (on by
default) give bit-identical outputs. Details: [results/gptq_validation.md](../results/gptq_validation.md),
[results/final_validation.md](../results/final_validation.md).

## Marlin vision vs the previous HQQ4 vision

The vision encoders moved from HQQ 4-bit on GemLite to GPTQ int4 on Marlin on 2026-09-29 (`OMNIVLA_VISION=hqq4`
restores the old path). Same boot:

| | Marlin vision (default) | HQQ4 vision |
|---|---|---|
| Pose goal | 271 ms | 374 ms |
| Image goal, new goal every frame | 775 ms | 981 ms |
| Language (modes 7 / 8) | 549-564 ms | 663-672 ms |
| Vision per encoded image (DINOv2 + SigLIP) | 65 ms | 168 ms |
| RAM peak, pose / image (of 7620 MB) | 6272 / 6286 MB | 6346 / 6351 MB |
| Image-goal driving error | 1.301 | 1.314 |
| Object goal, picks the named object | 82% | 84% |

Neither accuracy change is significant (image goal p = 0.33, object goal p = 0.22). Round-to-nearest per-channel vision
weights failed the object-goal rule (-5 points, p = 0.007), so the vision linears are GPTQ-calibrated too. Before the
change the README numbers were 375-395 ms pose, 980-1012 ms image goal and 650-735 ms language, with 930-1030 MB of
RAM left with all modes in one process. Details: [results/vismarlin_summary.md](../results/vismarlin_summary.md),
[results/latency_options.md](../results/latency_options.md).

## The original 7B on a cloud GPU

Context on different hardware, not a speed comparison: OmniVLA's own inference path in fp16 takes 511 / 520 / 537 ms
(pose / image / language) on a Kaggle machine with two Tesla T4 GPUs (the model does not fit on one, so it is split
across both). The OmniVLA paper ran the 7B model on a desktop RTX 4090 that controlled the robot over the internet
(action chunks at 3 Hz) and does not report a latency. Details and sources:
[results/benchmark_original_vs_ours.md](../results/benchmark_original_vs_ours.md).

## 7B vs OmniVLA-edge: notes

The table is in the [README](../README.md#7b-or-omnivla-edge); the full one with confidence intervals is
[results/edge_vs_7b.md](../results/edge_vs_7b.md).

- Image goal, 5 s pose goal and object goal: the deployed runtime's outputs on the Jetson (Marlin vision). 20 s pose goal
  and CAST instructions: the same int4 weights on a Kaggle T4 (fp16 kernels, previous HQQ4 vision; they match the
  Jetson to about 0.003 action units where both exist).
- OmniVLA-edge: accuracy off the Jetson with its 5 past frames (without them: image goal 1.451, 5 s pose 1.399).
  Latency and memory were measured earlier with OmniVLA's own `run_omnivla_edge.py`, not with this repo
  ([results/edge_jetson.md](../results/edge_jetson.md)); edge's power is not measured.
- With a pose goal the 7B is better at 5 s and not clearly better at 20 s, but the pose tests are weaker checks of
  perception: a shuffled camera image does not make the 7B model clearly worse on them.
- With a language goal, the 7B only keeps its edge because language modes skip the image-token pruning: with the 75%
  pruning used for pose and image goals it drops to 69%, level with edge (see below).
- Power at MAXN SUPER (measured before CUDA graphs and Marlin vision became the defaults): 20 W and 8.8 J per
  pose-goal prediction, 23 W and 24 J per image-goal prediction. Lower power modes save watts but not energy: at 15W a
  pose-goal prediction took 561 ms at 16 W, 9.3 J ([results/power_summary.md](../results/power_summary.md)).

## Object goals (LeLaN)

210 frames from LeLaN's robot recordings, each with two labeled objects in different directions; the prompt is
"move toward <object>" for one of them, the format the model was trained on. This is in-distribution: LeLaN is in the
checkpoint's training mix, so it checks that language mode works as trained, not that it generalizes. Every row has
blank-image, shuffled-image and shuffled-instruction controls, and every model follows the prompt and fails with a
blank or shuffled image. The rows below ran on a Kaggle T4 with the deployed int4 weights (fp16 kernels, HQQ4 vision);
the deployed runtime on the Jetson, with Marlin vision and no pruning, scores 82%
([results/cross_mode_summary.md](../results/cross_mode_summary.md)).

| | Picks the named object | With the other object's prompt |
|---|---|---|
| 7B full precision (bf16; the control run in fp16) | 84% | 19% |
| **7B int4, no image-token pruning (language modes now)** | **84%** | 20% |
| 7B int4, 50% pruning | 81% | 25% |
| 7B int4, 75% pruning (the first language setting) | 69% | 31% |
| 7B fp16, 75% pruning, no quantization | 71% | 28% |
| OmniVLA-edge | 70% | 29% |
| Straight ahead (no model) | 48% | |

The first run of the deployed config (69%) was clearly below full precision; the ablation, on the same int4 weights,
shows why: **the 75% image-token pruning causes the whole drop, not the int4 weights.** Without pruning the int4
weights match bf16 (84%, p = 1); 50% pruning costs 3 points (not significant); 75% pruning costs about the same with or
without quantization. So the runtime and the Python API skip pruning in language modes (7, 8), and keep 75% for pose and
image goals, where it did not change driving error. The price is latency and memory: on the Jetson a language
prediction takes 549-564 ms (pose goal: 271 ms), and about 70 MB more RAM. `lang_prune_frac=0.5` is a middle ground:
81%, and 538 ms without CUDA graphs and with HQQ4 vision
([results/jetson_validation_2026-09-28.md](../results/jetson_validation_2026-09-28.md)).

Pruning by relevance to the instruction does not help: keeping the image patches most similar to the object phrase
(SigLIP image-text similarity) scored 69% at 75% pruning, the same as the uniform grid, and 78% at 50% (uniform: 81%).

Language grounding turned out to be more sensitive to compression than pose and image goals: the pruning that left
their driving error unchanged cost 15 points here. Each goal type needs its own task-grounded test. Full tables:
[results/lelan_summary.md](../results/lelan_summary.md).

## Behavioral instructions (CAST)

237 CAST episodes with instructions like "move along the corridor", with the same three controls.

**Default weights (omnivla-original).** The checkpoint was not trained on this kind of instruction, and no model passes
the instruction check here: a shuffled instruction is not worse than the right one. The test therefore says nothing
about language mode as trained, and the int4-vs-bf16 difference on it (-0.26, p = 0.046) does not mean int4 is better
([results/lang_summary.md](../results/lang_summary.md)).

**CAST weights (omnivla-finetuned-cast, `setup_jetson.sh --cast`).** GPTQ int4 for the LLM and the vision encoders,
calibrated on 64 samples from recordings that are not in the test; 175 held-out episodes, language mode, no pruning:

| | bf16 | GPTQ int4 |
|---|---|---|
| Driving error | 1.396 | 1.433 (+0.037, p = 0.35) |
| Blank image / shuffled image | +64% / +90% | +95% / +82% |
| Shuffled instruction | -4% (p = 0.022) | -2% (p = 0.46) |
| Turn direction correct (98 turning episodes) | 91% | 87% (p = 0.29) |
| On the Jetson | - | 562 ms median, 590 ms p95; 100/100 repeats bit-identical |

Quantization kept the checkpoint's behavior (no significant accuracy loss, turn direction kept). But even in bf16 the
checkpoint does not pass the shuffled-instruction check on these episodes: the instruction rarely changes the path,
although the turn direction is right in 91% of the episodes that turn. Outside language mode the CAST weights are not
better than the default: no significant difference on the image-goal and object-goal tests, worse on the 5 s pose test
as int4 (+0.096, p = 0.002) but not in bf16 ([results/cast_gptq_summary.md](../results/cast_gptq_summary.md),
[results/cross_mode_summary.md](../results/cross_mode_summary.md)).
