# CAST checkpoint vs original checkpoint outside language mode

Both as the deployed int4 runtime on the Jetson (GPTQ int4 LLM + GPTQ int4 Marlin vision; 75% token pruning in pose and image-goal modes, none in language mode), same frames and controls. omnivla-finetuned-cast was fine-tuned from the original on CAST; these tests use FrodoBots (image goal, 5 s pose goal) and LeLaN (object goal), which are in the original's training mix. Errors in action units vs the human's driven path; lower is better.

## Image goal (img3m, 100 frames)

| Model | Error | Blank image | Shuffled image | Checks perception |
|---|---|---|---|---|
| original int4 (Jetson) | 1.301 | +102% (p=9.3e-10) | +27% (p=0.00025) | yes |
| CAST int4 (Jetson) | 1.314 | +127% (p=2.8e-11) | +17% (p=0.043) | no |
| original bf16 (GPU) | 1.330 | - | - | - |
| CAST bf16 (GPU) | 1.301 | - | - | - |

CAST vs original (Jetson int4, paired): +0.014 (Wilcoxon p=0.44) -> no significant difference.
CAST vs original in bf16 (GPU, paired): -0.030 (Wilcoxon p=0.42); int4 vs bf16 of the same checkpoint: original -0.029 (p=0.17), CAST +0.014 (p=0.82).

## Pose goal 5 s ahead (pose5)

| Model | Error | Blank image | Shuffled image | Checks perception |
|---|---|---|---|---|
| original int4 (Jetson) | 1.177 | +11% (p=0.13) | +5% (p=0.41) | no |
| CAST int4 (Jetson) | 1.273 | -0% (p=0.9) | +3% (p=0.64) | no |
| original bf16 (GPU) | 1.298 | - | - | - |
| CAST bf16 (GPU) | 1.268 | - | - | - |

CAST vs original (Jetson int4, paired): +0.096 (Wilcoxon p=0.0023) -> CAST is WORSE.
CAST vs original in bf16 (GPU, paired): -0.029 (Wilcoxon p=0.25); int4 vs bf16 of the same checkpoint: original -0.121 (p=0.00062), CAST +0.005 (p=0.89).

## Object goal (LeLaN, 210 frames, "move toward <object>")

| Model | Picks named object | Blank image | Shuffled image | Other object's prompt | Uses instruction |
|---|---|---|---|---|---|
| original int4 (Jetson) | 82% | 46% (-36 pts, p=4e-20) | 51% (-31 pts, p=2e-14) | 23% (-59 pts, p=5e-35) | yes |
| CAST int4 (Jetson) | 84% | 48% (-36 pts, p=2e-19) | 50% (-34 pts, p=3e-18) | 19% (-65 pts, p=1e-35) | yes |
| original bf16 (GPU) | 84% | - | - | - | - |
| CAST bf16 (GPU) | 85% | - | - | - | - |

CAST vs original (Jetson int4, paired): +1.4 points (McNemar p=0.45) -> no significant difference.

## Verdict

**By the rule (Jetson int4 on both): the CAST checkpoint loses accuracy on: Pose goal 5 s ahead (pose5).** Single open-loop predictions on in-distribution test frames. Read it together with the bf16 lines above: they show whether a gap comes from the checkpoint itself or from how each checkpoint reacts to quantization.
