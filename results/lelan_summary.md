# Object-goal language test (LeLaN, in-distribution)

210 frames from LeLaN's robot recordings (gs2 70, gs4 70, sacson 70). **In-distribution:** LeLaN is in the deployed checkpoint's training mix, and the prompts use its training format ("move toward <object>"). So this checks that language mode works as trained and survives quantization, not that it generalizes. Each frame has two labeled objects whose directions differ by at least 30 degrees; the prompt names one of them. Open-loop single predictions. 7B int4 = the deployed weights on a Kaggle T4 with fp16 kernels (not the Jetson's; language mode has not been run on the Jetson). OmniVLA-edge: prompt = the object phrase (its sample script's format), current frame repeated as history.

Straight ahead (no model) picks the named object in 48% of frames.

| Model | Picks named object | Bearing error (deg, median) | Blank image | Shuffled image | Prompt for the other object | Checks perception | Uses instruction |
|---|---|---|---|---|---|---|---|
| 7B bf16 | 84% (p vs 50%: 5e-25) | 15 | - | - | - | - | - |
| 7B fp16 | 84% (p vs 50%: 5e-25) | 15 | 49% (-35 pts, p=1e-19) | 50% (-34 pts, p=3e-16) | 19% (-65 pts, p=4e-40) | yes | yes |
| 7B fp16, 75% pruning (no quantization) | 71% (p vs 50%: 4e-10) | 23 | 49% (-23 pts, p=6e-10) | 47% (-24 pts, p=1e-10) | 28% (-43 pts, p=7e-23) | yes | yes |
| 7B int4 deployed (75% pruning) | 69% (p vs 50%: 8e-08) | 24 | 50% (-19 pts, p=1e-08) | 48% (-20 pts, p=2e-08) | 31% (-38 pts, p=3e-20) | yes | yes |
| 7B int4, 0% pruning | 84% (p vs 50%: 5e-25) | 16 | 47% (-37 pts, p=6e-20) | 51% (-33 pts, p=3e-15) | 20% (-64 pts, p=6e-38) | yes | yes |
| 7B int4, 50% pruning | 81% (p vs 50%: 3e-20) | 18 | 48% (-33 pts, p=2e-18) | 50% (-31 pts, p=2e-15) | 25% (-56 pts, p=6e-33) | yes | yes |
| 7B int4, 75% pruning (same-session rerun) | 69% (p vs 50%: 8e-08) | 24 | 50% (-19 pts, p=1e-08) | 48% (-20 pts, p=2e-08) | 31% (-38 pts, p=3e-20) | yes | yes |
| 7B int4, 25% uniform pruning | 83% (p vs 50%: 7e-23) | 17 | 49% (-34 pts, p=5e-19) | 50% (-32 pts, p=5e-16) | 20% (-62 pts, p=7e-40) | yes | yes |
| 7B int4, 25% prompt-aware pruning | 82% (p vs 50%: 1e-21) | 17 | 48% (-34 pts, p=2e-17) | 50% (-32 pts, p=9e-15) | 21% (-60 pts, p=6e-36) | yes | yes |
| 7B int4, 50% prompt-aware pruning | 78% (p vs 50%: 3e-16) | 19 | 47% (-30 pts, p=2e-15) | 50% (-28 pts, p=4e-12) | 25% (-53 pts, p=2e-32) | yes | yes |
| 7B int4, 75% prompt-aware pruning | 69% (p vs 50%: 3e-08) | 25 | 47% (-22 pts, p=8e-10) | 50% (-19 pts, p=5e-07) | 31% (-38 pts, p=8e-20) | yes | yes |
| 7B int4, 75% prompt-aware, original SigLIP (reference) | 72% (p vs 50%: 2e-10) | 23 | 50% (-22 pts, p=2e-10) | 50% (-22 pts, p=8e-09) | 31% (-41 pts, p=7e-24) | yes | yes |
| OmniVLA-edge | 70% (p vs 50%: 3e-09) | 19 | 48% (-23 pts, p=8e-09) | 46% (-25 pts, p=5e-09) | 29% (-42 pts, p=3e-20) | yes | yes |

Controls: percentages are the picks-named-object rate under the control; pts = change vs the real input; p = paired exact test. With the other object's prompt, a model that follows language should fall well below 50%.

Paired comparisons of picks-named-object with bf16: fp16 +0 pts (p=1); fp16_spatial75 -13 pts (p=3.5e-06); gptqx_pc4 -16 pts (p=3.6e-08); gptqx_pc4_none +0 pts (p=1); gptqx_pc4_spatial50 -3 pts (p=0.14); gptqx_pc4_spatial75 -16 pts (p=3.6e-08); gptqx_pc4_spatial25 -1 pts (p=0.45); gptqx_pc4_prompt25 -2 pts (p=0.18); gptqx_pc4_prompt50 -7 pts (p=0.0026); gptqx_pc4_prompt75 -15 pts (p=1.9e-08); gptqx_pc4_promptorig75 -12 pts (p=8.7e-07); edge -14 pts (p=1.3e-07).

## Cause of the int4 drop (ablation, same deployed int4 weights)

Rule fixed before the run: if the deployed int4 weights without pruning pick the named object in >= 77% of frames (half of the gap to bf16 closed), pruning is the cause and GPTQ is not recalibrated.

- int4, no pruning: 84% (bf16 84%; p=1) -> meets the 77% rule.
- int4, 50% pruning: 81% (p vs bf16 0.14).
- int4, 75% pruning: 69%; fp16 (no quantization), 75% pruning: 71% (the two do not differ significantly, p=0.18).

**The drop comes from the 75% image-token pruning, not from the int4 weights.** Pruning costs about the same with and without quantization, and the int4 weights alone match bf16. The runtime therefore does not prune in language modes (7, 8); pose and image-goal modes keep 75%, where pruning did not change driving error (token-pruning study on the Jetson: image goal -0.013, 20 s pose goal -0.004 action units vs unpruned, both not significant; deployed config vs bf16 in results/final_validation.md).

**Language grounding is more sensitive to compression than pose and image goals.** The same 75% pruning that left image-goal and pose-goal driving error unchanged cut object-goal accuracy by about 15 points. Fidelity (distance from bf16 actions) overstates the damage for pose and image goals, but a driving-error test on those modes would have understated it for language: each goal modality needs its own task-grounded test.

## Prompt-aware pruning (same deployed int4 weights)

Rule fixed before the run: adopt only if prompt-aware pruning keeps >= 81% (uniform 50% pruning) at a higher pruning rate than 50%, i.e. at 75%. Patches are scored by similarity to the object phrase in SigLIP's image-text space (deploy/prompt_prune.py); "original SigLIP" uses the released image tower instead of OmniVLA's fine-tuned one, as a reference for whether fine-tuning broke the image-text alignment.

| Pruning | Uniform grid | Prompt-aware (fine-tuned SigLIP) | Prompt-aware (original SigLIP) |
|---|---|---|---|
| 25% | 83% | 82% | - |
| 50% | 81% | 78% | - |
| 75% | 69% | 69% | 72% |

No pruning: 84%. Prompt-aware at 75%: 69% (vs uniform 75%: p=1) -> **not adopted** under the rule.

