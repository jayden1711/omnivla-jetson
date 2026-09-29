# Vision linears as Marlin per-channel int4: accuracy (Kaggle T4, fp16 fake-quant)

Same run and same deployed int4 LLM (GPTQ, hash-checked against the deployment) for every row; only the vision linears change. Baseline = the deployed HQQ4 vision (group 64). Marlin rows are per-channel int4 (groupsize -1; grouped Marlin is wrong on sm_87), simulated with fp16 dequantized weights. Image goal: img3m, 100 frames, 75% token pruning (as deployed). Object goal: LeLaN, 210 frames, no pruning (as deployed in language modes).

Rules fixed before the run: a variant fails if image-goal driving error is higher than the baseline with Wilcoxon p < 0.05, or if object-goal accuracy is lower with McNemar p < 0.05 or by more than 5 points.

| Vision weights | Image goal: error vs driven path | vs baseline (p) | Fidelity to bf16 | vs baseline outputs | Object goal: picks named object | vs baseline (p) | Pass |
|---|---|---|---|---|---|---|---|
| HQQ4 group 64 (deployed baseline) | 1.314 | - | 0.479 | 0.000 | 84% | - | - |
| RTN, all 204 vision linears | 1.246 | -0.068 (p=0.075) | 0.573 | 0.416 | 79% | -5.2 pts (p=0.0074) | NO |
| GPTQ, all 204 vision linears | 1.301 | -0.013 (p=0.32) | 0.536 | 0.229 | 82% | -1.9 pts (p=0.22) | yes |
| RTN, SigLIP MLP only (54) | 1.268 | -0.046 (p=0.0096) | 0.555 | 0.230 | 84% | -0.5 pts (p=1) | yes |

**Decision (rule above): adopt GPTQ, all 204 vision linears.**

The export for the Jetson (`LANG_TEST=visgptqx`, vision-only rerun of the same GPTQ calibration) uses the per-channel
scales GPTQ itself used. The first export recomputed them from the original weights and failed its own grid check:
GPTQ zeroes input columns that are never active in the calibration data before it sets the scale, which changes the
scale of a few SigLIP rows. The evaluated weights were not affected.

## Jetson validation (2026-09-28/29, one boot, MAXN SUPER; eval/jetson/vism_validate.sh, eval/jetson/vis_marlin_gate.py)

Marlin needs k % 128 == 0 and n % 256 == 0. DINOv2 fits natively. All 108 SigLIP linears are zero-padded (qkv
1152->3456 as 1152->3584, proj 1152->1152 as 1152->1280, MLP fc1 1152->4304 as 1152->4352, fc2 4304->1152 as
4352->1280): padded weights are exactly 0 and meet zero-padded inputs, padded outputs are sliced off.

- **Weights:** 204/204 layers bit-exact vs the Kaggle export (Marlin(I) == exported W^T; padded weights exactly 0).
- **Correctness vs dequantize-then-matmul (fp32), real token counts (one and two images):** relative mean error
  1.95-3.45e-4 for the padded SigLIP layers, 1.99-3.37e-4 for the unpadded DINOv2 layers (fp16 cuBLAS on the same
  weights: 1.8-2.8e-4). Padded and unpadded layers are in the same range.
- **Determinism:** 50/50 bit-identical repeats per layer and token count (32 cases); end to end, 50 fresh processes
  bit-identical on all 11 image-goal outputs (goal-cache hit and miss).
- **Padding vs no padding** (DINOv2 layers repacked with padding): not bit-identical (max difference 0.0078 = one fp16
  step at those magnitudes): Marlin splits the work differently. Not required: the SigLIP shapes cannot run unpadded.
- **Same weights as evaluated:** Jetson outputs vs the Kaggle-simulated GPTQ-vision outputs on img3m: 0.0032 action
  units on average (HQQ4 vision, same comparison: 0.0030).
- **Reference check:** re-recorded with Marlin vision, then 40/40 bit-identical (modes 4, 6, 7, 8) with the default
  runtime. The HQQ4 path is unchanged by the runtime edit: 40/40 bit-identical to the previous references first.

| Same boot | Marlin vision (default) | HQQ4 vision (`OMNIVLA_VISION=hqq4`) |
|---|---|---|
| Pose goal (prediction) | 271 ms | 374 ms |
| Image goal, new goal every frame | 775 ms | 981 ms |
| Image goal, goal unchanged | 711 ms | 814 ms |
| Language (7 / 8) | 549-564 ms | 663-672 ms |
| Vision per encoded image (DINOv2 + SigLIP) | 65 ms (27 + 37) | 168 ms (47 + 120) |
| RAM peak, validation runs pose / image (of 7620 MB) | 6272 / 6286 MB | 6346 / 6351 MB |
| Torch weights | 4.134 GiB | 4.145 GiB |
| Image-goal driving error (100 frames) | 1.301 | 1.314 |
| Fidelity to bf16 | 0.535 | 0.478 |

Marlin vision vs HQQ4: driving -0.013 (Wilcoxon p = 0.33); vs NF4 (1.317) p = 0.41; vs bf16 (1.330) p = 0.17.
10-minute soak (pose goal, 2129 predictions): mean 271 ms throughout, worst call 303 ms, GPU at most 74 C, torch memory
flat, process RSS +73 MB (the known heap retention; restart the node between runs).
