# Language-goal test on CAST (behavioral instructions the deployed checkpoint was not trained on)

**What this test is:** CAST's instructions describe behaviors ("move along the corridor", "follow the dirt path, making a gentle left turn"). The deployed checkpoint, omnivla-original, was not trained on CAST or on instructions of this kind: its language training is LeLaN-style object goals ("move toward <object>"). The OmniVLA authors released a separate checkpoint, omnivla-finetuned-cast, for CAST-style instructions; it is not evaluated here. So this is a test of the base checkpoint outside its training, not of language mode as trained. The in-distribution object-goal test is results/lelan_summary.md.

237 of 240 samples reliable (moving >= 1 action unit in 8 steps), one per CAST episode, from 8 robot groups. Instructions are CAST's (the deployed checkpoint was not trained on CAST); the images come from GNM robots, and GNM is in OmniVLA's training mix. 128x128 images upscaled to 224x224. Open-loop single predictions. Error = mean distance to the driven waypoints over 8 steps, in action units.

int4 = the deployed weights (GPTQ int4 LLM, identical tensor for tensor to the Jetson weights; HQQ4 vision; 75% token pruning), run on a Kaggle T4 with fp16 dequantized weights: Kaggle-simulated, not the Marlin/GemLite kernels, not bit-exact with the Jetson (on the image-goal test the two differ by 0.003 action units on average). Language mode has not been run on the Jetson.

## Does the model use the image and the instruction?

| Model | Real | Blank image | Shuffled image | Shuffled instruction | Checks perception | Uses instruction |
|---|---|---|---|---|---|---|
| 7B fp16 | 2.492 | 2.420 (-3%, p=0.3) | 2.564 (+3%, p=0.2) | 2.502 (+0%, p=0.5) | no | no |
| 7B int4 (Kaggle-simulated) | 2.231 | 2.277 (+2%, p=0.03) | 2.274 (+2%, p=0.3) | 2.235 (+0%, p=0.4) | no | no |
| OmniVLA-edge | 2.188 | 2.848 (+30%, p=4e-08) | 2.381 (+9%, p=0.02) | 2.133 (-3%, p=0.01) | no | no |

**Result: this test does not show that any of these models follows the held-out instructions.** No model passes the instruction check (a shuffled instruction is not clearly worse). The 7B model also fails the perception check here, so its error on this test cannot rank configs. Treat the accuracy table below as descriptive only.

Turn direction under the same controls (added after the checks above; samples whose driven path ends >= 1 unit to the side; chance = 50%):

| Model | Real | Blank image | Shuffled image | Shuffled instruction |
|---|---|---|---|---|
| 7B fp16 | 67% (p=0.0001) | 41% (p=0.05) | 47% (p=0.5) | 60% (p=0.03) |
| 7B int4 (Kaggle-simulated) | 65% (p=0.0004) | 46% (p=0.3) | 54% (p=0.3) | 65% (p=0.0004) |
| OmniVLA-edge | 62% (p=0.004) | 63% (p=0.003) | 51% (p=0.8) | 65% (p=0.0008) |

136 samples; p: two-sided binomial test against 50%.

## Accuracy

| Model | Error, 8 steps [95% CI] | Steps 1-5 | Final step | Turn direction correct | vs bf16 (95% CI), p | Fidelity to bf16 |
|---|---|---|---|---|---|---|
| 7B bf16 | 2.492 [2.302, 2.690] | 1.689 | 4.255 | 67% of 136 | +0.000 (+0.000, +0.000), p=nan | 0.000 |
| 7B fp16 | 2.492 [2.298, 2.683] | 1.689 | 4.254 | 67% of 136 | +0.000 (-0.002, +0.002), p=0.29 | 0.019 |
| 7B int4 (Kaggle-simulated) | 2.231 [2.092, 2.373] | 1.464 | 4.351 | 65% of 136 | -0.261 (-0.406, -0.119), p=0.046 | 1.171 |
| OmniVLA-edge | 2.188 [2.018, 2.368] | 1.378 | 4.551 | 62% of 136 | -0.304 (-0.448, -0.166), p=0.0023 | - |

OmniVLA-edge minus 7B int4: -0.043 action units (95% CI -0.220, +0.131), p=0.21.

Model-free references: standing still 4.627; straight ahead at the test set's median speed (1.04 units per step, uses the ground truth of all samples) 1.862.

## By robot group (error, 8 steps)

| Robot group | n | 7B bf16 | 7B int4 | edge |
|---|---|---|---|---|
| Cory Hall | 17 | 2.258 | 2.238 | 2.100 |
| GoStanford-style (dated) | 45 | 3.312 | 3.201 | 2.211 |
| Jackal (SCAND/RECON) | 35 | 1.391 | 1.706 | 1.423 |
| SACSoN-style indoor (bww/soda) | 43 | 4.371 | 2.354 | 3.878 |
| Seattle | 7 | 1.254 | 1.721 | 1.370 |
| Spot (SCAND) | 13 | 0.692 | 0.754 | 0.949 |
| TartanDrive | 41 | 1.893 | 2.117 | 1.832 |
| sim*/no* scenes | 36 | 1.976 | 2.144 | 1.937 |
