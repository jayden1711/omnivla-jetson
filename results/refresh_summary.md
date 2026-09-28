# Goal-cache refresh study (deploy runtime, GPTQ default weights; 24 sequential clips)

Reference: the same runtime with goal_refresh=1 (goal vision features cached = exact; no K/V reuse).

## Error vs cache age (no refresh within a clip; strides 5/10/21 pooled)

| age bin (s) | frames | err vs uncached (all 8) | err (exec @1.05 s) | drive diff vs uncached [95% CI] | p | rule ok |
|---|---|---|---|---|---|---|
| (0, 1] | 144 | 0.105 | 0.138 | +0.020 [+0.008, +0.033] | 0.0012 | NO |
| (1, 2] | 168 | 0.135 | 0.177 | +0.028 [+0.011, +0.044] | 0.00032 | NO |
| (2, 3] | 168 | 0.156 | 0.202 | -0.007 [-0.025, +0.012] | 0.86 | yes |
| (3, 4] | 168 | 0.177 | 0.232 | +0.005 [-0.018, +0.029] | 0.93 | yes |
| (4, 5] | 168 | 0.215 | 0.283 | +0.035 [+0.007, +0.062] | 0.019 | NO |

## Refresh interval at the deployed image-goal cadence (one prediction per 1.05 s, 120 frames)

| goal_refresh N | K/V reused | latency median / mean ms | err vs uncached (reuse frames) | drive diff vs uncached (all frames) [95% CI] | p |
|---|---|---|---|---|---|
| 1 | 0% | 863 / 897 | 0.000 | +0.000 [+0.000, +0.000] | nan |
| 2 | 40% | 876 / 740 | 0.112 | +0.002 [-0.009, +0.013] | 0.33 |
| 3 | 60% | 444 / 652 | 0.125 | +0.020 [+0.005, +0.036] | 0.0019 |
| 4 | 60% | 443 / 652 | 0.141 | +0.008 [-0.009, +0.025] | 0.12 |
| goal change only (>=5 here) | 80% | 444 / 567 | 0.155 | +0.014 [-0.006, +0.036] | 0.091 |

**Decision:** largest age with every bin passing = 0 s -> goal_refresh N = floor(0 / 1.05) + 1 = **1** (max cache age at 1.05 s cadence = 0.00 s). Clips are 5 s, so ages beyond 5 s are untested.
