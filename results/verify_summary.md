Reference: bf16cpu. 20 frames (one per ride) x modes 7/4/8, elision ON. Error in action units (dataset-normalized waypoint spacings); mean +- std over frame-mode pairs.

| Condition | n | Error, all 8 steps | 95% CI | Error, steps 1-5 | Rel. % | mode 7 | mode 4 | mode 8 |
|---|---|---|---|---|---|---|---|---|
| Kaggle 4-bit vision+LLM (fp16 compute) | 60 | 0.288 +- 0.209 | +-0.053 | 0.191 +- 0.138 | 7.6 | 0.244 | 0.290 | 0.330 |
| Kaggle 4-bit NF4, Jetson config (bf16 compute) | 60 | 0.286 +- 0.209 | +-0.053 | 0.191 +- 0.139 | 7.5 | 0.244 | 0.288 | 0.325 |
| Jetson 4-bit NF4 (vision+LLM, bf16 compute) | 60 | 0.284 +- 0.209 | +-0.053 | 0.189 +- 0.139 | 7.4 | 0.242 | 0.285 | 0.325 |
| (a) LLM 4-bit, vision fp16 | 60 | 0.230 +- 0.154 | +-0.039 | 0.157 +- 0.103 | 5.8 | 0.205 | 0.219 | 0.267 |
| (b) vision 4-bit, LLM fp16 | 60 | 0.176 +- 0.141 | +-0.036 | 0.115 +- 0.096 | 4.9 | 0.190 | 0.167 | 0.169 |
| fp16 full model (GPU) | 60 | 0.014 +- 0.006 | +-0.002 | 0.009 +- 0.004 | 0.3 | 0.012 | 0.017 | 0.012 |

Pairwise checks:

| comparison                               |   n |    err8 |   err8_std |    err5 |   max_abs_diff |   identical_pct |
|:-----------------------------------------|----:|--------:|-----------:|--------:|---------------:|----------------:|
| platform control: Kaggle vs Jetson 4-bit |  60 | 0.01296 |    0.00668 | 0.00884 |        0.0625  |               0 |
| determinism: fp16 run 2 vs run 1         |  60 | 0       |    0       | 0       |        0       |             100 |
| 4-bit compute dtype: fp16 vs bf16        |  60 | 0.01399 |    0.0065  | 0.00945 |        0.07812 |               0 |
