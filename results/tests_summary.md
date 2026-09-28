Frames: img3m 100, pose20 93, pose5 100 (reliable 2.4 s ground-truth windows).

## img3m
| Control / baseline | Error vs actual path (all 8) | Steps 1-5 | vs real fp16 model | p |
|---|---|---|---|---|
| real model (fp16) | 1.327 | 0.962 | - | - |
| blind: blank current image | 3.120 | 1.999 | +135% | 3.4e-11 |
| blind: shuffled current image | 1.714 | 1.091 | +29% | 0.0013 |
| model-free: straight ahead at current speed | 0.659 | 0.386 | -50% | 3.5e-11 |
| model-free: straight line to goal (oracle goal pose) | 0.605 | 0.472 | -54% | 2.9e-11 |
| model-free: goal interpolation (oracle goal pose) | 0.441 | 0.335 | -67% | 4.3e-15 |
**img3m: PRIMARY** (blind controls clearly worse: [np.True_, np.True_]).

| Config | Fidelity vs bf16 all 8 | Fid. 1-5 | Drive all 8 | Drive 1-5 | Drive vs bf16 (95% CI) | p |
|---|---|---|---|---|---|---|
| bf16 | 0.000 | 0.000 | 1.330 | 0.964 | +0.000 (+0.000, +0.000) | nan |
| fp16 | 0.018 | 0.010 | 1.327 | 0.962 | -0.003 (-0.006, +0.000) | 0.034 |
| nf4_all | 0.298 | 0.169 | 1.318 | 0.954 | -0.012 (-0.065, +0.046) | 0.39 |
| nf4llm_visfp16 | 0.262 | 0.145 | 1.306 | 0.957 | -0.024 (-0.070, +0.023) | 0.17 |
| hqq8_vis4 | 0.178 | 0.109 | 1.329 | 0.949 | -0.001 (-0.037, +0.036) | 0.65 |
| hqq8_visfp16 | 0.028 | 0.016 | 1.328 | 0.963 | -0.002 (-0.006, +0.002) | 0.13 |
| hqq4_vis4 | 0.319 | 0.198 | 1.335 | 0.973 | +0.005 (-0.049, +0.063) | 0.77 |
| hqq4_visfp16 | 0.312 | 0.199 | 1.357 | 1.010 | +0.027 (-0.027, +0.084) | 0.63 |
| hqq3_vis4 | 0.840 | 0.422 | 1.301 | 0.888 | -0.029 (-0.148, +0.083) | 0.73 |
| hqq3_visfp16 | 0.848 | 0.423 | 1.298 | 0.881 | -0.032 (-0.145, +0.085) | 0.63 |
| hqq2_vis4 | 2.326 | 1.370 | 2.212 | 1.343 | +0.882 (+0.615, +1.124) | 7.5e-09 |
| hqq2_visfp16 | 2.339 | 1.375 | 2.222 | 1.346 | +0.892 (+0.645, +1.132) | 6.6e-09 |

## pose20
| Control / baseline | Error vs actual path (all 8) | Steps 1-5 | vs real fp16 model | p |
|---|---|---|---|---|
| real model (fp16) | 1.609 | 1.141 | - | - |
| blind: blank current image | 1.475 | 0.928 | -8% | 0.75 |
| blind: shuffled current image | 1.732 | 1.196 | +8% | 0.054 |
| model-free: straight ahead at current speed | 0.668 | 0.390 | -58% | 1.1e-12 |
| model-free: straight line to goal | 1.610 | 1.103 | +0% | 0.92 |
| model-free: goal interpolation | 1.019 | 0.681 | -37% | 1.2e-05 |
**pose20: NOT primary** (blind controls clearly worse: [np.False_, np.False_]).

| Config | Fidelity vs bf16 all 8 | Fid. 1-5 | Drive all 8 | Drive 1-5 | Drive vs bf16 (95% CI) | p |
|---|---|---|---|---|---|---|
| bf16 | 0.000 | 0.000 | 1.610 | 1.141 | +0.000 (+0.000, +0.000) | nan |
| fp16 | 0.014 | 0.008 | 1.609 | 1.141 | -0.001 (-0.003, +0.001) | 0.19 |
| nf4_all | 0.215 | 0.114 | 1.541 | 1.112 | -0.069 (-0.102, -0.038) | 6.8e-05 |
| nf4llm_visfp16 | 0.176 | 0.096 | 1.576 | 1.135 | -0.034 (-0.063, -0.002) | 0.0062 |
| hqq8_vis4 | 0.113 | 0.062 | 1.600 | 1.132 | -0.010 (-0.032, +0.011) | 0.56 |
| hqq8_visfp16 | 0.027 | 0.014 | 1.603 | 1.136 | -0.007 (-0.011, -0.003) | 0.00035 |
| hqq4_vis4 | 0.209 | 0.104 | 1.565 | 1.130 | -0.045 (-0.076, -0.013) | 0.0048 |
| hqq4_visfp16 | 0.184 | 0.088 | 1.567 | 1.136 | -0.043 (-0.068, -0.019) | 0.00032 |
| hqq3_vis4 | 0.558 | 0.280 | 1.395 | 1.080 | -0.215 (-0.291, -0.138) | 3.6e-06 |
| hqq3_visfp16 | 0.550 | 0.272 | 1.390 | 1.073 | -0.220 (-0.299, -0.140) | 1.4e-06 |
| hqq2_vis4 | 2.244 | 1.248 | 1.819 | 1.162 | +0.210 (-0.065, +0.480) | 0.11 |
| hqq2_visfp16 | 2.282 | 1.271 | 1.860 | 1.182 | +0.251 (-0.031, +0.531) | 0.074 |

## pose5
| Control / baseline | Error vs actual path (all 8) | Steps 1-5 | vs real fp16 model | p |
|---|---|---|---|---|
| real model (fp16) | 1.296 | 0.936 | - | - |
| blind: blank current image | 1.676 | 1.006 | +29% | 0.0023 |
| blind: shuffled current image | 1.396 | 1.003 | +8% | 0.27 |
| model-free: straight ahead at current speed | 0.659 | 0.386 | -49% | 1e-10 |
| model-free: straight line to goal | 0.705 | 0.533 | -46% | 1e-07 |
| model-free: goal interpolation | 0.491 | 0.361 | -62% | 7.5e-14 |
**pose5: NOT primary** (blind controls clearly worse: [np.True_, np.False_]).

| Config | Fidelity vs bf16 all 8 | Fid. 1-5 | Drive all 8 | Drive 1-5 | Drive vs bf16 (95% CI) | p |
|---|---|---|---|---|---|---|
| bf16 | 0.000 | 0.000 | 1.298 | 0.936 | +0.000 (+0.000, +0.000) | nan |
| fp16 | 0.014 | 0.009 | 1.296 | 0.936 | -0.002 (-0.003, +0.000) | 0.065 |
| nf4_all | 0.214 | 0.124 | 1.299 | 0.935 | +0.001 (-0.040, +0.041) | 0.91 |
| nf4llm_visfp16 | 0.147 | 0.088 | 1.263 | 0.912 | -0.035 (-0.060, -0.010) | 0.0029 |
| hqq8_vis4 | 0.116 | 0.075 | 1.301 | 0.940 | +0.004 (-0.020, +0.026) | 0.17 |
| hqq8_visfp16 | 0.018 | 0.011 | 1.296 | 0.937 | -0.001 (-0.004, +0.001) | 0.37 |
| hqq4_vis4 | 0.207 | 0.122 | 1.304 | 0.957 | +0.007 (-0.030, +0.042) | 0.36 |
| hqq4_visfp16 | 0.170 | 0.099 | 1.298 | 0.955 | +0.001 (-0.029, +0.029) | 0.75 |
| hqq3_vis4 | 0.561 | 0.244 | 1.249 | 0.934 | -0.049 (-0.132, +0.035) | 0.27 |
| hqq3_visfp16 | 0.562 | 0.239 | 1.238 | 0.928 | -0.060 (-0.144, +0.021) | 0.26 |
| hqq2_vis4 | 2.138 | 1.180 | 2.008 | 1.294 | +0.710 (+0.480, +0.944) | 3.1e-07 |
| hqq2_visfp16 | 2.143 | 1.191 | 2.014 | 1.306 | +0.717 (+0.485, +0.942) | 3.5e-07 |
