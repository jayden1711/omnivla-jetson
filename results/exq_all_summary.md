# exq_all: img3m, 100 frames

| config | fid all8 | fid L0.43 | fid L1.05 | fid L1.43 | fid L2.2 | drive all8 | drive L1.05 | drive L2.2 |
|---|---|---|---|---|---|---|---|---|
| actq_act_g2 | 0.970 | 0.398 | 1.266 | 1.623 | 2.034 | 1.274 | 1.556 | 2.427 |
| actq_act_pc4 | 0.784 | 0.376 | 0.996 | 1.295 | 1.629 | 1.260 | 1.573 | 2.307 |
| actq_exec_deploy_g2 | 1.063 | 0.481 | 1.394 | 1.736 | 2.108 | 1.276 | 1.583 | 2.414 |
| actq_exec_deploy_pc4 | 0.686 | 0.318 | 0.867 | 1.147 | 1.469 | 1.275 | 1.582 | 2.288 |
| actq_exec_nf4_g2 | 0.953 | 0.368 | 1.240 | 1.635 | 2.085 | 1.270 | 1.540 | 2.418 |
| actq_exec_nf4_pc4 | 0.755 | 0.370 | 0.954 | 1.233 | 1.552 | 1.264 | 1.580 | 2.293 |
| gptq_act_g2 | 1.104 | 0.556 | 1.362 | 1.779 | 2.332 | 1.344 | 1.676 | 2.568 |
| gptq_act_pc4 | 0.530 | 0.268 | 0.674 | 0.837 | 1.022 | 1.294 | 1.613 | 2.220 |
| gptq_exec_deploy_g2 | 0.966 | 0.495 | 1.191 | 1.550 | 2.030 | 1.201 | 1.466 | 2.326 |
| gptq_exec_deploy_pc4 | 0.546 | 0.281 | 0.689 | 0.860 | 1.054 | 1.306 | 1.643 | 2.255 |
| gptq_exec_nf4_g2 | 1.139 | 0.585 | 1.393 | 1.819 | 2.413 | 1.348 | 1.636 | 2.709 |
| gptq_exec_nf4_pc4 | 0.515 | 0.262 | 0.658 | 0.808 | 0.970 | 1.304 | 1.627 | 2.240 |
| gptq_plain_g2 | 1.107 | 0.582 | 1.365 | 1.759 | 2.261 | 1.235 | 1.521 | 2.437 |
| gptq_plain_pc4 | 0.479 | 0.241 | 0.599 | 0.757 | 0.954 | 1.314 | 1.641 | 2.235 |
| hqq_plain_g2 | 2.245 | 0.850 | 2.857 | 3.650 | 4.482 | 2.116 | 2.549 | 4.272 |
| none_plain_fp16 | 0.494 | 0.259 | 0.622 | 0.775 | 0.952 | 1.297 | 1.622 | 2.222 |
| rtn_plain_pc4 | 1.087 | 0.520 | 1.376 | 1.790 | 2.229 | 1.339 | 1.708 | 2.538 |

## Paired comparisons (extension - baseline; negative = extension better)

| extension | baseline | metric | diff | 95% CI | p |
|---|---|---|---|---|---|
| gptq_plain_g2 | hqq_plain_g2 | fid_all8 | -1.1376 | (-1.2605, -1.0075) | 6.7e-18 |
| gptq_plain_g2 | hqq_plain_g2 | fid_L0.43 | -0.2676 | (-0.3428, -0.1944) | 3.3e-09 |
| gptq_plain_g2 | hqq_plain_g2 | fid_L1.05 | -1.4919 | (-1.6754, -1.2856) | 5.8e-17 |
| gptq_plain_g2 | hqq_plain_g2 | fid_L1.43 | -1.8913 | (-2.1276, -1.6576) | 3.2e-17 |
| gptq_plain_g2 | hqq_plain_g2 | fid_L2.2 | -2.2216 | (-2.4874, -1.9499) | 3.5e-17 |
| gptq_plain_g2 | hqq_plain_g2 | drive_all8 | -0.8811 | (-1.0288, -0.7278) | 4.1e-14 |
| gptq_plain_g2 | hqq_plain_g2 | drive_L1.05 | -1.0285 | (-1.2670, -0.7888) | 1e-10 |
| gptq_plain_g2 | hqq_plain_g2 | drive_L2.2 | -1.8349 | (-2.1541, -1.4987) | 9e-14 |
| gptq_act_g2 | gptq_plain_g2 | fid_all8 | -0.0032 | (-0.0936, +0.0835) | 0.65 |
| gptq_act_g2 | gptq_plain_g2 | fid_L0.43 | -0.0263 | (-0.0803, +0.0268) | 0.35 |
| gptq_act_g2 | gptq_plain_g2 | fid_L1.05 | -0.0034 | (-0.1310, +0.1269) | 0.64 |
| gptq_act_g2 | gptq_plain_g2 | fid_L1.43 | +0.0199 | (-0.1279, +0.1695) | 0.47 |
| gptq_act_g2 | gptq_plain_g2 | fid_L2.2 | +0.0718 | (-0.1138, +0.2598) | 0.33 |
| gptq_act_g2 | gptq_plain_g2 | drive_all8 | +0.1092 | (+0.0258, +0.1893) | 0.0065 |
| gptq_act_g2 | gptq_plain_g2 | drive_L1.05 | +0.1549 | (+0.0347, +0.2709) | 0.0059 |
| gptq_act_g2 | gptq_plain_g2 | drive_L2.2 | +0.1310 | (-0.0583, +0.3183) | 0.26 |
| gptq_exec_deploy_g2 | gptq_act_g2 | fid_all8 | -0.1383 | (-0.2139, -0.0595) | 0.001 |
| gptq_exec_deploy_g2 | gptq_act_g2 | fid_L0.43 | -0.0602 | (-0.1042, -0.0143) | 0.017 |
| gptq_exec_deploy_g2 | gptq_act_g2 | fid_L1.05 | -0.1709 | (-0.2823, -0.0604) | 0.0039 |
| gptq_exec_deploy_g2 | gptq_act_g2 | fid_L1.43 | -0.2292 | (-0.3580, -0.1047) | 0.00084 |
| gptq_exec_deploy_g2 | gptq_act_g2 | fid_L2.2 | -0.3024 | (-0.4622, -0.1471) | 0.00054 |
| gptq_exec_deploy_g2 | gptq_act_g2 | drive_all8 | -0.1430 | (-0.2200, -0.0684) | 0.0013 |
| gptq_exec_deploy_g2 | gptq_act_g2 | drive_L1.05 | -0.2097 | (-0.3245, -0.0978) | 0.0017 |
| gptq_exec_deploy_g2 | gptq_act_g2 | drive_L2.2 | -0.2418 | (-0.4120, -0.0862) | 0.016 |
| actq_act_g2 | hqq_plain_g2 | fid_all8 | -1.2751 | (-1.4034, -1.1497) | 6.5e-18 |
| actq_act_g2 | hqq_plain_g2 | fid_L0.43 | -0.4521 | (-0.5227, -0.3834) | 2.8e-16 |
| actq_act_g2 | hqq_plain_g2 | fid_L1.05 | -1.5917 | (-1.7746, -1.4018) | 1.6e-17 |
| actq_act_g2 | hqq_plain_g2 | fid_L1.43 | -2.0272 | (-2.2544, -1.8088) | 1.3e-17 |
| actq_act_g2 | hqq_plain_g2 | fid_L2.2 | -2.4476 | (-2.7236, -2.1621) | 1.9e-17 |
| actq_act_g2 | hqq_plain_g2 | drive_all8 | -0.8417 | (-1.0093, -0.6625) | 5.6e-12 |
| actq_act_g2 | hqq_plain_g2 | drive_L1.05 | -0.9935 | (-1.2564, -0.7374) | 2.4e-09 |
| actq_act_g2 | hqq_plain_g2 | drive_L2.2 | -1.8449 | (-2.2078, -1.4686) | 4.7e-12 |
| actq_exec_deploy_g2 | actq_act_g2 | fid_all8 | +0.0935 | (+0.0561, +0.1318) | 2.5e-05 |
| actq_exec_deploy_g2 | actq_act_g2 | fid_L0.43 | +0.0840 | (+0.0574, +0.1113) | 3.2e-08 |
| actq_exec_deploy_g2 | actq_act_g2 | fid_L1.05 | +0.1288 | (+0.0758, +0.1840) | 2.8e-05 |
| actq_exec_deploy_g2 | actq_act_g2 | fid_L1.43 | +0.1129 | (+0.0531, +0.1712) | 0.00068 |
| actq_exec_deploy_g2 | actq_act_g2 | fid_L2.2 | +0.0739 | (+0.0064, +0.1448) | 0.025 |
| actq_exec_deploy_g2 | actq_act_g2 | drive_all8 | +0.0013 | (-0.0378, +0.0416) | 0.88 |
| actq_exec_deploy_g2 | actq_act_g2 | drive_L1.05 | +0.0272 | (-0.0309, +0.0836) | 0.36 |
| actq_exec_deploy_g2 | actq_act_g2 | drive_L2.2 | -0.0131 | (-0.0870, +0.0621) | 0.71 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | fid_all8 | -0.0150 | (-0.0491, +0.0183) | 0.22 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | fid_L0.43 | -0.0058 | (-0.0248, +0.0143) | 0.39 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | fid_L1.05 | -0.0162 | (-0.0622, +0.0303) | 0.23 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | fid_L1.43 | -0.0290 | (-0.0834, +0.0287) | 0.14 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | fid_L2.2 | -0.0523 | (-0.1161, +0.0154) | 0.084 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | drive_all8 | +0.0105 | (-0.0203, +0.0424) | 0.35 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | drive_L1.05 | +0.0146 | (-0.0302, +0.0596) | 0.29 |
| gptq_exec_nf4_pc4 | gptq_act_pc4 | drive_L2.2 | +0.0206 | (-0.0499, +0.0954) | 0.38 |
| actq_exec_nf4_pc4 | actq_act_pc4 | fid_all8 | -0.0293 | (-0.0496, -0.0092) | 0.0027 |
| actq_exec_nf4_pc4 | actq_act_pc4 | fid_L0.43 | -0.0059 | (-0.0163, +0.0050) | 0.061 |
| actq_exec_nf4_pc4 | actq_act_pc4 | fid_L1.05 | -0.0422 | (-0.0697, -0.0151) | 0.0014 |
| actq_exec_nf4_pc4 | actq_act_pc4 | fid_L1.43 | -0.0617 | (-0.0939, -0.0283) | 0.00062 |
| actq_exec_nf4_pc4 | actq_act_pc4 | fid_L2.2 | -0.0767 | (-0.1187, -0.0362) | 0.0011 |
| actq_exec_nf4_pc4 | actq_act_pc4 | drive_all8 | +0.0041 | (-0.0116, +0.0207) | 0.79 |
| actq_exec_nf4_pc4 | actq_act_pc4 | drive_L1.05 | +0.0077 | (-0.0164, +0.0310) | 0.52 |
| actq_exec_nf4_pc4 | actq_act_pc4 | drive_L2.2 | -0.0140 | (-0.0544, +0.0266) | 0.51 |
| gptq_exec_nf4_g2 | gptq_act_g2 | fid_all8 | +0.0344 | (-0.0378, +0.1072) | 0.34 |
| gptq_exec_nf4_g2 | gptq_act_g2 | fid_L0.43 | +0.0295 | (-0.0102, +0.0713) | 0.15 |
| gptq_exec_nf4_g2 | gptq_act_g2 | fid_L1.05 | +0.0312 | (-0.0768, +0.1336) | 0.61 |
| gptq_exec_nf4_g2 | gptq_act_g2 | fid_L1.43 | +0.0406 | (-0.0788, +0.1564) | 0.5 |
| gptq_exec_nf4_g2 | gptq_act_g2 | fid_L2.2 | +0.0808 | (-0.0667, +0.2295) | 0.25 |
| gptq_exec_nf4_g2 | gptq_act_g2 | drive_all8 | +0.0035 | (-0.0599, +0.0683) | 0.92 |
| gptq_exec_nf4_g2 | gptq_act_g2 | drive_L1.05 | -0.0396 | (-0.1429, +0.0605) | 0.49 |
| gptq_exec_nf4_g2 | gptq_act_g2 | drive_L2.2 | +0.1418 | (+0.0038, +0.2842) | 0.034 |
| actq_exec_nf4_g2 | actq_act_g2 | fid_all8 | -0.0172 | (-0.0509, +0.0167) | 0.33 |
| actq_exec_nf4_g2 | actq_act_g2 | fid_L0.43 | -0.0297 | (-0.0481, -0.0113) | 0.0011 |
| actq_exec_nf4_g2 | actq_act_g2 | fid_L1.05 | -0.0252 | (-0.0757, +0.0233) | 0.29 |
| actq_exec_nf4_g2 | actq_act_g2 | fid_L1.43 | +0.0120 | (-0.0439, +0.0705) | 0.72 |
| actq_exec_nf4_g2 | actq_act_g2 | fid_L2.2 | +0.0505 | (-0.0177, +0.1238) | 0.2 |
| actq_exec_nf4_g2 | actq_act_g2 | drive_all8 | -0.0040 | (-0.0336, +0.0241) | 0.91 |
| actq_exec_nf4_g2 | actq_act_g2 | drive_L1.05 | -0.0154 | (-0.0607, +0.0304) | 0.75 |
| actq_exec_nf4_g2 | actq_act_g2 | drive_L2.2 | -0.0091 | (-0.0795, +0.0600) | 0.63 |
