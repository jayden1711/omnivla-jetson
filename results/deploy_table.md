# Deployable configurations on the Jetson Orin Nano (2026-09-26; final loader, fresh tegrastats log per run)
RAM peak out of 7620 MB; run-to-run variation of RAM peaks is about +-0.2 GB. Latency = median forward (s), elision on.
mode 4 = pose goal, mode 6 = image goal. n = frames per run.

| Config | Weights GiB | Pose ms | Torch peak 4 | RAM peak 4 | Headroom 4 | Image ms | Torch peak 6 | RAM peak 6 | Headroom 6 | n |
|---|---|---|---|---|---|---|---|---|---|---|
| NF4 (bnb) | 4.19 | 1416 | 4.56 | 6793 | 827 | 2146 | 4.76 | 7045 | 575 | 30 |
| mixc2 (HQQ portable) | 3.69 | 3478 | 4.21 | 6324 | 1296 | 4317 | 4.41 | 6525 | 1095 | 30 |
| q2lora (HQQ portable) | 3.51 | 3223 | 4.04 | 6201 | 1419 | 4057 | 4.23 | 6379 | 1241 | 30 |
| gl_q2lora (GemLite) | 3.51 | 1402 | 3.82 | 6234 | 1386 | 2460 | 4.02 | 6606 | 1014 | 30 |
| marpc (Marlin per-channel LLM + GemLite vision) | 4.14 | 754 | 4.43 | 6690 | 930 | 1391 | 4.64 | 6990 | 630 | 100 |
| gl_q2lora_sf (+ SigLIP MLP fp16) | 3.87 | 1178 | 4.17 | 6649 | 971 | 2358 | 4.38 | 7033 | 587 | 100 |
| marpc_sf (+ SigLIP MLP fp16) | 4.51 | 629 | 4.80 | 7106 | 514 | 1275 | 5.00 | 7198 | 422 | 100 |
marpc_sf@6: one CUDA OOM on the first attempt, completed on retry -> marginal.
