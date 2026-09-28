# Final validation of the deployed config (2026-09-26)
Marlin per-channel int4 LLM + GemLite HQQ4 vision + elision + uniform-grid pruning of 75% of current-image tokens,
pre-packed weights, deploy/omnivla_deploy.py on the Jetson Orin Nano. img3m image-goal test (100 reliable frames) and
pose mode on the same frames. Single-prediction, in-distribution (FrodoBots) evaluation. Fresh tegrastats log per run.

- frames (img3m): 100
- max |d| vs research pipeline pq_marpc@6@spatial75: 0.0
- pose latency median fwd / e2e ms: 425 / 434  (NF4 1416)
- image latency median fwd / e2e ms: 1050 / 1061  (NF4 2146)
- weights GiB: 4.14
- torch peak GiB pose / image: 4.28 / 4.49
- RAM peak MB pose / image: 6543 / 6736
- RAM headroom MB pose / image (+-0.2 GB): 1077 / 884  (NF4 827 / 575)
- fidelity vs bf16 [95% CI]: 1.087 [0.984, 1.195]  (NF4 0.299)
- driving error (all 8 steps): 1.339  (NF4 1.317, bf16 1.330)
- driving vs NF4 [95% CI], p: +0.022 [-0.119, 0.168], p=0.93
- driving vs bf16 [95% CI], p: +0.009 [-0.138, 0.157], p=0.97
