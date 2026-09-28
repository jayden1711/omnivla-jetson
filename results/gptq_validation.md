# GPTQ per-channel int4 weights: Jetson validation (2026-09-27)

Same format/kernel as the deployed RTN weights (Marlin per-channel int4 LLM + HQQ4 GemLite vision + elision + grid 75%); deploy runtime; img3m 100 frames + pose.

- frames (img3m): 100
- Jetson vs Kaggle outputs, same weights: GPTQ / RTN (different kernels: Marlin+GemLite vs fp16 fake-quant): 0.0030 (max 0.010) / 0.0040 (max 0.011)
- fidelity vs bf16 [95% CI]: GPTQ / RTN / NF4: 0.478 [0.424, 0.533] / 1.087 / 0.299; GPTQ - RTN -0.609 [-0.72, -0.501], p=8.9e-15
- driving error: GPTQ / RTN / NF4 / bf16: 1.314 / 1.339 / 1.317 / 1.330
- driving GPTQ - RTN [95% CI], p: -0.026 [-0.158, 0.105], p=0.67
- driving GPTQ - NF4 [95% CI], p: -0.003 [-0.087, 0.083], p=0.76
- driving GPTQ - bf16 [95% CI], p: -0.016 [-0.094, 0.062], p=0.40
- latency median fwd ms, image / pose: GPTQ vs RTN: 1056 / 424 vs 1050 / 425
- RAM peak MB, image / pose: GPTQ vs RTN: 6769 / 6565 vs 6736 / 6543; headroom GPTQ 851 / 1055 MB
- torch weights GiB: GPTQ / RTN: 4.14 / 4.14

## Pass criteria

- PASS: weights bit-exact vs Kaggle (224 layers)
- PASS: Marlin gate (3 shapes x 3 T, 50x determinism)
- PASS: Jetson-vs-Kaggle agreement no worse than RTN
- PASS: fidelity to bf16 lower than RTN
- PASS: driving not significantly worse than RTN
- PASS: driving not significantly worse than NF4
- PASS: driving not significantly worse than bf16
- PASS: RAM peak within +-0.2 GB of RTN
- PASS: latency within +-3% of RTN

**ALL PASS -> make GPTQ the deployed default**
