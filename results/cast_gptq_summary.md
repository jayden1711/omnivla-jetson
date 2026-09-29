# omnivla-finetuned-cast: GPTQ int4 on held-out CAST episodes (Kaggle T4, fp16 fake-quant)

175 reliable held-out samples (148 recordings); GPTQ calibrated on 64 samples from 58 other recordings (split by recording: by recording (across sources), round-robin over sources, seed 0). Language mode, no token pruning. int4 = GPTQ per-channel int4 (Marlin format) for the LLM and the vision linears; simulated with fp16 dequantized weights, not the Jetson's kernels. In-distribution: omnivla-finetuned-cast was trained on CAST.

| Model | Error (8 steps) | Blank image | Shuffled image | Shuffled instruction | Checks perception | Uses instruction | Turn direction correct |
|---|---|---|---|---|---|---|---|
| bf16 | 1.396 | +64% (p=2.8e-15) | +90% (p=6.5e-21) | -4% (p=0.022) | yes | no | 91% of 98 |
| GPTQ int4 | 1.433 | +95% (p=3.1e-20) | +82% (p=1.9e-18) | -2% (p=0.46) | yes | no | 87% of 98 |

int4 vs bf16: error +0.037 action units (Wilcoxon p=0.35); fidelity to bf16 0.546; turn direction -4.1 pts (McNemar p=0.29).

Rules: (a) error not significantly higher: pass; (b) instruction use kept: pass (bf16 does not pass the instruction check: rule not applicable); (c) turn direction kept: pass.

**Result: no significant accuracy loss from GPTQ int4.**

## On the Jetson (deploy/weights_cast, 2026-09-29; eval/jetson/cast_check.py)

Weights folder built by `deploy/tools/build_cast_weights.py` from the Kaggle export and the checkpoint's non-quantized
tensors (fetched by range requests, `deploy/tools/fetch_ckpt_subset.py`; NHirose/omnivla-finetuned-cast @ 7d3744a, heads
at step 210000). LLM 224/224 and vision 204/204 Marlin layers bit-exact vs the Kaggle export; vision gate passed.
Language mode, no pruning, the same 176 held-out episodes:

- Jetson vs Kaggle-simulated int4 outputs: 0.0049 action units on average (max 0.014); error 1.433 on both.
- Jetson int4 vs bf16: +0.037 (Wilcoxon p = 0.36); turn direction 87% vs 91% (McNemar p = 0.29).
- Latency 562 ms median, 590 ms p95; torch peak 4.24 GiB; 100/100 repeated predictions bit-identical.
- Found on the way: the runtime captured a CUDA graph for every new prompt length (unbounded), which ran the Orin out of
  memory after ~6 different instructions (each capture ~1.5 s, 65-120 MB of RAM). `deploy/cuda_graphs.py` now captures
  a token count on its second use and keeps at most 3 LLM graphs; others run eagerly (bit-identical). After the fix:
  RAM flat over 176 different instructions, 548-565 ms per prediction.
