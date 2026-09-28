# OmniVLA-edge on the Jetson Orin Nano 8 GB: latency and memory

Measured before this repo existed, with OmniVLA's own `inference/run_omnivla_edge.py` and a timing
wrapper around the model call. Not measured with this repo's code or the `omnivla_jetson` package, and the raw logs are
not in this repo.

- Hardware and software: Jetson Orin Nano Super 8 GB, JetPack 6.2, PyTorch 2.8, batch size 1.
- Method: 5 warm-up calls, then 50 timed calls of the model.

| Power mode | Latency per call (50 timed calls) | Rate |
|---|---|---|
| 25W | 151.5 ms and 131.7 ms (two runs) | 6.6-7.6 Hz |
| MAXN_SUPER + jetson_clocks | 112.8 ms | 8.9 Hz |

- Peak GPU memory (`torch.cuda.max_memory_allocated`): 1.10 GB.
- Power and energy per inference: not measured yet.
