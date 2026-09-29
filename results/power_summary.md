# Power and energy per inference on the Jetson Orin Nano 8 GB

Board input power (tegrastats VDD_IN, 100 ms samples), deployed runtime through launch.sh (jetson_clocks on),
10 reference frames, back-to-back predictions. Energy = mean VDD_IN x wall time per prediction. Cases: 4 pose goal,
6 image goal with a new goal every call, 6c image goal with an unchanged goal, 7 language (no pruning).
Script: eval/jetson/power_measure.py. Measured 2026-09-28 without CUDA graphs (the runtime default since then),
with the LLM key/value cache on; with graphs each prediction is 3-10% shorter.

| Power mode | Case | Latency (fwd, median) | Wall time / inference | VDD_IN | Idle | Energy / inference | Above idle | GPU clock | tj max |
|---|---|---|---|---|---|---|---|---|---|
| 15W | 4 | 561 ms | 580 ms | 16.04 W | 7.84 W | 9.30 J | 4.75 J | 612 MHz | 61 C |
| 15W | 6 | 1387 ms | 1399 ms | 17.79 W | 7.84 W | 24.89 J | 13.91 J | 612 MHz | 65 C |
| 15W | 6c | 1114 ms | 1129 ms | 18.49 W | 7.84 W | 20.88 J | 12.02 J | 612 MHz | 67 C |
| 15W | 7 | 1002 ms | 1011 ms | 17.04 W | 7.84 W | 17.22 J | 9.29 J | 612 MHz | 67 C |
| 25W | 4 | 446 ms | 458 ms | 18.77 W | 8.15 W | 8.60 J | 4.87 J | - | - |
| 25W | 6 | 1158 ms | 1168 ms | 20.41 W | 8.15 W | 23.83 J | 14.31 J | - | - |
| 25W | 6c | 951 ms | 964 ms | 21.00 W | 8.15 W | 20.24 J | 12.38 J | - | - |
| 25W | 7 | 755 ms | 765 ms | 20.57 W | 8.15 W | 15.73 J | 9.50 J | - | - |
| MAXN_SUPER | 4 | 426 ms | 434 ms | 20.18 W | 8.42 W | 8.76 J | 5.10 J | 1020 MHz | 64 C |
| MAXN_SUPER | 6 | 1042 ms | 1052 ms | 22.66 W | 8.42 W | 23.84 J | 14.98 J | 1020 MHz | 69 C |
| MAXN_SUPER | 6c | 863 ms | 875 ms | 23.31 W | 8.42 W | 20.40 J | 13.03 J | 1020 MHz | 71 C |
| MAXN_SUPER | 7 | 771 ms | 779 ms | 21.49 W | 8.42 W | 16.75 J | 10.18 J | 1020 MHz | 71 C |

GPU clock and tj: - = not logged (that run was made before the logging was added).
