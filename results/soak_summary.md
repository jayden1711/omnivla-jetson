# 30-min soaks, deployed config, pose mode (continuous inference, 2026-09-26)
Orin thermal throttling starts near 99-105 C junction; jetson_clocks holds clocks at max. System RAM from tegrastats includes the model load in the first minute.

## without malloc_trim
- duration min / inferences: 30.0 / 4133
- latency ms: first min / last min / max 30-s mean / worst single: 423.4 / 424.9 / 429.6 / 444.8
- latency drift ms per 10 min: +0.51
- torch allocated / reserved GiB (min=max?): 4.154-4.154 / 4.617-4.617
- system RAM MB at 10 min / end / max: 6672 / 6903 / 6907
- system RAM drift after 10 min, MB per 10 min: +118.1
- junction temp C max / last 5 min: 70.3 / 69.8
- CPU clock MHz min / max (all cores, all samples): 1728 / 1728
- GPU load % median / board power W mean: 99 / 20.6

## with malloc_trim every 20 predictions (deployed)
- duration min / inferences: 30.0 / 4121
- latency ms: first min / last min / max 30-s mean / worst single: 424.4 / 424.8 / 425.6 / 433.5
- latency drift ms per 10 min: +0.04
- torch allocated / reserved GiB (min=max?): 4.154-4.154 / 4.617-4.617
- system RAM MB at 10 min / end / max: 6584 / 6694 / 6704
- system RAM drift after 10 min, MB per 10 min: +62.9
- junction temp C max / last 5 min: 70.4 / 69.8
- CPU clock MHz min / max (all cores, all samples): 1728 / 1728
- GPU load % median / board power W mean: 99 / 20.6
- process RssAnon MB start / end; drift after 10 min per 10 min: 5987 / 6132; +63.9
- worst call incl. preprocessing ms / worst malloc_trim ms: 454.3 / 3.1
