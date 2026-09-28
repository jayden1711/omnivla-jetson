# 30-min soaks, image-goal mode, GPTQ default weights + goal cache (2026-09-27)
24 sequential clips cycled (goal changes every 5 predictions), one prediction per clip frame at the 1.05 s cadence.
Orin thermal throttling starts near 99-105 C junction; jetson_clocks holds clocks at max. System RAM from tegrastats includes the model load in the first minute.

## image goal, goal_refresh=1 (default: goal vision cache only)
- duration min / inferences: 30.0 / 1971
- latency ms: first min / last min / max 30-s mean / worst single: 900.4 / 900.6 / 905.1 / 1071.7
- latency drift ms per 10 min: +0.42
- torch allocated / reserved GiB (min=max?): 4.155-4.155 / 4.795-4.795
- system RAM MB at 10 min / end / max: 6767 / 6889 / 6898
- system RAM drift after 10 min, MB per 10 min: +56.8
- junction temp C max / last 5 min: 74.3 / 73.6
- CPU clock MHz min / max (all cores, all samples): 1728 / 1728
- GPU load % median / board power W mean: 99 / 23.0
- process RssAnon MB start / end; drift after 10 min per 10 min: 6160 / 6300; +61.2
- worst call incl. preprocessing ms / worst malloc_trim ms: 1080.6 / 2.8
- goal K/V reused on (fraction of predictions) / latency 30-s mean ms range: 0.00 / 896-905

## image goal, goal_refresh=3 (opt-in goal K/V reuse)
- duration min / inferences: 30.0 / 2714
- latency ms: first min / last min / max 30-s mean / worst single: 656.2 / 653.5 / 661.3 / 1063.2
- latency drift ms per 10 min: -0.26
- torch allocated / reserved GiB (min=max?): 4.329-4.329 / 4.811-4.811
- system RAM MB at 10 min / end / max: 6792 / 6888 / 6893
- system RAM drift after 10 min, MB per 10 min: +47.4
- junction temp C max / last 5 min: 72.5 / 71.2
- CPU clock MHz min / max (all cores, all samples): 1728 / 1728
- GPU load % median / board power W mean: 99 / 21.8
- process RssAnon MB start / end; drift after 10 min per 10 min: 6174 / 6305; +45.8
- worst call incl. preprocessing ms / worst malloc_trim ms: 1082.5 / 3.1
- goal K/V reused on (fraction of predictions) / latency 30-s mean ms range: 0.60 / 646-661
