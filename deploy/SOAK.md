# Soak and memory behavior (Jetson Orin Nano, deployed config, continuous inference)

Pose mode, 30 min each. Source: eval/jetson/final_validate.py --soak 30, eval/analysis/soak_analysis.py.

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

## Diagnosis
- The growth is process heap (glibc) retention of freed per-frame buffers, not held Python or torch objects: python-traced allocations stay at 0.4 MB and torch allocated/reserved are flat. One malloc_trim(0) after 600 identical predictions returned all 59 MB.
- Cycling real frames, a periodic trim leaves slow residual growth (+63 MB / 10 min in the 30-min soak, irregular). A fixed glibc mmap threshold (MALLOC_MMAP_THRESHOLD_=65536, MALLOC_ARENA_MAX=2) made it WORSE (+175 MB, not trimmable). Do not set it.
- Policy: omnivla_deploy trims every 20 predictions (<= 3 ms). The ROS node stops the rover (LOWMEM) below 300 MB available. At the soak rate that is about 2 h of continuous pose-goal inference: restart the node between runs. Root cause of the residual growth: open.

## Image-goal soaks (GPTQ weights, goal cache; results/soak6_summary.md)
24 sequential clips cycled, goal changes every 5 predictions, 30 min each:
- goal_refresh=1 (default, vision cache only): 1971 inferences, 30-s means 896-905 ms, drift +0.42 ms / 10 min, worst call
  1.08 s; torch 4.155 GiB flat; RAM +57 MB / 10 min (peak 6898 MB -> 722 MB headroom); tj max 74.3 C; clocks constant; 23 W.
- goal_refresh=3 (opt-in K/V reuse, 60% of predictions): 2714 inferences, 646-661 ms, drift -0.26 ms / 10 min; torch
  4.329 GiB flat (+0.17 GiB for the cached goal K/V); RAM +47 MB / 10 min (peak 6893 MB); tj max 72.5 C; 21.8 W.
- The slow RAM creep is the same glibc heap behavior as in pose mode: restart the node between runs.
