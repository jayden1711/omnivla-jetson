# Tested on real hardware, and what is not

On the Jetson Orin Nano 8 GB (JetPack 6.2, MAXN SUPER):

- Loading, latency and memory of the deployed runtime in all four modes: pose goal and image goal (100 frames each),
  object-goal language (210 LeLaN frames) and CAST instructions (176 held-out episodes with the CAST weights)
  ([results/vismarlin_summary.md](../results/vismarlin_summary.md),
  [results/cast_gptq_summary.md](../results/cast_gptq_summary.md)).
- Accuracy of the deployed runtime on the Jetson, with blank-image and shuffled-image controls: image goal, 5 s pose
  goal and object goal, for the default and the CAST weights
  ([results/cross_mode_summary.md](../results/cross_mode_summary.md)); the CAST weights on held-out CAST episodes.
- Bit-exact reproduction of the recorded outputs in all four modes (`deploy/tools/reference_check.py`, 10 frames x
  4 modes, 40/40) for the default Marlin-vision weights, the HQQ4 vision fallback (`reference_hqq4.npz`) and the CAST
  weights (`reference_cast.npz`).
- The Marlin vision encoders: all 204 quantized layers bit-exact against the Kaggle export, each layer checked against a
  dequantize-then-matmul reference at real token counts, 50/50 deterministic repeats per layer, and Jetson outputs
  within 0.0032 action units of the Kaggle-simulated ones (`eval/jetson/vis_marlin_gate.py`).
- CUDA graphs (the default): bit-exact in all four modes, 3-10% faster, a 30-minute soak cycling all modes (stable
  latency, no memory errors, 74 C at most), and the graph cap: RAM flat over 176 different language instructions.
- Determinism: 50 fresh processes give bit-identical image-goal outputs.
- The `omnivla_jetson` package: identical outputs to the runtime in all four modes (`deploy/tools/api_check.py`).
- `deploy/setup_jetson.sh` end to end in a fresh venv and OmniVLA clone with the weights downloaded from Hugging Face,
  a second run on top of it (`build/jetson_e2e_test.sh --hf`), and the README workflow from a fresh clone
  ([results/readme_rerun_2026-09-29.md](../results/readme_rerun_2026-09-29.md)).
- Power and energy per prediction at 15W, 25W and MAXN SUPER ([results/power_summary.md](../results/power_summary.md);
  measured before CUDA graphs and Marlin vision became the defaults).
- 30-minute soaks in pose and image-goal mode ([deploy/SOAK.md](../deploy/SOAK.md)) and a 10-minute pose-goal soak with
  Marlin vision: no slowdown, no throttling, slow memory growth.
- The manual setup steps in [deployment.md](deployment.md), in a fresh venv and a fresh OmniVLA clone.
- The ROS 2 node in offline replay (recorded camera frames, no rover), including fault injection
  ([results/fault_injection.md](../results/fault_injection.md)).
- The online-footage demo: the runtime replaying video as a live camera at its real rate ([demos.md](demos.md)).

Not tested on hardware:

- The rover: the model has never driven it. The servo bridge has not run on the rover's Pi.
- Any closed-loop driving. All accuracy numbers are open-loop single predictions.
- The 20 s pose-goal test and the CAST test of the default weights: a Kaggle GPU with the same int4 weights and the
  previous HQQ4 vision (the runs match the Jetson to about 0.003 action units where both exist).
- OmniVLA-edge: its accuracy was measured off the Jetson; its latency and memory on the Jetson come from an earlier
  measurement with OmniVLA's own script ([results/edge_jetson.md](../results/edge_jetson.md)).
- The system-changing steps of `setup_jetson.sh` (headless boot, swap, sudoers) on a fresh Jetson: the end-to-end test
  runs with `--no-system` on a Jetson where they were already done.
