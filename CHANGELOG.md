# Changelog

## 0.2.0 (unreleased)

### Added
- `omnivla_jetson` Python package: `OmniVLAJetson(weights_path).predict(image, goal_image=, goal_pose=, instruction=)`
  returns the waypoints and OmniVLA's velocity command. It wraps `deploy/omnivla_deploy.py` without changing it.
  `examples/` has one script per mode. `tests/test_api.py` replays the validated deployment's recorded outputs;
  `deploy/tools/api_check.py` runs the same comparison on the Jetson.
- Language tests with blank-image, shuffled-image and shuffled-instruction controls: CAST behavioral instructions,
  which the base checkpoint was not trained on (`eval/data/cast_extract.py`, `eval/analysis/lang_analysis.py`,
  `results/lang_summary.md`), and LeLaN object goals in its training format, in-distribution
  (`eval/data/lelan_extract.py`, `eval/analysis/lelan_analysis.py`, `results/lelan_summary.md`); both run by
  `eval/kaggle/lang_eval.sh`.
- OmniVLA-7B vs OmniVLA-edge comparison on all driving tests (`eval/edge/edge_eval.py`,
  `eval/data/history_extract.py`, `eval/analysis/edge_compare.py`, `results/edge_vs_7b.md`). Edge latency and memory
  on the Jetson from an earlier measurement: `results/edge_jetson.md`.
- Demo video and GIF of image-goal predictions with the deployed int4 weights over FrodoBots-2K clips (`docs/media/`,
  `eval/demo/render_demo.py`, `eval/demo/seq_demo_inputs.py`).
- Evaluation scripts stop instead of reporting results when a test has 0 samples or only part of its outputs.
- Power and energy per inference at 15W, 25W and MAXN_SUPER (`eval/jetson/power_measure.py`,
  `eval/jetson/run_power_sweep.sh`). Not run on the Jetson yet.
- `docs/troubleshooting.md`, a safety section in the README, `CITATION.cff`, and a GitHub Actions workflow (syntax
  checks, API tests, rover protocol tests, mock test of `setup_jetson.sh`).

### Changed
- `deploy/omnivla_deploy.py`: image-token pruning is now mode-dependent. Pose and image-goal modes keep 75% (same code
  path and outputs as before); language modes (7, 8) run unpruned (`lang_prune_frac=0.0`). An ablation on the
  object-goal test showed the 75% pruning, not the int4 weights, cut accuracy from 84% to 69% (`results/lelan_summary.md`).
  Language-mode latency rises to about 750 ms (estimated; not yet measured on the Jetson).
- `build/kaggle_compress.py`: new `lang`, `lelan` and `seq` tests, `lang` / `lelan` / `lelan_p75` studies, `BLIND` /
  `BLIND_TESTS` options for control runs, `EVAL_PRUNES` (evaluate the same GPTQ weights at several pruning levels), and a
  stop when a test named in `TESTS_ONLY` has no frames. The weight build
  (`STUDY=gptqx`) is unchanged: both language runs reproduced the deployed int4 weights tensor for tensor.

## 0.1.0 (2026-09-28)

- First release: OmniVLA-7B on a Jetson Orin Nano 8 GB with a GPTQ int4 LLM on Marlin, HQQ 4-bit vision on GemLite,
  modality elision and 75% image-token pruning; build scripts for the weights on Kaggle, pre-built weights on Hugging
  Face, Jetson setup script with a bit-exact reference check, ROS 2 node for the Troupe rover, and the evaluation
  scripts behind `results/`.
