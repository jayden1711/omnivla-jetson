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
- Power and energy per inference at 15W, 25W and MAXN_SUPER, measured on the Jetson (`eval/jetson/power_measure.py`,
  `eval/jetson/run_power_sweep.sh`, `results/power_summary.md`): 8.6-9.3 J per pose-goal prediction and 23.8-24.9 J per
  image-goal prediction in every mode.
- `docs/troubleshooting.md`, a safety section in the README, `CITATION.cff`, and a GitHub Actions workflow (syntax
  checks, API tests, rover protocol tests, mock test of `setup_jetson.sh`).

### Changed
- `deploy/omnivla_deploy.py`: image-token pruning is now mode-dependent. Pose and image-goal modes keep 75% (same code
  path and outputs as before); language modes (7, 8) run unpruned (`lang_prune_frac=0.0`). An ablation on the
  object-goal test showed the 75% pruning, not the int4 weights, cut accuracy from 84% to 69% (`results/lelan_summary.md`).
  Language-mode latency is 700-770 ms on the Jetson (it differs between boots), 538 ms with `lang_prune_frac=0.5`
  (results/jetson_validation_2026-09-28.md).
- Prompt-aware token pruning for language modes, evaluated and not adopted: at 75% pruning it keeps 69% object-goal
  accuracy, the same as the uniform grid (`deploy/prompt_prune.py`, `eval/kaggle/lang_eval.sh` with
  `LANG_TEST=promptprune`, results/lelan_summary.md).
- `eval/jetson/determinism.sh`: 50 fresh processes, image-goal outputs bit-identical in every run.
- `deploy/cuda_graphs.py`: the vision encoders and the 32-layer LLM stack run as CUDA graphs, **on by default**
  (`cuda_graphs=False` restores the previous path; off automatically with `goal_refresh > 1`). On the Jetson:
  bit-identical outputs in all four modes, pose 424 -> 395 ms, image goal 1041 -> 1012 ms, language 769 -> 733 ms (same
  boot), ~4.6 s longer warm-up, 30-minute soak stable. A first per-layer version ran out of memory and was replaced.
- The LLM no longer builds its key/value cache, which the runtime never reads (`llm_kv_cache=False`, **the default**;
  `True` restores it). Bit-identical in all four modes. Without CUDA graphs it frees 283 MB at the same speed; with
  graphs (which already skipped it inside the graph) latency and memory are unchanged. 10-minute soak with both defaults:
  stable.
- `deploy/tools/trt_vision.py`: TensorRT vision encoders, evaluated and not adopted: TensorRT 10.3 converts the (correct)
  DINOv2 ONNX model to an engine with wrong outputs (cosine 0.54 to the reference, also in FP32), and SigLIP did not
  build within the Orin's memory. Write-up: `docs/tensorrt_dinov2_issue.md`.
- `deploy/tests/reference/reference.npz` re-recorded on the Jetson for modes 4, 6, 7 and 8 (one image-goal value differs
  by 0.0039 from the 2026-09-27 recording, which neither the old nor the new runtime reproduces; see its README).
  `reference_check.py` now checks the language modes too; `api_check.py` covers all four modes.
- `build/jetson_e2e_test.sh --hf` tests setup with the weights downloaded from Hugging Face.
- `build/kaggle_compress.py`: new `lang`, `lelan` and `seq` tests, `lang` / `lelan` / `lelan_p75` studies, `BLIND` /
  `BLIND_TESTS` options for control runs, `EVAL_PRUNES` (evaluate the same GPTQ weights at several pruning levels), and a
  stop when a test named in `TESTS_ONLY` has no frames. The weight build
  (`STUDY=gptqx`) is unchanged: both language runs reproduced the deployed int4 weights tensor for tensor.

## 0.1.0 (2026-09-28)

- First release: OmniVLA-7B on a Jetson Orin Nano 8 GB with a GPTQ int4 LLM on Marlin, HQQ 4-bit vision on GemLite,
  modality elision and 75% image-token pruning; build scripts for the weights on Kaggle, pre-built weights on Hugging
  Face, Jetson setup script with a bit-exact reference check, ROS 2 node for the Troupe rover, and the evaluation
  scripts behind `results/`.
