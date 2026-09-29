# Results

Summary tables and figures written by the analysis scripts in `eval/analysis/`. Raw per-frame outputs are not included.

Terms used in the tables:
- img3m: image-goal test. The goal image is the frame where the robot's real path first reaches 3 m (100 frames).
- pose5 / pose20: pose-goal tests, goal 5.1 s / 20.1 s ahead on the real path (100 / 93 frames).
- lang: language test, 237 CAST episodes with behavioral instructions the base checkpoint was not trained on.
- lelan: object-goal language test, 210 LeLaN frames ("move toward <object>"), in-distribution.
- fidelity (fid): mean distance between a config's actions and the bf16 model's actions, in action units.
- drive: mean distance between the predicted waypoints and the path the human driver actually took (8 steps x 0.3 s).
- L0.43 / L1.05 / ...: the error weighted by which chunk steps are executed at that latency (seconds).
- Action units are normalized waypoint spacings (0.25 m in the FrodoBots training data).

| File | Content |
|---|---|
| final_validation.md | deployed runtime on the Jetson: latency, memory, fidelity, driving |
| gptq_validation.md | GPTQ vs round-to-nearest weights, pass criteria |
| deploy_table.md | latency and memory of all configurations tried on the Jetson |
| tests_summary.md | driving tests and blind-model controls (image, blank and shuffled goals) |
| lang_summary.md | language test on CAST (behavioral instructions, outside the base checkpoint's training) |
| lelan_summary.md | object-goal language test on LeLaN (in-distribution): controls, bf16 / int4 / OmniVLA-edge, pruning ablation |
| edge_vs_7b.md | OmniVLA-7B int4 vs OmniVLA-edge: accuracy on every test, Jetson latency and memory |
| edge_jetson.md | OmniVLA-edge latency and memory on the Jetson (earlier measurement, provenance inside) |
| power_summary.md | power and energy per inference on the Jetson, per power mode |
| jetson_validation_2026-09-28.md | on-device checks: reference outputs, language-mode latency and memory, Python package, setup |
| verify_summary.md | 4-bit vs full-precision reference, per condition |
| refresh_summary.md | goal key/value reuse: error vs refresh interval |
| cache_summary.md | temporal reuse between frames (goal cache, VLA-Cache style, ViT patch reuse) |
| soak_summary.md, soak6_summary.md | 30-minute soak tests, pose and image goal |
| fault_injection.md | fault injection into a copy of the ROS node, to check the replay tests |
| exq_all_summary.md, lora_seeds_test.md | execution-weighted quantization study |
| a8c_summary.md | activation quantization (W4A8 / W8A8) simulation |
| *.png | figures: fidelity vs driving, latency vs driving, elision speedup, 4-bit verification |
