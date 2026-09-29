# OmniVLA on a Jetson Orin Nano 8 GB: v1.0

OmniVLA-7B, the 7B vision-language-action navigation model, runs on a Jetson Orin Nano 8 GB at 271 ms per pose-goal
prediction, 775 ms per image-goal prediction and 549-564 ms per language prediction, with no significant accuracy loss
against the original bf16 model and better accuracy than OmniVLA-edge. Pre-built weights are on Hugging Face; the
setup script installs everything and ends with a bit-exact check.

## Highlights

- Image goal: driving error 1.301 on the Jetson vs 1.330 for the original in bf16 (no significant difference,
  p = 0.17) and 1.490 for OmniVLA-edge. Object goal: picks the named object in 82% of frames vs 84% bf16 (p = 0.22) and
  70% edge. Single open-loop predictions on in-distribution frames; the model has not driven a robot.
- 4.13 GiB of weights on the GPU, 1.3 GB of RAM left; the original's 15.1 GB does not fit the Orin's 7.6 GB.
- Everything measured on the Jetson is listed with what was not in `docs/validation.md`.

## What is new since 0.1.0

**Faster vision encoders.** DINOv2 and SigLIP now run as GPTQ per-channel int4 on the Marlin kernel, like the LLM
(SigLIP's shapes are zero-padded to Marlin's, exactly). On the Jetson: pose goal 374 -> 271 ms, image goal 981 -> 775 ms,
language 663-672 -> 549-564 ms, about 70 MB less RAM, no significant accuracy change. `OMNIVLA_VISION=hqq4` keeps the
previous HQQ 4-bit path; weights without `vis_marpc/` fall back to it automatically.

**CUDA graphs and no key/value cache, by default.** The vision encoders and the 32-layer LLM stack replay as CUDA
graphs (bit-identical outputs, 3-10% faster). The LLM no longer builds a key/value cache the runtime never reads.
`cuda_graphs=False` / `llm_kv_cache=True` restore the old paths.

**Language modes run without image-token pruning.** An ablation showed the 75% pruning, not the int4 weights, cut
object-goal accuracy from 84% to 69%; pose and image goals keep the pruning, where it did not change driving error.
Prompt-aware pruning was evaluated and not adopted (no better than the uniform grid).

**CAST instructions.** `setup_jetson.sh --cast` installs the authors' omnivla-finetuned-cast checkpoint as GPTQ int4
(`jayden1711/omnivla-7b-cast-jetson-int4`); run it with `OMNIVLA_WEIGHTS=weights_cast`. On held-out CAST episodes:
error 1.433 vs 1.396 in bf16 (p = 0.35), turn direction 87% vs 91% (p = 0.29), 562 ms on the Jetson. Outside language
mode it is not better than the default weights.

**Python package.** `OmniVLAJetson(weights_path).predict(image, goal_image=, goal_pose=, instruction=)` returns the
waypoints and OmniVLA's velocity command; `examples/` has one script per mode; `tests/test_api.py` replays the validated
outputs.

**Benchmarks.** Original 7B vs this repo vs OmniVLA-edge in one table (accuracy, memory, latency; the original timed in
fp16 on two T4s); OmniVLA-7B vs OmniVLA-edge on every test with confidence intervals; language tests with blank-image,
shuffled-image and shuffled-instruction controls (LeLaN object goals, CAST behavioral instructions); the CAST checkpoint
vs the original outside language mode; power and energy per prediction at 15W, 25W and MAXN SUPER.

**Demos.** The deployed runtime on the Jetson replaying online walking footage as a live camera at its real rate, and
image-goal predictions over FrodoBots-2K clips (`docs/media/`, CC BY-SA 4.0).

**Also:** `docs/troubleshooting.md`, a safety section, `CITATION.cff`, a GitHub Actions workflow (syntax checks, API
tests, rover protocol tests, mock setup test), a 50-process determinism check, and evaluation scripts that stop instead
of reporting partial results.

## Fixed

- CUDA graphs captured one LLM graph per new prompt length without a limit, so varied language instructions ran the
  Orin out of memory after about 6 prompts. A token count is now captured on its second use and at most 3 LLM graphs
  are kept; other lengths run eagerly (bit-identical, ~5% slower). RAM is flat over 176 different prompts.

## Evaluated and not adopted

- TensorRT vision encoders: TensorRT 10.3 on the Orin converts the DINOv2 model to an engine with wrong outputs, also in
  FP32, and SigLIP did not build within the Orin's memory (`docs/tensorrt_dinov2_issue.md`).
- Grouped Marlin (groupsize 128) gives wrong results on sm_87; only per-channel weights are used
  (`docs/marlin_sm87_bug.md`).
- GPU-side preprocessing and a different attention kernel: no gain worth the change (`results/latency_options.md`).

## Weights

- Default: `jayden1711/omnivla-7b-jetson-int4` at revision bd51d77 (adds `vis_marpc/`, the Marlin vision layers).
- CAST: `jayden1711/omnivla-7b-cast-jetson-int4` at revision 2c15a98.
- Both derive from Llama 2 and fall under the Llama 2 Community License; the code is MIT.

## Upgrading from 0.1.0

- Rerun `deploy/setup_jetson.sh`: it downloads the new weights revision (the Marlin vision layers) and checks the
  re-recorded reference outputs. Existing weights without `vis_marpc/` keep working with HQQ4 vision.
- CUDA graphs are on by default; they add about 4.6 s to the warm-up and are turned off automatically with
  `goal_refresh > 1`.
- Language predictions take about twice as long as pose-goal predictions now (no pruning); `lang_prune_frac=0.5` is a
  middle ground (81% instead of 82-84% on the object-goal test).

## Known limitations

- All accuracy numbers are open-loop single predictions on in-distribution frames. No closed-loop driving; the ROS 2
  node has been tested offline on recorded frames only.
- Of the driving tests, only the image-goal test passes the blind-model check.
- Process memory grows by 50-65 MB per 10 minutes; restart the node every ~2 hours. Latency differs by up to ~10%
  between boots of the same Jetson.

Full list of changes: `CHANGELOG.md`.
