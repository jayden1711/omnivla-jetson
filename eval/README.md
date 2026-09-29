# Evaluation

Scripts that produced the numbers in `results/`. Most of them need a GPU, a Kaggle account or the Jetson. No data is
included: step 1 downloads what it needs from FrodoBots-2K (CC BY-SA 4.0) with HTTP range requests (~1 GB).

## 1. Test data (any machine; `pip install -r eval/requirements-eval.txt` and ffmpeg)

Run from the repo root, in this order:

```
python eval/data/extract_frames.py        # 200 test frames from 22 rides -> data_frames/, results/frames_manifest.csv
python eval/data/gt_extract.py            # ground-truth future paths -> results/gt_frodobots.npz (fetches gt_src)
python eval/data/gt_tests.py              # image-goal (img3m) and pose-goal tests -> results/gt_tests.npz
python eval/data/heldout_extract.py       # held-out calibration samples (GPTQ) -> data_heldout/, results/heldout.npz
python eval/data/seq_extract.py           # sequential clips for the temporal-reuse tests -> data_seq/, results/seq.npz
python eval/data/history_extract.py       # 5 past frames per test frame, for OmniVLA-edge -> data_frames/history/
```

The language-goal test set comes from CAST (catglossop/CAST-dataset; its card states no license, so nothing from it is
in this repo). `eval/data/cast_extract.py` needs tensorflow and downloads 7 GB, so it runs on a Kaggle CPU kernel with
internet (no GPU time): 240 CAST episodes, one sample each, ground truth computed like OmniVLA's own CAST loader
-> `cast_lang.npz`. The Kaggle kernel must be named `<you>/omnivla-cast-extract` for step 2. CAST's behavioral
instructions are outside the deployed checkpoint's training (the authors' omnivla-finetuned-cast covers them).

The object-goal test uses LeLaN (NHirose/LeLaN_dataset_NoMaD_traj, MIT), the prompt style the base checkpoint was
trained on, so it is in-distribution: `python eval/data/lelan_extract.py` (range requests, ~15 MB) -> 210 frames with
a target and a distractor object -> `results/lelan_lang.npz`. `python eval/demo/seq_demo_inputs.py` packs the demo
clips' image-goal inputs -> `results/seq_demo.npz`. Both go in a private Kaggle dataset
`<you>/omnivla-lelan-demo-inputs` for `LANG_TEST=lelan eval/kaggle/lang_eval.sh`.

## 2. Kaggle (quantization studies and the bf16 reference)

`build/make_build_dataset.sh` runs step 1 if needed and uploads the frames, ground truth and OmniVLA inference code as
your private Kaggle dataset `<KAGGLE_USER>/omnivla-frodobots-frames` (packed by `eval/kaggle/make_kaggle_bundle.sh`).
`build/build_model.sh` uses the same dataset.
`eval/kaggle/make_kernel.py` wraps `build/kaggle_compress.py` in a notebook; the `STUDY` variable selects the study
(listed at the top of `kaggle_compress.py`: bit-width sweep with the bf16 reference, real-driving tests, sensitivity,
teacher labels, low-rank correction, 2:4 pruning, execution-weighted quantization, activation quantization, GPTQ export). All runs install the OpenVLA-OFT transformers
fork; stock transformers gives wrong actions.

`eval/kaggle/lang_eval.sh` (~1 GPU-hour) runs the language test: the 7B model in fp16 with blank-image, shuffled-image
and shuffled-instruction controls, bf16 as the reference, then the deployed int4 config (GPTQ int4 LLM, HQQ4 vision,
75% token pruning) on the language test and on img3m, pose5 and pose20. It checks that the int4 LLM weights are
identical, tensor for tensor, to the deployed ones (`build/reference_tensor_hashes.json`). The kernels are fp16
dequantized, not the Jetson's Marlin/GemLite, so these int4 numbers are close to the Jetson's but not bit-exact.
`LANG_TEST=lelan eval/kaggle/lang_eval.sh` (~0.8 GPU-hour) does the same for the LeLaN object-goal test (the "shuffled
instruction" control is the other object's prompt on the same image) and also predicts the demo clips with the deployed
int4 weights. `LANG_TEST=promptprune` (~0.7 GPU-hour) tests prompt-aware pruning (`deploy/prompt_prune.py`: patches ranked by SigLIP
image-text similarity to the object phrase) at 25/50/75%, with uniform 25% for comparison and the original SigLIP image
tower at 75% as a reference; it needs `open_clip_torch==2.24.0`. `LANG_TEST=ablation` (~0.8 GPU-hour) is the pruning ablation on the same test: fp16 with 75% pruning,
then the deployed int4 weights (GPTQ calibrated as built) at 0%, 50% and 75% pruning. `kaggle_compress.py` stops if a
test named in `TESTS_ONLY` has no frames.

### Not run: omnivla-finetuned-cast

The OmniVLA authors' checkpoint for CAST-style behavioral instructions (NHirose/omnivla-finetuned-cast) has the same
architecture and size as omnivla-original; its heads are at step 210000 instead of 120000 and its LoRA adapter is
larger. Evaluating it would take: making the checkpoint name and head step parameters of `build/kaggle_build.py`,
`build/kaggle_compress.py` and `deploy/omnivla_deploy.py` (about 2-3 hours of work); a weight build on Kaggle (about
0.4 GPU-hours); the CAST test in bf16, fp16 with controls, and int4 (about 0.75 GPU-hours; 1.2-1.5 GPU-hours in total
with a smoke run); and on the Jetson, new reference outputs plus a latency and memory check (1-2 hours).

## 3. Jetson

Copy `eval/jetson/*` into the OmniVLA clone on the Jetson (default root `/mnt/nvme/omnivla`, override with
`OMNIVLA_ROOT`) along with the test data, and run each script through `deploy/launch.sh`:

- `final_validate.py 6` / `final_validate.py 4`: latency, memory and outputs of the deployed runtime; `--soak 30` for soaks.
- `jetson_breakdown.py`: latency per stage.
- `run_refresh_sweep.sh` (`deploy_refresh_sweep.py`), `jetson_cache.py`: goal cache and temporal reuse.
- `run_replays.sh` (`verify_replay6.py`): ROS replay tests with bit-exact reproduction of every published chunk.
- `run_soak6.sh`: image-goal soaks.
- `marlin_pc_gate.py`, `marlin_sm87_repro.py`: Marlin correctness on sm_87 (see `docs/marlin_sm87_bug.md`).
- `wait_job.sh` (runs on the PC): waits for a tmux job on the Jetson with a timeout and a stall check.
- `lang_latency.py`: latency and memory per mode (`MODES=4,6,7,8`, `GRAPHS=1`, `NOCACHE=1`); `graph_soak.py`: 30-minute
  soak cycling modes 4, 6, 7; `determinism.sh`: N fresh processes, bit-identical outputs; `profile_launch.py`: per-stage
  GPU vs wall time (torch.profiler).
- `power_measure.py` / `run_power_sweep.sh`: board power and energy per inference at 15W, 25W and MAXN_SUPER (asks
  before each power-mode change; `power_analysis.py` writes `results/power_summary.md`). Not run yet.

## OmniVLA-edge (any machine)

`OMNIVLA_SRC=<OmniVLA clone> python eval/edge/edge_eval.py omnivla-edge.pth cast_lang.npz` runs OmniVLA-edge
(https://huggingface.co/NHirose/omnivla-edge) on img3m, pose5, pose20 and the language test with the same controls,
using OmniVLA's edge preprocessing and the real past frames. CPU is enough (fp32). Output: `results/edge_tests.npz`.
Add `results/lelan_lang.npz` as a third argument for the object-goal test; `EDGE_TESTS=frodobots,lang,lelan` picks
tests (unknown names and tests with 0 samples stop the run; skipped tests are reported as skipped).

## Demo

`python eval/demo/render_demo.py compress_gptqx_lelan.npz` (from the repo root) draws saved image-goal predictions of the
deployed int4 weights (the `LANG_TEST=lelan` Kaggle run) over the original FrodoBots-2K video and writes
`docs/media/demo.mp4` and `demo.gif` (CC BY-SA 4.0, see THIRD_PARTY_LICENSES.md).

## 4. Analysis (any machine)

Copy the Jetson and Kaggle outputs (`mf_results/`, `results/*.npz`) into `results/` and run from the repo root:
`tests_analysis.py` (driving tests), `final_analysis.py` and `gptq_validation.py` (deployed config),
`refresh_analysis.py` and `cache_analysis.py` (goal cache), `soak_analysis.py`, `exq_analysis.py` and
`lora_seed_analysis.py` (execution-weighted quantization), `lang_analysis.py` (CAST test), `lelan_analysis.py`
(object-goal test), `edge_compare.py`
(7B vs OmniVLA-edge, all tests; `JETSON_NPZ=results/cross_jet_orig.npz` takes the Jetson outputs of the current
Marlin-vision runtime from the cross-mode run and adds the object-goal row).
