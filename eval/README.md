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
```

## 2. Kaggle (quantization studies and the bf16 reference)

`build/make_build_dataset.sh` runs step 1 if needed and uploads the frames, ground truth and OmniVLA inference code as
your private Kaggle dataset `<KAGGLE_USER>/omnivla-frodobots-frames` (packed by `eval/kaggle/make_kaggle_bundle.sh`).
`build/build_model.sh` uses the same dataset.
`eval/kaggle/make_kernel.py` wraps `build/kaggle_compress.py` in a notebook; the `STUDY` variable selects the study
(listed at the top of `kaggle_compress.py`: bit-width sweep with the bf16 reference, real-driving tests, sensitivity,
teacher labels, low-rank correction, 2:4 pruning, execution-weighted quantization, activation quantization, GPTQ export). All runs install the OpenVLA-OFT transformers
fork; stock transformers gives wrong actions.

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

## 4. Analysis (any machine)

Copy the Jetson and Kaggle outputs (`mf_results/`, `results/*.npz`) into `results/` and run from the repo root:
`tests_analysis.py` (driving tests), `final_analysis.py` and `gptq_validation.py` (deployed config),
`refresh_analysis.py` and `cache_analysis.py` (goal cache), `soak_analysis.py`, `exq_analysis.py` and
`lora_seed_analysis.py` (execution-weighted quantization).
