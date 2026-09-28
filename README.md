# OmniVLA on a Jetson Orin Nano 8 GB

This repo runs OmniVLA, a 7B vision-language-action model for robot navigation, on a Jetson Orin Nano 8 GB. The LLM is
quantized to int4 with GPTQ and runs on the Marlin kernel. The vision encoders use HQQ 4-bit on GemLite. Unused goal
tokens are dropped and 75% of the camera image tokens are pruned. A ROS 2 node drives a small rover from the model's output.

## Results

Jetson Orin Nano 8 GB (MAXN SUPER), 100 FrodoBots frames, image-goal driving test. Errors are in action units
(normalized waypoint spacing); lower is better.

| | This repo | bitsandbytes NF4 | bf16 (cloud GPU) |
|---|---|---|---|
| Latency, pose goal | 424 ms | 1416 ms | - |
| Latency, image goal | 1056 ms (863 ms with an unchanged goal) | 2146 ms | - |
| RAM headroom, pose / image | 1055 / 851 MB | 827 / 575 MB | does not fit |
| Distance from bf16 actions | 0.48 | 0.30 | 0 |
| Driving error vs the human's path | 1.314 | 1.317 | 1.330 |

The driving error of this repo's config is not significantly different from NF4 (p = 0.76) or bf16 (p = 0.40). Details:
[results/gptq_validation.md](results/gptq_validation.md), [results/final_validation.md](results/final_validation.md),
[results/](results/).

## Requirements

- Jetson Orin Nano 8 GB with an NVMe SSD, JetPack 6.2 (CUDA 12.6, Python 3.10), booted without the desktop.
- To build the weights: a Kaggle account with GPU access, the `kaggle` CLI, ffmpeg and the packages in
  `eval/requirements-eval.txt` (the dataset step downloads a few GB of FrodoBots-2K). Not needed if you use the pre-built
  weights on Hugging Face: [huggingface.co/jayden1711/omnivla-7b-jetson-int4](https://huggingface.co/jayden1711/omnivla-7b-jetson-int4).
- ffmpeg on the Jetson for the one-time reference-image download (or copy the images from a PC).
- OmniVLA itself is not included. The setup script clones it at commit `5182600` and applies a one-line patch.

## Setup

Skip the first two commands if you use the pre-built weights: without `--weights-src`, `setup_jetson.sh` downloads
them from Hugging Face (`--hf-repo` / `--hf-revision` to override) and checks `SHA256SUMS`.

```
KAGGLE_USER=<you> ./build/make_build_dataset.sh             # on a PC, once: your private Kaggle dataset (no GPU)
KAGGLE_USER=<you> ./build/build_model.sh build/out          # on a PC: builds the weights on a free Kaggle T4 (~0.4 GPU-h)
./deploy/setup_jetson.sh --weights-src <copy of build/out/weights>   # on the Jetson
```

Both build scripts take `--dry-run`. `python build/verify_build.py build/out/weights` compares a build with the validated
weights tensor by tensor. `setup_jetson.sh` asks before every system change (power mode, headless boot, swap, sudoers entries), builds the venv
with the Jetson PyTorch wheel, and ends with a check that 10 reference frames (downloaded from FrodoBots-2K on first use) reproduce the validated
outputs bit for bit.
It is safe to rerun. Run the model with `deploy/launch.sh`; the ROS 2 node with `deploy/launch.sh --ros`.
See [docs/deployment.md](docs/deployment.md) for the manual steps, the Python API and the rover setup.
Every change made to OmniVLA is listed in [deploy/CHANGES_TO_OMNIVLA.md](deploy/CHANGES_TO_OMNIVLA.md).

## Reproducing the results

[eval/README.md](eval/README.md) lists the scripts: test-frame extraction from FrodoBots-2K, the ground-truth builder,
the Kaggle quantization studies, the Jetson latency and accuracy runs, and the analysis scripts that write `results/`.
The ROS tests are `deploy/ros2/run_ros_test.sh` (offline replay, no rover) and `deploy/fault_injection/run_faults.sh`.

## Limitations

- All accuracy numbers are open-loop, single predictions on in-distribution FrodoBots frames. No closed-loop driving yet.
- Only pose-goal and image-goal modes were validated on the Jetson.
- Process memory grows by 50-60 MB per 10 minutes; restart the node every ~2 hours.
- Grouped Marlin (groupsize 128) gives wrong results on sm_87; only per-channel weights are used
  ([docs/marlin_sm87_bug.md](docs/marlin_sm87_bug.md)).
- The rover's servo channel map differs between hardware versions. Test on a bench with the wheels off the ground first.

## Credit and citation

OmniVLA is by Noriaki Hirose, Catherine Glossop, Dhruv Shah and Sergey Levine
([github.com/NHirose/OmniVLA](https://github.com/NHirose/OmniVLA)). If you use this work, please cite their paper:

```
@article{hirose2025omnivla,
  title   = {{OmniVLA}: An Omni-Modal Vision-Language-Action Model for Robot Navigation},
  author  = {Hirose, Noriaki and Glossop, Catherine and Shah, Dhruv and Levine, Sergey},
  journal = {arXiv preprint arXiv:2509.19480},
  year    = {2025}
}
```

## License

The code in this repo is MIT ([LICENSE](LICENSE)). OmniVLA is MIT. The model weights are derived from Llama 2 and
fall under the Llama 2 Community License. The FrodoBots-2K data is CC BY-SA 4.0. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
