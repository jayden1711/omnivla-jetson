# README workflow from scratch on the Jetson (2026-09-29)

`eval/jetson/readme_rerun.sh`: a git export of the staged tree (as a clone would give), a new root
(/mnt/nvme/omnivla/readme_rerun), a fresh venv and a fresh OmniVLA clone, weights from Hugging Face, then the same for
the CAST weights. Jetson Orin Nano 8 GB, JetPack 6.2 (L4T R36.4.7), MAXN SUPER, already configured (`--no-system`: the
system checks all passed; `--yes-runtime`). No ffmpeg on this Jetson, so the 20 reference images were copied from a PC,
as the README says.

| Step | Time | Result |
|---|---|---|
| Get the repo + reference images | < 1 s | 20 images |
| `setup_jetson.sh`: preflight and system checks | < 1 s | all ok |
| venv: Jetson torch 2.8.0 wheel | 50 s | ok |
| venv: transformers fork, pinned packages, Marlin built for sm_87 | 127 s | ok |
| OmniVLA clone @5182600 + patch | 4 s | ok |
| Weights download (jayden1711/omnivla-7b-jetson-int4 @ bd51d77, 4.1 GB) + SHA256SUMS | 663 s | verified |
| Correctness check (load + 40 predictions) | 73 s | 40/40 bit-identical (modes 4, 6, 7, 8) |
| **`setup_jetson.sh` total** | **920 s (15.3 min)** | exit 0 |
| `launch.sh` smoke test | 49 s | pose mode 270 ms |
| `setup_jetson.sh --cast`: checks, reused venv and OmniVLA | 21 s | ok |
| CAST weights download (jayden1711/omnivla-7b-cast-jetson-int4 @ 2c15a98, 4.1 GB) + SHA256SUMS | 643 s | verified |
| CAST correctness check | 71 s | 40/40 bit-identical to reference_cast.npz |
| **`setup_jetson.sh --cast` total** | **739 s (12.3 min)** | exit 0 |
| `OMNIVLA_WEIGHTS=weights_cast launch.sh` smoke test | 47 s | pose mode 271 ms |
| **Everything** | **1755 s (29.3 min)** | all passed |

Downloads ran at about 6.3 MB/s; they take most of the time and depend on the connection (an earlier test from this
Jetson's network gave 0.8 MB/s, which would make each download about 85 minutes).
