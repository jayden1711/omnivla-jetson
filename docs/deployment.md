# OmniVLA-7B on a Jetson Orin Nano 8 GB: deployment

`deploy/` runs the validated deployment config (file paths in this document are relative to `deploy/`):

- **LLM:** int4 per-channel on the Marlin kernel, GPTQ-calibrated weights. `OMNIVLA_WEIGHTS=<folder>` selects another weights folder inside `deploy/`.
- **Vision:** GPTQ int4 per-channel on the Marlin kernel (default since 2026-09-29, `weights/vis_marpc`). SigLIP's
  shapes are not Marlin-sized (k % 128, n % 256), so its 108 linears are zero-padded (exact: padded weights are 0 and
  padded outputs are dropped). About 103 ms faster per encoded image than the previous HQQ 4-bit on GemLite, with no
  significant accuracy change ([results/vismarlin_summary.md](../results/vismarlin_summary.md)). `OMNIVLA_VISION=hqq4`
  (or weights without `vis_marpc`) uses HQQ 4-bit on GemLite as before.
- **Tokens:** modality elision, then uniform-grid pruning of 75% of the current-image tokens in pose and image-goal
  modes. Language modes (7, 8) are not pruned: pruning cut object-goal accuracy from 84% to 69%
  ([results/lelan_summary.md](../results/lelan_summary.md)). Their latency is therefore higher: 549-564 ms on the
  Jetson (650-735 ms with HQQ4 vision; it differed between boots).
  `OmniVLADeploy(..., lang_prune_frac=0.5)` trades some accuracy for speed (81% on the same test, not significantly below 84%).
- **Image goal:** the goal image's vision features are computed once per goal (exact). Reusing the goal tokens' keys/values
  as well is **off by default** (`goal_refresh=1`): OmniVLA's attention is bidirectional, so those keys/values depend on the
  current frame and reuse is an approximation that measurably costs about +0.02 driving error. Opt in with
  `OmniVLADeploy(..., goal_refresh=3)` / ROS `-p goal_refresh:=3` (full pass on goal change and every 3 predictions; ~2.4x faster).
- **Weights:** quantized offline and loaded pre-packed.
- **CUDA graphs** (default, `cuda_graphs=True`): the two vision encoders and the 32-layer LLM stack are captured once
  per input shape and replayed; outputs are bit-identical, pose goals 7-10% faster, image goals 3-5%, language 5-6%
  (`deploy/cuda_graphs.py`, `tools/graph_check.py`). Capturing adds ~4.6 s to the warm-up. They are turned off
  automatically with `goal_refresh > 1`. `cuda_graphs=False` restores the previous path. The LLM gets at most 3 graphs
  (pose goal, image goal, one language instruction), each captured the second time its token count occurs; other
  prompt lengths run without graphs (same outputs, ~5% slower). Before this cap, varied instructions ran the Orin out
  of memory.
- **CAST instructions:** `OMNIVLA_WEIGHTS=weights_cast` runs the authors' omnivla-finetuned-cast checkpoint as GPTQ int4
  (LLM and vision on Marlin), for behavioral instructions ("follow the dirt path"). Build it with
  `tools/fetch_ckpt_subset.py` + `tools/build_cast_weights.py` from the `LANG_TEST=castgptq` Kaggle export
  ([results/cast_gptq_summary.md](../results/cast_gptq_summary.md)). Language mode: 562 ms median on the Jetson.
- **No key/value cache** (default, `llm_kv_cache=False`): the LLM does not build the cache that the runtime never reads.
  Outputs are bit-identical; without CUDA graphs it frees 283 MB of RAM.

It also contains a ROS 2 node for the Troupe rover.

Measured on the Jetson (100 in-distribution FrodoBots frames, single predictions, not closed-loop):

| | Deployed config (GPTQ) | NF4 baseline |
|---|---|---|
| Latency, pose goal | 271 ms (HQQ4 vision: 375-395 ms) | 1416 ms |
| Latency, image goal | 775 ms; 711 ms while the goal stays fixed (HQQ4 vision: 980-1012 / 863 ms) | 2146 ms |
| RAM headroom (of 7620 MB) | 1334-1348 MB in the validation runs (one mode per process; HQQ4 vision in the same boot: 1269-1274 MB); all modes in one process, HQQ4 vision: 930-1030 MB | 827 / 575 MB |
| Fidelity to bf16 (lower = closer) | 0.54 (HQQ4 vision 0.48; round-to-nearest LLM weights 1.09) | 0.30 |
| Driving error vs NF4 (image-goal test) | 1.301 vs 1.317, p = 0.41 (HQQ4 vision: 1.314, p = 0.76) | |

Latency and memory with Marlin vision were measured on one boot (2026-09-29, CUDA graphs on; latency differed by up to
~10% between boots before). The 10-minute soak with Marlin vision: flat 271 ms, worst call 303 ms, 74 C at most.

30-minute soaks (details in `deploy/SOAK.md`; pose mode 2026-09-26, image goal 2026-09-27) showed:
- **No slowdown and no throttling:** latency drift below 0.5 ms per 10 minutes; junction temperature at most 74 C; about 21-23 W.
- **Slow process-memory growth:** about +63 MB per 10 minutes, even with the built-in periodic `malloc_trim`. **Restart the node between runs.** The ROS node stops the rover if available memory falls below 300 MB, which is about 2 hours of continuous pose-goal driving.

The ROS 2 offline replay test passes 13 of 13 checks in pose mode and in image-goal mode (default and opt-in goal cache); every image-goal chunk the node published was reproduced bit-exactly offline (`verify_replay6.py`).

With GPTQ weights its outputs are close to NF4's distance from bf16 (fidelity 0.48 vs 0.30); the driving-error change against the human's path is not significant. Test in closed loop before trusting it on a rover.

```
deploy/
  launch.sh                 run anything with the settings the 8 GB Orin needs (clocks, allocator, page-cache drop)
  omnivla_deploy.py         the runtime: OmniVLADeploy(dir).predict(image, mode, goal_pose|goal_image|lang)
  weights/                  pre-packed weights, GPTQ LLM (3.4 GB Marlin/HQQ shards + 0.4 GB base.safetensors + model/), vis_marpc/
                            (0.37 GB GPTQ Marlin vision, added by tools/add_vision_marlin.py), SHA256SUMS
  weights_cast/             optional: omnivla-finetuned-cast for CAST-style instructions (tools/build_cast_weights.py;
                            OMNIVLA_WEIGHTS=weights_cast)
  constraints.txt           torch 2.8.0 / torchvision 0.23.0 / numpy 1.26.4 (Jetson builds - never replace)
  requirements-deploy.txt   every other Python package, pinned
  patches/omnivla.patch     one-line import fix for the OmniVLA repo
  tools/                    reference_check.py (bit-exact check against tests/reference), gptq_weight_check.py,
                            freezes of the validated venvs
  ros2/                     omnivla_nav_node.py (Jetson), servo_udp_bridge.py (rover Pi), rover_protocol.py,
                            test_rover_protocol.py, replay_test.py + run_ros_test.sh (offline test, no rover)
```

## Reproduce everything

0. **Make your Kaggle dataset** (on a PC, once; downloads the FrodoBots-2K parts it needs, no GPU):
   ```
   KAGGLE_USER=<you> ./build/make_build_dataset.sh           # --dry-run first to check tools and login
   ```
1. **Build the weights** (on a PC, uses one free Kaggle T4 session, about 0.4 GPU-hours):
   ```
   KAGGLE_USER=<you> ./build/build_model.sh build/out        # from the repo root; --dry-run writes the notebook only
   python build/verify_build.py build/out/weights           # tensor-level comparison with the validated deployment
   ```
   Starts from `NHirose/omnivla-original`, runs GPTQ calibration on 128 held-out FrodoBots samples, quantizes the LLM to
   per-channel int4 packed for Marlin and the vision encoders to HQQ 4-bit (repacked for GemLite at load), assembles
   `weights/` (shards, `base.safetensors`, `model/`), writes `SHA256SUMS`, per-tensor hashes and `BUILD_MANIFEST.json`
   (checkpoint revision, package versions), downloads it and verifies the checksums. Every package is pinned in the
   script; Kaggle's torch version is recorded in the manifest.
2. **Set up the Jetson** (on a fresh Orin Nano 8 GB with JetPack 6.2, from this folder):
   ```
   ./setup_jetson.sh                                       # weights from Hugging Face
   ./setup_jetson.sh --weights-src <copy of build/out/weights>   # or your own build
   ```
   Checks the NVMe mount, asks before each system change (MAXN SUPER, headless boot, 16 GB swap, page-cache helper,
   sudoers entries), builds the venv (Jetson torch 2.8.0 wheel + `constraints.txt`, transformers fork, pinned
   packages, Marlin from source), clones OmniVLA @5182600 and applies `patches/omnivla.patch`, downloads (or copies) and verifies the
   weights, and ends with `tools/reference_check.py`: 10 reference frames x 2 modes must match the validated
   deployment bit-exactly. Safe to rerun (every step checks first). `--no-system` never changes system settings,
   `--yes` accepts all, `--yes-runtime` only the non-persistent ones (stopping services, max clocks during the check).

Every change to OmniVLA's code and behavior is listed in `CHANGES_TO_OMNIVLA.md`. The manual steps below are what the
two scripts automate.

## Setup from scratch on a fresh Orin Nano 8 GB

Tested on:

- Jetson Orin Nano Engineering Reference Developer Kit Super, NVMe SSD.
- JetPack 6.2 (L4T R36.4.7), Ubuntu 22.04, CUDA 12.6, Python 3.10.
- Power mode 2 (MAXN SUPER).

Steps 1-3 need sudo once. Everything after runs as a normal user. The paths below assume the NVMe is mounted at `/mnt/nvme`.

### 1. Boot headless and free memory
The 7B model needs about 5.7 GB of the 7.6 GB unified memory, so the desktop must not run.
```
sudo systemctl set-default multi-user.target      # no desktop (saves ~0.8 GB); undo with graphical.target
sudo nvpmodel -m 2                                # MAXN SUPER (Orin Nano Super); check with: sudo nvpmodel -q
sudo reboot
```
Before every run, stop the services that hold memory. `launch.sh` warns if less than 5.7 GB is available.
```
sudo systemctl stop docker docker.socket containerd snapd snapd.socket jtop
free -h        # "available" must be >= 5.7 GB
```

### 2. Swap
The 7B load has short peaks. Add a 16 GB NVMe swap file next to the default zram:
```
sudo fallocate -l 16G /mnt/nvme/swapfile && sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile && sudo swapon /mnt/nvme/swapfile
echo '/mnt/nvme/swapfile none swap sw,nofail 0 0' | sudo tee -a /etc/fstab
```

### 3. The page-cache helper and sudoers entries
The GPU shares RAM with the page cache, so `launch.sh` drops the cache every second while the model runs.
```
sudo tee /usr/local/bin/omni-dropcache >/dev/null <<'EOF'
#!/bin/sh
while true; do sync; echo 3 > /proc/sys/vm/drop_caches; sleep 1; done
EOF
sudo chmod 755 /usr/local/bin/omni-dropcache
sudo visudo -f /etc/sudoers.d/omnivla     # add (replace USER):
USER ALL=(root) NOPASSWD: /usr/bin/jetson_clocks, /usr/local/bin/omni-dropcache, /usr/bin/pkill -f omni-dropcache, /usr/bin/tegrastats, /usr/bin/systemctl stop *, /usr/sbin/reboot
```
Check that `sudo -n /usr/bin/jetson_clocks` runs without a password prompt.

### 4. Python venv with the Jetson PyTorch
Use NVIDIA's Jetson wheels only. A generic PyPI torch has no CUDA on the Orin.
```
sudo apt-get install -y python3-venv python3-dev git     # if missing
cd /mnt/nvme/omnivla/deploy
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # 2.8.0 True
```
From here on, **every** pip command takes `-c constraints.txt`, so torch, torchvision and numpy stay the Jetson builds.
The torch step may pull numpy 2.x as a dependency. The first `-c constraints.txt` install in step 5 puts it back to 1.26.4, which the pinned packages need. Step 5 ends with a check.

### 5. OmniVLA code, the transformers fork, the other packages, and Marlin
```
git clone https://github.com/NHirose/OmniVLA.git /mnt/nvme/omnivla/OmniVLA
cd /mnt/nvme/omnivla/OmniVLA && git checkout 5182600cb4a9ee07684e17cdd2a6cbafc56b8a68
git apply /mnt/nvme/omnivla/deploy/patches/omnivla.patch          # training-only import made optional
cd /mnt/nvme/omnivla/deploy
pip install -c constraints.txt "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab"
pip install -c constraints.txt -r requirements-deploy.txt
git clone https://github.com/IST-DASLab/marlin.git ~/src/marlin && git -C ~/src/marlin checkout 1f25790
TORCH_CUDA_ARCH_LIST=8.7 pip install -c constraints.txt --no-build-isolation ~/src/marlin     # ~2 min (nvcc 12.6)
python -c "import torch, numpy, marlin, gemlite, hqq; print(torch.__version__, numpy.__version__, torch.cuda.is_available())"   # 2.8.0 1.26.4 True
```
The exact validated environment is recorded in `tools/deploy_venv_freeze.txt`.
**Do not `pip install marlin`.** The PyPI package named `marlin` is an unrelated project.

OmniVLA **must** use the OpenVLA-OFT transformers fork. Stock `transformers` silently runs causal attention, which gives actions about 1 unit off.

### 6. Weights
**Option A: the pre-built weights on Hugging Face** (https://huggingface.co/jayden1711/omnivla-7b-jetson-int4). `setup_jetson.sh` downloads them by default (log in
first with `hf auth login` if the repo is private). To do it by hand, or to check a copy:
```
cd /mnt/nvme/omnivla/deploy/weights && sha256sum -c SHA256SUMS      # every line must say OK
```
**Option B: rebuild from scratch** with `build/build_model.sh` (see above). The result is identical, tensor for tensor,
to the validated weights (`build/verify_build.py` checks this).

### 7. Run
```
./launch.sh                      # smoke test: loads in ~45 s, 3 pose-mode predictions, prints a velocity command
```
Python API (through `./launch.sh your_script.py`):
```python
from omnivla_deploy import OmniVLADeploy
m = OmniVLADeploy("/mnt/nvme/omnivla/deploy")
out = m.predict(pil_img, mode=4, goal_pose=m.goal_pose_from_gps(lat, lon, compass_deg, goal_lat, goal_lon))
v, w = m.actions_to_cmd(out["actions"])        # OmniVLA's own controller (m/s, rad/s)
```
Modes:

| Mode | Goal | Argument |
|---|---|---|
| 4 | pose goal | `goal_pose` = [x, y, cos, sin] in model units |
| 6 | image goal | `goal_image` |
| 7 | language | `lang` |
| 8 | language + pose | `lang` and `goal_pose` |

Only modes 4 and 6 were validated on the Jetson.

The `omnivla_jetson` package in the repo root wraps the same runtime (use it from a checkout of this repo, where
`setup_jetson.sh` puts the weights in `deploy/weights`). It picks the mode from the goals you pass,
accepts PIL images, numpy arrays or file paths, and returns the command with the waypoints. Its outputs are the
runtime's, unchanged (`tools/api_check.py` checks this on the Jetson against the reference outputs):
```python
from omnivla_jetson import OmniVLAJetson
m = OmniVLAJetson("deploy/weights")                                          # path from the repo root
p = m.predict(img, goal_pose=OmniVLAJetson.goal_pose_from_xy_yaw(3.0, 0.0))   # or goal_image=..., instruction=...
p.waypoints, p.linear, p.angular
```
`examples/` has one script per mode (`./launch.sh ../examples/pose_goal.py frame.jpg 3 0`).

If something fails, see [troubleshooting.md](troubleshooting.md).

## Troupe rover integration (ROS 2)
There is no velocity topic on the rover. The Troupe drivers send `servo_cmd_t` UDP packets to `servo_daemon` on the BeagleBone (192.168.7.2:5005), and that address is reachable from the Pi's USB link.

- **On the Jetson:** `ros2/omnivla_nav_node.py` subscribes to `/troupe/camera/color/compressed`. It infers asynchronously on the newest frame, executes the current 8-step chunk at 20 Hz, and publishes `~/servo_cmd` as `[throttle_ch, throttle_pct, steer_ch, steer_pct]`.
- **On the Pi:** `ros2/servo_udp_bridge.py` forwards each command as two 5-byte `<Bf` packets. It has its own watchdog, because the daemon has none.

Run them:
```
./launch.sh --ros --ros-args -p mode:=4 -p goal_xy_yaw:="[5.0, 0.0, 0.0]"          # Jetson
python3 servo_udp_bridge.py --ros-args -r /servo_udp_bridge/servo_cmd:=/omnivla_nav/servo_cmd   # Pi
```
Both machines need the same `ROS_DOMAIN_ID` (Troupe uses 42).

**Safety:**
- **Stale frames:** neutral when the newest frame is more than 0.5 s old.
- **Used-up chunk:** neutral once the chunk has been executed (2.4 s after its frame).
- **E-stop:** `/omnivla_nav/estop` (Bool) latches neutral until it is released.
- **Manual override:** `/omnivla_nav/manual_override` (Bool) makes the node send neutral once and then go silent, so the Troupe keyboard teleop is the only writer.
- **Speed caps:** `max_v` 0.3 m/s and `max_w` 0.3 rad/s.
- **Hard pulse caps:** throttle at most +0.6 % above neutral in the node, +1.0 % in the bridge.
- **Shutdown:** the node sends neutral on SIGINT or SIGTERM, then the bridge watchdog sends neutral for 1 s and goes silent.
- **Low memory (LOWMEM):** neutral when available memory is below `min_mem_available_mb` (300).
- **Warm-up:** the node compiles its kernels (about 14 s) before it subscribes to the camera, so the first chunk is fresh.

**Before any real test, verify on the bench with the wheels off the ground.** The throttle and steering channels and their directions differ between Troupe versions:
- The newest `drive_controller.py`: throttle on channel 1, forward = pulse above 7.5.
- The older one: throttle on channel 2, forward = 5.0.

The node's defaults follow the newest version. Set `throttle_ch`, `steer_ch`, `throttle_invert`, `steer_invert`, `wheelbase_m`, `max_steer_rad` and `v_full` to match the chassis. `waypoint_spacing_m` (0.1 m, as in OmniVLA's controller) sets the metric scale of the actions; FrodoBots training data used 0.25 m.

Offline test (no rover; DDS stays on localhost; the bridge sends to 127.0.0.1): `ros2/run_ros_test.sh` replays FrodoBots frames as the camera topic and checks the command stream. Unit tests: `python3 ros2/test_rover_protocol.py`.

## Known pitfalls (each of these cost us a crash or a wrong result)
- **The transformers fork is required** (see step 5).
- **Grouped Marlin (groupsize 128) is wrong and non-deterministic on sm_87** at Llama-7B shapes. Only per-channel (`groupsize=-1`) is used here.
- **Out-of-memory crashes** (`NVML_SUCCESS == r INTERNAL ASSERT FAILED`) leak NvMap memory. Reboot before the next run.
- **Never load the whole pre-packed file with `torch.load(mmap=True)`.** Mapped pages stay resident until the last tensor is freed, and the page-cache drop can't reclaim them. The loader reads 256 MB shards instead.
- **Always `.detach()` when copying saved tensors to the GPU.** Saved biases are Parameters; a non-detached copy keeps the CPU tensor, and with it the file, alive.
- **The loader calls `malloc_trim(0)` after each shard.** Without it, glibc keeps about 0.5 GB of freed buffers.
- **Quantizing on the Jetson itself (HQQ) peaks too high and leaves about 0.4 GB of buffers behind.** Always quantize offline.
- **GemLite needs layer widths divisible by 32.** With HQQ4 vision, SigLIP's MLP (4304) therefore stays on HQQ's portable kernel (99 ms per image). Keeping it in fp16 saves 0.12 s but leaves only about 0.4 GB of headroom in image mode, and it ran out of memory once. Marlin vision (the default) avoids both by zero-padding to Marlin's sizes.
- **GPTQ zeroes dead input columns before it sets a channel's scale.** An export must use the scale GPTQ used (`gptq(..., ret_scale=True)`), not one recomputed from the original weights: the first vision export failed its grid check on SigLIP rows whose largest weight sat in a never-active input.
- **`tegrastats --logfile` appends.** Delete the log before measuring.
- **One 7B process at a time.** Loading two models, or switching goal modes over and over in one long-lived process, has run out of memory.
