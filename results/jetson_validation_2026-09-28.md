# On-device validation, 2026-09-28 (Jetson Orin Nano 8 GB, JetPack 6.2, MAXN_SUPER, headless)

Runtime: `deploy/omnivla_deploy.py` with mode-dependent pruning (75% for pose and image goals, none for language modes),
deployed GPTQ weights. One job at a time, services stopped, >= 5.7 GB available before each job.

## 1. Reference check (`deploy/tools/reference_check.py`)

- The new runtime and the previous one (kept as `omnivla_deploy.py.bak4`) give bit-identical outputs: 20/20 (10 frames,
  pose goal and image goal).
- Against the reference file recorded on 2026-09-27, both runtimes match 19/20. The one difference: image goal, frame
  fb_p00_16577_t00182, 0.0039 action units (about one fp16 rounding step). Neither runtime reproduces that recorded
  value (5 runs: with and without warm-up, goal-cache hit and miss, the recording script's own protocol); the cause was
  not found. The input images are byte-identical to the ones used for the recording.
- The references were re-recorded from the new runtime for all four modes (`--record-all`; language prompts "move toward
  the bench" and "stay on the path"), and re-checked in a new process: **40/40 bit-identical, PASS**.
- Determinism (`eval/jetson/determinism.sh 50`): 50 fresh processes, each running reference_check's protocol on all 10
  image-goal frames, plus the first frame as a goal-cache miss. **Every output bit-identical in all 50 runs (spread 0) and
  equal to the re-recorded reference**, including frame fb_p00_16577_t00182 on both cache paths. The runtime is
  deterministic; the 2026-09-27 value for that frame remains unexplained.

## 2. Language-mode latency and memory (`eval/jetson/lang_latency.py`)

50 predictions per mode after warm-up; fresh tegrastats log; RAM of 7620 MB.

| Language modes | Forward, median (p90) | End to end | Peak torch memory | RAM peak | RAM left |
|---|---|---|---|---|---|
| No pruning (default) | mode 7: 768 ms (772), mode 8: 770 ms (773) | 775 / 778 ms | 4.45 GiB | 6820 MB | 800 MB |
| 50% pruning (`lang_prune_frac=0.5`) | mode 7: 538 ms (540), mode 8: 539 ms (541) | 545 / 546 ms | 4.34 GiB | 6695 MB | 925 MB |

**Differs between boots.** These runs were made after about 45 hours of uptime. After a reboot the same script gave
mode 7: 699 ms, mode 8: 701 ms and 942 MB of RAM left, which first looked like an uptime effect. But after a later
reboot (5 minutes of uptime) mode 7 was back at 769 ms. So language-mode latency is about 700 ms on some boots and
about 770 ms on others, stable within a boot; pose and image goal differ by only 1-2%. The cause was not investigated;
compare configurations only within one boot.

For comparison (results/gptq_validation.md): pose goal 424 ms with 1055 MB left, image goal 1056 ms with 851 MB left.
Object-goal accuracy (results/lelan_summary.md): 84% without pruning, 81% at 50% (not significantly lower).

## 3. Python package (`deploy/tools/api_check.py`)

**40/40 PASS**: `omnivla_jetson.OmniVLAJetson.predict` returns the recorded outputs bit for bit, and the command equals
`OmniVLADeploy.actions_to_cmd`, in all four modes (10 frames each).

## 4. Setup from scratch (`build/jetson_e2e_test.sh --hf`)

`deploy/setup_jetson.sh --no-system --yes-runtime` in a fresh folder (`setup_test/`: new venv, new OmniVLA clone at
5182600 + patch, Marlin built from source), with the weights downloaded from Hugging Face (jayden1711/omnivla-7b-jetson-int4,
pinned revision 81cc6dc; public, no login needed) and checked against `SHA256SUMS`:

- First run: venv (torch 2.8.0, numpy 1.26.4, transformers fork, hqq 0.2.8.post1), weights verified, correctness check
  **40/40 bit-identical, PASS**, exit 0.
- Second run on top of the first (idempotency): steps skipped as done, correctness check 40/40 PASS. The test harness lost
  this run's exit code (its tmux session ended before the final `echo`); a third run started separately exited 0 with
  40/40 PASS.
- The downloaded weights are identical, file by file, to the existing deployment (base.safetensors, model/config.json,
  action head, SHARDS.json).
- Not covered: the system-changing steps (headless boot, swap, sudoers), which were already done on this Jetson.

## 5. Power (`eval/jetson/power_measure.py`)

15W, 25W and MAXN_SUPER measured (results/power_summary.md; power modes switched by hand with `sudo nvpmodel`, which
is not in the passwordless sudo list; no reboot was needed; MAXN_SUPER restored at the end). 50 predictions per case,
board input power (VDD_IN).

- Energy per prediction hardly depends on the power mode: pose goal 8.8 / 8.6 / 9.3 J, image goal 23.8 / 23.8 / 24.9 J,
  language 16.8 / 15.7 / 17.2 J (MAXN_SUPER / 25W / 15W). Lower modes draw less power but take longer.
- Latency: pose goal 426 / 446 / 561 ms, image goal 1042 / 1158 / 1387 ms, language 771 / 755 / 1002 ms.
- No throttling: the GPU stayed at the mode's maximum clock (1020 MHz at MAXN_SUPER, 612 MHz at 15W), junction
  temperature at most 71 C.
- Unexplained: language mode was about 2% faster at 25W (755 ms) than at MAXN_SUPER (771 ms, reproduced 3 times:
  768, 773, 771 ms), while every other case was 5-11% slower at 25W. The 25W run was made before clock logging was
  added, so its clocks are unknown; not re-measured. All power runs were made before the reboot described in section 2,
  which later made language mode about 9% faster; drifting system state may explain the difference.

## 6. CUDA graphs (`OmniVLADeploy(..., cuda_graphs=True)`, `deploy/cuda_graphs.py`; off by default)

- First version (one graph per LLM layer): ran out of GPU memory while capturing (NvMap error 12, then an NVML allocator
  assert): each capture used a new side stream, whose cached blocks stayed reserved, and 32 layers x 3 token counts kept
  their own input and output buffers. Jetson rebooted per the out-of-memory rule.
- Second version (the whole 32-layer stack as one graph per token count, one side stream): **bit-exact**, 30/30 outputs
  identical to eager mode (10 frames, modes 4, 6, 7) and modes 4/6 identical to the recorded references
  (`deploy/tools/graph_check.py`).
- Which part does what: eager, eager without the LLM's key/value cache (`llm_kv_cache=False`, bit-identical 30/30),
  and CUDA graphs (which also skip that cache), same boot, one process each, 50 predictions per mode
  (`eval/jetson/lang_latency.py`):

| Mode | Eager | Eager, no KV cache | CUDA graphs |
|---|---|---|---|
| 4 pose goal | 424 ms | 426 ms | 395 ms (-7%) |
| 6 image goal (new goal every call) | 1041 ms | 1045 ms | 1012 ms (-3%) |
| 7 language | 769 ms | 772 ms | 733 ms (-5%) |
| 8 language + pose | 769 ms | 771 ms | 734 ms (-5%) |
| RAM left (peak over all four modes) | 771 MB | 1054 MB (+283) | 983 MB (+212) |
| Warm-up (includes capturing) | 18.8 s | 18.8 s | 23.4 s |

  The speed-up comes from the graphs alone (skipping the cache changes latency by < 1%); the memory saving comes from
  skipping the unused cache (the graphs' own buffers take back ~70 MB). On an earlier boot the same comparison gave
  -10% / -5% / -6% (pose / image / language; eager 416 / 1035 / 701 ms, graphs 374 / 981 / 660 ms).
- An earlier comparison in this session (language 769 -> 648 ms, -16%) mixed runs from before and after the reboot and
  overstated the gain.
- 30-minute soak with CUDA graphs (`eval/jetson/graph_soak.py`, cycling modes 4, 6 and 7 on every prediction so all
  graph sets stay in use): 2490 predictions, no NvMap errors; latency drift < 1 ms per 10 minutes in every mode (pose
  395 -> 396 ms, image goal 1011 -> 1013 ms, language 730 -> 735 ms); torch memory flat (4.49 GiB reserved); RAM
  +52 MB per 10 minutes after the first 10 minutes (the same heap growth as without graphs); RAM peak 6690 MB (930 MB
  left); junction temperature at most 73.8 C; GPU clock constant at 1020 MHz.
- **CUDA graphs are now the default** (`cuda_graphs=True`; turned off automatically with `goal_refresh > 1`). With the
  new default, `reference_check.py` passes 40/40 bit-identical in all four modes.

- **No key/value cache by default** (`llm_kv_cache=False`), together with CUDA graphs: reference check 40/40 PASS; same
  boot, graphs with vs without the cache: pose 373 / 374 ms, image goal 980 / 981 ms, language 658 / 661 ms, RAM left
  993 / 985 MB (no difference: the graphs already skipped the cache). 10-minute soak cycling modes 4, 6, 7: 900
  predictions, no NvMap errors, RAM 6550 -> 6578 MB (peak 6588, 1032 MB left), torch memory flat, 73.5 C at most;
  language latency rose from 653 to 664 ms in the first 3 minutes while the chip warmed up, then stayed at 665 ms.

## 7. TensorRT vision encoders (`deploy/tools/trt_vision.py`): not adopted

- ONNX export of both fine-tuned encoders with the runtime's truncated forward: checked against the fp32 PyTorch model
  with ONNX Runtime (cosine 1.000000). The deployed HQQ4 encoders are close to the same fp32 reference (cosine 0.982
  DINOv2, 0.969 SigLIP), which also confirms the export matches the runtime.
- TensorRT 10.3 (JetPack 6.2) on the Orin: the DINOv2 engine builds (FP16: 23.2 ms per image vs 77 ms for HQQ4, 558 MB;
  FP32: 67.0 ms, 1110 MB) but its output is wrong: cosine 0.54 to the fp32 reference in FP16 and in FP32, also when run
  by `trtexec` itself (cosine 0.56), so it is the conversion, not the precision or this repo's wrapper. Not bisected.
- The SigLIP FP16 build ran out of GPU memory (a 1.59 GB block for the fp32 weight constants; NvMap error 12; Jetson
  rebooted afterwards). Not retried: same conversion path, and both engines together (~1.4 GB) would not fit next to
  the LLM (~0.9 GB left), while DINOv2 alone would leave ~0.4-0.5 GB.
- Not adopted: not accurate. Write-up with the reproduction: docs/tensorrt_dinov2_issue.md. The ONNX and engine
  files were deleted from the Jetson afterwards.

## 8. Determinism, prompt-aware pruning, cleanup

- Determinism: see section 1 (50 fresh processes, bit-identical).
- Prompt-aware pruning (Kaggle, not on the Jetson): not adopted, results/lelan_summary.md.
- `setup_test/` and the side folder used for the old-vs-new runtime comparison were deleted from the Jetson after their
  logs were copied.
