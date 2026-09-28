# Troubleshooting

The 7B model uses about 5.7 GB of the Orin Nano's 7.6 GB, and the CPU and GPU share that memory. Most problems come
from memory. Paths are relative to `deploy/`.

## CUDA out of memory while loading

The model needs at least 5.7 GB of available memory when it starts. `launch.sh` warns below that.

1. Check `free -h`. "available" should be 5.7 GB or more (about 5.8 GB is normal on a headless Orin Nano).
2. Boot without the desktop: `sudo systemctl set-default multi-user.target`, then reboot. The desktop takes about 0.8 GB.
3. Stop services that hold memory: `sudo systemctl stop docker docker.socket containerd snapd snapd.socket jtop`.
4. Check that the 16 GB NVMe swap file is on (`swapon --show`). The load has short peaks.
5. Always start through `launch.sh`. It sets the allocator options and drops the page cache every second (see below).
6. Run one model process at a time. A second Python process with torch on the GPU takes a few hundred MB.

If a load or a run crashed with an out-of-memory error, reboot before the next attempt (next section).

## NvMap errors, or an NVML / allocator assert

On Jetson, GPU memory is allocated through NvMap. Messages mentioning NvMap in the output or in `dmesg`, or an NVML
assertion from the CUDA caching allocator, mean the GPU could not get memory. After such a crash the memory is not
always returned: `free -h` then shows less than before, and the next load fails too.

- Reboot (`sudo reboot`), stop the services again, check `free -h`, then retry.
- If it crashes at the same point twice in a row, something else holds memory. Check `ps aux --sort=-rss | head`.
- Do not switch modes many times in one short-lived test process. Early experiments that ran three different modes
  back to back in one process crashed this way. The deployed runtime and the ROS node use one mode per goal and were soak-tested.

## Wrong actions, about 1 unit off, without any error

OmniVLA needs the OpenVLA-OFT transformers fork
(`git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9`). Stock `transformers` silently runs causal
attention in the LLM, and the actions come out about 1 action unit off. Check the installed version:

```
python -c "import inspect, transformers.models.llama.modeling_llama as m; print('fork OK' if 'is_causal=False' in inspect.getsource(m.LlamaSdpaAttention.forward) else 'STOCK transformers: wrong actions')"
```

`tools/reference_check.py` catches this too: it requires 20 predictions to match the validated deployment bit for bit.
Install packages only with `-c constraints.txt`, so torch, torchvision and numpy stay the Jetson builds.

## Available memory shrinks although the model is idle (page cache)

The page cache (file data Linux keeps in RAM) competes with the GPU for the same memory. Reading the 4 GB of weights
fills it, and the kernel does not always free it fast enough for GPU allocations. `launch.sh` runs `omni-dropcache`
(sync and drop caches every second) while the model runs, and stops it on exit. If you start Python directly instead,
nothing drops the cache. If `sudo -n /usr/local/bin/omni-dropcache` asks for a password, the sudoers entry from
`docs/deployment.md` step 3 is missing.

## Memory creep over hours, and restarts

Process memory grows by about 50-65 MB per 10 minutes of continuous inference in both modes (`deploy/SOAK.md`).
Latency and GPU memory stay flat. It is glibc keeping freed per-frame buffers. The runtime already calls
`malloc_trim` every 20 predictions, which halves the growth but does not stop it.

- Restart the node between runs, and at least every 2 hours of continuous driving.
- The ROS node stops the rover (LOWMEM) when available memory falls below 300 MB.
- Do not set `MALLOC_MMAP_THRESHOLD_` or `MALLOC_ARENA_MAX`: they made the growth worse (+175 MB per 10 minutes).

## Slow first predictions

The first calls compile the GPU kernels (about 14 s on the Orin). Call `warmup()` before serving; the ROS node does this at startup.

## `import marlin` fails or imports something else

Marlin is built from source (`setup_jetson.sh` does this, or `docs/deployment.md` step 5). The PyPI package named
`marlin` is an unrelated project. Do not `pip install marlin`.
