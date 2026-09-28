### Summary
On a Jetson Orin Nano (Ampere, sm_87, 8 SMs), `marlin.mul` with **groupsize=128** fails Marlin's own accuracy check at Llama-2-7B layer shapes. The error is non-deterministic: identical calls on identical inputs return different outputs.

- **Upstream test shape passes:** with the shape used by `test_tiles` (k=1024, n=512), groupsize=128 passes at every row count and tile setting.
- **Per-channel is correct:** `groupsize=-1` passes and is deterministic on the same device, for every shape we tried.

The Llama-shape test in `test.py` is disabled, which is probably why this has gone unnoticed.

### Environment
- **Device:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super (8 GB). `torch.cuda.get_device_properties(0)` reports "Orin", compute capability 8.7, 8 SMs.
- **Software:** L4T R36.4.7 (JetPack 6), CUDA 12.6 (nvcc 12.6.68), Python 3.10.12.
- **PyTorch:** 2.8.0, NVIDIA Jetson build (`torch.cuda.get_arch_list()` = `['sm_87']`).
- **Marlin:** commit `1f25790`, built with `TORCH_CUDA_ARCH_LIST=8.7 pip install --no-build-isolation .`

### Minimal reproduction
This uses `gen_quant4` copied verbatim from `test.py` @1f25790, `marlin.mul` with automatic tiles, and Marlin's own criterion `mean|C - C_ref| / mean|C_ref| < 1e-3`. Each shape is called 5 times on identical inputs.

<details><summary>marlin_min_repro.py</summary>

```python
"""Minimal repro: Marlin groupsize=128 gives wrong, non-deterministic results at Llama-7B shapes on sm_87 (Jetson Orin).
Uses gen_quant4 from test.py @1f25790 (verbatim) and Marlin's own criterion mean|C-C_ref|/mean|C_ref| < 1e-3."""
import torch, torch.nn as nn
import marlin

DEV = torch.device("cuda:0")
torch.manual_seed(0)


def gen_quant4(m, n, groupsize=-1):  # verbatim from IST-DASLab/marlin test.py @ 1f25790
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    return ref, layer.B, layer.s


p = torch.cuda.get_device_properties(0)
print(f"{p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} torch {torch.__version__} cuda {torch.version.cuda}")
for gs in (128, -1):
    for (m, k, n) in [(16, 1024, 512), (300, 4096, 4096), (300, 4096, 11008), (300, 11008, 4096)]:
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n, groupsize=gs)
        C_ref = torch.matmul(A, B_ref)
        outs = []
        for _ in range(5):                                   # identical calls
            C = torch.zeros((m, n), dtype=torch.half, device=DEV)
            ws = torch.zeros(n // 128 * 16, device=DEV)
            marlin.mul(A, B, C, s, ws, -1, -1, -1)
            torch.cuda.synchronize(); outs.append(C)
        err = [float((C - C_ref).abs().float().mean() / C_ref.abs().float().mean()) for C in outs]
        det = all(torch.equal(outs[0], C) for C in outs[1:])
        print(f"groupsize {gs:>4}  m={m:<4} k={k:<6} n={n:<6} rel_err {' '.join(f'{e:.1e}' for e in err)}  "
              f"identical: {det}  -> {'OK' if max(err) < 1e-3 and det else 'FAIL'}")
```
</details>

### Expected
Every case passes the criterion (rel_err < 1e-3), and repeated identical calls return bitwise-identical outputs. That is what we observe for groupsize=-1, and for groupsize=128 at k=1024, n=512.

### Actual (Jetson Orin Nano, sm_87)
```
Orin sm_87 SMs=8 torch 2.8.0 cuda 12.6
groupsize  128  m=16   k=1024   n=512    rel_err 2.4e-04 2.4e-04 2.4e-04 2.4e-04 2.4e-04  identical: True  -> OK
groupsize  128  m=300  k=4096   n=4096   rel_err 5.1e-03 4.0e-03 3.6e-04 6.7e-04 2.0e-03  identical: False  -> FAIL
groupsize  128  m=300  k=4096   n=11008  rel_err 3.3e-03 3.4e-03 6.5e-03 2.4e-03 5.1e-03  identical: False  -> FAIL
groupsize  128  m=300  k=11008  n=4096   rel_err 8.4e-03 3.9e-03 4.2e-03 7.3e-03 5.4e-03  identical: False  -> FAIL
groupsize   -1  m=16   k=1024   n=512    rel_err 3.5e-04 3.5e-04 3.5e-04 3.5e-04 3.5e-04  identical: True  -> OK
groupsize   -1  m=300  k=4096   n=4096   rel_err 2.7e-04 2.7e-04 2.7e-04 2.7e-04 2.7e-04  identical: True  -> OK
groupsize   -1  m=300  k=4096   n=11008  rel_err 2.4e-04 2.4e-04 2.4e-04 2.4e-04 2.4e-04  identical: True  -> OK
groupsize   -1  m=300  k=11008  n=4096   rel_err 2.3e-04 2.3e-04 2.3e-04 2.3e-04 2.3e-04  identical: True  -> OK
```

A fuller sweep over rows m in {16, 64, 65, 80, 128, 300, 555}, shapes (k, n) in {(1024, 512), (4096, 4096), (4096, 11008), (11008, 4096)}, and thread tiles in {auto, (64, 256), (128, 128)} found the following for groupsize=128:
- **36 failing configurations** at the three Llama-7B shapes, with rel_mean 1.3e-3 to 2e-2 and rel_max up to ~0.13.
  - Errors often corrupt whole 64-row blocks (the first bad row is 0, 64, 128, ...).
  - Failures occur with both auto and (64, 256) tiles.
  - They are intermittent even at m <= 64: m = 64 failed in the sweep but passed 6/6 in a separate determinism check.
- **Splitting the input into <= 64-row calls is not a reliable workaround:** 5 of 9 chunked cases still failed.
- **All k=1024, n=512 cases pass.**
- (128, 128) tiles raise `No kernel implementation for thread_k=128, thread_n=128, groupsize=128` for m > 16. This looks like an intended limitation, not part of this bug.

The non-determinism suggests a race, for example in the grouped path's global reduction or lock workspace on an 8-SM part.

### Why the tests don't catch it
- `test_llama_shapes` in `test.py` @1f25790 starts with a bare `return`, so it is disabled. Even if enabled, it only uses batch sizes 1 and 16 with (128, 128) tiles.
- `test_tiles` covers m up to 1024, but only at k=1024, n=512, which is a shape that passes on sm_87.
- So no test exercises groupsize=128 at Llama-sized k/n with m > 16 and automatic tiles.

### Per-channel (groupsize=-1) result
Correct (rel_mean <= 4e-4) and deterministic for every shape, row count, and supported tile setting above.

We also checked real Llama-2-7B weights: layers 0 q_proj, 15 up_proj and 31 down_proj, which cover all three LLM shapes. They were packed with `Layer.pack` on a different GPU (T4, sm_75) and run on the Orin:
- At m = 75, 300 and 555, rel_mean was 2.8e-4 to 3.1e-4 against the dequantize-then-matmul reference.
- 50 repeated calls per shape were bitwise identical (9/9 cases).

Per-channel Marlin runs our 7B VLA end to end on the Orin at ~1.9x the speed of bitsandbytes NF4. So a fix for the grouped path on sm_87 would be very welcome. We are happy to run a patched build on the device.

### Context
We found this while evaluating fused low-bit kernels for OmniVLA (a 7B vision-language-action model) on the Jetson Orin Nano.

Side note: HQQ 0.2.8's `patch_hqq_to_marlin` cannot reach Marlin, because `BaseQuantizeConfig(group_size=None)` is rewritten to `group_size = in_features`, which the patcher then rejects. We called Marlin directly.
