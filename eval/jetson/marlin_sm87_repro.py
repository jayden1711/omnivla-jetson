"""Minimal reproduction: Marlin FP16xINT4 returns wrong results for m > 64 rows on Jetson Orin (sm_87).

Uses Marlin's own test helpers (gen_quant4 + marlin.mul, copied verbatim from test.py at commit 1f25790) and
Marlin's own pass criterion (mean |C - C_ref| / mean |C_ref| < 1e-3). Sweeps the number of rows m (tokens),
LLaMA-2-7B layer shapes, and the thread-tile setting, then checks the chunking workaround (<= 64 rows per call).

Run:  python marlin_sm87_repro.py
"""
import numpy as np
import torch
import torch.nn as nn

import marlin

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)
DEV = torch.device("cuda:0")


def gen_quant4(m, n, groupsize=-1):          # verbatim from IST-DASLab/marlin test.py @ 1f25790
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


def run(m, k, n, thread_k, thread_n, groupsize=128, chunk=None):
    A = torch.randn((m, k), dtype=torch.half, device=DEV)
    B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)
    C = torch.zeros((m, n), dtype=torch.half, device=DEV)
    step = chunk or m
    for j in range(0, m, step):
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        marlin.mul(A[j:j + step], B, C[j:j + step], s, workspace, thread_k, thread_n, -1)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B_ref)
    err = (C - C_ref).abs().float()
    rel_mean = float(err.mean() / C_ref.abs().float().mean())
    row_bad = (err.mean(1) / C_ref.abs().float().mean() > 1e-2).nonzero().flatten()
    first_bad = int(row_bad[0]) if len(row_bad) else None
    return rel_mean, float(err.max() / C_ref.abs().float().max()), first_bad, len(row_bad)


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} | torch {torch.__version__} "
          f"cuda {torch.version.cuda} | arch list {torch.cuda.get_arch_list()}")
    SHAPES = [(1024, 512), (4096, 4096), (4096, 11008), (11008, 4096)]  # (k=in, n=out): upstream test_tiles shape, attn, MLP up/gate, MLP down
    ROWS = [16, 64, 65, 80, 128, 300, 555]
    TILES = [(-1, -1), (64, 256), (128, 128)]
    print("\ngroup k      n      thread  m     rel_mean   rel_max   first_bad_row  bad_rows  status")
    fails = 0
    for gs in (128, -1):
        for (k, n) in SHAPES:
            for tk, tn in TILES:
                for m in ROWS:
                    try:
                        rm, rx, fb, nb = run(m, k, n, tk, tn, groupsize=gs)
                    except RuntimeError as e:
                        print(f"{gs:<5} {k:<6} {n:<6} {tk:>3},{tn:<4} {m:<5} n/a ({str(e)[:60]})")
                        break
                    ok = rm < 1e-3
                    fails += not ok
                    print(f"{gs:<5} {k:<6} {n:<6} {tk:>3},{tn:<4} {m:<5} {rm:.2e}  {rx:.2e}  {str(fb):>13}  {nb:>8}  {'OK' if ok else 'FAIL'}")
    print(f"\n{fails} failing configurations (Marlin's own criterion: rel_mean < 1e-3)")
    print("\nWorkaround: call marlin.mul on <= 64 rows at a time (auto thread tiles)")
    for (k, n) in SHAPES:
        for m in (75, 300, 555):
            rm, rx, fb, nb = run(m, k, n, -1, -1, chunk=64)
            print(f"  k={k} n={n} m={m}: rel_mean {rm:.2e} rel_max {rx:.2e} -> {'OK' if rm < 1e-3 else 'FAIL'}")
