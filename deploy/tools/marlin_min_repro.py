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
