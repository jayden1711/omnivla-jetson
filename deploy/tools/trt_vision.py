# trt_vision.py - OmniVLA's two vision encoders (DINOv2-L/14 reg4 and SigLIP-SO400M/14, fine-tuned weights) as TensorRT
# FP16 engines, and the checks that decide whether to use them. Nothing here changes the runtime.
#
#   python deploy/tools/trt_vision.py export OUT_DIR       any machine with torch, timm 0.9.10, onnx (not the Jetson venv):
#       downloads only the vision tensors of NHirose/omnivla-original (range requests, ~1.5 GB), rebuilds both encoders with
#       the runtime's truncated forward (output of the second-to-last block, prefix tokens removed), exports ONNX
#       (fp32, input 1x3x224x224) and saves fixed test inputs with fp32 reference outputs (check.npz).
#   python3 deploy/tools/trt_vision.py build OUT_DIR       Jetson, system python: trtexec --fp16 per encoder; GPU time per
#       call from trtexec -> OUT_DIR/build.json
#   ./deploy/launch.sh deploy/tools/trt_vision.py features OUT_DIR    Jetson: TensorRT and the deployed HQQ4 encoders vs
#       the fp32 reference on the same inputs (cosine, max |d|) -> OUT_DIR/features.json
#   ./deploy/launch.sh deploy/tools/trt_vision.py e2e OUT_DIR dino|both    Jetson: the runtime with TensorRT encoders
#       (HQQ4 weights of the replaced encoders freed): 10 reference frames x modes 4, 6, 7 vs tests/reference, latency,
#       RAM peak -> OUT_DIR/e2e_<which>.json
# TensorRT comes from JetPack (10.3 on JetPack 6.2); the venv does not have it, so it is imported from the system
# dist-packages (same Python 3.10).
import json, os, re, struct, subprocess, sys, time, urllib.request
import numpy as np

CKPT = "https://huggingface.co/NHirose/omnivla-original/resolve/e36a84d4923c041149d441f93f3bdb7092bb5f07/{}"   # the build's pinned revision
ENC = {"dino": ("featurizer", "vit_large_patch14_reg4_dinov2.lvd142m"), "siglip": ("fused_featurizer", "vit_so400m_patch14_siglip_224")}
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(HERE)


def _range(url, a, b):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()


def vision_state_dict(prefix):
    """fine-tuned weights of vision_backbone.<prefix>.* (bf16 in the checkpoint -> fp32), via HTTP range requests"""
    import torch
    idx = json.load(urllib.request.urlopen(CKPT.format("model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if k.startswith(f"vision_backbone.{prefix}.")]
    out = {}
    for shard in sorted({idx[k] for k in keys}):
        url = CKPT.format(shard); n = struct.unpack("<Q", _range(url, 0, 7))[0]
        hdr = json.loads(_range(url, 8, 7 + n)); off = 8 + n
        for k in [k for k in keys if idx[k] == shard]:
            m = hdr[k]; a, b = m["data_offsets"]; raw = _range(url, off + a, off + b - 1)
            if m["dtype"] == "BF16":
                x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                x = np.frombuffer(raw, {"F16": np.float16, "F32": np.float32}[m["dtype"]]).astype(np.float32)
            name = k[len(f"vision_backbone.{prefix}."):].replace(".scale_factor", ".gamma")   # OpenVLA renamed LayerScale
            out[name] = torch.from_numpy(x.reshape(m["shape"]).copy())
    return out


def truncated(model):
    """the runtime's forward (omnivla_deploy._install_vit_trunc): stop after block len-2, drop prefix tokens"""
    import torch

    class T(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, img):
            m = self.m
            x = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(img))))
            for blk in m.blocks[: len(m.blocks) - 1]:
                x = blk(x)
            return x[:, m.num_prefix_tokens:]
    return T(model).eval()


def export(out, only=None, half=False):
    """half=True exports fp16 weights (halves TensorRT's build memory; SigLIP did not build from fp32 on the 8 GB Orin)"""
    import torch, timm
    os.makedirs(out, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    inputs = torch.randn(4, 3, 224, 224, generator=g)                   # fixed test inputs (already normalized scale)
    check = {"inputs": inputs.numpy()}
    for name, (prefix, timm_name) in ENC.items():
        if only and name != only:
            continue
        m = timm.create_model(timm_name, pretrained=False, num_classes=0, img_size=224)
        sd = vision_state_dict(prefix)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith(("head", "fc_norm"))]
        assert not missing and not unexpected, f"{name}: missing {missing[:5]}, unexpected {unexpected[:5]}"
        net = truncated(m)
        with torch.no_grad():
            check[f"ref_{name}"] = np.stack([net(inputs[i:i + 1])[0].numpy() for i in range(len(inputs))])
        if half:
            net = net.half()
        torch.onnx.export(net, inputs[:1].half() if half else inputs[:1], os.path.join(out, f"{name}.onnx"),
                          input_names=["img"], output_names=["feat"], opset_version=17, dynamo=False)
        print(f"[TRT] exported {name}: {sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params, output "
              f"{tuple(check[f'ref_{name}'].shape[1:])}, {len(sd)} tensors loaded", flush=True)
    if only is None:
        np.savez(os.path.join(out, "check.npz"), **check)


def build(out, only=None, fp32=False, tag=""):
    res = json.load(open(os.path.join(out, "build.json"))) if only and os.path.exists(os.path.join(out, "build.json")) else {}
    for name in ENC:
        if only and name != only:
            continue
        eng = os.path.join(out, f"{name}{tag}.engine")
        cmd = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={os.path.join(out, name + '.onnx')}"] + ([] if fp32 else ["--fp16"]) + \
              [f"--saveEngine={eng}", "--memPoolSize=workspace:1024", "--iterations=50", "--warmUp=2000"]
        t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True)
        log = p.stdout + p.stderr; open(os.path.join(out, f"build_{name}{tag}.log"), "w").write(log)
        med = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = [\d.]+ ms, median = ([\d.]+) ms", log)
        res[name + tag] = dict(fp32=fp32, ok=p.returncode == 0 and os.path.exists(eng), build_s=round(time.time() - t),
                         gpu_ms_median=float(med[1]) if med else None,
                         engine_mb=round(os.path.getsize(eng) / 2**20) if os.path.exists(eng) else None)
        print(f"[TRT] {name}{tag}: {res[name + tag]}", flush=True)
    json.dump(res, open(os.path.join(out, "build.json"), "w"), indent=1)


class Engine:
    """a TensorRT engine called on torch CUDA tensors (fp32 in and out, as exported)"""

    def __init__(self, path):
        sys.path.append("/usr/lib/python3.10/dist-packages")        # JetPack's TensorRT bindings
        import tensorrt as trt, torch
        self.torch = torch
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = self.rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.eng.create_execution_context()
        names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inp = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.out = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.out_shape = tuple(self.eng.get_tensor_shape(self.out))
        dt = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self.in_dt, self.out_dt = dt[self.eng.get_tensor_dtype(self.inp)], dt[self.eng.get_tensor_dtype(self.out)]

    def __call__(self, x):
        torch = self.torch
        x = x.to(self.in_dt).contiguous(); y = torch.empty(self.out_shape, dtype=self.out_dt, device=x.device)
        self.ctx.set_tensor_address(self.inp, x.data_ptr()); self.ctx.set_tensor_address(self.out, y.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return y


def _cmp(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return dict(cos=float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), max_abs=float(np.abs(a - b).max()),
                rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)))


def features(out, tag=""):
    import torch
    sys.path.insert(0, D)
    from omnivla_deploy import OmniVLADeploy
    C = np.load(os.path.join(out, "check.npz")); x = torch.from_numpy(C["inputs"]).cuda()
    res = {}
    for name in ENC:
        if not os.path.exists(os.path.join(out, f"{name}{tag}.engine")):
            print(f"[TRT] {name}{tag}: no engine, skipped", flush=True); continue
        e = Engine(os.path.join(out, f"{name}{tag}.engine"))
        t = np.stack([e(x[i:i + 1]).float().cpu().numpy()[0] for i in range(len(x))])
        res[f"trt_{name}{tag}"] = _cmp(t, C[f"ref_{name}"])
        del e; torch.cuda.empty_cache()
    m = OmniVLADeploy(D, verbose=False)                                   # the deployed HQQ4 encoders, same inputs
    for name, (prefix, _) in ENC.items():
        mod = getattr(m.vla.vision_backbone, prefix)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = np.stack([mod(x[i:i + 1].half()).float().cpu().numpy()[0] for i in range(len(x))])
        res[f"hqq4_{name}"] = _cmp(h, C[f"ref_{name}"])
    for k, v in res.items():
        print(f"[TRT] {k} vs fp32: cos {v['cos']:.6f}, rel L2 {v['rel_l2']:.4f}, max |d| {v['max_abs']:.3g}", flush=True)
    json.dump(res, open(os.path.join(out, f"features{tag}.json"), "w"), indent=1)


def e2e(out, which):
    import torch
    from PIL import Image
    sys.path.insert(0, D)
    from omnivla_deploy import OmniVLADeploy
    teg_log = f"/tmp/tegra_trt_{which}.log"
    if os.path.exists(teg_log):
        os.remove(teg_log)
    teg = subprocess.Popen(["tegrastats", "--interval", "500", "--logfile", teg_log])
    R = np.load(os.path.join(D, "tests", "reference", "reference.npz")); REF = os.path.join(D, "tests", "reference")
    frames = sorted(k[6:] for k in R.files if k.startswith("goal__"))
    m = OmniVLADeploy(D, verbose=False)
    names = ["dino"] if which == "dino" else ["dino", "siglip"]
    for name in names:                                                    # replace, then free the HQQ4 weights
        mod = getattr(m.vla.vision_backbone, ENC[name][0])
        eng = Engine(os.path.join(out, f"{name}.engine"))
        mod.blocks = torch.nn.ModuleList()
        mod.forward = (lambda e: (lambda img: e(img).half()))(eng)
        mod._trt = eng
    torch.cuda.empty_cache()
    m.warmup(modes=(4, 6, 7))
    ld = lambda s, f: Image.open(os.path.join(REF, s, f + ".jpg")).convert("RGB")
    lat, dist, acts = {4: [], 6: [], 7: []}, {4: [], 6: [], 7: []}, {}
    for f in frames:
        for mode, kw in ((4, dict(goal_pose=R[f"goal__{f}"])), (6, dict(goal_image=ld("goals", f))), (7, dict(lang=str(R["lang__m7"])))):
            o = m.predict(ld("frames", f), mode=mode, **kw)
            lat[mode].append(o["t_fwd"]); acts[f"act_m{mode}__{f}"] = o["actions"]
            dist[mode].append(float(np.linalg.norm(o["actions"][:, :2] - R[f"act_m{mode}__{f}"][:, :2], axis=1).mean()))
    time.sleep(1.0); teg.terminate()
    ram = max([int(v) for v in re.findall(r"RAM (\d+)/\d+MB", open(teg_log).read())] or [0])
    res = dict(which=which, ram_peak_mb=ram, ram_headroom_mb=7620 - ram, torch_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
               modes={md: dict(fwd_ms_median=float(np.median(lat[md]) * 1000),
                               mean_dist_to_reference=float(np.mean(dist[md]))) for md in lat})
    np.savez(os.path.join(out, f"e2e_{which}_actions.npz"), **acts)
    json.dump(res, open(os.path.join(out, f"e2e_{which}.json"), "w"), indent=1)
    print("[TRT] e2e " + json.dumps(res), flush=True)


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "export":
        export(out, only=next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("--only=")), None), half="--half" in sys.argv)
    elif cmd == "build":
        build(out, only=next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("--only=")), None), fp32="--fp32" in sys.argv,
              tag=next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("--tag=")), ""))
    elif cmd == "features":
        features(out, tag=next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("--tag=")), ""))
    else:
        e2e(out, sys.argv[3])
