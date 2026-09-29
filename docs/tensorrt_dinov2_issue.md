### Summary
On a Jetson Orin Nano (JetPack 6.2, TensorRT 10.3), the TensorRT engine built from OmniVLA's DINOv2 vision encoder
returns wrong features, although the ONNX model it is built from is correct. This happens in FP32 as well as FP16, and
also when the engine is run by `trtexec` itself, so it is not a precision problem and not this repo's Python wrapper.
The cause (which layer or fusion is wrong) was not located. Not reported upstream.

| Model run | Cosine similarity to the PyTorch fp32 output | Relative L2 error |
|---|---|---|
| ONNX model in ONNX Runtime 1.23.2 (Mac, CPU) | 1.000000 | 0.0000 |
| Deployed runtime's HQQ 4-bit DINOv2 (Jetson) | 0.982 | 0.19 |
| TensorRT engine, FP16 (Jetson, `trt_vision.py features`) | 0.543 | 1.06 |
| TensorRT engine, FP32 (Jetson, `trt_vision.py features`) | 0.544 | 1.06 |
| TensorRT engine, FP32, run by `trtexec --loadInputs/--exportOutput` | 0.560 | 1.03 |

The first three rows use 4 fixed random inputs (seed 0, 1x3x224x224, already on the normalized scale); the `trtexec` row
uses the first of them. The engines themselves build and run: FP16 23.2 ms per image (vs 77 ms for the deployed HQQ 4-bit
encoder), 558 MB; FP32 67.0 ms, 1110 MB (`trtexec` median GPU compute time, MAXN_SUPER).

The second encoder, SigLIP-SO400M, did not build at all: the FP16 build ran out of GPU memory while allocating a 1.59 GB
block for the fp32 weight constants (`region-alloc.cpp:allocate:60`, NvMap error 12), then failed with "Could not find
any implementation for node {ForeignNode[/blocks.25/attn/Slice_output_0[Constant].../blocks.25/Add_1]}".

### Model
- DINOv2 ViT-L/14 with 4 register tokens, timm `vit_large_patch14_reg4_dinov2.lvd142m`, img_size 224, with OmniVLA's
  fine-tuned weights (NHirose/omnivla-original @e36a84d, `vision_backbone.featurizer.*`; the checkpoint names the
  LayerScale parameters `scale_factor`, mapped to timm's `gamma`).
- Forward as in the runtime: patch embedding, position embedding, blocks 0..22 (all but the last), prefix tokens (class
  + 4 registers) removed. Output 1x256x1024.

### Environment
- **Export:** macOS, PyTorch 2.14.0 (CPU), timm 0.9.10, onnx 1.23.0; `torch.onnx.export(..., opset_version=17,
  dynamo=False)`, fp32 weights, static input 1x3x224x224.
- **Device:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super (8 GB), MAXN_SUPER.
- **Software:** L4T R36.4.7 (JetPack 6.2), CUDA 12.6, TensorRT 10.3.0 (JetPack package libnvinfer 10.3.0.30-1+cuda12.5),
  `/usr/src/tensorrt/bin/trtexec`.

### Reproduction
```
# any machine with torch, timm 0.9.10, onnx (downloads only the vision tensors, ~1.5 GB):
python deploy/tools/trt_vision.py export OUT
# check the ONNX model (ONNX Runtime): cosine 1.000000 to OUT/check.npz["ref_dino"]
# on the Jetson:
/usr/src/tensorrt/bin/trtexec --onnx=OUT/dino.onnx --saveEngine=OUT/dino_fp32.engine --memPoolSize=workspace:1024
python3 -c "import numpy as np; np.load('OUT/check.npz')['inputs'][0:1].astype(np.float32).tofile('OUT/input0.bin')"
/usr/src/tensorrt/bin/trtexec --loadEngine=OUT/dino_fp32.engine --loadInputs=img:OUT/input0.bin \
    --exportOutput=OUT/out0.json --iterations=1 --warmUp=0 --duration=0
# compare out0.json ("feat", 256x1024) with check.npz["ref_dino"][0]: cosine 0.56
```
`python3 deploy/tools/trt_vision.py build OUT` builds both engines (`--only=dino --fp32 --tag=_fp32` for the FP32 one),
and `./deploy/launch.sh deploy/tools/trt_vision.py features OUT` prints the comparison table above.

### Not tried
Bisecting the network (exporting the first N blocks) to find the first wrong layer; other opsets, the dynamo exporter
or onnx-simplifier; a newer TensorRT release. The ONNX and engine files were
deleted from the Jetson after this write-up; `trt_vision.py export` recreates them.
