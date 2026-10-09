# Running the student on the Dragon Q6A NPU

What has been measured on the board, how to reproduce it, and what to keep in mind when fine-tuning the
student or training a bigger one (on any machine) so that it still deploys. The method and the silent
failure modes come from the yolo-test repo on the board (`~/Documents/yolo-test/docs/hexagon-npu.md`).
That document covers them in more depth; this one is specific to minisam3d.

## Benchmarks (measured 2026-10-09)

Model: `runs/v3rot_loso_all/student.onnx`. MobileNetV2 width 1.0, grayscale 1x256x256 input, 14 joints,
global xyz head and 9 segment rotations, 2.69M params. Errors are against fp32 on CPU, over the 50 NOARK
calibration crops in `calib/`. Timing covers 1000 frames, single stream, with the crop already prepared.

| | fp32 CPU (4x A78) | **W8A16 NPU** | W8A8 NPU |
|---|---|---|---|
| AI Hub quantize job | - | `j5wy9o3jg` | `j5wy9oq6g` |
| network | 41-58 ms | **4.4 ms** | 2.4 ms |
| network + numpy decode | - | **8.7 ms (114 fps)** | 4.4 ms (226 fps) |
| 2D keypoints, mean / p95 / max | ref | 0.15 / 0.46 / 1.5 px | 0.26 / 0.87 / 2.7 px |
| 3D xyz, mean / p95 / max | ref | 1.2 / 3.0 / 4.8 mm | 3.0 / 6.5 / 11.8 mm |
| segment rotations, mean / p95 / max | ref | 0.28 / 0.61 / 1.1 deg | 0.73 / 1.6 / 2.2 deg |
| session init (HTP compile) | - | 1.8 s | 0.8 s |

- **Use W8A16 by default.** Its error is far below the student-vs-teacher gap, and 4.4 ms is not the
  bottleneck: capture, cropping and the CPU decode are. Move to W8A8 only when you run several cameras or
  people per frame, and score it against mocap first.
- Unlike YOLO26 in yolo-test, int8 activations do not break this model on the hardware. All outputs keep
  their full range: heatmap logits about [-17, 15], xyz within ±0.6 m, rot6d within ±3.4.
- **The numpy decode (2-4 ms) costs about as much as the network.** It is a softmax over 14x64x64, the
  soft-argmax and rot6d->matrix. If frame rate matters, optimise this before anything on the NPU.
- The pixel errors are in the 256x256 crop. In the full fisheye frame they shrink by `box_size / 256`.
- The test crops are the calibration crops, so the error is slightly optimistic. The number that matters
  is NOARK's mocap score using the quantized model.
- Temperature: the CPU idles at 49-50 °C with the heatsink and drops to about 52 °C while the NPU does the
  work. Ignore the `ufs-thermal` zone. It reads a constant ~78 °C with no trip points and does not
  correspond to any hardware on this board (it has no UFS, only NVMe). `npu_bench.py` reads only the
  `cpu*` zones.

## The hardware and runtime

| | |
|---|---|
| Board | Radxa Dragon Q6A, QCM6490 (AI Hub calls it `chipset:qcs6490`) |
| NPU | Hexagon **v68** HTP. Its fast path is INT8 weights with INT8/INT16 activations. fp16 also runs but is slower |
| CPU | 4x A78 + 4x A55, 7.4 GB RAM. Use 4 threads for CPU work; the A55s only slow it down |
| OS / Python | Ubuntu 24.04 aarch64, Python 3.13 (required: `onnxruntime-qnn` only ships a cp313 aarch64 wheel) |
| Runtime | QDQ ONNX on ONNX Runtime + `onnxruntime-qnn` (QNN execution provider). The wheel ships the v68 HTP libs, so no QAIRT SDK or Qualcomm account is needed to run |
| Quantizer | Qualcomm AI Hub (cloud). The board's `qai-hub` is already configured with a token |
| AI Hub device | `Dragonwing RB3 Gen 2 Vision Kit` (v68). `QCS6490 (Proxy)` no longer exists |

This replaces the TFLite + `libQnnTFLiteDelegate.so` path in the README (section 4) and in
`scripts/aihub_compile.py` / `run_q6a.py`. The QDQ ONNX keeps float I/O, so no int8 boundary handling is
needed, and the EP compiles it for whatever HTP it finds.

## Reproduce

On the board, in `~/minisam3d`, after `uv sync --locked`:

```bash
# calibration crops (needs the NOARK recordings; run on the PC, then copy calib/ over)
python scripts/make_calib.py --per-subject 10                          # -> calib/calib.npy (50,1,256,256)

.venv/bin/python scripts/npu_quantize.py --onnx runs/<run>/student.onnx               # W8A16
.venv/bin/python scripts/npu_quantize.py --onnx runs/<run>/student.onnx --activations INT8 \
    --dataset d7x8wkq59                                                # reuse the uploaded calib set
.venv/bin/python scripts/npu_bench.py --fp32 runs/<run>/student.onnx \
    --model runs/<run>/models/w8a16_<job>/model.onnx [--runs 1000]
```

`d7x8wkq59` is the AI Hub dataset ID of the current 50 crops. It is valid for any model with the same
input (gray, 1x256x256, `image`, [0,1]).

## Training a new or bigger model: what keeps it deployable

These constraints apply on whatever machine you train on. Nothing about training itself needs the NPU, but
the exported graph has to satisfy all of them.

**Graph and export**
- Keep the exported op set HTP-native. Today it is only `Conv, Clip (ReLU6), Add, Resize (nearest),
  ReduceMean`, and nothing needs to be cut out of the graph. Avoid `TopK`, `GatherElements`, `NonMaxSuppression`,
  `Mod` and dynamic-shape ops. Attention / LayerNorm / GELU blocks do run on the HTP but are much less
  quantization-friendly than conv + ReLU6, so expect to need W8A16 and to re-measure accuracy.
- Static shapes, batch 1, opset 17 (`scripts/export_onnx.py` already does this). The HTP compiles for a
  fixed shape anyway.
- Keep the decode (softmax, soft-argmax, rot6d) **outside** the graph, as now. Bounded, well-spread outputs
  quantize well. Outputs where the signal lives in rare outliers do not (that is what broke YOLO's int8 head).
- Before quantizing, check the new graph's partition: load the fp32 ONNX with `npu_session` and confirm
  nothing lands on CPU apart from the boundary quantize/dequantize nodes.

**Input / preprocessing**
- The input contract is grayscale, crop via `crop_affine` from the person box, 256x256, float32 /255, name
  `image`. If you change any of it (size, channels, normalisation, crop scale), regenerate `calib/` with
  `make_calib.py` to match and upload a new dataset. A calibration set preprocessed differently from
  inference is a silent accuracy loss.
- Network cost scales roughly with input area: 256x192 ≈ 0.75x, 192x192 ≈ 0.56x.

**Budget for a bigger model** (rough, extrapolated from one data point)
- The current student is about 0.85 GMAC and takes 4.4 ms at W8A16 / 2.4 ms at W8A8. Expect roughly linear
  scaling for conv nets: a model of about 3-4 GMAC (e.g. MobileNetV2 width 1.4 at 256, or a small HRNet)
  lands around 15-20 ms at W8A16, which still allows 30 fps for one person. For reference, AI Hub lists
  HRNetPose (28.5M params, about 7 GMAC, 256x192) at 5.2 ms W8A8 on this SoC.
- The CPU decode does not grow with the backbone, only with K and heatmap resolution (it is O(K x 64 x 64)).
- Measure a candidate before training it: export it with random weights
  (`export_onnx.py` without `--ckpt`), quantize, and bench the speed. Latency does not depend on the weights.

**Every new checkpoint**
- Re-quantize. Quantization parameters are tied to the weights, so a QDQ model from an old checkpoint
  cannot be reused.
- Run `npu_bench.py` and compare against the table above. As a guide, accept W8A16 when the mean xyz error is
  under about 3 mm and the max under about 10 mm vs fp32. Anything much worse points to an outlier-heavy tensor;
  look at the raw output ranges the bench prints.
- Always judge accuracy on the NPU, never on the CPU simulation. For W8A8 the CPU-QDQ run gave a max xyz
  error of 21.9 mm where the hardware gave 11.8 mm. In yolo-test they disagreed in the other direction.
- Add the result as a row to the benchmark table, with its AI Hub job ID.

## Silent failure checklist (from yolo-test)

- `sess.get_providers()` must include `QNNExecutionProvider`. Attach the EP with `add_provider_for_devices`
  (as `minisam3d/npu_session.py` does). `providers=[...]` silently runs on CPU.
- Session init time is a weak signal for this model (0.8-1.8 s even on the NPU, because the graph is
  small). The reliable check is network time: about 4 ms on the NPU vs about 50 ms on CPU.
- Keep `intra_op_num_threads=1` and spinning off in the NPU session, otherwise ORT's idle threads starve
  the CPU decode.
- Check `job.wait().success` on every AI Hub job: a failed job still returns a model stub.
- HTP burst mode (`htp_performance_mode`) is rejected on this board. All numbers above are at default
  NPU clocks.
