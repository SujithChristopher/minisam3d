# miniSAM3D: upper-body 3D pose distilled from SAM 3D Body, for the Radxa Dragon Q6A

## Recommendation in one paragraph

Use SAM 3D Body (`facebook/sam-3d-body-dinov3`, 840M params) only as an **offline teacher** on a GPU.
Distill it into a **2.1M-parameter MobileNetV2 + FPN-lite student** (0.84 GMACs at 256x256) that
predicts, for 14 upper-body joints, a 2D heatmap plus a root-relative 3D location (metres, neck = root).
Quantize to INT8 and run it on the QCS6490 Hexagon NPU through TFLite + the QNN delegate, compiled with
Qualcomm AI Hub. There is no detector in the loop: each frame's crop comes from the previous frame's
keypoints, with a whole-frame fallback when tracking is lost. Compute is not the bottleneck: the
expected NPU time is a few milliseconds per frame, so 20 fps is comfortable and 30 fps is plausible;
camera capture and resize on the CPU will likely dominate.

## 1. What the teacher outputs (checked in the sam-3d-body source)

`SAM3DBodyEstimator.process_one_image(img)` returns one dict per detected person:

| key | shape | meaning |
|---|---|---|
| `bbox` | (4,) | person box, xyxy px (from ViTDet detector inside the estimator) |
| `pred_keypoints_3d` | (70,3) | MHR70 keypoints, metres, camera-oriented, before translation |
| `pred_cam_t` | (3,) | camera translation; `kp3d + cam_t` = camera-frame coordinates |
| `pred_keypoints_2d` | (70,2) | projection of the above into the full image (px) |
| `focal_length` | () | focal used (MoGe-2 FOV estimate, or default FOV) |
| `pred_vertices` | (18439,3) | MHR mesh (not used by the student) |
| `body_pose_params`, `hand_pose_params`, `shape_params`, `global_rot`, ... | | MHR rig parameters |

Upper-body subset used by the student (`minisam3d/keypoints.py`), MHR70 index in brackets:
nose [0], eyes [1,2], ears [3,4], **neck [69] (root)**, shoulders [5,6], elbows [7,8],
wrists L [62] / R [41], hips [9,10]. Hips are kept because they anchor torso orientation;
when they are off-screen their 2D loss is masked and their 3D loss is down-weighted.

Hands (MHR70 21..61, 20 finger joints per hand) are behind a `--hands` flag. At a 256px
upper-body crop a hand is roughly 25-40px, so finger joints from the same crop will be poor.
If fingers matter, the cheap path is a second small hand-crop model later, not more channels here.

## 2. Student

`minisam3d/model.py`

- Backbone: torchvision MobileNetV2 (ImageNet init), taps at stride 8/16/32.
- Decoder: 1x1 laterals + nearest upsample + 3x3 conv, out at stride 4 (64x64 for a 256x256 input).
- Head: `4K` channels = K heatmap logits + 3K location maps (x, y, z).
- Decode (outside the NPU graph, ~0.1 ms of NumPy): softmax over each heatmap, soft-argmax for 2D,
  heatmap-weighted average of the location maps for 3D (VNect-style).
- Exported ONNX op set is only `Conv, Clip (ReLU6), Add, Resize(nearest)`: all HTP-native, INT8-friendly.

| config | params | GMACs |
|---|---|---|
| width 1.0, 256x256 (default) | 2.11M | 0.84 |
| width 1.0, 256x192 | 2.11M | 0.63 |
| width 1.0, 192x192 | 2.11M | 0.47 |
| width 0.75, 224x224 | 1.33M | 0.56 |

For scale: Qualcomm AI Hub lists HRNetPose (28.5M params, ~7 GMACs, 256x192) at **5.2 ms on QCS6490,
TFLite w8a8**. The student is ~8x less compute, so the NPU part should land in roughly 1.5-3 ms
(inferred; confirm with `scripts/aihub_compile.py`, which profiles on a real QCS6490).
On this x86 container it runs in 6.9 ms fp32 on 4 CPU threads, so even CPU-only on the Q6A's
Kryo cores is likely near the 10 fps floor.

## 3. Distillation data

`scripts/label_with_teacher.py` runs the teacher over image folders and videos and writes
`labels/shard_*.npz` (bbox, full MHR70 2D/3D, cam_t, focal per person). Full 70 keypoints are stored,
so the student's joint set can change later without relabeling.

Suggested mix (~150-300k person crops):
1. COCO train2017 persons (filtered to boxes >= 96px), plus AI Challenger / MPII for pose variety.
2. **Your own footage from the camera you will deploy** (same lens, lighting, framing; 20-50k frames at
   stride 5). This is the most valuable data per frame.
3. Optional: upper-body-heavy video (talking heads, gestures, fitness) for truncated framings.

Teacher cost is the main expense: body-only mode (`inference_type="body"`) on one 24GB-class GPU is
probably a few images/s, so 200k crops is on the order of 10-20 GPU-hours (estimate, not measured).

Training augmentation (in `minisam3d/data.py`) crops around the upper body exactly like the runtime
tracker does, then jitters scale/shift, rotates ±30° (rotating 3D about the camera axis to match),
flips with L/R swap, and applies colour/blur/noise.

Loss (`minisam3d/losses.py`): L1 on soft-argmax 2D vs teacher, cross-entropy to a Gaussian heatmap
(keeps the softmax peaky so INT8 decode is stable), L1 on root-relative 3D, and bone-length L1.

## 4. Deployment toolchain

1. `scripts/export_onnx.py` → static 1x3xHxW ONNX.
2. `scripts/aihub_compile.py` → AI Hub quantize job (INT8 weights, INT8 or INT16 activations, calibrated on
   pseudo-labelled crops) → compile for `QCS6490 (Proxy)` → profile on a real device → download `.tflite`
   (or `qnn_context_binary`).
3. On the board: Radxa's QAIRT/QNN runtime libs + `libQnnTFLiteDelegate.so`, then
   `scripts/run_q6a.py --model minisam3d.tflite --camera 0`.
   Known gotcha: "Failed to create device: 14001" usually means a QNN skel/runtime version mismatch.

## 5. Expected accuracy (inferred, not measured)

- Teacher: 54.8 mm MPJPE on 3DPW (full body, per the SAM 3D Body README).
- Student vs teacher: expect a gap of roughly 25-45 mm root-relative MPJPE on upper-body joints,
  smaller on in-domain footage. Wrists and depth (z) will be the weakest.
- INT8 usually costs 1-3 mm more; if it costs more, switch activations to INT16 (`--act int16`),
  which is roughly 1.5-2x slower and still well inside budget.
- 2D quality should be comparable to lightweight top-down 2D models (RTMPose-t/s class) on in-domain data.
- No absolute depth / metric root position: the student gives root-relative 3D plus 2D. Absolute
  position can be added later as one extra regressed value if you need it.

## 6. Status

- [x] Student, data pipeline, losses, training loop, ONNX export, AI Hub script, on-device runner
- [x] CPU smoke test (`scripts/smoke_test.py`): synthetic labels → train → export → ONNX/PyTorch parity
  (max diff 3e-7) → runner loop on a test video; mono model, resume, fisheye runner,
  fisheye_to_views, and the labeler (with a stub teacher) for intrinsics passing and resume
- [ ] Run the teacher on real data (needs a CUDA GPU + Hugging Face access to the gated checkpoint)
- [ ] Train v1
- [ ] Profile on QCS6490 via AI Hub (needs an AI Hub API token)
- [ ] Bring-up on the board

## 7. Camera: Waveshare OV9281, 160-degree fisheye, mono (stereo pair)

- **Fisheye.** Neither the teacher nor the student understands a 160-degree lens, so every image they see is
  a virtual pinhole view rendered from the fisheye frame (`minisam3d/fisheye.py`, OpenCV fisheye model).
  At runtime the view is re-aimed each frame at the tracked person (rotation + FOV), costing ~0.8 ms to build
  the remap table and ~0.2 ms to remap on x86. 3D comes back in the physical camera frame by undoing the view
  rotation. Calibrate once with `scripts/calibrate_fisheye.py` (checkerboard, aim for RMS < 0.5 px).
- **Mono / NIR.** The OV9281 is monochrome and NIR-sensitive. Train with `--gray` (1-channel input, pretrained
  RGB filters folded into one channel). Training converts COCO to gray with randomized channel weights and gamma
  to cover the NIR look; your own footage is what really closes that gap.
- **Own footage for the teacher.** `scripts/fisheye_to_views.py` renders recordings into left/centre/right
  90-degree views with exact intrinsics, which `label_with_teacher.py` passes to SAM 3D Body instead of
  guessing the FOV.
- **Stereo.** Not needed for this pipeline. Later it can give absolute distance by running the student on
  both cameras and triangulating the 2D joints (still well inside the NPU budget).

## 8. Colab

`colab/minisam3d_colab.ipynb` runs the whole thing: install the teacher, COCO person subset, your footage,
resumable labeling into Drive, resumable training, export, and AI Hub profiling. Labeling wants an A100 or L4;
training runs fine on a T4.

## 9. Running on an 8 GB laptop GPU (e.g. RTX 4060, native Windows or WSL2)

The full teacher stack (SAM 3D Body + ViTDet detector + MoGe) likely does not fit in 8 GB. Drop the
detector and MoGe and use bf16 (memory figures are estimates):

```
# COCO: ground-truth person boxes instead of the detector, default FOV instead of MoGe
set PYTHONPATH=C:\path\to\sam-3d-body
python scripts\label_with_teacher.py --inputs D:\coco\coco_person --out D:\msam_labels\coco ^
    --coco-ann D:\coco\annotations\person_keypoints_train2017.json --no-fov --half
# your fisheye footage: exact intrinsics from calibration; whole view = one person
python scripts\label_with_teacher.py --inputs D:\own_views\images --out D:\msam_labels\own --no-detector --half
# student
python scripts\train.py --data D:\msam_labels\coco D:\msam_labels\own --repeat 1 4 ^
    --image-dirs D:\coco D:\own_views --gray --pretrained --bs 128 --workers 4 --out runs\v1 --resume runs\v1\last.pt
```

Label paths are stored with forward slashes, so labels made on Windows also train on Colab or Linux.
On Windows the labeler cannot always create symlinks, which is why `--image-dirs` points training at the images.

## Layout

```
minisam3d/   keypoints.py  model.py  data.py  losses.py  postprocess.py  fisheye.py
scripts/     calibrate_fisheye.py  fisheye_to_views.py  label_with_teacher.py  train.py
             export_onnx.py  aihub_compile.py  run_q6a.py  smoke_test.py
colab/       minisam3d_colab.ipynb
```

Sources: github.com/facebookresearch/sam-3d-body, huggingface.co/qualcomm/HRNetPose (QCS6490 numbers),
github.com/ZephyrSai/dragon_q6a_yolo (Q6A QNN toolchain notes).
