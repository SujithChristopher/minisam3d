"""Quantize + compile + profile the student on a real QCS6490 via Qualcomm AI Hub.

Needs a free AI Hub account: `pip install qai-hub && qai-hub configure --api_token <token>`.

  python scripts/aihub_compile.py --onnx student.onnx --calib /data/msam_labels --runtime tflite
Prints on-device latency and downloads the compiled model.
"""
import argparse
import os
import sys

import numpy as np
import qai_hub as hub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def calib_data(root, n, H, W, gray=False, image_dirs=()):
    if root is None:
        print("no --calib: using random calibration data (latency only, accuracy meaningless)")
        return {"image": [np.random.rand(1, 1 if gray else 3, H, W).astype(np.float32) for _ in range(8)]}
    from minisam3d.data import PseudoLabelDataset
    ds = PseudoLabelDataset(root, H, W, train=False, gray=gray, image_dirs=image_dirs)
    ids = np.random.default_rng(0).choice(len(ds), min(n, len(ds)), replace=False)
    return {"image": [ds[int(i)]["img"][None].numpy() for i in ids]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", help="pseudo-label root for calibration images")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--device", default="QCS6490 (Proxy)",
                    help='see `qai-hub list-devices`; "RB3 Gen 2 (Proxy)" is the same SoC')
    ap.add_argument("--runtime", default="tflite", choices=["tflite", "qnn_context_binary", "onnx", "qnn_dlc"])
    ap.add_argument("--act", default="int8", choices=["int8", "int16"])
    ap.add_argument("--gray", action="store_true", help="model was trained with --gray")
    ap.add_argument("--image-dirs", nargs="*", default=[])
    ap.add_argument("--fp", action="store_true", help="skip quantization (fp16 on HTP)")
    ap.add_argument("--outdir", default="aihub_out")
    a = ap.parse_args()
    H, W = a.size
    dev = hub.Device(a.device)
    model = a.onnx
    if not a.fp:
        qj = hub.submit_quantize_job(
            model=a.onnx, calibration_data=calib_data(a.calib, a.n_calib, H, W, a.gray, a.image_dirs),
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8 if a.act == "int8" else hub.QuantizeDtype.INT16,
            name="minisam3d-quant")
        model = qj.get_target_model()
    cj = hub.submit_compile_job(model=model, device=dev, name="minisam3d",
                                options=f"--target_runtime {a.runtime}" +
                                (" --quantize_io" if not a.fp else ""))
    target = cj.get_target_model()
    pj = hub.submit_profile_job(model=target, device=dev, name="minisam3d-profile")
    prof = pj.download_profile()
    s = prof["execution_summary"]
    print(f"on-device estimated inference: {s['estimated_inference_time'] / 1000:.2f} ms")
    os.makedirs(a.outdir, exist_ok=True)
    print("downloaded", target.download(os.path.join(a.outdir, "minisam3d")))
    print("compile:", cj.url, "\nprofile:", pj.url)


if __name__ == "__main__":
    main()
