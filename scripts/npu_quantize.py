"""Quantize the student ONNX for the Dragon Q6A's Hexagon NPU on Qualcomm AI Hub (port of yolo-test/submit_aihub.py).

The result is a QDQ ONNX that the QNN EP (onnxruntime-qnn) compiles for the HTP at session creation, saved to
<onnx dir>/models/w8a<bits>_<job id>/model.onnx. Calibration is calib/calib.npy from make_calib.py (NOARK crops,
preprocessed exactly as at inference). Needs `qai-hub configure --api_token <token>`.

  python scripts/npu_quantize.py --onnx runs/v3rot_loso_all/student.onnx                  # W8A16
  python scripts/npu_quantize.py --onnx runs/v3rot_loso_all/student.onnx --activations INT8
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import qai_hub as hub

# Only used for the optional --profile run. The QDQ ONNX itself is compiled on the board for whatever HTP it finds.
DEVICE = "Dragonwing RB3 Gen 2 Vision Kit"  # chipset:qcs6490, hexagon:v68 -- same silicon as the Q6A


def _target_or_die(job, label):
    """A failed job still hands back a Model stub; check the status, not just `is not None`."""
    status = job.wait()
    if not status.success:
        raise SystemExit(f"{label} job failed: {status.message}\n  {job.url}")
    model = job.get_target_model()
    if model is None:
        raise SystemExit(f"{label} job produced no model\n  {job.url}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="fp32 student ONNX from export_onnx.py")
    ap.add_argument("--calib", default="calib/calib.npy", help="(N,1,H,W) float32 in [0,1] from make_calib.py")
    ap.add_argument("--limit", type=int, default=100)
    # INT16 activations first (known-good reference), then try INT8 against it.
    ap.add_argument("--activations", default="INT16", choices=["INT8", "INT16"])
    ap.add_argument("--quant-options", default="", help='e.g. "--range_scheme min_max"')
    ap.add_argument("--dataset", help="reuse an uploaded calibration dataset id")
    ap.add_argument("--quantize-job", help="re-download an existing successful quantize job")
    ap.add_argument("--profile", action="store_true",
                    help="also compile to an AI Hub-hosted ONNX and profile it on " + DEVICE)
    a = ap.parse_args()

    onnx_path = Path(a.onnx)
    bits = f"w8a{a.activations[3:]}"
    if a.quantize_job:
        quantize = hub.get_job(a.quantize_job)
    else:
        if a.dataset:
            calib = hub.get_dataset(a.dataset)
        else:
            x = np.load(a.calib)[: a.limit].astype(np.float32)
            print(f"calibrating on {len(x)} images {x.shape[1:]}")
            calib = hub.upload_dataset({"image": [xi[None] for xi in x]})
            print(f"calibration dataset: {calib.dataset_id} (reuse with --dataset)")
        quantize = hub.submit_quantize_job(
            model=hub.upload_model(str(onnx_path)),
            calibration_data=calib,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=getattr(hub.QuantizeDtype, a.activations),
            options=a.quant_options,
            name=f"minisam3d-{onnx_path.parent.name}-{bits}{a.quant_options}",
        )
    quantized = _target_or_die(quantize, "quantize")
    print(f"quantize job: {quantize.url}")

    # AI Hub hands the QDQ ONNX back as a zip holding model.onnx (+ model.data).
    out = onnx_path.parent / "models" / f"{bits}_{quantize.job_id}"
    out.parent.mkdir(parents=True, exist_ok=True)
    zipped = Path(quantized.download(str(out) + ".onnx"))
    with zipfile.ZipFile(zipped) as z:
        inner = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
        out.mkdir(exist_ok=True)
        for name in ("model.onnx", "model.data"):
            if name in inner:
                (out / name).write_bytes(z.read(inner[name]))
    zipped.unlink()
    print(f"saved {out / 'model.onnx'}")

    if a.profile:
        dev = hub.Device(DEVICE)
        compiled = _target_or_die(hub.submit_compile_job(model=quantized, device=dev, options="--target_runtime onnx",
                                                         name=f"minisam3d-{bits}-onnx"), "compile")
        print("profile:", hub.submit_profile_job(model=compiled, device=dev, name=f"minisam3d-{bits}-profile").url)


if __name__ == "__main__":
    main()
