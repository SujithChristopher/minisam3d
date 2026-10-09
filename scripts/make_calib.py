"""Build a quantization calibration set from NOARK fisheye frames (the deployment domain).

Takes N frames per subject, evenly spaced over each recording, and crops them from the person box exactly as
infer_frames.py does (gray, crop_affine, /255). Writes <out>/calib_XX_<subject>_<frame>.png (uint8 crops) and
<out>/calib.npy (N,1,H,W) float32 in [0,1] -- ready to feed a quantizer as the "image" input.

  python scripts/make_calib.py --session E:/CMC/.../sep23_26 --out calib --per-subject 10
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minisam3d.data import crop_affine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone\data\sep23_26")
    ap.add_argument("--out", default="calib")
    ap.add_argument("--per-subject", type=int, default=10)
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256])
    a = ap.parse_args()
    H, W = a.size
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(a.session, "noark_game", "data", "*", "Session-*", "Video", "rec_*", "sam3d")))
    xs = []
    for d in dirs:
        subj = os.path.relpath(d, os.path.join(a.session, "noark_game", "data")).split(os.sep)[0]
        meta = np.load(os.path.join(d, "meta.npz"))
        valid = [j for j, b in enumerate(meta["boxes"]) if np.isfinite(b).all()]
        pick = [valid[int(k)] for k in np.linspace(0, len(valid) - 1, a.per_subject)]
        for j in pick:
            i = int(meta["idx"][j])
            img = cv2.imread(os.path.join(d, f"img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
            x0, y0, x1, y1 = meta["boxes"][j]
            M = crop_affine(np.array([(x0 + x1) / 2, (y0 + y1) / 2]), max(x1 - x0, y1 - y0), 0.0, W, H)
            crop = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(os.path.join(a.out, f"calib_{len(xs):02d}_{subj}_{i:05d}.png"), crop)
            xs.append(crop)
        print(f"{subj}: {len(pick)} frames from {len(valid)} valid")
    arr = np.stack(xs)[:, None].astype(np.float32) / 255.0
    np.save(os.path.join(a.out, "calib.npy"), arr)
    print(f"{len(xs)} crops -> {a.out}/calib.npy {arr.shape}")


if __name__ == "__main__":
    main()
