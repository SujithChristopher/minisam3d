"""Subject boxes for the SAM 3D teacher from our own person detector (det.onnx), instead of 19's seg-model box.

Runs on a 19_sam3d_export.py folder (<rec>/sam3d/img_*.png + meta.npz). Each frame is resized to the detector
input (256x160), decoded with decode_det, and the square upper-body box is mapped back to frame pixels, median-
smoothed over time (--smooth samples; a covered torso or a reaching arm moves a single-frame box), widened by
--expand and written to meta.npz as boxes_det (+ det_center / det_side / det_score; misses are filled from the
nearest hit). --apply also makes it the box 20_sam3d_infer.py uses: meta["boxes"] <- boxes_det, with the
seg-model box kept as boxes_seg. Running it again re-applies from boxes_seg.

  python scripts/det_boxes.py <rec>/sam3d --apply
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minisam3d.detector import decode_det  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="19_sam3d_export.py output folders (glob ok)")
    ap.add_argument("--onnx", default=os.path.join(os.path.dirname(__file__), "..", "runs", "det_v1", "det.onnx"))
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--expand", type=float, default=1.2, help="box side / detector side")
    ap.add_argument("--smooth", type=int, default=9, help="temporal median window, in exported samples")
    ap.add_argument("--apply", action="store_true", help="make boxes_det the box 20_sam3d_infer.py reads")
    a = ap.parse_args()
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    _, _, IH, IW = sess.get_inputs()[0].shape
    for d in sorted(sum((glob.glob(p) for p in a.dirs), [])):
        meta = dict(np.load(os.path.join(d, "meta.npz")))
        W, H = (int(v) for v in meta["size"])
        sx, sy = W / IW, H / IH
        cen, side, score = [], [], []
        for i in meta["idx"]:
            img = cv2.imread(os.path.join(d, f"img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
            x = cv2.resize(img, (IW, IH), interpolation=cv2.INTER_AREA).astype(np.float32)[None, None] / 255.0
            r = decode_det(sess.run(None, {"image": x})[0], (IW, IH), a.thr)
            cen.append(r[0] * [sx, sy] if r else [np.nan, np.nan])
            side.append(r[1] * sy if r else np.nan)
            score.append(r[2] if r else 0.0)
        cen, side, score = np.array(cen, float), np.array(side, float), np.array(score, float)
        hit = np.isfinite(side)
        if not hit.any():
            print(f"{d}: detector found nobody in {len(hit)} frames, nothing written")
            continue
        n = np.arange(len(hit))
        fill = lambda v: np.interp(n, n[hit], v[hit])  # noqa: E731
        c = np.c_[fill(cen[:, 0]), fill(cen[:, 1])]
        s = fill(side)
        if a.smooth > 1:
            c = median_filter(c, size=(a.smooth, 1), mode="nearest")
            s = median_filter(s, size=a.smooth, mode="nearest")
        h = a.expand * s / 2
        boxes = np.c_[np.clip(c[:, 0] - h, 0, W - 1), np.clip(c[:, 1] - h, 0, H - 1),
                      np.clip(c[:, 0] + h, 0, W - 1), np.clip(c[:, 1] + h, 0, H - 1)]
        meta.update(boxes_det=boxes, det_center=cen, det_side=side, det_score=score)
        if a.apply:
            meta.setdefault("boxes_seg", meta["boxes"])
            meta["boxes"] = boxes
        np.savez(os.path.join(d, "meta.npz"), **meta)
        print(f"{d}: {hit.sum()}/{len(hit)} detected, score median {np.median(score[hit]):.2f}, "
              f"side median {np.median(side[hit]):.0f}px{' (applied)' if a.apply else ''}")


if __name__ == "__main__":
    main()
