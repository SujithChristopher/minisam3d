"""Turn fisheye recordings into pinhole views the teacher can label.

Each frame becomes a few virtual perspective views (default: centre, left, right, 90-degree FOV,
768x768) saved as jpg, plus <out>/intrinsics.json mapping image path -> 3x3 K, which
label_with_teacher.py passes to SAM 3D Body instead of guessing the FOV.

  python fisheye_to_views.py --calib calib_left.json --videos rec/*.mp4 --out /data/own_views --stride 5
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from minisam3d.fisheye import load_calib, look_at, virtual_view_maps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--yaws", type=float, nargs="+", default=[0.0, -45.0, 45.0])
    ap.add_argument("--pitches", type=float, nargs="+", default=[0.0])
    a = ap.parse_args()
    K, D, _ = load_calib(a.calib)
    views = []
    for p in a.pitches:
        for y in a.yaws:
            d = np.array([np.sin(np.radians(y)) * np.cos(np.radians(p)), np.sin(np.radians(p)),
                          np.cos(np.radians(y)) * np.cos(np.radians(p))])
            mx, my, Kv = virtual_view_maps(K, D, look_at(d), a.fov, a.size, a.size)
            views.append((f"y{int(y):+d}p{int(p):+d}", mx, my, Kv))
    intr_path = os.path.join(a.out, "intrinsics.json")
    intr = json.load(open(intr_path)) if os.path.exists(intr_path) else {}
    for vp in [f for p in a.videos for f in sorted(glob.glob(p))]:
        cap, i = cv2.VideoCapture(vp), 0
        stem = os.path.splitext(os.path.basename(vp))[0]
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % a.stride == 0:
                for name, mx, my, Kv in views:
                    rel = os.path.join("images", stem, name, f"{i:07d}.jpg")
                    os.makedirs(os.path.join(a.out, os.path.dirname(rel)), exist_ok=True)
                    cv2.imwrite(os.path.join(a.out, rel), cv2.remap(fr, mx, my, cv2.INTER_LINEAR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    intr[rel] = Kv.tolist()
            i += 1
        print(vp, i, "frames")
    json.dump(intr, open(intr_path, "w"))


if __name__ == "__main__":
    main()
