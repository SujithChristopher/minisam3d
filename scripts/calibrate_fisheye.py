"""Calibrate the 160-degree OV9281 with a printed checkerboard (cv2.fisheye model).

  python calibrate_fisheye.py --images calib/*.png --board 9 6 --square 0.025 --out calib_left.json
Capture 20-40 shots with the board covering the edges and corners of the frame.
"""
import argparse
import glob
import json

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--board", type=int, nargs=2, default=[9, 6], help="inner corners (cols rows)")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    ap.add_argument("--out", default="calib.json")
    a = ap.parse_args()
    cols, rows = a.board
    obj = np.zeros((1, cols * rows, 3), np.float32)
    obj[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * a.square
    O, I, size = [], [], None
    files = [f for p in a.images for f in sorted(glob.glob(p))]
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
            O.append(obj)
            I.append(c.reshape(1, -1, 2))
    print(f"board found in {len(O)}/{len(files)} images")
    K, D = np.zeros((3, 3)), np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, _, _ = cv2.fisheye.calibrate(O, I, size, K, D, flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7))
    print(f"RMS reprojection error {rms:.3f} px (aim for < 0.5)")
    json.dump(dict(K=K.tolist(), D=D.ravel().tolist(), size=list(size), rms=rms), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
