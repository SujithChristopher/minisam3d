"""Run a trained student over a folder of frames (NOARK 19_sam3d_export.py layout) and save its outputs.

Input dir: img_<frame>.png + meta.npz (idx, boxes, K). Two passes per frame: crop from the given person box,
then re-crop from the student's own upper-body keypoints exactly as training crops (UPPER_FOR_CROP, x1.35).

Writes <out>/student_out.npz: idx, ok, uv (K,2) image px, xyz (K,3) root-relative m, conf (K,),
rot (R,3,3) camera-frame segment rotations, and -- so NOARK's 21_sam3d_score.py can score it unchanged --
pred_global_rots (127,3,3) in MHR axes (identity where the student has no joint), pred_keypoints_2d/3d and
pred_cam_t filled with NaN (the student does not predict them in MHR70 order).

  python scripts/infer_frames.py --ckpt runs/v3rot/best.pt --frames <rec>/sam3d --out <dir>
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minisam3d.data import UPPER_FOR_CROP, crop_affine  # noqa: E402
from minisam3d.keypoints import MHR_TO_CAM, joint_set  # noqa: E402
from minisam3d.model import Student, decode, rot6d_to_mat  # noqa: E402


def crop_from_points(pts, scale=1.35):
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, max(max(hi[1] - lo[1], hi[0] - lo[0]) * scale, 16.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = Student(ck["K"], width=ck["width"], in_ch=ck["in_ch"], dec_ch=ck["dec_ch"], head3d=ck["head3d"],
                num_rot=ck.get("num_rot", 0))
    m.load_state_dict(ck["model"])
    m.eval().to(a.device)
    js = joint_set(ck["hands"])
    H, W = ck["size"]
    crop_ids = [js["names"].index(n) for n in UPPER_FOR_CROP]
    meta = np.load(os.path.join(a.frames, "meta.npz"))

    def run(img, center, size):
        M = crop_affine(center, size, 0.0, W, H)
        crop = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        x = torch.from_numpy(crop).float().div(255)[None, None].to(a.device)
        if ck["in_ch"] == 3:
            x = x.repeat(1, 3, 1, 1)
        with torch.no_grad():
            out = m(x)
            uv, xyz, conf = decode(out, ck["K"])
            rot = rot6d_to_mat(out[2].reshape(1, -1, 6))[0].cpu().numpy() if ck.get("num_rot") else None
        Mi = cv2.invertAffineTransform(M)
        p = uv[0].cpu().numpy() * [W, H]
        return np.c_[p, np.ones(len(p))] @ Mi.T, xyz[0].cpu().numpy(), conf[0].cpu().numpy(), rot

    n = len(meta["idx"])
    R = ck.get("num_rot", 0)
    res = dict(idx=meta["idx"], ok=np.zeros(n, bool), uv=np.full((n, ck["K"], 2), np.nan, np.float32),
               xyz=np.full((n, ck["K"], 3), np.nan, np.float32), conf=np.zeros((n, ck["K"]), np.float32),
               rot=np.tile(np.eye(3, dtype=np.float32), (n, max(R, 1), 1, 1))[:, :R])
    for j, i in enumerate(meta["idx"]):
        img = cv2.imread(os.path.join(a.frames, f"img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        b = meta["boxes"][j]
        if not np.isfinite(b).all():
            continue
        x0, y0, x1, y1 = b
        uv, _, _, _ = run(img, np.array([(x0 + x1) / 2, (y0 + y1) / 2]), max(x1 - x0, y1 - y0))
        center, size = crop_from_points(uv[crop_ids])
        uv, xyz, conf, rot = run(img, center, size)
        res["uv"][j], res["xyz"][j], res["conf"][j], res["ok"][j] = uv, xyz, conf, True
        if rot is not None:
            res["rot"][j] = rot
        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{n}", flush=True)
    # NOARK 21_sam3d_score.py layout: MHR-axes global rotations at the MHR joint indices
    C = np.array(MHR_TO_CAM, np.float32)
    G = np.tile(np.eye(3, dtype=np.float32), (n, 127, 1, 1))
    if R:
        G[:, js["rot_idx"]] = C[:, None] * res["rot"] * C[None, :]
    nan = lambda *s: np.full((n,) + s, np.nan, np.float32)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "student_out.npz"), **res, pred_global_rots=G,
                        pred_keypoints_2d=nan(70, 2), pred_keypoints_3d=nan(70, 3), pred_cam_t=nan(3))
    print(f"{int(res['ok'].sum())}/{n} frames -> {os.path.join(a.out, 'student_out.npz')}")


if __name__ == "__main__":
    main()
