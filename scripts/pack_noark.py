"""Pack the NOARK teacher-labelled takes into self-contained crop files for training elsewhere.

The session folder holds both cameras' raw msgpack streams (~47 GB); training only needs the reference
camera's undistorted, upright frames that 19_sam3d_export.py wrote to <take>/sam3d/ plus the teacher output.
Each take becomes <out>/<take>/:
  pack.bin                  PNG bytes, back to back: per frame a crop around the subject, then a small full frame
  labels/shard_00000.npz    the import_sam3d_npz.py label set, but in crop pixels, plus the pack index

Crop: square, centred on the upper-body box data.py derives from the teacher keypoints, side PACK_SIDE x that
box, stored at --crop px. data.py's train-time jitter (shift 0.1, scale up to 1.35, rotation 30 deg) stays
inside it, so training from the pack matches training from the full frame. The full frame (--full, default
256x160 = 640x400 / 2.5) is what the person detector trains on; `box` is its target in original-frame pixels.

  python scripts/pack_noark.py --out data/noark_pack
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minisam3d.data import UPPER_FOR_CROP, crop_affine  # noqa: E402
from minisam3d.keypoints import MHR_TO_CAM, ROT_JOINTS, joint_set  # noqa: E402

PACK_SIDE = 2.0  # pack crop side / training box side


def takes(session):
    out = {}
    # sep23 layout: <session>/<subj>_w(o)c, noark_game/data/<subj>/...; oct layout: std_movement/..., game/<subj>/...
    for d in sorted(glob.glob(os.path.join(session, "*_wc", "sam3d")) + glob.glob(os.path.join(session, "*_woc", "sam3d"))
                    + glob.glob(os.path.join(session, "std_movement", "*_w*c", "sam3d"))):
        out[os.path.basename(os.path.dirname(d))] = d
    for pat in (("noark_game", "data"), ("game",)):
        for d in sorted(glob.glob(os.path.join(session, *pat, "*", "Session-*", "Video", "rec_*", "sam3d"))):
            out[os.path.relpath(d, os.path.join(session, *pat)).replace("\\", "/").split("/")[0] + "_game"] = d
    return {k: v for k, v in out.items() if os.path.exists(os.path.join(v, "sam3d_out.npz"))}


def upper_box(kp2d, W, H, crop_mhr):
    """data.py's crop box without jitter: (center, side) around the visible upper-body joints."""
    pts = kp2d[crop_mhr]
    vis = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    if vis.sum() >= 3:
        pts = pts[vis]
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, max(hi[1] - lo[1], hi[0] - lo[0]) * 1.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone\data\sep23_26")
    ap.add_argument("--out", default="data/noark_pack")
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--full", type=int, nargs=2, default=[256, 160], metavar=("W", "H"))
    ap.add_argument("--take", nargs="*", help="only these takes (default: all with teacher output)")
    a = ap.parse_args()
    js = joint_set(False)
    crop_mhr = [js["mhr_idx"][js["names"].index(n)] for n in UPPER_FOR_CROP]
    C = np.array(MHR_TO_CAM, np.float32)
    ridx = [i for _, i in ROT_JOINTS]
    P, (FW, FH) = a.crop, a.full
    png = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    found = takes(a.session)
    for name, d in found.items():
        if a.take and name not in a.take:
            continue
        meta, res = np.load(os.path.join(d, "meta.npz")), np.load(os.path.join(d, "sam3d_out.npz"))
        assert (meta["idx"] == res["idx"]).all(), f"{name}: meta / sam3d_out frame mismatch"
        ok = res["ok"] & np.isfinite(res["pred_keypoints_2d"]).all((1, 2))
        K, (W, H) = meta["K"], meta["size"]
        root = os.path.join(a.out, name)
        os.makedirs(os.path.join(root, "labels"), exist_ok=True)
        rec = {k: [] for k in ("kp2d", "in_img", "M", "box", "crop_off", "full_off", "frame")}
        pos = 0
        with open(os.path.join(root, "pack.bin"), "wb") as fb:
            for j in np.nonzero(ok)[0]:
                fi = int(meta["idx"][j])
                img = cv2.imread(os.path.join(d, f"img_{fi:05d}.png"), cv2.IMREAD_GRAYSCALE)
                assert img.shape == (H, W), f"{name} {fi}: {img.shape} != {(H, W)}"
                kp = res["pred_keypoints_2d"][j].astype(np.float64)
                c, s = upper_box(kp, W, H, crop_mhr)
                M = crop_affine(c, PACK_SIDE * s, 0.0, P, P)
                crop = cv2.warpAffine(img, M, (P, P), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                full = cv2.resize(img, (FW, FH), interpolation=cv2.INTER_AREA)
                for key, im in (("crop_off", crop), ("full_off", full)):
                    b = cv2.imencode(".png", im, png)[1].tobytes()
                    fb.write(b)
                    rec[key].append((pos, len(b)))
                    pos += len(b)
                rec["kp2d"].append(np.c_[kp, np.ones(len(kp))] @ M.T)
                rec["in_img"].append((kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H))
                rec["M"].append(M)
                rec["box"].append(np.r_[c - s / 2, c + s / 2])
                rec["frame"].append(fi)
        n = len(rec["frame"])
        rot = res["pred_global_rots"][ok][:, ridx]
        lab = dict(image=np.array([f"pack:{name}:{i}" for i in range(n)]),
                   img_wh=np.tile(np.array([P, P], np.int32), (n, 1)),
                   bbox=np.tile(np.array([P / 4, P / 4, 3 * P / 4, 3 * P / 4], np.float32), (n, 1)),
                   kp2d=np.array(rec["kp2d"], np.float32),
                   kp3d=res["pred_keypoints_3d"][ok].astype(np.float32),
                   cam_t=res["pred_cam_t"][ok].astype(np.float32),
                   focal=np.full(n, K[0, 0], np.float32),
                   rot=(C[:, None] * rot * C[None, :]).astype(np.float32),
                   in_img=np.array(rec["in_img"]),
                   pack_M=np.array(rec["M"], np.float32),
                   box=np.array(rec["box"], np.float32),
                   crop_off=np.array(rec["crop_off"], np.int64),
                   full_off=np.array(rec["full_off"], np.int64),
                   frame=np.array(rec["frame"], np.int32),
                   frame_wh=np.tile(np.array([W, H], np.int32), (n, 1)),
                   full_wh=np.tile(np.array([FW, FH], np.int32), (n, 1)))
        np.savez(os.path.join(root, "labels", "shard_00000.npz"), **lab)
        np.savez(os.path.join(root, "camera.npz"), K=K, R_model_to_board=meta["R_model_to_board"], size=meta["size"])
        print(f"{name}: {n}/{len(ok)} frames, {pos / 2**20:.0f} MB -> {root}", flush=True)


if __name__ == "__main__":
    main()
