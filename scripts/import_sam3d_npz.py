"""Turn SAM 3D Body results already computed per take into student label shards.

Input per take: a folder holding img_<frame>.png, meta.npz (idx, boxes, K, size) and sam3d_out.npz
(idx, ok, pred_keypoints_2d/3d, pred_cam_t, pred_global_rots) -- the layout NOARK's
19_sam3d_export.py / 20_sam3d_infer.py write. Each take becomes <out>/<name>/ with
labels/shard_00000.npz and images/<name> -> the take folder (junction / symlink).

  python scripts/import_sam3d_npz.py --out E:/NOARK_CV/msam_labels/noark \
      --take gokul_game=D:/.../rec_16-40-17/sam3d --take gokul_woc=D:/.../gokul_woc/sam3d

Train with --data <out>/<name> for each take (hold one subject out for validation).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minisam3d.keypoints import MHR_TO_CAM, ROT_JOINTS  # noqa: E402


def link_dir(src, dst):
    if os.path.lexists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.name == "nt":  # a junction needs no admin rights
        subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True, capture_output=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--take", action="append", required=True, metavar="NAME=DIR")
    a = ap.parse_args()
    C = np.array(MHR_TO_CAM, np.float32)
    ridx = [i for _, i in ROT_JOINTS]
    for spec in a.take:
        name, d = spec.split("=", 1)
        meta, res = np.load(os.path.join(d, "meta.npz")), np.load(os.path.join(d, "sam3d_out.npz"))
        assert (meta["idx"] == res["idx"]).all(), f"{name}: meta / sam3d_out frame mismatch"
        ok = res["ok"] & np.isfinite(res["pred_keypoints_2d"]).all((1, 2))
        n = int(ok.sum())
        K, (W, H) = meta["K"], meta["size"]
        rot = res["pred_global_rots"][ok][:, ridx]
        lab = dict(image=np.array([f"images/{name}/img_{i:05d}.png" for i in meta["idx"][ok]]),
                   img_wh=np.tile(np.array([W, H], np.int32), (n, 1)),
                   bbox=meta["boxes"][ok].astype(np.float32),
                   kp2d=res["pred_keypoints_2d"][ok].astype(np.float32),
                   kp3d=res["pred_keypoints_3d"][ok].astype(np.float32),
                   cam_t=res["pred_cam_t"][ok].astype(np.float32),
                   focal=np.full(n, K[0, 0], np.float32),
                   rot=(C[:, None] * rot * C[None, :]).astype(np.float32))
        root = os.path.join(a.out, name)
        os.makedirs(os.path.join(root, "labels"), exist_ok=True)
        np.savez(os.path.join(root, "labels", "shard_00000.npz"), **lab)
        link_dir(os.path.abspath(d), os.path.join(root, "images", name))
        print(f"{name}: {n}/{len(ok)} frames -> {root}")


if __name__ == "__main__":
    main()
