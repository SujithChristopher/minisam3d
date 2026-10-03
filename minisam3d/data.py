"""Dataset over teacher pseudo-labels written by scripts/label_with_teacher.py.

Each shard labels/shard_XXXXX.npz holds, per detected person:
  image (str, relative to data root), img_wh (2,), bbox (4, xyxy),
  kp2d (70,2) px, kp3d (70,3) m (camera-oriented, before cam_t), cam_t (3,), focal ()
"""
import glob
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .keypoints import joint_set

UPPER_FOR_CROP = ["nose", "left_ear", "right_ear", "neck", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist"]


def load_labels(root):
    cols = {}
    for f in sorted(glob.glob(os.path.join(root, "labels", "shard_*.npz"))):
        d = np.load(f, allow_pickle=False)
        for k in d.files:
            cols.setdefault(k, []).append(d[k])
    if not cols:
        raise FileNotFoundError(f"no shards under {root}/labels")
    return {k: np.concatenate(v) for k, v in cols.items()}


def crop_affine(center, size, rot_deg, out_w, out_h):
    """2x3 affine mapping a (size*out_w/out_h, size) box at center, rotated, to the output crop."""
    s = out_h / size
    a = math.radians(rot_deg)
    c, sn = math.cos(a) * s, math.sin(a) * s
    M = np.array([[c, sn, 0], [-sn, c, 0]], np.float64)
    M[:, 2] = np.array([out_w / 2, out_h / 2]) - M[:, :2] @ center
    return M


class PseudoLabelDataset(Dataset):
    def __init__(self, root, in_h=256, in_w=256, hands=False, train=True, out_stride=4, gray=False, image_dirs=()):
        self.root, self.in_h, self.in_w, self.train, self.stride = root, in_h, in_w, train, out_stride
        self.gray, self.image_dirs = gray, list(image_dirs)
        self.js = joint_set(hands)
        self.L = load_labels(root)
        self.idx = np.array(self.js["mhr_idx"])
        self.crop_ids = [self.js["names"].index(n) for n in UPPER_FOR_CROP]

    def __len__(self):
        return len(self.L["image"])

    def __getitem__(self, i):
        L, js = self.L, self.js
        img = cv2.imread(self._path(str(L["image"][i])), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        kp2d = L["kp2d"][i][self.idx].astype(np.float64)
        kp3d = L["kp3d"][i][self.idx].astype(np.float32)
        in_img = (kp2d[:, 0] >= 0) & (kp2d[:, 0] < W) & (kp2d[:, 1] >= 0) & (kp2d[:, 1] < H)

        # crop box around the upper body (what the on-device tracker will produce)
        pts = kp2d[self.crop_ids][in_img[self.crop_ids]]
        if len(pts) < 3:
            pts = kp2d[self.crop_ids]
        lo, hi = pts.min(0), pts.max(0)
        center = (lo + hi) / 2
        aspect = self.in_w / self.in_h
        size = max(hi[1] - lo[1], (hi[0] - lo[0]) / aspect) * 1.35
        rot, flip = 0.0, False
        if self.train:
            center = center + np.random.uniform(-0.1, 0.1, 2) * size
            size *= np.random.uniform(0.85, 1.35)
            rot = np.random.uniform(-30, 30) if np.random.rand() < 0.6 else 0.0
            flip = np.random.rand() < 0.5
        size = max(size, 16.0)
        M = crop_affine(center, size, rot, self.in_w, self.in_h)
        crop = cv2.warpAffine(img, M, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        uv = (np.c_[kp2d, np.ones(len(kp2d))] @ M.T) / [self.in_w, self.in_h]

        # 3D: root-relative, rotate about the camera z axis with the in-plane rotation
        xyz = kp3d - kp3d[js["root"]]
        a = math.radians(rot)
        R = np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]], np.float32)
        xyz = xyz @ R.T

        vis = in_img & (uv[:, 0] >= 0) & (uv[:, 0] < 1) & (uv[:, 1] >= 0) & (uv[:, 1] < 1)
        if flip:
            crop = crop[:, ::-1].copy()
            uv[:, 0] = 1 - uv[:, 0]
            xyz[:, 0] *= -1
            f = js["flip"]
            uv, xyz, vis = uv[f], xyz[f], vis[f]

        if self.gray:
            crop = self._to_gray(crop)
        if self.train:
            crop = self._color(crop)
        x = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        w3d = np.where(in_img if not flip else in_img[js["flip"]], 1.0, 0.5).astype(np.float32)
        return dict(img=x, uv=torch.from_numpy(uv.astype(np.float32)),
                    vis=torch.from_numpy(vis.astype(np.float32)),
                    xyz=torch.from_numpy(xyz.astype(np.float32)), w3d=torch.from_numpy(w3d))

    def _path(self, rel):
        """Labels store images/<input dir name>/...; look under the label root first, then under
        each --image-dirs entry (used when images were not linked into the label root, e.g. on Drive)."""
        p = os.path.join(self.root, rel)
        if os.path.exists(p) or not self.image_dirs:
            return p
        sub = rel.split("/", 1)[1] if rel.startswith("images/") else rel
        for d in self.image_dirs:
            q = os.path.join(d, sub)
            if os.path.exists(q):
                return q
        raise FileNotFoundError(f"{rel} not under {self.root} or {self.image_dirs}")

    def _to_gray(self, rgb):
        """Mono sensor stand-in. The OV9281 is also NIR-sensitive, so channel weights are randomized
        in training (red-heavy mixes look more like NIR: brighter skin, flatter fabric)."""
        if self.train:
            w = np.random.dirichlet([2.0, 2.0, 1.0])
            g = rgb.astype(np.float32) @ w
            g = 255.0 * (g / 255.0) ** np.random.uniform(0.7, 1.4)  # gamma
        else:
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.clip(g, 0, 255).astype(np.uint8)[..., None]

    @staticmethod
    def _color(img):
        img = img.astype(np.float32)
        img = img * np.random.uniform(0.7, 1.3) + np.random.uniform(-25, 25)
        img = img * np.random.uniform(0.85, 1.15, img.shape[-1])  # per-channel white balance
        if np.random.rand() < 0.2:
            img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.5, 1.5)).reshape(img.shape)
        if np.random.rand() < 0.3:
            img = img + np.random.normal(0, np.random.uniform(2, 8), img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)
