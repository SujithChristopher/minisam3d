"""Person box detector that picks the subject: the person closest to the frame centre.

Single-class CenterNet on the whole (undistorted, upright, gray) frame at 256x160. One output tensor
(1, 4, H/8, W/8): centre heatmap logit, sub-cell offset (dx, dy) and log box side / input height. Same
op set as the student (conv, ReLU6, nearest resize, add), so it quantizes cleanly and runs on the HTP.
Peak picking is outside the graph (decode_det). The box it learns is the square upper-body box the
student's crops are cut from (pack_noark.upper_box), so detector -> crop_affine -> student needs no
re-tuning between the two.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from .data import load_labels, read_pack
from .model import Student, cbr

STRIDE = 8


class PersonDet(nn.Module):
    def __init__(self, pretrained=False, in_ch=1, ch=64):
        super().__init__()
        mb = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None).features
        Student._fold_input(mb, in_ch)
        self.s8, self.s16, self.s32 = mb[:7], mb[7:14], mb[14:18]
        self.lat8, self.lat16, self.lat32 = cbr(32, ch, 1), cbr(96, ch, 1), cbr(320, ch, 1)
        self.up16 = cbr(ch, ch)
        self.up8 = nn.Sequential(cbr(ch, ch), cbr(ch, ch))
        self.head = nn.Conv2d(ch, 4, 1)
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():
            self.head.bias[0] = -4.0  # start with "no person" everywhere (focal loss prior)

    def forward(self, x):
        f8 = self.s8(x)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        p = self.lat32(f32)
        p = self.up16(self.lat16(f16) + F.interpolate(p, scale_factor=2, mode="nearest"))
        p = self.up8(self.lat8(f8) + F.interpolate(p, scale_factor=2, mode="nearest"))
        return self.head(p)


def det_loss(out, b):
    """CenterNet focal loss on the heatmap + L1 on offset and log size at the target cell."""
    heat = out[:, 0].sigmoid().clamp(1e-4, 1 - 1e-4)
    g = b["heat"]
    pos = g.eq(1).float()
    pos_l = -(torch.log(heat) * (1 - heat) ** 2 * pos).sum()
    neg_l = -(torch.log(1 - heat) * heat ** 2 * (1 - g) ** 4 * (1 - pos)).sum()
    n = pos.sum().clamp(min=1)
    hm = (pos_l + neg_l) / n
    B = out.shape[0]
    cell = b["cell"].long()
    reg = out[torch.arange(B), 1:, cell[:, 1], cell[:, 0]]  # (B, 3)
    off = F.l1_loss(reg[:, :2], b["off"])
    size = F.l1_loss(reg[:, 2], b["logsize"])
    return hm + off + 2 * size, dict(hm=hm.item(), off=off.item(), size=size.item())


def _peaks(h):
    """3x3 non-max suppression on a 2D map."""
    p = np.pad(h, 1, constant_values=-np.inf)
    m = np.max([p[dy:dy + h.shape[0], dx:dx + h.shape[1]] for dy in range(3) for dx in range(3)], 0)
    return h >= m


def decode_det(out, in_wh=(256, 160), thr=0.3):
    """out: (1,4,h,w) or (4,h,w) -> (center (2,), side, score) in input px, or None.
    Of the peaks above thr, returns the one closest to the frame centre (the seated subject); a bystander
    can score higher but sits off-centre."""
    o = np.asarray(out, np.float32).reshape(4, *np.asarray(out).shape[-2:])
    h = 1 / (1 + np.exp(-o[0]))
    ys, xs = np.nonzero(_peaks(h) & (h > thr))
    if not len(ys):
        return None
    cx, cy = (xs + o[1, ys, xs]) * STRIDE, (ys + o[2, ys, xs]) * STRIDE
    k = int(np.argmin(np.hypot(cx - in_wh[0] / 2, cy - in_wh[1] / 2)))
    return np.array([cx[k], cy[k]]), float(np.exp(o[3, ys[k], xs[k]]) * in_wh[1]), float(h[ys[k], xs[k]])


class DetDataset(Dataset):
    """Full frames + subject box from a pack_noark.py take, with geometric / photometric augmentation."""

    def __init__(self, root, train=True, sigma=1.5):
        self.root, self.train, self.sigma = root, train, sigma
        L = load_labels(root)
        self.full_off, self.box = L["full_off"], L["box"].astype(np.float64)
        self.scale = L["full_wh"][:, 0] / L["frame_wh"][:, 0]  # original frame px -> full frame px
        self.wh = tuple(int(v) for v in L["full_wh"][0])

    def __len__(self):
        return len(self.box)

    def __getitem__(self, i):
        W, H = self.wh
        img = read_pack(self.root, self.full_off[i], cv2.IMREAD_GRAYSCALE)
        x0, y0, x1, y1 = self.box[i] * self.scale[i]
        c, s = np.array([(x0 + x1) / 2, (y0 + y1) / 2]), x1 - x0
        if self.train:
            sc = np.random.uniform(0.75, 1.3)
            a = math.radians(np.random.uniform(-10, 10)) if np.random.rand() < 0.5 else 0.0
            t = np.random.uniform(-0.15, 0.15, 2) * [W, H]
            M = cv2.getRotationMatrix2D((W / 2, H / 2), math.degrees(a), sc)
            M[:, 2] += t
            if np.random.rand() < 0.5:
                M = np.array([[-1, 0, W - 1], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            c, s = M[:, :2] @ c + M[:, 2], s * sc
            g = img.astype(np.float32)
            g = 255.0 * (g / 255.0) ** np.random.uniform(0.7, 1.4)
            g = g * np.random.uniform(0.7, 1.3) + np.random.uniform(-25, 25)
            if np.random.rand() < 0.2:
                g = cv2.GaussianBlur(g, (0, 0), np.random.uniform(0.5, 1.5))
            if np.random.rand() < 0.3:
                g = g + np.random.normal(0, np.random.uniform(2, 8), g.shape)
            img = np.clip(g, 0, 255).astype(np.uint8)
        gh, gw = H // STRIDE, W // STRIDE
        cg = np.clip(c / STRIDE, 0, [gw - 1e-3, gh - 1e-3])
        cell = np.floor(cg).astype(np.int64)
        yy, xx = np.mgrid[:gh, :gw]
        heat = np.exp(-((xx - cell[0]) ** 2 + (yy - cell[1]) ** 2) / (2 * self.sigma ** 2)).astype(np.float32)
        return dict(img=torch.from_numpy(img[None].astype(np.float32) / 255.0), heat=torch.from_numpy(heat),
                    cell=torch.from_numpy(cell), off=torch.from_numpy((cg - cell).astype(np.float32)),
                    logsize=torch.tensor(math.log(max(s, 4.0) / H), dtype=torch.float32),
                    center=torch.from_numpy(c.astype(np.float32)), side=torch.tensor(s, dtype=torch.float32))
