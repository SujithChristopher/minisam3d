"""Fisheye (OpenCV Kannala-Brandt / cv2.fisheye) <-> virtual pinhole views.

The OV9281 160-degree lens is far from pinhole, while SAM 3D Body and the student
both assume a pinhole camera. So every image the networks see is a *virtual
perspective view* rendered from the fisheye frame:
  - for teacher labeling: a few wide views per frame (see scripts/fisheye_to_views.py)
  - at runtime: one view per frame, aimed at the tracked person (this file's PersonView)
3D predicted in a virtual view is rotated back into the physical camera frame with R.T.
"""
import json
import math

import cv2
import numpy as np


def load_calib(path):
    c = json.load(open(path))
    return np.array(c["K"], np.float64), np.array(c["D"], np.float64).reshape(4), tuple(c["size"])


def look_at(direction, up=(0.0, 1.0, 0.0)):
    """Rotation R (world->virtual) whose +z axis points along direction; roll keeps image 'down' ~ camera +y."""
    z = np.asarray(direction, np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(np.asarray(up, np.float64), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows


def fisheye_project(rays, K, D):
    """rays (N,3) in the physical camera frame -> fisheye pixels (N,2). Handles angles > 90 deg."""
    r = np.linalg.norm(rays[:, :2], axis=1)
    th = np.arctan2(r, rays[:, 2])
    th2 = th * th
    thd = th * (1 + D[0] * th2 + D[1] * th2 ** 2 + D[2] * th2 ** 3 + D[3] * th2 ** 4)
    s = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.0)
    xd = rays[:, :2] * s[:, None]
    return np.c_[K[0, 0] * xd[:, 0] + K[0, 2], K[1, 1] * xd[:, 1] + K[1, 2]]


def fisheye_unproject(px, K, D):
    """fisheye pixels (N,2) -> unit rays (N,3). Valid for the whole 160-degree field."""
    xd = np.c_[(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1]]
    thd = np.linalg.norm(xd, axis=1)
    th = thd.copy()
    for _ in range(10):  # Newton on theta_d(theta) = thd
        t2 = th * th
        f = th * (1 + D[0] * t2 + D[1] * t2 ** 2 + D[2] * t2 ** 3 + D[3] * t2 ** 4) - thd
        df = 1 + 3 * D[0] * t2 + 5 * D[1] * t2 ** 2 + 7 * D[2] * t2 ** 3 + 9 * D[3] * t2 ** 4
        th = th - f / df
    s = np.where(thd > 1e-9, np.sin(th) / np.maximum(thd, 1e-9), 1.0)
    return np.c_[xd * s[:, None], np.cos(th)]


def virtual_view_maps(K, D, R, fov_deg, out_w, out_h, step=1):
    """Remap tables rendering a pinhole view (rotation R, horizontal-or-vertical FOV) from the fisheye image.
    step>1 evaluates the (smooth) mapping on a coarse grid and upsamples it, ~step^2 faster.
    Returns map_x, map_y (float32) and the virtual intrinsics Kv."""
    f = (max(out_w, out_h) / 2) / math.tan(math.radians(fov_deg) / 2)
    Kv = np.array([[f, 0, out_w / 2], [0, f, out_h / 2], [0, 0, 1]])
    if step > 1:
        gw, gh = -(-out_w // step) + 1, -(-out_h // step) + 1
        u, v = np.meshgrid(np.arange(gw) * step + 0.5, np.arange(gh) * step + 0.5)
        rays_v = np.stack([(u - Kv[0, 2]) / f, (v - Kv[1, 2]) / f, np.ones_like(u)], -1).reshape(-1, 3)
        grid = fisheye_project(rays_v @ R, K, D).astype(np.float32).reshape(gh, gw, 2)
        # bilinear upsample: output pixel i sits at node coordinate i/step
        gx, gy = np.meshgrid(np.arange(out_w, dtype=np.float32) / step, np.arange(out_h, dtype=np.float32) / step)
        px = cv2.remap(grid, gx, gy, cv2.INTER_LINEAR)
        return px[..., 0].copy(), px[..., 1].copy(), Kv
    u, v = np.meshgrid(np.arange(out_w) + 0.5, np.arange(out_h) + 0.5)
    rays_v = np.stack([(u - Kv[0, 2]) / f, (v - Kv[1, 2]) / f, np.ones_like(u)], -1).reshape(-1, 3)
    rays_c = rays_v @ R  # virtual -> camera (R is world->virtual, rows are axes)
    px = fisheye_project(rays_c, K, D).astype(np.float32)
    return px[:, 0].reshape(out_h, out_w), px[:, 1].reshape(out_h, out_w), Kv


class PersonView:
    """Per-frame crop for the student: a virtual pinhole camera aimed at the person.

    Usage per frame:
        crop = pv.render(frame)                  # (in_h, in_w) image for the student
        kp_fish, xyz_cam = pv.to_camera(uv, xyz) # back to fisheye pixels / physical camera frame
        pv.update(kp_fish[crop_ids])             # aim next frame's view at these keypoints
    """

    def __init__(self, K, D, in_w=256, in_h=256, margin=1.35, default_fov=120.0):
        self.K, self.D, self.in_w, self.in_h = K, D, in_w, in_h
        self.margin, self.default_fov = margin, default_fov
        self.reset()

    def reset(self):
        self.R, self.fov = np.eye(3), self.default_fov
        self._build()

    def _build(self):
        self.mx, self.my, self.Kv = virtual_view_maps(self.K, self.D, self.R, self.fov, self.in_w, self.in_h, step=8)

    def render(self, frame):
        return cv2.remap(frame, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def to_camera(self, uv, xyz_rel):
        """uv in [0,1] of the crop -> fisheye pixels; root-relative xyz (virtual frame) -> camera frame."""
        rays_v = np.c_[(uv[:, 0] * self.in_w - self.Kv[0, 2]) / self.Kv[0, 0],
                       (uv[:, 1] * self.in_h - self.Kv[1, 2]) / self.Kv[1, 1], np.ones(len(uv))]
        return fisheye_project(rays_v @ self.R, self.K, self.D), xyz_rel @ self.R

    def update(self, kp_fish):
        rays = fisheye_unproject(np.asarray(kp_fish, np.float64), self.K, self.D)
        c = rays.mean(0)
        R = look_at(c)
        rv = rays @ R.T
        a = np.degrees(np.arctan2(np.abs(rv[:, :2]), rv[:, 2:]))  # per-axis angle from view centre
        half = float(a.max()) * self.margin
        self.R, self.fov = R, float(np.clip(2 * half, 20.0, 150.0))
        self._build()
