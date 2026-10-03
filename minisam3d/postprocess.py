"""NumPy decode for on-device use (mirror of model.decode)."""
import numpy as np


def decode_np(out, K):
    out = out[0] if out.ndim == 4 else out  # (4K,h,w)
    _, h, w = out.shape
    lg = out[:K].reshape(K, -1)
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    gx = (np.tile(np.arange(w), h) + 0.5) / w
    gy = (np.repeat(np.arange(h), w) + 0.5) / h
    uv = np.stack([p @ gx, p @ gy], -1)
    loc = out[K:].reshape(3, K, -1)
    xyz = np.einsum("ckn,kn->kc", loc, p)
    conf = p.max(1) * np.sqrt(h * w)
    return uv, xyz, conf


def uv_to_image(uv, affine_inv, in_w, in_h):
    """Map normalized crop coords back to full image pixels with the inverse crop affine (2x3)."""
    pts = np.c_[uv[:, 0] * in_w, uv[:, 1] * in_h, np.ones(len(uv))]
    return pts @ affine_inv.T
