"""NumPy decode for on-device use (mirror of model.decode)."""
import numpy as np


def rot6d_to_mat_np(r6):
    """(R,6) -> (R,3,3), mirror of model.rot6d_to_mat."""
    a, b = r6[:, :3], r6[:, 3:]
    x = a / np.linalg.norm(a, axis=1, keepdims=True)
    y = b - (x * b).sum(1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -1)


def split_outputs(out, K):
    """Model outputs in any order (ONNX/TFLite) -> (maps, xyz or None, rot6d (R,6) or None)."""
    if not isinstance(out, (list, tuple)):
        return out, None, None
    outs = sorted(out, key=lambda o: -o.size)  # maps are by far the largest
    xyz = next((o for o in outs[1:] if o.size == 3 * K), None)
    rot = next((o for o in outs[1:] if o is not xyz), None)
    return outs[0], xyz, (None if rot is None else np.asarray(rot).reshape(-1, 6))


def decode_np(out, K):
    """out: maps array, or [maps, xyz(, rot)] for the global-3D head (any batch-1 layout ONNX/TFLite returns).
    Rotations are not decoded here: use split_outputs + rot6d_to_mat_np."""
    out, xyz_g, _ = split_outputs(out, K)
    out = out[0] if out.ndim == 4 else out  # (C,h,w)
    _, h, w = out.shape
    lg = out[:K].reshape(K, -1)
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    gx = (np.tile(np.arange(w), h) + 0.5) / w
    gy = (np.repeat(np.arange(h), w) + 0.5) / h
    uv = np.stack([p @ gx, p @ gy], -1)
    if xyz_g is not None:
        xyz = np.asarray(xyz_g).reshape(K, 3)
    else:
        loc = out[K:].reshape(3, K, -1)
        xyz = np.einsum("ckn,kn->kc", loc, p)
    conf = p.max(1) * np.sqrt(h * w)
    return uv, xyz, conf


def uv_to_image(uv, affine_inv, in_w, in_h):
    """Map normalized crop coords back to full image pixels with the inverse crop affine (2x3)."""
    pts = np.c_[uv[:, 0] * in_w, uv[:, 1] * in_h, np.ones(len(uv))]
    return pts @ affine_inv.T
