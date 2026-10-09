import torch
import torch.nn.functional as F

from .model import decode, rot6d_to_mat


def gaussian_targets(uv, h, w, sigma=1.5):
    ys = torch.arange(h, device=uv.device, dtype=uv.dtype) + 0.5
    xs = torch.arange(w, device=uv.device, dtype=uv.dtype) + 0.5
    cx = uv[..., 0:1, None] * w
    cy = uv[..., 1:2, None] * h
    gx = torch.exp(-((xs.view(1, 1, 1, -1) - cx) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ys.view(1, 1, -1, 1) - cy.transpose(-1, -2)) ** 2) / (2 * sigma ** 2))
    return gy * gx  # (B,K,h,w)


def geodesic(Rp, Rt):
    """Angle (rad) between rotation matrices (...,3,3)."""
    c = ((Rp.transpose(-1, -2) @ Rt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def rot_out(out, R):
    """(B,R,3,3) predicted rotations from the model output's third element."""
    return rot6d_to_mat(out[2].reshape(len(out[2]), R, 6))


def distill_loss(out, batch, js, w2d=1.0, w3d=1.0, whm=1.0, wbone=0.5, wrot=1.0):
    K = js["num"]
    uv_p, xyz_p, _ = decode(out, K)
    maps = out[0] if isinstance(out, (tuple, list)) else out
    vis, w3 = batch["vis"], batch["w3d"]
    l2d = ((uv_p - batch["uv"]).abs().sum(-1) * vis).sum() / vis.sum().clamp(min=1)
    # auxiliary heatmap loss sharpens the softmax so decoding is stable after INT8
    h, w = maps.shape[-2:]
    tgt = gaussian_targets(batch["uv"], h, w)
    tgt = tgt / tgt.flatten(2).sum(-1).clamp(min=1e-6)[..., None, None]
    logp = F.log_softmax(maps[:, :K].flatten(2), -1)
    lhm = (-(tgt.flatten(2) * logp).sum(-1) * vis).sum() / vis.sum().clamp(min=1)
    l3d = ((xyz_p - batch["xyz"]).abs().sum(-1) * w3).sum() / w3.sum()
    a, b = zip(*js["bones"])
    bl_p = (xyz_p[:, list(a)] - xyz_p[:, list(b)]).norm(dim=-1)
    bl_t = (batch["xyz"][:, list(a)] - batch["xyz"][:, list(b)]).norm(dim=-1)
    lbone = (bl_p - bl_t).abs().mean()
    total = w2d * l2d + whm * 0.01 * lhm + w3d * l3d + wbone * lbone
    parts = dict(l2d=l2d.item(), lhm=lhm.item(), l3d=l3d.item(), lbone=lbone.item())
    if isinstance(out, (tuple, list)) and len(out) > 2:  # rotation head; wrot = 0 where the label has none
        wr = batch["wrot"]
        lrot = (geodesic(rot_out(out, wr.shape[1]), batch["rot"]) * wr).sum() / wr.sum().clamp(min=1)
        total = total + wrot * lrot
        parts["lrot"] = lrot.item()
    return total, parts


@torch.no_grad()
def metric_sums(out, batch, js, in_px):
    """Per-joint sums for aggregation over a dataset: 2D px error over visible joints, 3D error (mm) over all."""
    out = tuple(o.float() for o in out) if isinstance(out, (tuple, list)) else out.float()  # bf16 under --amp
    uv_p, xyz_p, _ = decode(out, js["num"])
    vis = batch["vis"]
    px =((uv_p - batch["uv"]) * in_px).norm(dim=-1)
    mm = 1000 * (xyz_p - batch["xyz"]).norm(dim=-1)
    s = dict(px=(px * vis).sum(0), vis=vis.sum(0), mm=mm.sum(0), n=torch.tensor(float(len(mm)), device=mm.device))
    if isinstance(out, (tuple, list)) and len(out) > 2:
        wr = batch["wrot"]
        s["deg"] = (torch.rad2deg(geodesic(rot_out(out, wr.shape[1]), batch["rot"])) * wr).sum(0)
        s["nrot"] = wr.sum(0)
    return s


@torch.no_grad()
def evaluate(model, loader, js, in_px, dev):
    acc = None
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items()}
        s = metric_sums(model(b["img"]), b, js, in_px)
        acc = s if acc is None else {k: acc[k] + s[k] for k in s}
    px_j = (acc["px"] / acc["vis"].clamp(min=1)).cpu()
    mm_j = (acc["mm"] / acc["n"]).cpu()
    r = dict(px_err=(acc["px"].sum() / acc["vis"].sum().clamp(min=1)).item(), mpjpe_mm=mm_j.mean().item(),
             px_joint=px_j.tolist(), mm_joint=mm_j.tolist())
    if "deg" in acc:  # geodesic rotation error per rotation joint, over samples that have rotation labels
        deg_j = (acc["deg"] / acc["nrot"].clamp(min=1)).cpu()
        r.update(rot_deg=(acc["deg"].sum() / acc["nrot"].sum().clamp(min=1)).item(), deg_joint=deg_j.tolist())
    return r
