import torch
import torch.nn.functional as F

from .model import decode


def gaussian_targets(uv, h, w, sigma=1.5):
    ys = torch.arange(h, device=uv.device, dtype=uv.dtype) + 0.5
    xs = torch.arange(w, device=uv.device, dtype=uv.dtype) + 0.5
    cx = uv[..., 0:1, None] * w
    cy = uv[..., 1:2, None] * h
    gx = torch.exp(-((xs.view(1, 1, 1, -1) - cx) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ys.view(1, 1, -1, 1) - cy.transpose(-1, -2)) ** 2) / (2 * sigma ** 2))
    return gy * gx  # (B,K,h,w)


def distill_loss(out, batch, js, w2d=1.0, w3d=1.0, whm=1.0, wbone=0.5):
    K = js["num"]
    uv_p, xyz_p, _ = decode(out, K)
    vis, w3 = batch["vis"], batch["w3d"]
    l2d = ((uv_p - batch["uv"]).abs().sum(-1) * vis).sum() / vis.sum().clamp(min=1)
    # auxiliary heatmap loss sharpens the softmax so decoding is stable after INT8
    h, w = out.shape[-2:]
    tgt = gaussian_targets(batch["uv"], h, w)
    tgt = tgt / tgt.flatten(2).sum(-1).clamp(min=1e-6)[..., None, None]
    logp = F.log_softmax(out[:, :K].flatten(2), -1)
    lhm = (-(tgt.flatten(2) * logp).sum(-1) * vis).sum() / vis.sum().clamp(min=1)
    l3d = ((xyz_p - batch["xyz"]).abs().sum(-1) * w3).sum() / w3.sum()
    a, b = zip(*js["bones"])
    bl_p = (xyz_p[:, list(a)] - xyz_p[:, list(b)]).norm(dim=-1)
    bl_t = (batch["xyz"][:, list(a)] - batch["xyz"][:, list(b)]).norm(dim=-1)
    lbone = (bl_p - bl_t).abs().mean()
    total = w2d * l2d + whm * 0.01 * lhm + w3d * l3d + wbone * lbone
    return total, dict(l2d=l2d.item(), lhm=lhm.item(), l3d=l3d.item(), lbone=lbone.item())


@torch.no_grad()
def metrics(out, batch, js, in_px):
    uv_p, xyz_p, _ = decode(out, js["num"])
    vis = batch["vis"]
    px = ((uv_p - batch["uv"]) * in_px).norm(dim=-1)
    mpjpe = (xyz_p - batch["xyz"]).norm(dim=-1).mean()
    return dict(px_err=((px * vis).sum() / vis.sum().clamp(min=1)).item(), mpjpe_mm=1000 * mpjpe.item())
