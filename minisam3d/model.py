"""Student: MobileNetV2 + FPN-lite decoder -> per-joint heatmaps, plus root-relative 3D.

Only conv / BN / ReLU6 / nearest-upsample / add, so it quantizes cleanly to
INT8 and every op runs on the Hexagon HTP. Decoding (soft-argmax) is done
outside the exported graph, see postprocess.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def cbr(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Student(nn.Module):
    """v2 changes vs v1 (v1 = stride-8 skips only, location-map 3D, no input normalization):
      - stride-4 skip from the 64x64 backbone stage, so 2D peaks are not upsampled from stride 8
      - ImageNet normalization folded into the first conv (pretrained filters see the input they expect)
      - wider decoder with two convs at the output resolution
      - head3d="global": root-relative 3D regressed from pooled stride-32 features. 3D pose is holistic,
        and joints outside the crop (hips in upper-body framing) have no heatmap location to read from.
    """

    def __init__(self, num_joints=14, width=1.0, dec_ch=128, pretrained=False, in_ch=3, head3d="global",
                 num_rot=0):
        super().__init__()
        if pretrained and width != 1.0:
            print("warning: torchvision has ImageNet weights only for width 1.0; training from scratch")
        weights = "IMAGENET1K_V1" if pretrained and width == 1.0 else None
        mb = torchvision.models.mobilenet_v2(weights=weights, width_mult=width).features
        self._fold_input(mb, in_ch)
        # stride 4 / 8 / 16 / 32 taps
        self.s4, self.s8, self.s16, self.s32 = mb[:4], mb[4:7], mb[7:14], mb[14:18]
        c4, c8, c16, c32 = [self._out_ch(m) for m in (self.s4, self.s8, self.s16, self.s32)]
        self.lat32 = cbr(c32, dec_ch, 1)
        self.lat16 = cbr(c16, dec_ch, 1)
        self.lat8 = cbr(c8, dec_ch, 1)
        self.lat4 = cbr(c4, dec_ch, 1)
        self.up16 = cbr(dec_ch, dec_ch)
        self.up8 = cbr(dec_ch, dec_ch)
        self.up4 = nn.Sequential(cbr(dec_ch, dec_ch), cbr(dec_ch, dec_ch))
        self.head3d = head3d
        self.K = num_joints
        if head3d == "global":
            self.head = nn.Conv2d(dec_ch, num_joints, 1)
            self.reg = nn.Sequential(nn.Conv2d(c32, 256, 1), nn.ReLU6(inplace=True),
                                     nn.Conv2d(256, num_joints * 3, 1))  # 1x1 convs = FC on the pooled vector
        else:
            self.head = nn.Conv2d(dec_ch, num_joints * 4, 1)
        # global rotations (camera frame) of trunk / arm segments, as 6D (first two matrix columns)
        self.num_rot = num_rot
        assert not num_rot or head3d == "global", "rotation head needs head3d='global'"
        if num_rot:
            self.rot = nn.Sequential(nn.Conv2d(c32, 256, 1), nn.ReLU6(inplace=True), nn.Conv2d(256, num_rot * 6, 1))

    @staticmethod
    def _fold_input(mb, in_ch):
        """Replace the first conv so it takes raw [0,1] input: w' = w/std, bias = -sum(w*mean/std).
        For in_ch=1 (mono) the RGB filters are summed, i.e. the gray image is treated as R=G=B."""
        c0 = mb[0][0]
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        new = nn.Conv2d(in_ch, c0.out_channels, 3, 2, 1, bias=True)
        with torch.no_grad():
            w = c0.weight / std
            new.weight.copy_(w if in_ch == 3 else w.sum(1, keepdim=True).repeat(1, in_ch, 1, 1) / in_ch)
            new.bias.copy_(-(c0.weight * mean / std).sum((1, 2, 3)))
        mb[0][0] = new

    @staticmethod
    def _out_ch(seq):
        for m in reversed(list(seq.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels

    def forward(self, x):
        """x: (B,in_ch,H,W) float in [0,1].
        head3d="global": returns (maps (B,K,H/4,W/4) heatmap logits, xyz (B,3K,1,1) root-relative metres).
        head3d="loc":    returns maps (B,4K,H/4,W/4) = K heatmap logits + 3K xyz location maps (xyz-major).
        With num_rot > 0 a third output rot (B,6R,1,1) follows (6D rotations, see rot6d_to_mat)."""
        f4 = self.s4(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        y = self.lat32(f32)
        y = self.up16(F.interpolate(y, scale_factor=2, mode="nearest") + self.lat16(f16))
        y = self.up8(F.interpolate(y, scale_factor=2, mode="nearest") + self.lat8(f8))
        y = self.up4(F.interpolate(y, scale_factor=2, mode="nearest") + self.lat4(f4))
        maps = self.head(y)
        pooled = f32.mean((2, 3), keepdim=True)
        out = (maps, self.reg(pooled)) if self.head3d == "global" else (maps,)
        if self.num_rot:
            out = out + (self.rot(pooled),)
        return out if len(out) > 1 else out[0]


def rot6d_to_mat(r6):
    """(...,6) -> (...,3,3): Gram-Schmidt on the two predicted columns (Zhou et al. 2019)."""
    a, b = r6[..., :3], r6[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


def mat_to_rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def decode(out, K, beta=1.0):
    """Soft-argmax decode (torch, differentiable). `out` is the model output (tensor or (maps, xyz)).
    Returns uv in [0,1] (B,K,2), xyz root-relative metres (B,K,3), conf (B,K)."""
    if isinstance(out, (tuple, list)):  # (maps, xyz[, rot]) for head3d="global"; rot is decoded by the caller
        out, xyz_g = out[0], (out[1] if len(out) > 1 else None)
    else:
        xyz_g = None
    B, _, h, w = out.shape
    logits = out[:, :K].reshape(B, K, -1) * beta
    p = logits.softmax(-1)
    ys = (torch.arange(h, device=out.device, dtype=out.dtype) + 0.5) / h
    xs = (torch.arange(w, device=out.device, dtype=out.dtype) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    u = (p * gx.reshape(1, 1, -1)).sum(-1)
    v = (p * gy.reshape(1, 1, -1)).sum(-1)
    if xyz_g is not None:
        xyz = xyz_g.reshape(B, K, 3)
    else:
        loc = out[:, K:].reshape(B, 3, K, -1)  # xyz-major
        xyz = (loc * p.unsqueeze(1)).sum(-1).permute(0, 2, 1)
    conf = p.max(-1).values * (h * w) ** 0.5  # peakiness, rough confidence
    return torch.stack([u, v], -1), xyz, conf
