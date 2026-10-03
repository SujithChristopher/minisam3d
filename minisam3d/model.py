"""Student: MobileNetV2 + FPN-lite decoder -> per-joint heatmap + 3 location maps.

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


class Student(nn.Module):
    def __init__(self, num_joints=14, width=1.0, dec_ch=96, pretrained=False, in_ch=3):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained and width == 1.0 else None
        mb = torchvision.models.mobilenet_v2(weights=weights, width_mult=width).features
        if in_ch != 3:  # e.g. 1 for the mono OV9281; fold pretrained RGB filters into one channel
            c0 = mb[0][0]
            new = nn.Conv2d(in_ch, c0.out_channels, 3, 2, 1, bias=False)
            with torch.no_grad():
                new.weight.copy_(c0.weight.sum(1, keepdim=True).repeat(1, in_ch, 1, 1) / in_ch)
            mb[0][0] = new
        # stride 8 / 16 / 32 taps
        self.s8, self.s16, self.s32 = mb[:7], mb[7:14], mb[14:18]
        c8, c16, c32 = [self._out_ch(m) for m in (self.s8, self.s16, self.s32)]
        self.lat32 = cbr(c32, dec_ch, 1)
        self.lat16 = cbr(c16, dec_ch, 1)
        self.lat8 = cbr(c8, dec_ch, 1)
        self.up16 = cbr(dec_ch, dec_ch)
        self.up8 = cbr(dec_ch, dec_ch)
        self.up4 = cbr(dec_ch, dec_ch)
        self.head = nn.Conv2d(dec_ch, num_joints * 4, 1)
        self.K = num_joints

    @staticmethod
    def _out_ch(seq):
        for m in reversed(list(seq.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels

    def forward(self, x):
        """x: (B,in_ch,H,W) float in [0,1]. Returns (B, 4K, H/4, W/4):
        channels [0:K] heatmap logits, [K:4K] xyz location maps (K*3, xyz-major)."""
        f8 = self.s8(x)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        y = self.lat32(f32)
        y = self.up16(F.interpolate(y, scale_factor=2, mode="nearest") + self.lat16(f16))
        y = self.up8(F.interpolate(y, scale_factor=2, mode="nearest") + self.lat8(f8))
        y = self.up4(F.interpolate(y, scale_factor=2, mode="nearest"))
        return self.head(y)


def decode(out, K, beta=1.0):
    """Soft-argmax decode (torch, differentiable).
    Returns uv in [0,1] (B,K,2), xyz root-relative metres (B,K,3), conf (B,K)."""
    B, _, h, w = out.shape
    logits = out[:, :K].reshape(B, K, -1) * beta
    p = logits.softmax(-1)
    ys = (torch.arange(h, device=out.device, dtype=out.dtype) + 0.5) / h
    xs = (torch.arange(w, device=out.device, dtype=out.dtype) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    u = (p * gx.reshape(1, 1, -1)).sum(-1)
    v = (p * gy.reshape(1, 1, -1)).sum(-1)
    loc = out[:, K:].reshape(B, 3, K, -1)  # xyz-major
    xyz = (loc * p.unsqueeze(1)).sum(-1).permute(0, 2, 1)
    conf = p.max(-1).values * (h * w) ** 0.5  # peakiness, rough confidence
    return torch.stack([u, v], -1), xyz, conf
