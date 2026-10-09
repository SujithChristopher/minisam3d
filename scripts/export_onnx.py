"""Export a trained student to ONNX (static shape, batch 1, NCHW float input in [0,1]).

  python scripts/export_onnx.py --ckpt runs/v1/best.pt --out student.onnx
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from minisam3d.model import Student  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="omit to export random weights (latency profiling only)")
    ap.add_argument("--out", default="student.onnx")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--joints", type=int, default=14)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--in-ch", type=int, default=3)
    a = ap.parse_args()
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu")
        if ck.get("arch") != "v2":
            sys.exit("checkpoint is from the v1 model; export it with the v1 code (git history)")
        m = Student(ck["K"], width=ck["width"], in_ch=ck.get("in_ch", 3), dec_ch=ck["dec_ch"], head3d=ck["head3d"],
                    num_rot=ck.get("num_rot", 0))
        m.load_state_dict(ck["model"])
        H, W = ck["size"]
    else:
        m, (H, W) = Student(a.joints, width=a.width, in_ch=a.in_ch), a.size
    m.eval()
    torch.onnx.export(m, torch.rand(1, m.s4[0][0].in_channels, H, W), a.out, input_names=["image"],
                      output_names=(["maps", "xyz"] if m.head3d == "global" else ["maps"]) + (["rot"] if m.num_rot else []),
                      opset_version=17, dynamo=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
