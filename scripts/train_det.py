"""Train the subject box detector (minisam3d/detector.py) on pack_noark.py takes, then export ONNX.

  python scripts/train_det.py --data data/noark_pack/* --val kabil --out runs/det_v1 --pretrained

--val holds every take of that subject out (default: 2% random split of all takes). Metrics are in
original-frame pixels (640x400): centre error, box side error, IoU of the square boxes, and the miss rate
(no peak above threshold, or the picked peak is not the subject: IoU < 0.5).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from minisam3d.detector import DetDataset, PersonDet, decode_det, det_loss  # noqa: E402


def square_iou(c0, s0, c1, s1):
    lo, hi = np.maximum(c0 - s0 / 2, c1 - s1 / 2), np.minimum(c0 + s0 / 2, c1 + s1 / 2)
    inter = np.prod(np.clip(hi - lo, 0, None))
    return inter / (s0 ** 2 + s1 ** 2 - inter)


@torch.no_grad()
def evaluate(model, dl, dev, wh, to_frame, thr):
    model.eval()
    ce, se, iou, miss, n = [], [], [], 0, 0
    for b in dl:
        out = model(b["img"].to(dev)).float().cpu().numpy()
        for k in range(len(out)):
            n += 1
            r = decode_det(out[k], wh, thr)
            c, s = b["center"][k].numpy(), float(b["side"][k])
            if r is None:
                miss += 1
                continue
            u = square_iou(r[0], r[1], c, s)
            miss += u < 0.5
            iou.append(u)
            ce.append(np.hypot(*(r[0] - c)) * to_frame)
            se.append(abs(r[1] - s) / s)
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")  # noqa: E731
    return dict(center_px=q(ce, 50), center_p95=q(ce, 95), side_err=q(se, 50), iou=q(iou, 50),
                iou_p5=q(iou, 5), miss=float(miss) / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="pack_noark.py take folders")
    ap.add_argument("--val", help="hold out this subject's takes (folder names <subject>_<kind>)")
    ap.add_argument("--out", default="runs/det_v1")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    roots = [r for r in a.data if os.path.isdir(os.path.join(r, "labels"))]
    subj = lambda r: os.path.basename(os.path.normpath(r)).rsplit("_", 1)[0]  # noqa: E731
    if a.val:
        tr = ConcatDataset([DetDataset(r, True) for r in roots if subj(r) != a.val])
        va = ConcatDataset([DetDataset(r, False) for r in roots if subj(r) == a.val])
        if not len(va):
            sys.exit(f"no takes for --val {a.val} in {roots}")
    else:
        full = ConcatDataset([DetDataset(r, True) for r in roots])
        n_val = max(1, len(full) // 50)
        tr, vi = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
        va = Subset(ConcatDataset([DetDataset(r, False) for r in roots]), vi.indices)
    d0 = DetDataset(roots[0], False)
    wh, to_frame = d0.wh, 1 / float(d0.scale[0])
    print(f"train {len(tr)}  val {len(va)}  input {wh}  roots {[os.path.basename(os.path.normpath(r)) for r in roots]}")
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.bs, num_workers=a.workers)
    model = PersonDet(pretrained=a.pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(a.epochs * len(dl), 1), pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    step, best = 0, -1.0
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss, parts = det_loss(model(b["img"]), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0 or a.max_steps:
                print(f"ep {ep} step {step} loss {loss.item():.4f} " +
                      " ".join(f"{k} {v:.4f}" for k, v in parts.items()), flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        m = evaluate(model, vl, dev, wh, to_frame, a.thr)
        print(f"== ep {ep} val centre {m['center_px']:.1f}px (p95 {m['center_p95']:.1f})  side {100 * m['side_err']:.1f}%  "
              f"IoU {m['iou']:.3f} (p5 {m['iou_p5']:.3f})  miss {100 * m['miss']:.2f}%  ({time.time() - t0:.0f}s)",
              flush=True)
        ck = dict(model=model.state_dict(), size=(wh[1], wh[0]), metrics=m, epoch=ep)
        score = float(np.nan_to_num(m["iou"])) * (1 - m["miss"])  # all missed -> iou nan
        if score > best:
            best = score
            torch.save(ck, os.path.join(a.out, "best.pt"))
        torch.save(ck, os.path.join(a.out, "last.pt"))
        if a.max_steps and step >= a.max_steps:
            break
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev)["model"])
    model.eval().cpu()
    onnx = os.path.join(a.out, "det.onnx")
    torch.onnx.export(model, torch.rand(1, 1, wh[1], wh[0]), onnx, input_names=["image"], output_names=["det"],
                      opset_version=17, dynamo=False)
    print("wrote", onnx)


if __name__ == "__main__":
    main()
