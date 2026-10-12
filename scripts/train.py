"""Train the student on teacher pseudo-labels.

  python scripts/train.py --data /data/msam_labels --out runs/v1 --epochs 60 --pretrained
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from minisam3d.data import PseudoLabelDataset  # noqa: E402
from minisam3d.losses import distill_loss, evaluate  # noqa: E402
from minisam3d.model import Student  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="one or more pseudo-label roots")
    ap.add_argument("--repeat", type=int, nargs="*", default=[],
                    help="per --data oversampling factor, e.g. 1 4 to weight your own footage 4x")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    ap.add_argument("--backbone", default="mbv2", choices=["mbv2", "resnet18"])
    ap.add_argument("--width", type=float, default=1.0, help="MobileNetV2 width multiplier")
    ap.add_argument("--dec-ch", type=int, default=128)
    ap.add_argument("--head3d", default="global", choices=["global", "loc"])
    ap.add_argument("--rot", action="store_true", help="also distill the teacher's global rotations of the trunk "
                    "and arm segments (keypoints.ROT_JOINTS); needs labels with 'rot'")
    ap.add_argument("--init", help="start weights from this checkpoint (non-matching heads stay random)")
    ap.add_argument("--hands", action="store_true")
    ap.add_argument("--gray", action="store_true", help="1-channel input for the mono OV9281")
    ap.add_argument("--image-dirs", nargs="*", default=[], help="where the labelled input folders live, "
                    "if images were not linked into --data")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the network (about half the "
                    "activation memory; keeps an 8 GB GPU from spilling into shared memory)")
    ap.add_argument("--resume", help="checkpoint to continue from (Colab disconnects)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = a.size

    kw = dict(hands=a.hands, gray=a.gray, image_dirs=a.image_dirs)
    trs, vas, tes = [], [], []
    for k, root in enumerate(a.data):
        full = PseudoLabelDataset(root, H, W, train=True, **kw)
        n_val = max(1, len(full) // 50)
        tr, va = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
        trs += [tr] * (a.repeat[k] if k < len(a.repeat) else 1)
        clean = PseudoLabelDataset(root, H, W, train=False, **kw)
        vas.append(torch.utils.data.Subset(clean, va.indices))
        tes.append(torch.utils.data.Subset(clean, tr.indices[:n_val]))  # same size, no augmentation
    tr, va, te, js = ConcatDataset(trs), ConcatDataset(vas), ConcatDataset(tes), full.js
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.bs, pin_memory=True)
    vl = DataLoader(va, a.bs, num_workers=a.workers)
    tl = DataLoader(te, a.bs, num_workers=a.workers)  # train-set error: if close to val, the model underfits

    in_ch = 1 if a.gray else 3
    num_rot = js["num_rot"] if a.rot else 0
    model = Student(js["num"], width=a.width, dec_ch=a.dec_ch, pretrained=a.pretrained, in_ch=in_ch,
                    head3d=a.head3d, num_rot=num_rot, backbone=a.backbone).to(dev)
    if a.init:  # e.g. a keypoint-only checkpoint; new heads (rotation) start from scratch
        miss, unexp = model.load_state_dict(torch.load(a.init, map_location=dev)["model"], strict=False)
        print(f"init from {a.init}: missing {len(miss)} keys {sorted({k.split('.')[0] for k in miss})}, "
              f"unexpected {len(unexp)}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(total, 1), pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    step, best, ep0 = 0, 1e9, 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            ep0, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        print(f"resumed from {a.resume} at epoch {ep0}")
    for ep in range(ep0, a.epochs):
        model.train()
        t0 = time.time()
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev == "cuda"):
                out = model(b["img"])
            out = tuple(o.float() for o in out) if isinstance(out, tuple) else out.float()  # loss in fp32
            loss, parts = distill_loss(out, b, js)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or a.max_steps:
                print(f"ep {ep} step {step} loss {loss.item():.4f} " +
                      " ".join(f"{k} {v:.4f}" for k, v in parts.items()), flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        model.eval()
        in_px = torch.tensor([W, H], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev == "cuda"):
            agg, tra = evaluate(model, vl, js, in_px, dev), evaluate(model, tl, js, in_px, dev)
        rot_s = (f"  rot {agg['rot_deg']:.1f}deg" if "rot_deg" in agg else "",
                 f"  rot {tra['rot_deg']:.1f}deg" if "rot_deg" in tra else "")
        print(f"== ep {ep} val px_err {agg['px_err']:.2f}px  mpjpe {agg['mpjpe_mm']:.1f}mm{rot_s[0]} | "
              f"train px_err {tra['px_err']:.2f}px  mpjpe {tra['mpjpe_mm']:.1f}mm{rot_s[1]}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if ep == a.epochs - 1 or ep % 10 == 0:
            print("   per joint (val px / mm): " + ", ".join(
                f"{n} {p:.1f}/{m:.0f}" for n, p, m in zip(js["names"], agg["px_joint"], agg["mm_joint"])), flush=True)
            if "deg_joint" in agg:
                print("   per rotation joint (val deg): " + ", ".join(
                    f"{n} {d:.1f}" for n, d in zip(js["rot_names"], agg["deg_joint"])), flush=True)
        ck = dict(model=model.state_dict(), K=js["num"], hands=a.hands, size=(H, W), width=a.width, in_ch=in_ch,
                  dec_ch=a.dec_ch, head3d=a.head3d, num_rot=num_rot, arch="v2", backbone=a.backbone)
        score = agg.get("rot_deg", agg["mpjpe_mm"])  # with --rot, keep the checkpoint with the best rotations
        if score < best:
            best = score
            torch.save(ck, os.path.join(a.out, "best.pt"))
        torch.save(dict(ck, opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, step=step, best=best),
                   os.path.join(a.out, "last.pt"))
        if a.max_steps and step >= a.max_steps:
            break


if __name__ == "__main__":
    main()
