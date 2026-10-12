"""Agreement with the teacher, per take, on one split of a versioned dataset (default: its test split).

  python scripts/eval_teacher.py --dataset noark_v2 --ckpt runs/r18b_loso_all/best.pt runs/r18_v2_all/best.pt

Prints px / mm / rotation error against the teacher labels plus the arm and trunk joints that matter for
compensation, one row per take and checkpoint, and a frame-weighted mean per checkpoint. --out also writes JSON.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from minisam3d.data import PseudoLabelDataset  # noqa: E402
from minisam3d.losses import evaluate  # noqa: E402
from minisam3d.model import Student  # noqa: E402
from scripts.dataset import load as load_manifest, parse_maps, verify as verify_manifest  # noqa: E402

COLS = ("px", "mm", "rot", "wrist", "elbow", "trunk", "forearm")


def load_student(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Student(ck["K"], width=ck["width"], in_ch=ck["in_ch"], dec_ch=ck["dec_ch"], head3d=ck["head3d"],
                num_rot=ck["num_rot"], backbone=ck.get("backbone", "mbv2")).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--map", nargs="*")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="write the numbers here as JSON")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_manifest(a.dataset)
    maps, roots = parse_maps(a.map), m["splits"][a.split]
    bad = verify_manifest(m, a.data_root, maps, roots=roots)
    if bad:
        sys.exit(f"data on disk does not match {m['id']}:\n  " + "\n  ".join(bad[:20]))

    res = {}
    print(f"{m['id']} split {a.split}: {len(roots)} takes\n")
    print(f"{'ckpt':<28}{'take':<26}{'n':>6}" + "".join(f"{c:>9}" for c in COLS))
    for path in a.ckpt:
        model, ck = load_student(path, dev)
        name = os.path.basename(os.path.dirname(path))
        trained = (ck.get("dataset") or {}).get("roots", [])
        leak = sorted(set(trained) & set(roots))
        if leak:
            print(f"WARNING {name} trained on {leak}")
        res[name] = {}
        for rel in roots:
            ds = PseudoLabelDataset(maps.get(rel, os.path.join(a.data_root, rel)), 256, 256, train=False, gray=True)
            js = ds.js
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                r = evaluate(model, DataLoader(ds, a.bs, num_workers=a.workers), js,
                             torch.tensor([256, 256], device=dev), dev)
            jm = dict(zip(js["names"], r["mm_joint"]))
            jr = dict(zip(js["rot_names"], r.get("deg_joint", [np.nan] * len(js["rot_names"]))))
            row = dict(n=len(ds), px=r["px_err"], mm=r["mpjpe_mm"], rot=r.get("rot_deg", np.nan),
                       wrist=np.mean([jm["left_wrist"], jm["right_wrist"]]),
                       elbow=np.mean([jm["left_elbow"], jm["right_elbow"]]), trunk=jr["trunk"],
                       forearm=np.mean([jr["left_forearm"], jr["right_forearm"]]))
            res[name][rel] = {k: float(v) for k, v in row.items()}
            print(f"{name:<28}{os.path.basename(rel):<26}{row['n']:>6}" +
                  "".join(f"{row[c]:>9.2f}" for c in COLS), flush=True)
        n = np.array([v["n"] for v in res[name].values()])
        mean = {c: float(np.sum(n * [v[c] for v in res[name].values()]) / n.sum()) for c in COLS}
        res[name]["_mean"] = dict(mean, n=int(n.sum()))
        print(f"{name:<28}{'MEAN (frame-weighted)':<26}{n.sum():>6}" + "".join(f"{mean[c]:>9.2f}" for c in COLS))
        print()
    if a.out:
        json.dump(dict(dataset=m["id"], split=a.split, results=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
