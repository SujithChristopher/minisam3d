"""Pseudo-label images / videos with SAM 3D Body (run on a CUDA GPU machine).

Setup (once):
  git clone https://github.com/facebookresearch/sam-3d-body && follow its INSTALL.md
  accept the license on https://huggingface.co/facebook/sam-3d-body-dinov3 and `huggingface-cli login`

Usage:
  PYTHONPATH=/path/to/sam-3d-body python label_with_teacher.py \
      --inputs /data/coco/train2017 /data/my_videos --out /data/msam_labels \
      --video-stride 5 --shard-size 2000
Writes <out>/images/... (video frames as jpg) and <out>/labels/shard_XXXXX.npz.
"""
import argparse
import os
import sys

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def iter_frames(inputs, out_dir, video_stride, skip=()):
    for root in inputs:
        files = [root] if os.path.isfile(root) else sorted(
            os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT:
                rel = os.path.join("images", os.path.relpath(f, os.path.dirname(root.rstrip("/"))))
                if rel in skip:
                    continue
                img = cv2.imread(f)
                if img is None:
                    continue
                dst = os.path.join(out_dir, rel)
                if not os.path.lexists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    try:
                        os.symlink(os.path.abspath(f), dst)
                    except OSError:  # e.g. Google Drive: train with --image-dirs instead
                        pass
                yield rel, img
            elif ext in VID_EXT:
                cap = cv2.VideoCapture(f)
                stem = os.path.splitext(os.path.basename(f))[0]
                i = 0
                while True:
                    ok, img = cap.read()
                    if not ok:
                        break
                    rel = os.path.join("images", "video", stem, f"{i:07d}.jpg")
                    if i % video_stride == 0 and rel not in skip:
                        dst = os.path.join(out_dir, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        yield rel, img
                    i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-repo", default="facebook/sam-3d-body-dinov3")
    ap.add_argument("--video-stride", type=int, default=5)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--min-box", type=float, default=96, help="skip people smaller than this (px)")
    ap.add_argument("--no-fov", action="store_true", help="skip MoGe FOV estimation (faster, default FOV)")
    ap.add_argument("--hands", action="store_true", help="run the hand decoder too (slower)")
    ap.add_argument("--intrinsics", help="json {image rel path: 3x3 K} from fisheye_to_views.py; "
                    "defaults to <input>/intrinsics.json when present")
    ap.add_argument("--gray", action="store_true", help="feed the teacher grayscale (3x replicated), "
                    "to check/match how it behaves on mono OV9281 footage")
    a = ap.parse_args()

    from notebook.utils import setup_sam_3d_body  # from the sam-3d-body repo
    est = setup_sam_3d_body(hf_repo_id=a.hf_repo, fov_name="" if a.no_fov else "moge2")

    import json
    import torch
    # keyed by the last 3 path parts (video/view/frame.jpg) so any input root layout matches
    tail = lambda p: "/".join(p.replace(os.sep, "/").split("/")[-3:])  # noqa: E731
    intr = {}
    cands = [a.intrinsics] if a.intrinsics else \
        [os.path.join(d, "intrinsics.json") for r in a.inputs for d in (r, os.path.dirname(r.rstrip("/")))]
    for p in cands:
        if p and os.path.exists(p):
            intr.update({tail(k): v for k, v in json.load(open(p)).items()})
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    # resume: skip images already in existing shards (Colab sessions get cut off)
    done, shard = set(), 0
    for f in sorted(os.listdir(os.path.join(a.out, "labels"))):
        if f.startswith("shard_") and f.endswith(".npz"):
            done.update(str(x) for x in np.load(os.path.join(a.out, "labels", f))["image"])
            shard = max(shard, int(f[6:11]) + 1)
    done_log = os.path.join(a.out, "labels", "done.txt")  # also images with nobody in them
    if os.path.exists(done_log):
        done.update(x for x in open(done_log).read().split("\n") if x)
    if done:
        print(f"resuming: {len(done)} images already labelled", file=sys.stderr)
    buf, n_img = {}, 0
    pending = []

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        path = os.path.join(a.out, "labels", f"shard_{shard:05d}.npz")
        np.savez(path, **{k: np.stack(v) for k, v in buf.items()})
        print(f"wrote {path} ({len(buf['image'])} people)", file=sys.stderr)
        buf, shard = {}, shard + 1
        with open(done_log, "a") as fh:
            fh.write("\n".join(pending) + "\n")
        pending.clear()

    for rel, bgr in iter_frames(a.inputs, a.out, a.video_stride, skip=done):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if a.gray:
            rgb = np.repeat(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
        Kmat = intr.get(tail(rel))
        cam_int = torch.tensor(Kmat, dtype=torch.float32)[None] if Kmat is not None else None
        outs = est.process_one_image(rgb, cam_int=cam_int, inference_type="full" if a.hands else "body")
        n_img += 1
        pending.append(rel)
        for o in outs:
            x0, y0, x1, y1 = o["bbox"]
            if max(x1 - x0, y1 - y0) < a.min_box:
                continue
            rec = dict(image=np.array(rel), img_wh=np.array(bgr.shape[1::-1], np.int32),
                       bbox=np.asarray(o["bbox"], np.float32),
                       kp2d=np.asarray(o["pred_keypoints_2d"], np.float32),
                       kp3d=np.asarray(o["pred_keypoints_3d"], np.float32),
                       cam_t=np.asarray(o["pred_cam_t"], np.float32),
                       focal=np.asarray(o["focal_length"], np.float32).reshape(()))
            for k, v in rec.items():
                buf.setdefault(k, []).append(v)
        if buf and len(buf["image"]) >= a.shard_size:
            flush()
        if n_img % 500 == 0:
            print(f"{n_img} images", file=sys.stderr)
    flush()


if __name__ == "__main__":
    main()
