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
import queue
import sys
import threading
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minisam3d.keypoints import MHR_TO_CAM, ROT_JOINTS  # noqa: E402

ROT_IDX = [i for _, i in ROT_JOINTS]
C_FLIP = np.array(MHR_TO_CAM, np.float32)  # R_cam = C @ R_mhr @ C, C = diag(1,-1,-1)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def iter_frames(inputs, out_dir, video_stride, skip=(), keep=None):
    for root in inputs:
        files = [root] if os.path.isfile(root) else sorted(
            os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT:
                # forward slashes so labels made on Windows load anywhere
                rel = "images/" + os.path.relpath(f, os.path.dirname(root.rstrip("/\\"))).replace(os.sep, "/")
                if rel in skip or (keep and not keep(rel)):
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
                    rel = f"images/video/{stem}/{i:07d}.jpg"
                    if i % video_stride == 0 and rel not in skip and (not keep or keep(rel)):
                        dst = os.path.join(out_dir, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        yield rel, img
                    i += 1


def prefetch(it, n=8):
    """Run the generator (image decoding) on a thread so the GPU does not wait on cv2.imread."""
    q, end = queue.Queue(n), object()

    def run():
        try:
            for x in it:
                q.put(x)
        finally:
            q.put(end)
    threading.Thread(target=run, daemon=True).start()
    while (x := q.get()) is not end:
        yield x


def setup_teacher(hf_repo_id, fov_name="", detector_name="", device="cuda"):
    """notebook.utils.setup_sam_3d_body without its drawing imports; hf_repo_id may be a local dir."""
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body, load_sam_3d_body_hf
    if os.path.isdir(hf_repo_id):  # e.g. `hf download ... --local-dir checkpoints/sam-3d-body-dinov3`
        model, cfg = load_sam_3d_body(os.path.join(hf_repo_id, "model.ckpt"), device=device,
                                      mhr_path=os.path.join(hf_repo_id, "assets", "mhr_model.pt"))
    else:
        model, cfg = load_sam_3d_body_hf(hf_repo_id)
    det = fov = None
    if detector_name:
        from tools.build_detector import HumanDetector
        det = HumanDetector(name=detector_name, device=device)
    if fov_name:
        from tools.build_fov_estimator import FOVEstimator
        fov = FOVEstimator(name=fov_name, device=device)
    return SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=det,
                              human_segmentor=None, fov_estimator=fov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-repo", default="facebook/sam-3d-body-dinov3",
                    help="HF repo id, or a local dir holding model.ckpt + assets/mhr_model.pt")
    ap.add_argument("--video-stride", type=int, default=5)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--min-box", type=float, default=96, help="skip people smaller than this (px)")
    ap.add_argument("--no-fov", action="store_true", help="skip MoGe FOV estimation (faster, default FOV)")
    ap.add_argument("--hands", action="store_true", help="run the hand decoder too (slower)")
    ap.add_argument("--intrinsics", help="json {image rel path: 3x3 K} from fisheye_to_views.py; "
                    "defaults to <input>/intrinsics.json when present")
    ap.add_argument("--coco-ann", help="COCO person_keypoints_*.json: use its person boxes and skip the "
                    "teacher's ViTDet detector (faster, ~2-3 GB less GPU memory)")
    ap.add_argument("--min-box-h", type=float, default=200, help="with --coco-ann, min person box height (px)")
    ap.add_argument("--no-detector", action="store_true", help="don't load the detector; images without "
                    "COCO boxes use the whole frame as one person (fine for single-person footage)")
    ap.add_argument("--half", action="store_true", help="no-op, kept for old commands: the released teacher "
                    "configs already run the backbone in bf16 (~3.9 GB peak with --coco-ann --no-fov), and "
                    "global bf16 autocast crashes the MHR head (addmm_sparse_cuda has no bf16 kernel)")
    ap.add_argument("--gray", action="store_true", help="feed the teacher grayscale (3x replicated), "
                    "to check/match how it behaves on mono OV9281 footage")
    ap.add_argument("--num-workers", type=int, default=1, help="labeler processes sharing the GPU; start one "
                    "per --worker 0..N-1, each takes every Nth image and writes <out>/labels/w<worker>/")
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()

    try:
        from notebook.utils import setup_sam_3d_body  # from the sam-3d-body repo
    except ImportError as e:  # it needs pyrender (EGL) + detectron2 just for drawing, e.g. on Windows
        print(f"notebook.utils unavailable ({e}); building the teacher from sam_3d_body", file=sys.stderr)
        setup_sam_3d_body = setup_teacher
    if os.path.isdir(a.hf_repo):
        setup_sam_3d_body = setup_teacher
    import json
    import torch
    coco_boxes = None
    if a.coco_ann:
        ann = json.load(open(a.coco_ann))
        names = {im["id"]: im["file_name"] for im in ann["images"]}
        coco_boxes = {}
        for an in ann["annotations"]:
            x, y, w, h = an["bbox"]
            if an["category_id"] == 1 and not an["iscrowd"] and h >= a.min_box_h:
                coco_boxes.setdefault(names[an["image_id"]], []).append([x, y, x + w, y + h])
    use_det = not (a.coco_ann or a.no_detector)
    est = setup_sam_3d_body(hf_repo_id=a.hf_repo, fov_name="" if a.no_fov else "moge2",
                            detector_name="vitdet" if use_det else "")

    # keyed by the last 3 path parts (video/view/frame.jpg) so any input root layout matches
    tail = lambda p: "/".join(p.replace(os.sep, "/").split("/")[-3:])  # noqa: E731
    intr = {}
    cands = [a.intrinsics] if a.intrinsics else \
        [os.path.join(d, "intrinsics.json") for r in a.inputs for d in (r, os.path.dirname(r.rstrip("/")))]
    for p in cands:
        if p and os.path.exists(p):
            intr.update({tail(k): v for k, v in json.load(open(p)).items()})
    lab_dir = os.path.join(a.out, "labels", f"w{a.worker}") if a.num_workers > 1 else os.path.join(a.out, "labels")
    os.makedirs(lab_dir, exist_ok=True)
    # resume: skip images already in existing shards (Colab sessions get cut off)
    done, shard = set(), 0
    for f in sorted(os.listdir(lab_dir)):
        if f.startswith("shard_") and f.endswith(".npz"):
            done.update(str(x) for x in np.load(os.path.join(lab_dir, f))["image"])
            shard = max(shard, int(f[6:11]) + 1)
    done_log = os.path.join(lab_dir, "done.txt")  # also images with nobody in them
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
        path = os.path.join(lab_dir, f"shard_{shard:05d}.npz")
        np.savez(path, **{k: np.stack(v) for k, v in buf.items()})
        print(f"wrote {path} ({len(buf['image'])} people)", file=sys.stderr)
        buf, shard = {}, shard + 1
        with open(done_log, "a") as fh:
            fh.write("\n".join(pending) + "\n")
        pending.clear()

    def keep(rel):  # this worker's share; with COCO boxes, don't even decode images without a person
        if a.num_workers > 1 and zlib.crc32(rel.encode()) % a.num_workers != a.worker:
            return False
        return coco_boxes is None or a.no_detector or os.path.basename(rel) in coco_boxes

    for rel, bgr in prefetch(iter_frames(a.inputs, a.out, a.video_stride, skip=done, keep=keep)):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if a.gray:
            rgb = np.repeat(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
        Kmat = intr.get(tail(rel))
        cam_int = torch.tensor(Kmat, dtype=torch.float32)[None] if Kmat is not None else None
        boxes = None
        if coco_boxes is not None:
            b = coco_boxes.get(os.path.basename(rel))
            if b is None and not a.no_detector:
                pending.append(rel)
                continue
            boxes = np.array(b, np.float32) if b is not None else None
        if boxes is None and not use_det:
            boxes = np.array([[0, 0, bgr.shape[1], bgr.shape[0]]], np.float32)
        outs = est.process_one_image(rgb, bboxes=boxes, cam_int=cam_int,
                                     inference_type="full" if a.hands else "body")
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
            if "pred_global_rots" in o:  # trunk / arm segment rotations, conjugated into OpenCV camera axes
                Rg = np.asarray(o["pred_global_rots"], np.float32)[ROT_IDX]
                rec["rot"] = (C_FLIP[:, None] * Rg * C_FLIP[None, :]).astype(np.float32)
            for k, v in rec.items():
                buf.setdefault(k, []).append(v)
        if buf and len(buf["image"]) >= a.shard_size:
            flush()
        if n_img % 500 == 0:
            print(f"{n_img} images", file=sys.stderr)
    flush()


if __name__ == "__main__":
    main()
