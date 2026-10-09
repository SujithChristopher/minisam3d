"""Offline, resumable, start-to-finish NOARK student pipeline in one command.

For every subject (leave-one-subject-out) and then once on all subjects:
  train (rotation student, from a keypoint checkpoint) -> export ONNX -> run on the held-out subject's
  game + protocol takes -> score against mocap with NOARK's 21_sam3d_score.py (unchanged) -> summary.
The teacher (SAM 3D Body) is scored on the same takes for reference.

Needs no network: labels, images, checkpoints and both environments must already be on disk. Each venv's
python is called directly (no `uv run`, which may try to resolve packages). Re-running skips finished
steps and resumes an interrupted training run from its last.pt. Keeps Windows awake while it runs.

  E:/CMC/pyprojects/sam-3d-body/.venv/Scripts/python scripts/run_noark_loso.py
  ... --folds kabil ram      only these held-out subjects (default: all, then the all-subject model)
  ... --quick                1 epoch / 20 steps per fold, for checking the pipeline end to end

Writes runs/<prefix>_<fold>/ (checkpoints, ONNX, logs, noark_eval/) and runs/<prefix>_summary.md.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROW = re.compile(r"all in-window\s+(flexion|lateral|axial|magnitude)\s+([\d.]+)/\s*([\d.]+) r([+-][\d.]+) n\s*(\d+)"
                 r"(?:\s+([\d.]+)/\s*([\d.]+) r([+-][\d.]+) n\s*(\d+))?")


def keep_awake():
    if os.name == "nt":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while this process runs
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def log(msg, fh):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run(cmd, logfile, cwd=ROOT, env=None):
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg", **(env or {})))
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(map(str, cmd))} -- see {logfile}")


def frames_dir(session, subject, game):
    if not game:
        return os.path.join(session, f"{subject}_woc", "sam3d")
    d = glob.glob(os.path.join(session, "noark_game", "data", subject, "Session-*", "Video", "rec_*", "sam3d"))
    if len(d) != 1:
        raise SystemExit(f"{subject}: expected one game sam3d folder, found {d}")
    return d[0]


def trained_epochs(last):
    if not os.path.exists(last):
        return 0
    return int(torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1)) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=r"E:\NOARK_CV\msam_labels\noark", help="import_sam3d_npz.py output")
    ap.add_argument("--coco", default=r"E:\NOARK_CV\msam_labels\coco2014_vith", help="COCO labels with rotations")
    ap.add_argument("--coco-images", default=r"E:\NOARK_CV\minisam3d\data")
    ap.add_argument("--init", default=os.path.join(ROOT, "runs", "v2_coco2014", "best.pt"))
    ap.add_argument("--session", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone\data\sep23_26")
    ap.add_argument("--noark", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone", help="repo with 21")
    ap.add_argument("--noark-python", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone\.venv\Scripts\python.exe")
    ap.add_argument("--export-python", default=os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
                    help="a python with torch + onnx (the training venv usually lacks onnx)")
    ap.add_argument("--folds", nargs="*", help="held-out subjects to run (default: all, then 'all')")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefix", default="v3rot_loso")
    ap.add_argument("--quick", action="store_true", help="1 epoch, 20 steps: pipeline check")
    a = ap.parse_args()
    keep_awake()
    py = sys.executable
    if a.quick:
        a.epochs, a.prefix = 1, a.prefix + "_quick"
    takes = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.labels, "*_*"))
                   if os.path.isdir(os.path.join(d, "labels")))
    subjects = sorted({t.rsplit("_", 1)[0] for t in takes})
    folds = a.folds or subjects + ["all"]
    for p in (a.coco, a.init, a.noark_python, a.export_python):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    master = open(os.path.join(ROOT, "runs", f"{a.prefix}.log"), "a", encoding="utf-8")
    log(f"subjects {subjects}, takes {takes}, folds {folds}, epochs {a.epochs}", master)

    results = {}
    for fold in folds:
        out = os.path.join(ROOT, "runs", f"{a.prefix}_{fold}")
        os.makedirs(out, exist_ok=True)
        flog = os.path.join(out, "pipeline.log")
        done_json = os.path.join(out, "scores.json")
        if os.path.exists(done_json):
            results[fold] = json.load(open(done_json))
            log(f"{fold}: already done, skipping", master)
            continue
        train_takes = [t for t in takes if fold == "all" or t.rsplit("_", 1)[0] != fold]
        # 1. train (resumes from last.pt after an interruption)
        last = os.path.join(out, "last.pt")
        if trained_epochs(last) < a.epochs:
            log(f"{fold}: training on {train_takes} ({trained_epochs(last)}/{a.epochs} epochs done)", master)
            cmd = [py, "-u", os.path.join(HERE, "train.py"), "--data", a.coco] + \
                  [os.path.join(a.labels, t) for t in train_takes] + \
                  ["--image-dirs", a.coco_images, "--gray", "--rot", "--amp", "--init", a.init,
                   "--epochs", a.epochs, "--bs", a.bs, "--workers", a.workers, "--out", out]
            if os.path.exists(last):
                cmd += ["--resume", last]
            if a.quick:
                cmd += ["--max-steps", 20]
            run(cmd, os.path.join(out, "train.log"))
        # 2. export
        onnx = os.path.join(out, "student.onnx")
        if not os.path.exists(onnx):
            run([a.export_python, os.path.join(HERE, "export_onnx.py"), "--ckpt", os.path.join(out, "best.pt"),
                 "--out", onnx], flog)
        log(f"{fold}: trained + exported -> {onnx}", master)
        if fold == "all":
            results[fold] = {}
            json.dump(results[fold], open(done_json, "w"))
            continue
        # 3. held-out inference + scoring (student and teacher) on both takes
        scores = {}
        for kind, game in (("game", True), ("woc", False)):
            fdir = frames_dir(a.session, fold, game)
            ev = os.path.join(out, "noark_eval", f"{fold}_{kind}")
            stu = os.path.join(ev, "student_out.npz")
            if not os.path.exists(stu):
                run([py, os.path.join(HERE, "infer_frames.py"), "--ckpt", os.path.join(out, "best.pt"),
                     "--frames", fdir, "--out", ev, "--device", "cuda" if torch.cuda.is_available() else "cpu"],
                    flog)
            subj = fold if game else f"{fold}_woc"
            extra = ["--game"] if game else []
            for who, npz in (("student", stu), ("teacher", os.path.join(fdir, "sam3d_out.npz"))):
                slog = os.path.join(ev, f"score_{who}.txt")
                if not os.path.exists(slog) or "all in-window" not in open(slog, encoding="utf-8").read():
                    if os.path.exists(slog):
                        os.remove(slog)
                    try:  # 21 can refuse (e.g. too few well-excited frames to fit S): note it, keep going
                        run([a.noark_python, os.path.join(HERE, "score_student_noark.py"), "--student", npz,
                             "--save-dir", os.path.join(ev, who), "--", "--subject", subj] + extra, slog,
                            cwd=a.noark)
                    except RuntimeError:
                        tail = open(slog, encoding="utf-8").read().strip().splitlines()[-1:]
                        log(f"{fold} {kind} {who}: scoring failed: {tail}", master)
                        continue
                for m in ROW.finditer(open(slog, encoding="utf-8").read()):
                    ax = m.group(1)
                    scores.setdefault(kind, {}).setdefault(ax, {})[who] = \
                        dict(med=float(m.group(2)), p95=float(m.group(3)), r=float(m.group(4)), n=int(m.group(5)))
                    if m.group(6) and who == "teacher":
                        scores[kind][ax]["icp"] = dict(med=float(m.group(6)), p95=float(m.group(7)),
                                                       r=float(m.group(8)), n=int(m.group(9)))
            mag = scores.get(kind, {}).get("magnitude", {})
            fmt = lambda w: f"{mag[w]['med']}/{mag[w]['p95']}" if w in mag else "failed"  # noqa: E731
            log(f"{fold} {kind}: magnitude student {fmt('student')} teacher {fmt('teacher')}", master)
        results[fold] = scores
        json.dump(scores, open(done_json, "w"), indent=1)

    # 4. summary table
    lines = [f"# NOARK leave-one-subject-out ({a.prefix}, {a.epochs} epochs)", "",
             "Trunk rotation vs mocap from 21_sam3d_score.py, all in-window frames: median / p95 deg (r).", ""]
    for kind in ("game", "woc"):
        lines += [f"## {kind}", "", "| held-out | axis | student | teacher | ICP |", "|---|---|---|---|---|"]
        for fold, sc in results.items():
            for ax, v in sc.get(kind, {}).items():
                cell = lambda d: f"{d['med']:.1f}/{d['p95']:.1f} ({d['r']:+.2f})" if d else "-"  # noqa: E731
                lines.append(f"| {fold} | {ax} | {cell(v.get('student'))} | {cell(v.get('teacher'))} | "
                             f"{cell(v.get('icp'))} |")
        lines.append("")
    summ = os.path.join(ROOT, "runs", f"{a.prefix}_summary.md")
    open(summ, "w", encoding="utf-8").write("\n".join(lines))
    log(f"ALL DONE -> {summ}", master)


if __name__ == "__main__":
    main()
