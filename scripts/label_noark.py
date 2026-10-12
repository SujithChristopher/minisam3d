"""Resumable teacher labelling of NOARK recordings: frames -> SAM 3D Body labels -> training sets. No training.

Per take: NOARK 19_sam3d_export.py (undistorted upright frames in the GPIO window) -> det_boxes.py --apply (our
person detector picks the teacher's box) -> NOARK 20_sam3d_infer.py (sam-3d-body venv) -> import_sam3d_npz.py
(label shard + image junction under --labels) -> pack_noark.py (crop pack under --pack). Each step is skipped
when its output exists, so an interrupted run picks up where it stopped. Keeps Windows awake while it runs.

  .venv/Scripts/python scripts/label_noark.py            # oct9 + oct10 (all takes) and sep23 _wc
  ... --only s3_lee_woc                                   # one take

Takes are <session>:<subject>[:game]; protocol takes run at stride 1, game takes at --game-stride.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOARK = r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone"
SAM3D = r"E:\CMC\pyprojects\sam-3d-body"
NB = os.path.join(NOARK, "trunkpose", "dual_notebooks")
OCT = os.path.join(NOARK, "data", "trunk_take_oct")
SEP23 = os.path.join(NOARK, "data", "sep23_26")

OCT_SUBJECTS = {"oct9_26": ["s3_gokul", "s3_lee", "s3_ram"],
                "oct10_26": ["s3_arjun", "s3_diwakar", "s3_jerome", "s3_js"]}
SEP23_WC = ["diwakar", "gokul", "js", "kabil", "ram"]


def take_list():
    """(name, session dir, subject arg, game, labels out, pack out)"""
    out = []
    for sess, subs in OCT_SUBJECTS.items():
        sd = os.path.join(OCT, sess)
        for s in subs:
            for kind in ("woc", "wc"):
                out.append((f"{s}_{kind}", sd, f"{s}_{kind}", False, "noark_oct", "noark_pack_oct"))
            out.append((f"{s}_game", sd, s, True, "noark_oct", "noark_pack_oct"))
    for s in SEP23_WC:
        out.append((f"{s}_wc", SEP23, f"{s}_wc", False, "noark", "noark_pack"))
    return out


def sam3d_dir(session, subject, game):
    sys.path.insert(0, NB)
    import importlib
    from pathlib import Path
    return str(importlib.import_module("15_subject_icp").resolve_recording(Path(session), subject, game)[0] / "sam3d")


def log(msg, fh):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run(cmd, logfile, cwd):
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg"))
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(map(str, cmd))} -- see {logfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="take names (default: all)")
    ap.add_argument("--game-stride", type=int, default=6)
    ap.add_argument("--labels-root", default=r"E:\NOARK_CV\msam_labels")
    ap.add_argument("--pack-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--logdir", default=os.path.join(ROOT, "runs", "label_noark"))
    a = ap.parse_args()
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    os.makedirs(a.logdir, exist_ok=True)
    noark_py = os.path.join(NOARK, ".venv", "Scripts", "python.exe")
    sam_py = os.path.join(SAM3D, ".venv", "Scripts", "python.exe")
    me_py = sys.executable
    failed = []
    with open(os.path.join(a.logdir, "label_noark.log"), "a", encoding="utf-8") as fh:
        for name, sess, subj, game, lab, pack in take_list():
            if a.only and name not in a.only:
                continue
            lf = os.path.join(a.logdir, f"{name}.log")
            game_arg = ["--game"] if game else []
            try:
                d = sam3d_dir(sess, subj, game)
                meta = os.path.join(d, "meta.npz")
                if not os.path.exists(meta):
                    log(f"{name}: export", fh)
                    stride = ["--stride", a.game_stride] if game else []
                    run([noark_py, os.path.join(NB, "19_sam3d_export.py"), "--session", sess, "--subject", subj,
                         *game_arg, *stride], lf, NOARK)
                if "boxes_seg" not in np.load(meta).files:
                    log(f"{name}: detector boxes", fh)
                    run([me_py, os.path.join(HERE, "det_boxes.py"), d, "--apply"], lf, ROOT)
                if not os.path.exists(os.path.join(d, "sam3d_out.npz")):
                    log(f"{name}: teacher ({len(np.load(meta)['idx'])} frames)", fh)
                    run([sam_py, os.path.join(NB, "20_sam3d_infer.py"), "--session", sess, "--subject", subj,
                         *game_arg], lf, SAM3D)
                out_lab = os.path.join(a.labels_root, lab, name, "labels", "shard_00000.npz")
                if not os.path.exists(out_lab):
                    log(f"{name}: labels -> {lab}", fh)
                    run([me_py, os.path.join(HERE, "import_sam3d_npz.py"), "--out", os.path.join(a.labels_root, lab),
                         "--take", f"{name}={d}"], lf, ROOT)
                out_pack = os.path.join(a.pack_root, pack, name, "camera.npz")
                if not os.path.exists(out_pack):
                    log(f"{name}: pack -> {pack}", fh)
                    run([me_py, os.path.join(HERE, "pack_noark.py"), "--session", sess,
                         "--out", os.path.join(a.pack_root, pack), "--take", name], lf, ROOT)
                log(f"{name}: done", fh)
            except Exception as e:  # keep going: one bad take should not stop the night's run
                failed.append(name)
                log(f"{name}: FAILED {e}", fh)
        log(f"finished; failed: {failed or 'none'}", fh)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
