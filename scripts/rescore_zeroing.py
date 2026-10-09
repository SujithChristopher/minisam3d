"""Re-score existing LOSO outputs (student + teacher) with longer zeroing windows. Offline, CPU only.

  python scripts/rescore_zeroing.py --zero-frames 15 100 300
Writes runs/<prefix>_zeroing.md.
"""
import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_noark_loso import ROW, frames_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v3rot_loso")
    ap.add_argument("--zero-frames", type=int, nargs="+", default=[15, 100, 300])
    ap.add_argument("--session", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone\data\sep23_26")
    ap.add_argument("--noark", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone")
    ap.add_argument("--noark-python", default=r"E:\CMC\pyprojects\programs_rpi\NOARK_backbone\.venv\Scripts\python.exe")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    subjects = sorted(d.split(f"{a.prefix}_", 1)[1] for d in os.listdir(os.path.join(ROOT, "runs"))
                      if d.startswith(f"{a.prefix}_") and os.path.isdir(os.path.join(ROOT, "runs", d))
                      and os.path.isdir(os.path.join(ROOT, "runs", d, "noark_eval")))
    jobs = []
    for s in subjects:
        for kind, game in (("game", True), ("woc", False)):
            ev = os.path.join(ROOT, "runs", f"{a.prefix}_{s}", "noark_eval", f"{s}_{kind}")
            for who, npz in (("student", os.path.join(ev, "student_out.npz")),
                             ("teacher", os.path.join(frames_dir(a.session, s, game), "sam3d_out.npz"))):
                for k in a.zero_frames:
                    jobs.append((s, kind, game, who, npz, k, ev))

    def one(j):
        s, kind, game, who, npz, k, ev = j
        out = os.path.join(ev, f"zero{k}", who)
        os.makedirs(out, exist_ok=True)
        cmd = [a.noark_python, os.path.join(HERE, "score_student_noark.py"), "--student", npz, "--save-dir", out,
               "--zero-frames", str(k), "--", "--subject", s if game else f"{s}_woc"] + (["--game"] if game else [])
        r = subprocess.run(cmd, cwd=a.noark, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg"))
        open(os.path.join(out, "score.txt"), "w", encoding="utf-8").write(r.stdout + r.stderr)
        res = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4))) for m in ROW.finditer(r.stdout)}
        print(f"{s} {kind} {who} zero{k}: magnitude {res.get('magnitude')}", flush=True)
        return j, res

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(one, jobs))
    tab = {(s, kind, who, k): res for (s, kind, _, who, _, k, _), res in results}
    cell = lambda v: f"{v[0]:.1f}/{v[1]:.1f} ({v[2]:+.2f})" if v else "failed"  # noqa: E731
    lines = [f"# Zeroing window vs trunk error ({a.prefix})", "",
             "Same outputs as the LOSO summary, re-scored by 21 with both systems zeroed over the first K common "
             "frames (06 default K=15). median/p95 deg (r).", ""]
    for kind in ("game", "woc"):
        for ax in ("magnitude", "axial"):
            lines += [f"## {kind}: {ax}", "",
                      "| held-out | " + " | ".join(f"student K={k} | teacher K={k}" for k in a.zero_frames) + " |",
                      "|---|" + "---|---|" * len(a.zero_frames)]
            for s in subjects:
                lines.append(f"| {s} | " + " | ".join(
                    f"{cell(tab[(s, kind, 'student', k)].get(ax))} | {cell(tab[(s, kind, 'teacher', k)].get(ax))}"
                    for k in a.zero_frames) + " |")
            lines.append("")
    p = os.path.join(ROOT, "runs", f"{a.prefix}_zeroing.md")
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    print("->", p)


if __name__ == "__main__":
    main()
