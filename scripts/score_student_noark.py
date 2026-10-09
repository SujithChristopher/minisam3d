"""Score student outputs with NOARK's 21_sam3d_score.py, unchanged, without touching the teacher's files.

Redirects 21's read of <rec>/sam3d/sam3d_out.npz to --student (infer_frames.py output) and its write of
sam3d_vs_mocap*.npz into the student's folder. Everything after `--` goes to 21 as-is.

  cd <NOARK_backbone>
  uv run python <minisam3d>/scripts/score_student_noark.py --student <dir>/student_out.npz -- --subject kabil --game
"""
import os
import runpy
import sys

import numpy as np

i = sys.argv.index("--")
own = sys.argv[:i]
student = os.path.abspath(own[own.index("--student") + 1])
# where 21's sam3d_vs_mocap*.npz goes (default: next to --student)
save_dir = os.path.abspath(own[own.index("--save-dir") + 1]) if "--save-dir" in own else os.path.dirname(student)
args21 = sys.argv[i + 1:]
_load, _savez, _savezc = np.load, np.savez, np.savez_compressed


def load(f, *a, **k):
    if str(f).replace("\\", "/").endswith("sam3d/sam3d_out.npz"):
        f = student
    return _load(f, *a, **k)


def redirect(fn):
    def save(f, *a, **k):
        if "sam3d_vs_mocap" in os.path.basename(str(f)):
            os.makedirs(save_dir, exist_ok=True)
            f = os.path.join(save_dir, os.path.basename(str(f)))
        return fn(f, *a, **k)
    return save


np.load, np.savez, np.savez_compressed = load, redirect(_savez), redirect(_savezc)

# --zero-frames K: zero both systems over the first K common frames instead of 06's N_NEUTRAL (15).
# Applied only inside 07.compare_mocap, so 06's anatomical axes (assemble) are unchanged.
if "--zero-frames" in own:
    import importlib
    K = int(own[own.index("--zero-frames") + 1])
    _imp = importlib.import_module

    def import_module(name, *a, **k):
        mod = _imp(name, *a, **k)
        if name == "07_icp_trunk" and not getattr(mod, "_zero_patched", False):
            m6, orig = _imp("06_trunk_axis"), mod.compare_mocap

            def compare_mocap(*ca, **ck):
                old, m6.N_NEUTRAL = m6.N_NEUTRAL, K
                try:
                    return orig(*ca, **ck)
                finally:
                    m6.N_NEUTRAL = old
            mod.compare_mocap, mod._zero_patched = compare_mocap, True
        return mod
    importlib.import_module = import_module
script = os.path.join(os.getcwd(), "trunkpose", "dual_notebooks", "21_sam3d_score.py")
sys.path.insert(0, os.path.dirname(script))
sys.argv = [script] + args21
runpy.run_path(script, run_name="__main__")
