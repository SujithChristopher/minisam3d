"""Validate and benchmark a quantized student on the Hexagon NPU (port of yolo-test/trunk_seg/bench.py).

  python scripts/npu_bench.py --fp32 runs/v3rot_loso_all/student.onnx \
                              --model runs/v3rot_loso_all/models/w8a16_<job>/model.onnx

Compares fp32 on CPU against the QDQ graph on CPU (a simulation only -- it can agree with fp32 where the HTP
produces garbage) and the QDQ graph on the HTP: raw output ranges, then decoded keypoints (px in the 256 crop),
global xyz (mm) and segment rotations (deg). Then times network and decode over --runs frames, logging die temp.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minisam3d.npu_session import cpu_session, npu_session  # noqa: E402
from minisam3d.postprocess import decode_np, rot6d_to_mat_np, split_outputs  # noqa: E402


def die_temp():
    """Hottest CPU zone. Not the max over all zones: ufs-thermal (the onboard UFS chip) idles near 78 C."""
    ts = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(f"{z}/type").read().startswith(("cpu", "cpuss")):
                ts.append(int(open(f"{z}/temp").read()) / 1000)
        except (OSError, ValueError):
            pass
    return max(ts) if ts else float("nan")


def decode_all(out, K):
    uv, xyz, conf = decode_np(out, K)
    _, _, r6 = split_outputs(out, K)
    return uv, xyz, conf, (None if r6 is None else rot6d_to_mat_np(r6))


def rot_err_deg(Ra, Rb):
    c = (np.einsum("rij,rij->r", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="quantized QDQ ONNX")
    ap.add_argument("--fp32", required=True, help="fp32 ONNX it was quantized from")
    ap.add_argument("--images", default="calib/calib.npy", help="(N,1,H,W) crops from make_calib.py")
    ap.add_argument("--cache", default=None, help="HTP context cache dir (skips the ~15 s compile next time)")
    ap.add_argument("--runs", type=int, default=300)
    a = ap.parse_args()

    xs = list(np.load(a.images).astype(np.float32)[:, None])
    H, W = xs[0].shape[-2:]

    t0 = time.perf_counter()
    npu = npu_session(a.model, a.cache)
    init = time.perf_counter() - t0
    print(f"session init {init:.1f} s, providers {npu.get_providers()}")
    if "QNNExecutionProvider" not in npu.get_providers():
        sys.exit("not running on the NPU")
    if init < 2 and not a.cache:
        print("WARNING: init < 2 s without a cache -- check the graph actually compiled for the HTP")
    ref = cpu_session(a.fp32)
    sim = cpu_session(a.model)
    names = [o.name for o in ref.get_outputs()]
    K = ref.get_outputs()[0].shape[1]

    # --- accuracy: fp32 CPU vs QDQ-on-CPU (simulation) vs QDQ-on-HTP ----------
    lo = {k: [np.inf] * 3 for k in names}
    hi = {k: [-np.inf] * 3 for k in names}
    err = {"sim": {"uv": [], "xyz": [], "rot": [], "conf": []}, "npu": {"uv": [], "xyz": [], "rot": [], "conf": []}}
    for x in xs:
        outs = [s.run(None, {"image": x}) for s in (ref, sim, npu)]
        for col, o in enumerate(outs):
            for k, t in zip(names, o):
                lo[k][col] = min(lo[k][col], float(t.min()))
                hi[k][col] = max(hi[k][col], float(t.max()))
        r = decode_all(outs[0], K)
        for key, o in (("sim", outs[1]), ("npu", outs[2])):
            d = decode_all(o, K)
            e = err[key]
            e["uv"].append(np.linalg.norm((d[0] - r[0]) * [W, H], axis=1))
            e["xyz"].append(np.linalg.norm(d[1] - r[1], axis=1) * 1000)
            e["conf"].append(d[2] - r[2])
            if r[3] is not None:
                e["rot"].append(rot_err_deg(d[3], r[3]))

    print("\nraw output ranges  (fp32 CPU | QDQ on CPU | QDQ on HTP)")
    for k in names:
        print(f"  {k:<5}" + "".join(f"  [{lo[k][c]:9.3f}, {hi[k][c]:9.3f}]" for c in range(3)))

    print(f"\nvs fp32 over {len(xs)} crops ({K} joints)       mean    p95     max")
    for key, e in err.items():
        for m, unit in (("uv", "px"), ("xyz", "mm"), ("rot", "deg"), ("conf", "")):
            if not e[m]:
                continue
            v = np.abs(np.concatenate(e[m]))
            print(f"  {key} {m:<4} {unit:<3}                  {v.mean():7.2f} {np.percentile(v, 95):7.2f} {v.max():7.2f}")

    # --- speed ---------------------------------------------------------------
    for i in range(20):
        npu.run(None, {"image": xs[i % len(xs)]})
    t_net = t_dec = 0.0
    temp0 = die_temp()
    for i in range(a.runs):
        x = xs[i % len(xs)]
        b = time.perf_counter(); o = npu.run(None, {"image": x})
        c = time.perf_counter(); decode_all(o, K)
        d = time.perf_counter()
        t_net += c - b; t_dec += d - c
    t_net, t_dec = t_net / a.runs * 1000, t_dec / a.runs * 1000
    print(f"\n{a.runs} frames on the NPU, die {temp0:.0f} -> {die_temp():.0f} C")
    print(f"  net    {t_net:6.2f} ms\n  decode {t_dec:6.2f} ms\n  total  {t_net + t_dec:6.2f} ms "
          f"({1000 / (t_net + t_dec):.1f} fps, crop preprocessing excluded)")

    cpu = cpu_session(a.fp32)
    for i in range(5):
        cpu.run(None, {"image": xs[i]})
    n = min(a.runs, 50)
    b = time.perf_counter()
    for i in range(n):
        cpu.run(None, {"image": xs[i % len(xs)]})
    print(f"  (fp32 CPU, 4 threads: {(time.perf_counter() - b) / n * 1000:.2f} ms net)")


if __name__ == "__main__":
    main()
