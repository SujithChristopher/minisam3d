"""CPU smoke test: synthetic pseudo-labels -> 20 train steps -> ONNX export -> ORT parity."""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from minisam3d.model import Student, decode  # noqa: E402
from minisam3d.postprocess import decode_np  # noqa: E402

tmp = tempfile.mkdtemp()
os.makedirs(f"{tmp}/images"), os.makedirs(f"{tmp}/labels")
rng = np.random.default_rng(0)
N = 64
rec = {k: [] for k in ["image", "img_wh", "bbox", "kp2d", "kp3d", "cam_t", "focal"]}
for i in range(N):
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(f"{tmp}/images/{i}.jpg", img)
    kp3d = rng.normal(0, 0.3, (70, 3)).astype(np.float32)
    kp3d[:, 2] *= 0.3
    t = np.array([0, 0, 3.0], np.float32)
    f = 600.0
    p = kp3d + t
    kp2d = (p[:, :2] / p[:, 2:] * f + [320, 240]).astype(np.float32)
    rec["image"].append(np.array(f"images/{i}.jpg")); rec["img_wh"].append(np.array([640, 480]))
    rec["bbox"].append(np.r_[kp2d.min(0), kp2d.max(0)].astype(np.float32))
    rec["kp2d"].append(kp2d); rec["kp3d"].append(kp3d); rec["cam_t"].append(t); rec["focal"].append(np.float32(f))
np.savez(f"{tmp}/labels/shard_00000.npz", **{k: np.stack(v) for k, v in rec.items()})

py = sys.executable
subprocess.run([py, f"{HERE}/train.py", "--data", tmp, "--out", f"{tmp}/run", "--epochs", "1",
                "--bs", "8", "--workers", "0", "--max-steps", "20"], check=True)
subprocess.run([py, f"{HERE}/export_onnx.py", "--ckpt", f"{tmp}/run/last.pt", "--out", f"{tmp}/s.onnx"], check=True)

import onnxruntime as ort  # noqa: E402
ck = torch.load(f"{tmp}/run/last.pt", map_location="cpu")
m = Student(ck["K"]); m.load_state_dict(ck["model"]); m.eval()
x = torch.rand(1, 3, 256, 256)
ref = m(x).detach()
got = ort.InferenceSession(f"{tmp}/s.onnx").run(None, {"image": x.numpy()})[0]
print("onnx max abs diff", float(np.abs(ref.numpy() - got).max()))
uv_t, xyz_t, _ = decode(ref, ck["K"])
uv_n, xyz_n, _ = decode_np(got, ck["K"])
print("decode parity uv", float(np.abs(uv_t[0].numpy() - uv_n).max()), "xyz", float(np.abs(xyz_t[0].numpy() - xyz_n).max()))
print("SMOKE OK", tmp)

# ---- mono + fisheye path ----
subprocess.run([py, f"{HERE}/train.py", "--data", tmp, "--out", f"{tmp}/run_g", "--epochs", "1", "--gray",
                "--bs", "8", "--workers", "0", "--max-steps", "3"], check=True)
subprocess.run([py, f"{HERE}/train.py", "--data", tmp, "--out", f"{tmp}/run_g", "--epochs", "2", "--gray",
                "--bs", "8", "--workers", "0", "--max-steps", "3", "--resume", f"{tmp}/run_g/last.pt"], check=True)
subprocess.run([py, f"{HERE}/export_onnx.py", "--ckpt", f"{tmp}/run_g/best.pt", "--out", f"{tmp}/g.onnx"], check=True)
assert ort.InferenceSession(f"{tmp}/g.onnx").get_inputs()[0].shape == [1, 1, 256, 256]
import json  # noqa: E402
json.dump(dict(K=[[330., 0, 640], [0, 330, 400], [0, 0, 1]], D=[0.02, -0.01, 0.003, -0.0005], size=[1280, 800]),
          open(f"{tmp}/calib.json", "w"))
vw = cv2.VideoWriter(f"{tmp}/fish.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 800))
for _ in range(30):
    vw.write(rng.integers(0, 255, (800, 1280, 3), dtype=np.uint8))
vw.release()
r = subprocess.run([py, f"{HERE}/run_q6a.py", "--model", f"{tmp}/g.onnx", "--camera", f"{tmp}/fish.mp4",
                    "--calib", f"{tmp}/calib.json", "--conf", "0"], check=True, capture_output=True, text=True)
print("runner:", r.stdout.strip().split("\r")[-1].strip())
subprocess.run([py, f"{HERE}/fisheye_to_views.py", "--calib", f"{tmp}/calib.json", "--videos", f"{tmp}/fish.mp4",
                "--out", f"{tmp}/views", "--stride", "10", "--size", "128"], check=True)
n_views = len(json.load(open(f"{tmp}/views/intrinsics.json")))
assert n_views == 9, n_views

# ---- labeler with a fake teacher (checks intrinsics lookup + resume) ----
fake = os.path.join(tmp, "fake_teacher", "notebook")
os.makedirs(fake)
open(os.path.join(fake, "__init__.py"), "w").close()
open(os.path.join(fake, "utils.py"), "w").write('''
import numpy as np
class E:
    def process_one_image(self, img, cam_int=None, inference_type="body"):
        assert cam_int is not None and cam_int.shape == (1, 3, 3), "intrinsics not passed"
        return [dict(bbox=np.array([0, 0, 120, 120.]), pred_keypoints_2d=np.zeros((70, 2)),
                     pred_keypoints_3d=np.zeros((70, 3)), pred_cam_t=np.zeros(3), focal_length=np.float32(100))]
def setup_sam_3d_body(**kw):
    return E()
''')
env = dict(os.environ, PYTHONPATH=os.path.join(tmp, "fake_teacher"))
lab = [py, f"{HERE}/label_with_teacher.py", "--inputs", f"{tmp}/views/images", "--out", f"{tmp}/lab", "--shard-size", "4"]
subprocess.run(lab, check=True, env=env)
r = subprocess.run(lab, check=True, env=env, capture_output=True, text=True)
n = sum(len(np.load(f"{tmp}/lab/labels/{f}")["image"]) for f in os.listdir(f"{tmp}/lab/labels") if f.endswith(".npz"))
assert n == 9 and "resuming: 9" in r.stderr, (n, r.stderr)
print("labeler + resume OK")
print("SMOKE OK (mono/fisheye)")
