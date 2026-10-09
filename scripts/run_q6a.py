"""Live upper-body 3D pose on the Dragon Q6A (or any machine for testing).

  # on the board, TFLite + QNN delegate on the Hexagon NPU
  python3 run_q6a.py --model minisam3d.tflite --camera 0
  # anywhere, ONNX on CPU (for checking the pipeline)
  python3 run_q6a.py --model student.onnx --camera 0
  # OV9281 160-degree fisheye: pass the calibration, crops become virtual pinhole views
  python3 run_q6a.py --model minisam3d.tflite --camera 0 --calib calib_left.json

Crop strategy: the crop for frame t comes from the keypoints of frame t-1
(no detector in the loop). When confidence drops, it falls back to a centered
crop of the whole frame until the person is reacquired.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from minisam3d.data import UPPER_FOR_CROP, crop_affine  # noqa: E402
from minisam3d.keypoints import joint_set  # noqa: E402
from minisam3d.postprocess import decode_np, uv_to_image  # noqa: E402


class Runner:
    def __init__(self, path, qnn_delegate):
        self.kind = os.path.splitext(path)[1]
        if self.kind == ".onnx":
            import onnxruntime as ort
            self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.C, self.H, self.W = self.s.get_inputs()[0].shape[1:]
        else:
            try:
                from ai_edge_litert.interpreter import Interpreter, load_delegate
            except ImportError:
                from tflite_runtime.interpreter import Interpreter, load_delegate
            dl = [load_delegate(qnn_delegate, {"backend_type": "htp"})] if qnn_delegate else []
            self.s = Interpreter(model_path=path, experimental_delegates=dl, num_threads=4)
            self.s.allocate_tensors()
            self.i = self.s.get_input_details()[0]
            shp = self.i["shape"]
            self.nchw = shp[1] == 3
            self.C, self.H, self.W = (shp[1], shp[2], shp[3]) if self.nchw else (shp[3], shp[1], shp[2])

    def __call__(self, img_u8):
        """img_u8: (H,W,C) uint8, RGB or single-channel."""
        x = img_u8.reshape(self.H, self.W, -1).astype(np.float32) / 255.0
        if self.kind == ".onnx":
            outs = self.s.run(None, {"image": x.transpose(2, 0, 1)[None]})
            return outs if len(outs) > 1 else outs[0]
        if self.nchw:
            x = x.transpose(2, 0, 1)
        x = x[None]
        if self.i["dtype"] != np.float32:  # quantized I/O
            sc, zp = self.i["quantization"]
            x = np.clip(np.round(x / sc + zp), np.iinfo(self.i["dtype"]).min,
                        np.iinfo(self.i["dtype"]).max).astype(self.i["dtype"])
        self.s.set_tensor(self.i["index"], x)
        self.s.invoke()
        ys = []
        for o in self.s.get_output_details():
            y = self.s.get_tensor(o["index"])
            if o["dtype"] != np.float32:
                sc, zp = o["quantization"]
                y = (y.astype(np.float32) - zp) * sc
            if y.ndim == 4 and y.shape[1] == self.H // 4:
                y = y.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ys.append(y)
        return ys if len(ys) > 1 else ys[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--camera", default="0")
    ap.add_argument("--qnn-delegate", default="libQnnTFLiteDelegate.so")
    ap.add_argument("--hands", action="store_true")
    ap.add_argument("--conf", type=float, default=2.0, help="min mean peak confidence to keep tracking")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--calib", help="fisheye calibration json from calibrate_fisheye.py")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    a = ap.parse_args()
    js = joint_set(a.hands)
    K = js["num"]
    crop_ids = [js["names"].index(n) for n in UPPER_FOR_CROP]
    net = Runner(a.model, a.qnn_delegate if a.model.endswith(".tflite") else None)
    pv = None
    if a.calib:
        from minisam3d.fisheye import PersonView, load_calib
        Kc, Dc, _ = load_calib(a.calib)
        pv = PersonView(Kc, Dc, net.W, net.H)
    cap = cv2.VideoCapture(int(a.camera) if a.camera.isdigit() else a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    prev, xyz_s, t_last, fps = None, None, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if net.C == 1 and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif net.C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB if frame.ndim == 2 else cv2.COLOR_BGR2RGB)
        Hf, Wf = frame.shape[:2]
        if pv is not None:
            crop = pv.render(frame)
        else:
            if prev is None:
                center, size = np.array([Wf / 2, Hf / 2]), float(max(Hf, Wf * net.H / net.W))
            else:
                lo, hi = prev[crop_ids].min(0), prev[crop_ids].max(0)
                center = (lo + hi) / 2
                size = max(hi[1] - lo[1], (hi[0] - lo[0]) * net.H / net.W) * 1.35
            M = crop_affine(center, max(size, 32.0), 0.0, net.W, net.H)
            crop = cv2.warpAffine(frame, M, (net.W, net.H), flags=cv2.INTER_LINEAR)
        t0 = time.time()
        out = net(crop)
        t_net = time.time() - t0
        uv, xyz, conf = decode_np(out, K)
        if pv is not None:
            kp, xyz = pv.to_camera(uv, xyz)  # fisheye pixels, xyz in the physical camera frame
        else:
            kp = uv_to_image(uv, cv2.invertAffineTransform(M), net.W, net.H)
        tracked = conf[crop_ids].mean() > a.conf
        prev = kp if tracked else None
        if pv is not None:
            pv.update(kp[crop_ids]) if tracked else pv.reset()
        xyz_s = xyz if xyz_s is None else 0.5 * xyz_s + 0.5 * xyz  # light smoothing
        now = time.time()
        inst = 1 / max(now - t_last, 1e-6)
        fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        t_last = now
        print(f"fps {fps:5.1f}  net {1000*t_net:5.1f} ms  lwrist_xyz {np.round(xyz_s[js['names'].index('left_wrist')], 3)}",
              end="\r", flush=True)
        if a.show:
            vis = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            for (i, j) in js["bones"]:
                cv2.line(vis, tuple(map(int, kp[i])), tuple(map(int, kp[j])), (0, 255, 0), 2)
            cv2.imshow("minisam3d", vis)
            if cv2.waitKey(1) == 27:
                break


if __name__ == "__main__":
    main()
