"""ONNX Runtime sessions on the Hexagon NPU (QNN EP) and on CPU. Ported verbatim from yolo-test.

See yolo-test/docs/hexagon-npu.md for why each setting is what it is -- several of the defaults fail silently.
"""
import onnxruntime as ort

_registered = False


def npu_session(model, cache_dir=None):
    """Session with the graph on the HTP. Raises if no NPU device is found."""
    import onnxruntime_qnn as oq

    global _registered
    if not _registered:
        # A second registration in one process raises.
        ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
        _registered = True
    devices = [d for d in ort.get_ep_devices() if d.ep_name == oq.EP_NAME]
    if not devices:
        raise SystemExit("QNN EP registered but no NPU device found")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    # ORT's CPU pool only runs the boundary quantize/dequantize nodes. Left at defaults its threads spin-wait
    # after every run and starve the CPU-side post-processing.
    so.intra_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")

    opts = {
        "backend_path": oq.get_qnn_htp_path(),
        "htp_graph_finalization_optimization_mode": "3",
    }
    if cache_dir:
        opts["enable_htp_context_cache"] = "1"
        opts["context_cache_path"] = cache_dir

    # Plugin EPs must be attached per-device. Passing providers=[...] instead silently falls back to CPU.
    so.add_provider_for_devices(devices, opts)
    return ort.InferenceSession(str(model), so)


def cpu_session(model, threads=4):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads  # the four A78 cores; A55s only slow it down
    return ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
