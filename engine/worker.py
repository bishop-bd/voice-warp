"""App-owned RVC worker: bounded protocol v2 over private pipes, no sockets."""
import base64
import json
import logging
import os
from pathlib import Path
import queue
import statistics
import sys
import threading
import time
import traceback

from voice_warp_backend import (
    PROTOCOL, RUNTIME_VERSION, ExplicitDeviceManager, OnnxAudit,
    discover_devices, exported_bundle_paths, select_device, torch_placement, validate_request,
    verify_acceleration,
    configure_torch_policy,
)

MAX_LINE = 300_000
protocol = None
protocol_lock = threading.Lock()


def receive(stream=None):
    stream = stream or sys.stdin.buffer
    line = stream.readline(MAX_LINE + 1)
    if not line:
        return None
    if len(line) > MAX_LINE or not line.endswith(b"\n"):
        raise ValueError("Engine request exceeds the size limit")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected an engine request object")
    return value


def send(value):
    with protocol_lock:
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()


class InputMonitor:
    """Watch the owning pipe even while imports or inference occupy the main thread.

    A container's PID 1 has no useful host parent PID. EOF on the private pipe is
    the ownership signal there. The protocol permits one outstanding request;
    reject excess queued work instead of blocking this reader and hiding EOF.
    """
    def __init__(self, stream, report_error=None):
        self.stream = stream
        self.requests = queue.Queue(maxsize=1)
        self.report_error = report_error
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            while True:
                request = receive(self.stream)
                if request is None:
                    # Bypass destructors which may block behind a wedged GPU or
                    # native thread. --rm/Job Object/process ownership is handled
                    # by the launcher. No audio can be sent after owner loss.
                    os._exit(0)
                try:
                    self.requests.put_nowait(request)
                except queue.Full:
                    raise ValueError("Only one AI request may be outstanding") from None
                if request.get("action") == "shutdown":
                    # Allow the main loop to acknowledge an explicit shutdown
                    # even when the client closes its write end immediately.
                    return
        except Exception as error:
            if self.report_error is not None:
                try:
                    self.report_error(error)
                except Exception:
                    pass
            os._exit(1)

    def next(self):
        return self.requests.get()


def watch_parent():
    # Windows is supervised by a Job Object. Unix workers must also exit if the
    # app disappears while importing libraries or running a blocked inference.
    if os.name != "posix":
        return
    parent = os.getppid()

    def monitor():
        while True:
            time.sleep(0.5)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=monitor, daemon=True).start()


def load_pipeline(config, torch, ort, selected, audit):
    import faiss
    from mods.log_control import VoiceChangaerLogger
    from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
    from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer
    from voice_changer.RVC.inferencer.OnnxRVCInferencerNono import OnnxRVCInferencerNono
    from voice_changer.RVC.pitchExtractor.RMVPEOnnxPitchExtractor import RMVPEOnnxPitchExtractor
    from voice_changer.RVC.pipeline.Pipeline import Pipeline
    from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
    from voice_warp_hubert import OnnxHubertEmbedder, TorchHubertEmbedder
    from voice_warp_rvc import RvcStream, TorchRvcInferencer, model_metadata

    # The runtime payload may be read-only. The launcher owns persistent logs;
    # upstream's initialize() would create vcclient.log in the working directory.
    upstream_logger = VoiceChangaerLogger.get_instance().getLogger()
    upstream_logger.handlers = [logging.StreamHandler(sys.stderr)]
    upstream_logger.setLevel(logging.INFO)
    upstream_logger.propagate = False
    manager = ExplicitDeviceManager(torch, ort, selected)
    manager.policy = configure_torch_policy(torch, selected)
    manager.verify_allocation()
    # This replaces the worker-local singleton, not framework global device state.
    DeviceManager._instance = manager
    model_path = str(Path(config["model_path"]).resolve())
    index_path = config.get("index_path")
    assets = Path(config["assets_dir"])
    slot, checkpoint = model_metadata(model_path, ort)
    slot.indexFile = str(Path(index_path).resolve()) if index_path else ""
    index = faiss.read_index(slot.indexFile) if slot.indexFile else None
    if index is not None and index.d != slot.embChannels:
        raise ValueError("The index does not match this model's feature dimensions")
    params = VoiceChangerParams(
        model_dir=str(Path(model_path).parent), content_vec_500="", content_vec_500_onnx="",
        content_vec_500_onnx_on=False, hubert_base=str(assets / "hubert_base.pt"),
        hubert_base_jp="", hubert_soft="", nsf_hifigan="", sample_mode="",
        crepe_onnx_full="", crepe_onnx_tiny="", rmvpe="",
        rmvpe_onnx=config.get("rmvpe_path") or str(assets / "rmvpe.onnx"), whisper_tiny="",
    )
    model = RvcStream(params, slot)
    gpu = selected["index"]
    model.settings.gpu = gpu
    # Construct the same pinned upstream Pipeline with explicit source paths.
    # Upstream createPipeline strips directories and assumes model_dir/slot/name,
    # which is unsuitable for native file pickers and separate index folders.
    with audit.session_factory():
        if slot.isONNX:
            inferencer = (OnnxRVCInferencer() if slot.f0 else OnnxRVCInferencerNono()).loadModel(model_path, gpu, slot.version)
        else:
            inferencer = TorchRvcInferencer(checkpoint, slot, manager.device)
        embedder = OnnxHubertEmbedder(config["hubert_path"], manager, ort, slot) if config.get("hubert_path") else TorchHubertEmbedder(params.hubert_base, manager.device)
        with manager.onnx_stage("pitch"):
            pitch = RMVPEOnnxPitchExtractor(params.rmvpe_onnx, gpu)
    model.pipeline = Pipeline(embedder, inferencer, pitch, index,
                              slot.samplingRate, manager.device, False)
    model.settings.tran = config["pitch"]
    model.settings.indexRatio = config["index_ratio"] if index is not None else 0
    model.settings.protect = slot.defaultProtect
    model.settings.f0Detector = "rmvpe_onnx"
    calls = {"hubert": 0, "synthesizer": 0}

    def count_stage(stage, function):
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            calls[stage] += 1
            return result
        return call

    embedder.extractFeatures = count_stage("hubert", embedder.extractFeatures)
    inferencer.infer = count_stage("synthesizer", inferencer.infer)
    return model, params, manager, calls


def main():
    config = receive()
    if config is None:
        return
    # A one-shot probe intentionally permits the caller to close stdin after its
    # request. A live/init session must keep stdin open until the final reply or
    # explicit shutdown; monitor before bundle hashing and potentially slow imports.
    requests = None
    if config.get("action") in ("init", "export"):
        requests = InputMonitor(sys.stdin.buffer, lambda error: send(
            {"protocol": PROTOCOL, "error": str(error), "code": type(error).__name__}))
    config = exported_bundle_paths(config)
    validate_request(config)
    # A user environment setting must not silently change a requested MPS stage
    # into CPU execution. Unsupported operations fail during actual warmup.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch
    import onnxruntime as ort

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if config["action"] == "export":
        from export_models import export_bundle
        result = export_bundle(config["model_path"], config["assets_dir"], config["cache_dir"],
                               include_hubert=config.get("include_hubert", True))
        send({"protocol": PROTOCOL, "exported": True, **result})
        # The cancellation reader deliberately remains blocked on stdin while
        # the client consumes this final reply. Python's normal finalization can
        # abort while that daemon holds the buffered-reader lock. The completed
        # bundle and flushed reply need no further teardown in this one-shot worker.
        os._exit(0)
    report = discover_devices(torch, ort)
    if config["action"] == "probe":
        send(report)
        return
    selected = select_device(report, config.get("backend", "cpu"),
                             config.get("device", "0"), Path(config["model_path"]).suffix.lower())
    block_size = config["block_size"]
    from voice_changer.VoiceChangerV2 import VoiceChangerV2

    audit = OnnxAudit(ort)
    try:
        started = time.monotonic()
        model, params, manager, calls = load_pipeline(config, torch, ort, selected, audit)
        converter = VoiceChangerV2(params)
        converter.setModel(model)

        def convert(pcm):
            result = converter.on_request(pcm)
            if result is None:
                raise RuntimeError("RVC inference failed; inspect the engine log")
            audio, _ = result
            if len(audio) != block_size or not np.isfinite(audio).all():
                raise RuntimeError("RVC returned an invalid audio block")
            return audio.astype("<i2").tobytes()

        # Silence skips inference upstream: exercise voiced input and verify that
        # model stages ran, not just that device enumeration succeeded.
        tone = (3000 * np.sin(np.arange(block_size) * 2 * np.pi * 180 / 48000)).astype("<i2")
        for _ in range(2):
            convert(tone)
        if not all(calls.values()):
            raise RuntimeError("AI warmup did not execute HuBERT and voice synthesis")
        pipeline = model.pipeline
        stages = {
            "hubert": audit.placement(pipeline.embedder.onnx_session) if hasattr(pipeline.embedder, "onnx_session") else torch_placement(pipeline.embedder.model, manager.device),
            "pitch": audit.placement(pipeline.pitchExtractor.onnx_session, model.slotInfo.f0),
            "synthesizer": audit.placement(pipeline.inferencer.model) if model.slotInfo.isONNX else torch_placement(pipeline.inferencer.model, manager.device),
            "index": "CPU" if pipeline.index is not None else "not used",
            "resample": "CPU", "overlap": "CPU",
        }
        gpu_verified = verify_acceleration(selected["backend"], stages)
        if manager.policy.get("reason"):
            report["warnings"].append(manager.policy["reason"] + "; convolutions remain on the GPU")
        if selected["backend"] == "migraphx" and model.slotInfo.f0:
            report["warnings"].append("MIGraphX uses CPU RMVPE pitch extraction to avoid its lengthy graph compilation; HuBERT and synthesis providers are verified separately")
        # Provider audit ends ONNX profiling above. Measure the same execution
        # mode as live streaming, without per-node profiling overhead.
        performance = {}
        if config.get("measure_performance", False):
            durations = []
            for _ in range(8):
                measured = time.monotonic()
                convert(tone)
                durations.append((time.monotonic() - measured) * 1000)
            performance["steady_state_ms"] = {
                "p50": statistics.median(durations), "max": max(durations), "blocks": len(durations),
            }
        model.audio_buffer = model.pitchf_buffer = model.feature_buffer = None
        model.prevVol = 0
        converter = VoiceChangerV2(params)
        converter.setModel(model)
        send({"protocol": PROTOCOL, "ready": True, "sample_rate": 48000,
              "runtime_version": RUNTIME_VERSION, "device": selected["name"],
              "device_id": selected["id"], "backend": selected["backend"],
              "gpu_verified": gpu_verified,
              "torch_policy": manager.policy,
              "precision": "float32; ONNX synthesis float16" if model.slotInfo.isONNX and pipeline.inferencer.isHalf else "float32", "stages": stages,
              "model_name": Path(config["model_path"]).stem,
              "warmup_ms": round((time.monotonic() - started) * 1000),
              "torch_version": torch.__version__, "onnx_version": ort.__version__,
              "warnings": report["warnings"], **performance})
    finally:
        audit.close()
    sequence = 0
    while True:
        request = requests.next()
        if request.get("action") == "shutdown":
            send({"protocol": PROTOCOL, "stopped": True})
            return
        sequence += 1
        if type(request.get("sequence")) is not int or request["sequence"] != sequence:
            raise ValueError("Unexpected audio sequence")
        pcm = base64.b64decode(request["audio"], validate=True)
        if len(pcm) != block_size * 2:
            raise ValueError("Unexpected audio block length")
        started = time.monotonic()
        output = convert(np.frombuffer(pcm, dtype="<i2"))
        send({"sequence": sequence, "audio": base64.b64encode(output).decode("ascii"),
              "processing_ms": (time.monotonic() - started) * 1000})


def entrypoint():
    global protocol
    # Reserve the protocol descriptor before any ML/native library can print.
    # Binary mode avoids Windows newline translation on the duplicated handle.
    import io
    protocol = io.TextIOWrapper(os.fdopen(os.dup(sys.stdout.fileno()), "wb"),
                                encoding="utf-8", newline="\n", line_buffering=True)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    watch_parent()
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        send({"protocol": PROTOCOL, "error": str(error) or type(error).__name__,
              "code": type(error).__name__})
        # Failed init/export can leave the cancellation reader blocked on stdin.
        # Finish after flushing the diagnostic instead of finalizing Python while
        # that daemon holds the buffered-reader lock.
        sys.stderr.flush()
        os._exit(1)
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(entrypoint())
