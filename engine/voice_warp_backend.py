"""Explicit device policy for the pinned RVC libraries.

This module deliberately imports no ML packages at module load time, so protocol
validation and device selection can be tested without a GPU or model weights.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import platform
import tempfile

PROTOCOL = 2
RUNTIME_VERSION = "2.0.0"
BACKENDS = ("auto", "cpu", "cuda", "rocm", "mps", "directml")
BLOCK_SIZES = (4096, 6144, 8192, 12288)


def validate_request(config):
    if not isinstance(config, dict) or config.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible AI worker protocol; repair the AI runtime")
    if config.get("action") not in ("probe", "init"):
        raise ValueError("Expected an init or probe request")
    if config.get("backend", "cpu") not in BACKENDS:
        raise ValueError("Unknown AI backend")
    if not isinstance(config.get("device", "0"), str):
        raise ValueError("AI device must be an identifier string")
    if config["action"] == "probe":
        return
    block = config.get("block_size")
    if type(block) is not int or block not in BLOCK_SIZES:
        raise ValueError("Unsupported audio block size")
    pitch = config.get("pitch")
    if type(pitch) is not int or not -24 <= pitch <= 24:
        raise ValueError("Pitch must be between -24 and 24 semitones")
    ratio = config.get("index_ratio")
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("Index strength must be finite and between 0 and 1")
    for key, suffixes in (("model_path", (".pth", ".onnx")), ("index_path", (".index",))):
        value = config.get(key)
        if key == "index_path" and not value:
            continue
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{key} must be an absolute file path")
        path = Path(value)
        if path.suffix.lower() not in suffixes or not path.is_file():
            raise ValueError(f"Choose an existing {' or '.join(suffixes)} file")
    assets = config.get("assets_dir")
    if not isinstance(assets, str) or not Path(assets).is_absolute():
        raise ValueError("assets_dir must be an absolute directory")
    for name in ("hubert_base.pt", "rmvpe.onnx"):
        if not (Path(assets) / name).is_file():
            raise ValueError(f"Missing shared AI asset {name}; repair the AI runtime")


def directml_adapters():
    """Enumerate DXGI adapters with persistent-within-boot LUIDs, not names only."""
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16),
                    ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]

    class Desc(ctypes.Structure):
        _fields_ = [("name", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint32),
                    ("device", ctypes.c_uint32), ("subsystem", ctypes.c_uint32),
                    ("revision", ctypes.c_uint32), ("video_memory", ctypes.c_size_t),
                    ("system_memory", ctypes.c_size_t), ("shared_memory", ctypes.c_size_t),
                    ("luid", LUID), ("flags", ctypes.c_uint32)]

    def call(ptr, slot, restype, argtypes, *args):
        table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[slot])
        return fn(ptr, *args)

    iid = GUID(0x770aae78, 0xf26f, 0x4dba,
               (ctypes.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87))
    factory = ctypes.c_void_p()
    create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
        raise RuntimeError("Cannot enumerate Windows graphics adapters")
    result = []
    try:
        for ordinal in range(64):
            adapter = ctypes.c_void_p()
            status = call(factory, 12, ctypes.c_long,
                          [wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)],
                          ordinal, ctypes.byref(adapter))
            if status < 0:
                break
            try:
                desc = Desc()
                if call(adapter, 10, ctypes.c_long, [ctypes.POINTER(Desc)], ctypes.byref(desc)) >= 0 and not desc.flags & 2:
                    luid = f"{desc.luid.high & 0xffffffff:08x}{desc.luid.low:08x}"
                    result.append({"backend": "directml", "id": f"luid:{luid}",
                                   "name": desc.name, "index": ordinal,
                                   "id_scope": "boot", "memory_bytes": desc.video_memory})
            finally:
                call(adapter, 2, ctypes.c_ulong, [])
    finally:
        call(factory, 2, ctypes.c_ulong, [])
    return result


def discover_devices(torch, ort):
    devices = [{"backend": "cpu", "id": "cpu", "name": "CPU", "index": -1}]
    warnings = []
    try:
        if torch.cuda.is_available():
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                uuid = getattr(props, "uuid", None)
                name = torch.cuda.get_device_name(index)
                devices.append({"backend": backend, "id": f"uuid:{uuid}" if uuid else f"name:{name}",
                                "name": name, "index": index,
                                "memory_bytes": props.total_memory})
    except Exception as error:
        warnings.append(f"CUDA/ROCm device probe failed: {error}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append({"backend": "mps", "id": "mps", "name": "Metal GPU (MPS)", "index": 0})
    providers = ort.get_available_providers()
    if platform.system() == "Windows" and "DmlExecutionProvider" in providers:
        try:
            devices.extend(directml_adapters())
        except Exception as error:
            warnings.append(f"DirectML device probe failed: {error}")
    # This is discovery, not a claim that a user's model has run on these devices.
    return {"protocol": PROTOCOL, "runtime_version": RUNTIME_VERSION,
            "devices": devices, "warnings": warnings, "providers": providers,
            "status": "detected", "model_verified": False,
            "torch_version": torch.__version__, "onnx_version": ort.__version__}


def select_device(report, backend, identity="0", model_suffix=None):
    devices = report["devices"]
    if backend == "auto":
        devices = sorted(devices, key=lambda d: (d["backend"] == "cpu", d["backend"] == "directml"))
        # The DirectML lane requires an ONNX voice model, not arbitrary checkpoints.
        devices = [d for d in devices if d["backend"] != "directml" or model_suffix == ".onnx"]
        if identity not in ("", "0", "auto"):
            devices = [d for d in devices if d["id"] == identity]
            if len(devices) > 1:
                raise ValueError("Saved AI device identity is ambiguous; select an explicit adapter")
        if not devices:
            raise ValueError("The saved AI device is no longer available; select a device again")
        return devices[0]
    if backend == "directml" and model_suffix not in (None, ".onnx"):
        raise ValueError("DirectML currently requires an RVC ONNX model; HuBERT runs on CPU")
    candidates = [d for d in devices if d["backend"] == backend]
    if backend in ("cpu", "mps") and identity in ("", "0", "auto", backend):
        matches = candidates
    elif identity.isdecimal():
        matches = [d for d in candidates if d["index"] == int(identity)]
    else:
        matches = [d for d in candidates if d["id"] == identity]
    if len(matches) != 1:
        raise ValueError(f"Selected {backend} device is unavailable or ambiguous; probe devices and select again")
    return matches[0]


class ExplicitDeviceManager:
    """Adapter for upstream's singleton, whose CPU mode otherwise selects MPS."""
    def __init__(self, torch, ort, selected):
        self.torch = torch
        self.selected = selected
        backend = selected["backend"]
        self.device = torch.device("cuda", selected["index"]) if backend in ("cuda", "rocm") else torch.device("mps" if backend == "mps" else "cpu")
        available = ort.get_available_providers()
        self.provider = "CPUExecutionProvider"
        if backend == "cuda" and "CUDAExecutionProvider" in available:
            self.provider = "CUDAExecutionProvider"
        elif backend == "rocm" and "MIGraphXExecutionProvider" in available:
            self.provider = "MIGraphXExecutionProvider"
        elif backend == "directml":
            if "DmlExecutionProvider" not in available:
                raise RuntimeError("This runtime does not include DirectML")
            self.provider = "DmlExecutionProvider"

    def getDevice(self, _index):
        return self.device

    def getOnnxExecutionProvider(self, _index):
        options = {} if self.provider == "CPUExecutionProvider" else {"device_id": str(self.selected["index"])}
        return [self.provider], [options]

    def halfPrecisionAvailable(self, _index):
        return False

    def setForceTensor(self, _force):
        pass  # This adapter always uses float32, irrespective of upstream heuristics.

    def getDeviceMemory(self, _index):
        return self.selected.get("memory_bytes", 0)

    def verify_allocation(self):
        # `.item()` synchronizes the operation, exposing unsupported GPU kernels.
        value = self.torch.ones(8, device=self.device, dtype=self.torch.float32)
        if (value * value).sum().item() != 8:
            raise RuntimeError("GPU allocation/arithmetic probe failed")


class OnnxAudit:
    """Configure upstream sessions and verify actual GPU kernels after warmup."""
    def __init__(self, ort):
        self.ort = ort
        self.records = {}
        self.finished = set()
        self.temporary = tempfile.TemporaryDirectory(prefix="voice-warp-onnx-")

    @contextmanager
    def session_factory(self):
        original = self.ort.InferenceSession

        def create(path, sess_options=None, providers=None, provider_options=None, **kwargs):
            providers = providers or ["CPUExecutionProvider"]
            requested = providers[0]
            options = sess_options or self.ort.SessionOptions()
            options.intra_op_num_threads = 6
            options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
            if requested == "DmlExecutionProvider":
                options.enable_mem_pattern = False
            if requested != "CPUExecutionProvider":
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(self.temporary.name) / f"stage-{len(self.records)}")
            session = original(path, sess_options=options, providers=providers,
                               provider_options=provider_options, **kwargs)
            session.disable_fallback()
            self.records[id(session)] = (session, requested)
            if requested not in session.get_providers():
                raise RuntimeError(f"{requested} failed to initialize; refusing silent CPU fallback")
            return session

        self.ort.InferenceSession = create
        try:
            yield
        finally:
            self.ort.InferenceSession = original

    def placement(self, session, used=True):
        _, requested = self.records[id(session)]
        if not used:
            if requested != "CPUExecutionProvider":
                session.end_profiling()
                self.finished.add(id(session))
            return "not used (model has no pitch input)"
        if requested == "CPUExecutionProvider":
            return "CPU"
        path = session.end_profiling()
        self.finished.add(id(session))
        events = json.loads(Path(path).read_text(encoding="utf-8"))
        providers = {event.get("args", {}).get("provider") for event in events}
        providers.discard(None)
        if requested not in providers:
            raise RuntimeError(f"Warmup executed no kernels on {requested}; GPU use was not verified")
        return requested + (" + CPU nodes" if "CPUExecutionProvider" in providers else "")

    def close(self):
        for key, (session, requested) in self.records.items():
            if requested != "CPUExecutionProvider" and key not in self.finished:
                session.end_profiling()
        self.temporary.cleanup()


def torch_placement(module, expected):
    devices = {parameter.device for parameter in module.parameters()}
    if not devices or any(device.type != expected.type or
                          (expected.index is not None and device.index != expected.index)
                          for device in devices):
        raise RuntimeError(f"Model placement mismatch: expected {expected}, found {sorted(devices)}")
    return ", ".join(sorted(str(device) for device in devices))
