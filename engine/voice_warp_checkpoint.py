"""Dependency-free metadata normalization shared by inference and ONNX export."""


def voice_metadata(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("The selected checkpoint is not an RVC voice model")
    config = checkpoint.get("config")
    version = checkpoint.get("version", "v1")
    if version is None:
        version = "v1"
    if not isinstance(config, (list, tuple)) or len(config) != 18 or version not in ("v1", "v2"):
        raise ValueError("Only standard HuBERT RVC v1/v2 checkpoints are supported")
    pitch = checkpoint.get("f0")
    if pitch not in (0, 1, False, True):
        raise ValueError("RVC checkpoint is missing valid pitch metadata")
    rate = config[-1]
    if isinstance(rate, str):
        rate = {"32k": 32000, "40k": 40000, "48k": 48000}.get(rate)
    if rate not in (32000, 40000, 48000):
        raise ValueError("Unsupported RVC model sample rate")
    return {"application": "VC_CLIENT", "version": "2.1", "modelType": "onnxRVC",
            "samplingRate": int(rate), "f0": bool(pitch),
            "embChannels": 256 if version == "v1" else 768,
            "embedder": "hubert_base", "embOutputLayer": 9 if version == "v1" else 12,
            "useFinalProj": version == "v1", "voice_warp_rvc_version": version}


def normalized_voice_config(checkpoint):
    """Normalize architecture arguments without mutating the trusted checkpoint.

    Standard exports can carry an obsolete configured speaker count. The actual
    embedding weight shape is authoritative for inference and export alike.
    """
    info = voice_metadata(checkpoint)
    weights = checkpoint.get("weight")
    if not isinstance(weights, dict):
        raise ValueError("The selected checkpoint is missing RVC weights")
    shape = getattr(weights.get("emb_g.weight"), "shape", None)
    if shape is None or len(shape) != 2 or shape[0] <= 0:
        raise ValueError("RVC speaker embedding weights must be a nonempty matrix")
    config = list(checkpoint["config"])
    config[-1] = info["samplingRate"]
    config[-3] = int(shape[0])
    return config
