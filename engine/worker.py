"""Voice Warp RVC worker: bounded JSON-lines over private stdin/stdout, no sockets."""
import base64
import json
import os
import sys
import traceback

# Libraries print to stdout (including native code). Reserve a separate descriptor
# for the protocol before importing them and redirect all diagnostics to stderr.
protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
MAX_LINE = 300_000


def receive():
    line = sys.stdin.buffer.readline(MAX_LINE + 1)
    if not line:
        return None
    if len(line) > MAX_LINE or not line.endswith(b"\n"):
        raise ValueError("Engine request exceeds the size limit")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected an engine request object")
    return value


def send(value):
    protocol.write(json.dumps(value, allow_nan=False) + "\n")
    protocol.flush()


def main():
    config = receive()
    if config is None:
        return
    if config.get("protocol") != 1:
        raise ValueError("Unsupported Voice Warp engine protocol")
    block_size = config["block_size"]
    pitch = config["pitch"]
    index_ratio = config["index_ratio"]
    if block_size not in (4096, 6144, 8192, 12288):
        raise ValueError("Unsupported audio block size")
    if not isinstance(pitch, int) or not -24 <= pitch <= 24:
        raise ValueError("Pitch must be between -24 and 24 semitones")
    if not isinstance(index_ratio, (int, float)) or not 0 <= index_ratio <= 1:
        raise ValueError("Index strength must be between 0 and 1")

    import numpy as np
    import torch
    from data.ModelSlot import RVCModelSlot
    from mods.log_control import VoiceChangaerLogger
    from voice_changer.RVC.RVCModelSlotGenerator import RVCModelSlotGenerator
    from voice_changer.RVC.RVCr2 import RVCr2
    from voice_changer.VoiceChangerV2 import VoiceChangerV2
    from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

    VoiceChangaerLogger.get_instance().initialize(initialize=True)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    model_file = config["model_file"]
    if model_file not in ("model.pth", "model.onnx"):
        raise ValueError("Choose an RVC .pth or .onnx model")
    model_path = "/models/0/" + model_file
    slot = RVCModelSlot(slotIndex=0, modelFile=model_file)
    if model_file.endswith(".onnx"):
        slot = RVCModelSlotGenerator._setInfoByONNX(model_path, slot)
        if slot.deprecated:
            raise ValueError("This ONNX model has no supported RVC metadata; use its .pth version")
    else:
        slot = RVCModelSlotGenerator._setInfoByPytorch(model_path, slot)
    if slot.embedder != "hubert_base":
        raise ValueError("This engine currently supports standard HuBERT RVC v1/v2 models")
    slot.indexFile = "voice.index" if config.get("has_index") else ""
    if slot.indexFile:
        import faiss
        index = faiss.read_index("/models/0/voice.index")
        if index.d != slot.embChannels:
            raise ValueError("The index does not match this model's feature dimensions")
        del index
    params = VoiceChangerParams(
        model_dir="/models", content_vec_500="", content_vec_500_onnx="",
        content_vec_500_onnx_on=False, hubert_base="/assets/hubert_base.pt",
        hubert_base_jp="", hubert_soft="", nsf_hifigan="", sample_mode="",
        crepe_onnx_full="", crepe_onnx_tiny="", rmvpe="",
        rmvpe_onnx="/assets/rmvpe.onnx", whisper_tiny="",
    )
    model = RVCr2(params, slot)
    model.settings.gpu = -1
    model.initialize()
    if model.pipeline is None:
        raise RuntimeError("Could not initialize this RVC model; inspect the engine log")
    model.settings.tran = pitch
    model.settings.indexRatio = index_ratio if slot.indexFile else 0
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

    # Exercise the actual inference path before advertising readiness. Silence
    # alone skips inference in RVC. Reset streaming history after warmup.
    tone = (3000 * np.sin(np.arange(block_size) * 2 * np.pi * 180 / 48000)).astype("<i2")
    for _ in range(2):
        convert(tone)
    model.audio_buffer = model.pitchf_buffer = model.feature_buffer = None
    model.prevVol = 0
    converter = VoiceChangerV2(params)
    converter.setModel(model)
    send({"protocol": 1, "ready": True, "device": "CPU", "sample_rate": 48000})
    sequence = 0
    while (request := receive()) is not None:
        sequence += 1
        if request.get("sequence") != sequence:
            raise ValueError("Unexpected audio sequence")
        pcm = base64.b64decode(request["audio"], validate=True)
        if len(pcm) != block_size * 2:
            raise ValueError("Unexpected audio block length")
        output = convert(np.frombuffer(pcm, dtype="<i2"))
        send({"sequence": sequence, "audio": base64.b64encode(output).decode("ascii")})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        send({"error": str(error) or type(error).__name__})
        sys.exit(1)
