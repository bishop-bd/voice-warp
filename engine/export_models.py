"""Export trusted standard RVC checkpoints and HuBERT graphs for local inference.

Exports are float32, CPU-validated, and cached immutably by source and toolchain
hashes. A successful CPU export is not a GPU compatibility or listening test.
Legacy .pth files are trusted executable input; never invoke this on untrusted
checkpoints. The original checkpoint is never modified.
"""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import inspect
import json
import marshal
import os
from pathlib import Path
import shutil
import tempfile

from voice_warp_checkpoint import normalized_voice_config, voice_metadata

EXPORT_VERSION = "1"
UPSTREAM = "d8ef15799470193f7c8176ef471245753a656626"
OPSET = 17
VOICE_FRAMES = (32, 64, 113, 192)
HUBERT_SAMPLES = (8192, 16000, 24576, 48000)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_digest(name, source):
    """Hash source, or its packaged code when PyInstaller stores it in PYZ."""
    if Path(source).is_file():
        return sha256(source)
    spec = importlib.util.find_spec(name)
    loader = spec.loader if spec else None
    code = loader.get_code(name) if loader and hasattr(loader, "get_code") else None
    if code is None:
        raise ValueError(f"Cannot fingerprint packaged exporter module: {name}")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def fingerprint(source, assets, versions, include_hubert):
    """No provider-specific compiled graphs are persisted in this export cache."""
    payload = {"exporter": EXPORT_VERSION,
               "exporter_sha256": module_digest("export_models", __file__),
               "upstream": UPSTREAM, "opset": OPSET,
               "voice_sha256": sha256(source),
               "hubert_sha256": sha256(Path(assets) / "hubert_base.pt"),
               "rmvpe_sha256": sha256(Path(assets) / "rmvpe.onnx"),
               "versions": versions, "include_hubert": bool(include_hubert),
               "precision": "float32", "voice_validation_frames": VOICE_FRAMES,
               "hubert_validation_samples": HUBERT_SAMPLES}
    helper = Path(__file__).with_name("voice_warp_hubert.py")
    payload["hubert_adapter_sha256"] = module_digest("voice_warp_hubert", helper)
    checkpoint_helper = Path(__file__).with_name("voice_warp_checkpoint.py")
    payload["checkpoint_adapter_sha256"] = module_digest("voice_warp_checkpoint", checkpoint_helper)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return key, payload


def cached_manifest(directory, key):
    try:
        manifest = json.loads((directory / "export.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("cache_key") != key or not manifest.get("cpu_validated"):
            return None
        files = manifest.get("files")
        if not isinstance(files, dict) or not {"voice.onnx", "rmvpe.onnx"}.issubset(files):
            return None
        sources = manifest.get("sources")
        if not isinstance(sources, dict):
            return None
        if sources.get("include_hubert") and "hubert.onnx" not in files:
            return None
        for filename, digest in files.items():
            if Path(filename).name != filename or sha256(directory / filename) != digest:
                return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def remove_abandoned_exports(root, key):
    """Call only while holding this key's lock; EOF cancellation may skip finally."""
    for candidate in root.glob(f".{key}-*"):
        # Never follow a cache-directory symlink or touch another export key.
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)


@contextmanager
def zero_torch_noise(torch):
    """Numerical parity only: eliminate independent Torch/ORT random streams."""
    original = {name: getattr(torch, name) for name in ("rand", "randn", "rand_like", "randn_like")}
    def zeros(*size, **kwargs):
        kwargs.pop("generator", None)
        return torch.zeros(*size, **kwargs)
    def zeros_like(value, **kwargs):
        kwargs.pop("generator", None)
        return torch.zeros_like(value, **kwargs)
    torch.rand = torch.randn = zeros
    torch.rand_like = torch.randn_like = zeros_like
    try:
        yield
    finally:
        for name, function in original.items():
            setattr(torch, name, function)


@contextmanager
def exportable_attention(torch):
    """Trace inference SDPA as portable matmul/softmax for the legacy exporter.

    HuBERT uses noncausal self-attention with dropout disabled during inference.
    Reject unsupported semantics rather than exporting a different operation.
    The original implementation is restored before parity comparisons.
    """
    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention
    def attention(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
        if dropout_p or is_causal or enable_gqa:
            raise ValueError("Unsupported HuBERT attention export configuration")
        factor = scale if scale is not None else float(query.shape[-1]) ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * factor
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        return torch.matmul(torch.softmax(scores, dim=-1), value)
    functional.scaled_dot_product_attention = attention
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


def deterministic_graph(onnx, graph):
    """Copy a graph and replace random outputs with zero tensors for parity.

    The production graph keeps its stochastic operations. Unsupported random
    operators fail validation instead of being silently ignored.
    """
    model = onnx.ModelProto()
    model.CopyFrom(graph)
    nodes = []
    random_count = 0
    for ordinal, node in enumerate(model.graph.node):
        if not node.op_type.startswith("Random"):
            nodes.append(node)
            continue
        if node.op_type not in ("RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike"):
            raise ValueError(f"Unsupported stochastic operator: {node.op_type}")
        random_count += 1
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        dtype = attrs.get("dtype", onnx.TensorProto.FLOAT)
        if dtype != onnx.TensorProto.FLOAT:
            raise ValueError("Only float32 stochastic export is supported")
        if node.op_type.endswith("Like"):
            shape = f"voice_warp_parity_shape_{ordinal}"
            nodes.append(onnx.helper.make_node("Shape", [node.input[0]], [shape]))
        else:
            shape = f"voice_warp_parity_shape_{ordinal}"
            tensor = onnx.helper.make_tensor(shape, onnx.TensorProto.INT64,
                                            [len(attrs["shape"])], attrs["shape"])
            nodes.append(onnx.helper.make_node("Constant", [], [shape], value=tensor))
        zero = onnx.helper.make_tensor("zero", dtype, [1], [0.0])
        nodes.append(onnx.helper.make_node("ConstantOfShape", [shape], list(node.output), value=zero))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model, random_count


def session(ort, model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 6
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    result = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    result.disable_fallback()
    return result


def metadata(onnx, graph, values):
    entry = graph.metadata_props.add()
    entry.key = "metadata"
    entry.value = json.dumps(values, sort_keys=True)
    entry = graph.metadata_props.add()
    entry.key = "voice_warp_exporter"
    entry.value = EXPORT_VERSION
    onnx.checker.check_model(graph)


def validate_close(np, expected, actual, label, *, atol, rtol):
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError(f"{label}: incorrect shape or nonfinite exported output")
    maximum = float(np.max(np.abs(expected - actual)))
    mean = float(np.mean(np.abs(expected - actual)))
    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        raise ValueError(f"{label}: CPU export parity failed (max={maximum:.6g}, mean={mean:.6g})")
    return {"shape": list(actual.shape), "max_absolute_error": maximum,
            "mean_absolute_error": mean, "root_mean_square_error": float(np.sqrt(np.mean((expected - actual) ** 2))),
            "atol": atol, "rtol": rtol}


def export_voice(source, destination, torch, onnx, ort, np):
    # Explicit legacy trust boundary, scoped to this user-requested checkpoint.
    checkpoint = torch.load(str(source), map_location="cpu", weights_only=False)
    info = voice_metadata(checkpoint)
    models = importlib.import_module("voice_changer.RVC.inferencer.rvc_models.infer_pack.models")
    suffix = "" if info["f0"] else "_nono"
    model_type = getattr(models, f"SynthesizerTrnMs{info['embChannels']}NSFsid{suffix}")
    config = normalized_voice_config(checkpoint)
    model = model_type(*config, is_half=False).float().eval()
    loaded = model.load_state_dict(checkpoint["weight"], strict=False)
    missing = [name for name in loaded.missing_keys if not name.startswith("enc_q.")]
    unexpected = [name for name in loaded.unexpected_keys if not name.startswith("enc_q.")]
    if missing or unexpected:
        raise ValueError(f"Unsupported RVC checkpoint weights: missing={missing}, extra={unexpected}")
    del checkpoint
    # Training-only posterior encoder is not part of inference/export.
    del model.enc_q

    class Voice(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
        def forward(self, *inputs):
            return self.wrapped.infer(*inputs)[0][0, 0].clamp(-1.0, 1.0)

    voice = Voice(model).eval()
    names = ["feats", "p_len", "pitch", "pitchf", "sid"] if info["f0"] else ["feats", "p_len", "sid"]
    generator = torch.Generator().manual_seed(193)
    def inputs(frames, speaker=0):
        values = [torch.randn(1, frames, info["embChannels"], generator=generator) * 0.1,
                  torch.tensor([frames], dtype=torch.int64)]
        if info["f0"]:
            values += [torch.full((1, frames), 90, dtype=torch.int64),
                       torch.full((1, frames), 180.0, dtype=torch.float32)]
        values.append(torch.tensor([speaker], dtype=torch.int64))
        return tuple(values)
    dynamic = {"feats": {1: "frames"}, "audio": {0: "samples"}}
    if info["f0"]:
        dynamic.update(pitch={1: "frames"}, pitchf={1: "frames"})
    with torch.no_grad():
        torch.onnx.export(voice, inputs(64), str(destination), input_names=names,
                          output_names=["audio"], dynamic_axes=dynamic,
                          opset_version=OPSET, do_constant_folding=False,
                          **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}))
    graph = onnx.load(str(destination))
    metadata(onnx, graph, info)
    onnx.save(graph, str(destination))
    deterministic, random_count = deterministic_graph(onnx, graph)
    reference_session = session(ort, deterministic.SerializeToString())
    production = session(ort, str(destination))
    reports = []
    for frames in VOICE_FRAMES:
        values = inputs(frames, min(config[-3] - 1, 1))
        feed = {name: value.numpy() for name, value in zip(names, values)}
        with torch.no_grad(), zero_torch_noise(torch):
            expected = voice(*values).numpy()
        actual = reference_session.run(["audio"], feed)[0]
        report = validate_close(np, expected, actual, f"RVC {frames} frames", atol=0.01, rtol=0.03)
        # Cumulative phase/interpolation can differ locally between providers;
        # bound average error as well as the per-sample tolerance above.
        if report["mean_absolute_error"] > 0.001:
            raise ValueError("RVC export average error exceeds 0.001 full scale")
        stochastic = production.run(["audio"], feed)[0]
        expected_samples = frames * (info["samplingRate"] // 100)
        if stochastic.shape != (expected_samples,) or not np.isfinite(stochastic).all():
            raise ValueError("Stochastic RVC graph returned an invalid waveform")
        report.update(frames=frames, stochastic_output_checked=True)
        reports.append(report)
    info["speakers"] = config[-3]
    return info, {"cases": reports, "random_operators": random_count,
                  "parity_policy": "independent Torch/ONNX RNG outputs zeroed in validation only",
                  "production_randomness_preserved": True}


def export_hubert(source, destination, info, torch, onnx, ort, np):
    from voice_warp_hubert import load_hubert
    model = load_hubert(str(source), device="cpu").float().eval()
    output_layer = info["embOutputLayer"]
    projection = info["useFinalProj"]

    class Hubert(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
        def forward(self, waveform):
            return self.wrapped(waveform, output_layer=output_layer, use_final_proj=projection)

    wrapped = Hubert(model).eval()
    waveform = torch.sin(torch.arange(16000, dtype=torch.float32) * (180.0 * 2 * torch.pi / 16000)).reshape(1, -1) * 0.1
    with torch.no_grad(), exportable_attention(torch):
        torch.onnx.export(wrapped, (waveform,), str(destination), input_names=["waveform"],
                          output_names=["features"], dynamic_axes={"waveform": {1: "samples"}, "features": {1: "frames"}},
                          opset_version=OPSET, do_constant_folding=True,
                          **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}))
    graph = onnx.load(str(destination))
    metadata(onnx, graph, {"application": "VOICE_WARP_HUBERT", "version": "1",
                          "hubert_version": info["voice_warp_rvc_version"], "sample_rate": 16000,
                          "output_layer": output_layer, "use_final_proj": projection,
                          "channels": info["embChannels"], "source_sha256": sha256(source)})
    onnx.save(graph, str(destination))
    exported = session(ort, str(destination))
    reports = []
    for samples in HUBERT_SAMPLES:
        waveform = (torch.sin(torch.arange(samples, dtype=torch.float32) * (177.0 * 2 * torch.pi / 16000)) * 0.12).reshape(1, -1)
        with torch.no_grad():
            expected = wrapped(waveform).numpy()
        actual = exported.run(["features"], {"waveform": waveform.numpy()})[0]
        report = validate_close(np, expected, actual, f"HuBERT {samples} samples", atol=0.0005, rtol=0.003)
        report["samples"] = samples
        reports.append(report)
    return {"cases": reports, "sample_rate": 16000, "output_layer": output_layer,
            "use_final_proj": projection, "channels": info["embChannels"]}


def export_bundle(model_path, assets_dir, cache_dir, *, include_hubert=True):
    """Return absolute artifacts and validation report; never overwrite source."""
    import numpy as np
    import torch
    import torchaudio
    import onnx
    import onnxruntime as ort
    from filelock import FileLock

    source, assets, root = Path(model_path).resolve(), Path(assets_dir).resolve(), Path(cache_dir).resolve()
    if not source.is_file() or source.suffix.lower() != ".pth":
        raise ValueError("Export requires a trusted RVC .pth checkpoint")
    for name in ("hubert_base.pt", "rmvpe.onnx"):
        if not (assets / name).is_file():
            raise ValueError(f"Missing shared asset {name}")
    torch.set_num_threads(6)
    versions = {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
                "numpy": np.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}
    key, sources = fingerprint(source, assets, versions, include_hubert)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / key
    with FileLock(str(root / f"{key}.lock"), timeout=0):
        remove_abandoned_exports(root, key)
        manifest = cached_manifest(directory, key)
        if manifest is not None:
            return resolve_artifacts(directory, manifest, cached=True)
        if directory.exists():
            raise ValueError(f"Export cache entry failed integrity checks: {directory}. Remove that entry before retrying.")
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
        try:
            info, voice_validation = export_voice(source, temporary / "voice.onnx", torch, onnx, ort, np)
            validation = {"voice": voice_validation}
            files = {"voice.onnx": sha256(temporary / "voice.onnx")}
            if include_hubert:
                validation["hubert"] = export_hubert(assets / "hubert_base.pt", temporary / "hubert.onnx", info, torch, onnx, ort, np)
                files["hubert.onnx"] = sha256(temporary / "hubert.onnx")
            # Keep the qualified upstream pitch graph unchanged; package it with
            # the exported bundle so it can be mounted without a separate asset.
            shutil.copyfile(assets / "rmvpe.onnx", temporary / "rmvpe.onnx")
            files["rmvpe.onnx"] = sha256(temporary / "rmvpe.onnx")
            manifest = {"schema": 1, "cache_key": key, "sources": sources,
                        "model": info, "files": files, "cpu_validated": True,
                        "gpu_validated": False, "speech_quality_verified": False,
                        "validation": validation}
            (temporary / "export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return resolve_artifacts(directory, manifest, cached=False)


def resolve_artifacts(directory, manifest, *, cached):
    return {**manifest, "cached": cached, "directory": str(directory),
            "voice_model": str(directory / "voice.onnx"),
            "hubert_model": str(directory / "hubert.onnx") if "hubert.onnx" in manifest["files"] else None,
            "rmvpe_model": str(directory / "rmvpe.onnx")}


def main():
    import argparse
    import sys
    from contextlib import redirect_stdout
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--voice-only", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = export_bundle(args.model, args.assets, args.cache, include_hubert=not args.voice_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
