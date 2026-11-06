"""HuBERT inference without Fairseq's training/runtime dependencies.

The checkpoint key conversion is supplied by torchaudio's MIT-licensed official
Fairseq importer. The fixed shared asset is authenticated before deserializing;
only its unused Dictionary metadata is replaced during this individual load.
"""
import hashlib
from pathlib import Path
import pickle
from types import SimpleNamespace

import torch
from torch import nn
import torchaudio
from torchaudio.models.wav2vec2.utils.import_fairseq import _convert_state_dict

HUBERT_SHA256 = "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96"


class _DictionaryMetadata:
    """Unused training dictionary carried by the authenticated checkpoint."""
    pass


class _SharedAssetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("fairseq.data.dictionary", "Dictionary"):
            return _DictionaryMetadata
        if module.startswith("fairseq"):
            raise pickle.UnpicklingError(f"Unexpected Fairseq object in shared asset: {module}.{name}")
        return super().find_class(module, name)


def load_shared_checkpoint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != HUBERT_SHA256:
            raise ValueError("HuBERT shared asset checksum mismatch; repair the AI runtime")
        source.seek(0)
        # This exception applies only to this checksum-authenticated shared asset.
        # Never modify torch.load globally or silently trust arbitrary checkpoints.
        module = SimpleNamespace(Unpickler=_SharedAssetUnpickler, load=pickle.load,
                                 __name__="voice_warp_shared_asset_pickle")
        return torch.load(source, map_location="cpu", pickle_module=module, weights_only=False)


class HubertFeatures(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.hubert = torchaudio.models.hubert_base()
        converted = _convert_state_dict(state)
        converted.pop("label_embs_concat", None)  # training-only target embeddings
        self.hubert.load_state_dict(converted, strict=True)
        self.final_proj = nn.Linear(768, 256)
        self.final_proj.load_state_dict({"weight": state["final_proj.weight"],
                                        "bias": state["final_proj.bias"]}, strict=True)
        self.eval()

    def forward(self, waveform, output_layer=9, use_final_proj=True):
        features, _ = self.hubert.extract_features(waveform, num_layers=output_layer)
        result = features[-1]
        return self.final_proj(result) if use_final_proj else result


def load_hubert(path, device="cpu"):
    checkpoint = load_shared_checkpoint(path)
    model = HubertFeatures(checkpoint["model"])
    return model.to(device=device, dtype=torch.float32).eval()


class TorchHubertEmbedder:
    def __init__(self, path, device):
        self.model = load_hubert(path, device)
        self.dev = device

    def extractFeatures(self, feats, embOutputLayer=9, useFinalProj=True):
        with torch.no_grad():
            return self.model(feats.to(self.dev), embOutputLayer, useFinalProj)

    def getEmbedderInfo(self):
        return {"embedderType": "hubert_base", "device": str(self.dev),
                "implementation": "torchaudio", "isHalf": False}


class OnnxHubertEmbedder:
    """Validated exported HuBERT, with provider placement audited by the caller."""
    def __init__(self, path, manager, ort, slot):
        import json
        providers, options = manager.getOnnxExecutionProvider(manager.selected["index"])
        self.onnx_session = ort.InferenceSession(str(path), providers=providers,
                                                provider_options=options)
        metadata = json.loads(self.onnx_session.get_modelmeta().custom_metadata_map.get("metadata", "{}"))
        expected = {"application": "VOICE_WARP_HUBERT", "version": "1", "sample_rate": 16000,
                    "output_layer": slot.embOutputLayer, "use_final_proj": slot.useFinalProj,
                    "channels": slot.embChannels, "source_sha256": HUBERT_SHA256}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("The exported HuBERT graph does not match this voice or shared asset")
        if [value.name for value in self.onnx_session.get_inputs()] != ["waveform"] or [value.name for value in self.onnx_session.get_outputs()] != ["features"]:
            raise ValueError("Unsupported exported HuBERT input/output interface")
        self.dev = manager.device
        self.output_layer = slot.embOutputLayer
        self.use_final_proj = slot.useFinalProj

    def extractFeatures(self, feats, embOutputLayer=9, useFinalProj=True):
        if (embOutputLayer, useFinalProj) != (self.output_layer, self.use_final_proj):
            raise ValueError("HuBERT export feature configuration changed")
        output = self.onnx_session.run(["features"], {"waveform": feats.detach().cpu().float().numpy()})[0]
        return torch.from_numpy(output).to(self.dev)

    def getEmbedderInfo(self):
        return {"embedderType": "hubert_base", "implementation": "onnx",
                "providers": self.onnx_session.get_providers()}
