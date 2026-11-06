"""Small inference-only adapter for pinned w-okada RVC streaming code.

The buffer/resampling methods derive from RVCr2.py at upstream revision
 d8ef15799470193f7c8176ef471245753a656626 (MIT, see UPSTREAM-LICENSE).
Training managers and Fairseq are deliberately not imported by this adapter.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import logging
from typing import cast

import numpy as np
import resampy
import torch

from Exceptions import DeviceCannotSupportHalfPrecisionException, PipelineNotInitializedException
from voice_changer.RVC.RVCSettings import RVCSettings
from voice_warp_checkpoint import normalized_voice_config, voice_metadata

logger = logging.getLogger(__name__)


class RvcStream:
    voiceChangerType = "RVC"

    def __init__(self, params, slot):
        self.params = params
        self.slotInfo = slot
        self.settings = RVCSettings()
        self.pipeline = None
        self.audio_buffer = self.pitchf_buffer = self.feature_buffer = None
        self.prevVol = 0.0

    def setSamplingRate(self, inputSampleRate, outputSampleRate):
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate

    def get_processing_sampling_rate(self):
        return self.slotInfo.samplingRate

    def generate_input(
        self,
        newData: np.ndarray,
        crossfadeSize: int,
        solaSearchFrame: int,
        extra_frame: int,
    ):
        # 16k で入ってくる。
        inputSize = newData.shape[0]
        newData = newData.astype(np.float32) / 32768.0
        newFeatureLength = inputSize // 160  # hopsize:=160

        if self.audio_buffer is not None:
            # 過去のデータに連結
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)
            if self.slotInfo.f0:
                self.pitchf_buffer = np.concatenate([self.pitchf_buffer, np.zeros(newFeatureLength)], 0)
            self.feature_buffer = np.concatenate(
                [
                    self.feature_buffer,
                    np.zeros([newFeatureLength, self.slotInfo.embChannels]),
                ],
                0,
            )
        else:
            self.audio_buffer = newData
            if self.slotInfo.f0:
                self.pitchf_buffer = np.zeros(newFeatureLength)
            self.feature_buffer = np.zeros([newFeatureLength, self.slotInfo.embChannels])

        convertSize = inputSize + crossfadeSize + solaSearchFrame + extra_frame

        if convertSize % 160 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (160 - (convertSize % 160))
        outSize = int(((convertSize - extra_frame) / 16000) * self.slotInfo.samplingRate)

        # バッファがたまっていない場合はzeroで補う
        if self.audio_buffer.shape[0] < convertSize:
            self.audio_buffer = np.concatenate([np.zeros([convertSize]), self.audio_buffer])
            if self.slotInfo.f0:
                self.pitchf_buffer = np.concatenate([np.zeros([convertSize // 160]), self.pitchf_buffer])
            self.feature_buffer = np.concatenate(
                [
                    np.zeros([convertSize // 160, self.slotInfo.embChannels]),
                    self.feature_buffer,
                ]
            )

        # 不要部分をトリミング
        convertOffset = -1 * convertSize
        featureOffset = convertOffset // 160
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出
        if self.slotInfo.f0:
            self.pitchf_buffer = self.pitchf_buffer[featureOffset:]
        self.feature_buffer = self.feature_buffer[featureOffset:]

        # 出力部分だけ切り出して音量を確認。(TODO:段階的消音にする)
        cropOffset = -1 * (inputSize + crossfadeSize)
        cropEnd = -1 * (crossfadeSize)
        crop = self.audio_buffer[cropOffset:cropEnd]
        vol = np.sqrt(np.square(crop).mean())
        vol = max(vol, self.prevVol * 0.0)
        self.prevVol = vol

        return (
            self.audio_buffer,
            self.pitchf_buffer,
            self.feature_buffer,
            convertSize,
            vol,
            outSize,
        )

    def inference(self, receivedData: np.ndarray, crossfade_frame: int, sola_search_frame: int):
        if self.pipeline is None:
            logger.info("[Voice Changer] Pipeline is not initialized.")
            raise PipelineNotInitializedException()

        # 処理は16Kで実施(Pitch, embed, (infer))
        receivedData = cast(
            np.ndarray,
            resampy.resample(
                receivedData,
                self.inputSampleRate,
                16000,
                filter="kaiser_fast",
            ),
        )
        crossfade_frame = int((crossfade_frame / self.inputSampleRate) * 16000)
        sola_search_frame = int((sola_search_frame / self.inputSampleRate) * 16000)
        extra_frame = int((self.settings.extraConvertSize / self.inputSampleRate) * 16000)

        # 入力データ生成
        data = self.generate_input(receivedData, crossfade_frame, sola_search_frame, extra_frame)

        audio = data[0]
        pitchf = data[1]
        feature = data[2]
        convertSize = data[3]
        vol = data[4]
        outSize = data[5]

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16) * np.sqrt(vol)

        device = self.pipeline.device

        audio = torch.from_numpy(audio).to(device=device, dtype=torch.float32)
        repeat = 1 if self.settings.rvcQuality else 0
        sid = self.settings.dstId
        f0_up_key = self.settings.tran
        index_rate = self.settings.indexRatio
        protect = self.settings.protect

        if_f0 = 1 if self.slotInfo.f0 else 0
        embOutputLayer = self.slotInfo.embOutputLayer
        useFinalProj = self.slotInfo.useFinalProj

        try:
            audio_out, self.pitchf_buffer, self.feature_buffer = self.pipeline.exec(
                sid,
                audio,
                pitchf,
                feature,
                f0_up_key,
                index_rate,
                if_f0,
                # 0,
                self.settings.extraConvertSize / self.inputSampleRate if self.settings.silenceFront else 0.0,  # extaraDataSizeの秒数。入力のサンプリングレートで算出
                embOutputLayer,
                useFinalProj,
                repeat,
                protect,
                outSize,
            )
            # result = audio_out.detach().cpu().numpy() * np.sqrt(vol)
            result = audio_out[-outSize:].detach().cpu().numpy() * np.sqrt(vol)

            result = cast(
                np.ndarray,
                resampy.resample(
                    result,
                    self.slotInfo.samplingRate,
                    self.outputSampleRate,
                    filter="kaiser_fast",
                ),
            )

            return result
        except DeviceCannotSupportHalfPrecisionException as error:
            raise RuntimeError("The selected backend failed float32 inference") from error



def load_voice_checkpoint(path):
    """Deserialize only the user-selected, explicitly trusted legacy voice file.

    This keeps the legacy .pth trust boundary visible and avoids changing the
    behavior of any other torch.load call in the process.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    info = voice_metadata(checkpoint)
    return {**checkpoint, "config": normalized_voice_config(checkpoint),
            "version": info["voice_warp_rvc_version"], "f0": info["f0"]}


def model_metadata(path, ort):
    path = Path(path)
    if path.suffix.lower() == ".onnx":
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        metadata = json.loads(session.get_modelmeta().custom_metadata_map.get("metadata", "{}"))
        channels = metadata.get("embChannels")
        if channels not in (256, 768) or metadata.get("embedder", "hubert_base") != "hubert_base":
            raise ValueError("This ONNX voice has no supported HuBERT RVC metadata")
        version = str(metadata.get("version", ""))
        if version not in ("1", "2", "2.1", "2.2"):
            raise ValueError("Unsupported RVC ONNX export version")
        rate = metadata.get("samplingRate")
        if rate not in (32000, 40000, 48000):
            raise ValueError("Unsupported RVC model sample rate")
        f0 = metadata.get("f0")
        if f0 not in (True, False, 0, 1):
            raise ValueError("Invalid RVC ONNX pitch metadata")
        inferencer_version = ("v1.1" if version == "2.1" else "v1") if channels == 256 else "v" + version
        slot = SimpleNamespace(isONNX=True, embChannels=channels,
                               embOutputLayer=metadata.get("embOutputLayer", 9 if channels == 256 else 12),
                               useFinalProj=channels == 256, f0=bool(f0), samplingRate=rate,
                               version=inferencer_version, defaultProtect=0.5,
                               modelFile=str(path), indexFile="", embedder="hubert_base")
        return slot, None
    checkpoint = load_voice_checkpoint(path)
    version = checkpoint.get("version") or "v1"
    channels = 256 if version == "v1" else 768
    rate = checkpoint["config"][-1]
    if rate not in (32000, 40000, 48000):
        raise ValueError("Unsupported RVC model sample rate")
    slot = SimpleNamespace(isONNX=False, embChannels=channels,
                           embOutputLayer=9 if channels == 256 else 12,
                           useFinalProj=channels == 256, f0=bool(checkpoint.get("f0", 1)),
                           samplingRate=rate, version=version, defaultProtect=0.5,
                           modelFile=str(path), indexFile="", embedder="hubert_base")
    return slot, checkpoint


class TorchRvcInferencer:
    def __init__(self, checkpoint, slot, device):
        from voice_changer.RVC.inferencer.rvc_models.infer_pack.models import (
            SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid_nono,
        )
        constructor = {
            (256, True): SynthesizerTrnMs256NSFsid,
            (768, True): SynthesizerTrnMs768NSFsid,
            (256, False): SynthesizerTrnMs256NSFsid_nono,
            (768, False): SynthesizerTrnMs768NSFsid_nono,
        }[(slot.embChannels, slot.f0)]
        self.model = constructor(*normalized_voice_config(checkpoint), is_half=False)
        # Posterior encoder is training-only and absent from many voice exports.
        del self.model.enc_q
        missing, unexpected = self.model.load_state_dict(checkpoint["weight"], strict=False)
        unexpected = [key for key in unexpected if not key.startswith("enc_q.")]
        if missing or unexpected:
            raise ValueError(f"RVC weights do not match the model: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.model.to(device=device, dtype=torch.float32).eval()
        self.f0 = slot.f0
        self.isHalf = False

    def infer(self, feats, pitch_length, pitch, pitchf, sid, convert_length):
        if self.f0:
            result = self.model.infer(feats, pitch_length, pitch, pitchf, sid, convert_length=convert_length)
        else:
            result = self.model.infer(feats, pitch_length, sid, convert_length=convert_length)
        return result[0][0, 0].float().clamp(-1, 1)

    def getInferencerInfo(self):
        return {"inferencerType": "RVC", "isHalf": False}
