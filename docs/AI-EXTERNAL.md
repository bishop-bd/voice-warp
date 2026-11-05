# Local AI voice conversion

Voice Warp's optional AI mode sends microphone blocks to a **local w-okada v1 RVC server**, then routes the converted audio into the same virtual microphone used by its regular effects and soundboard. Voice Warp itself is Rust. The separately installed inference server uses Python and model weights; neither is bundled or automatically downloaded by Voice Warp.

The adapter targets the v1 REST protocol at upstream commit [`d8ef15799470193f7c8176ef471245753a656626`](https://github.com/w-okada/voice-changer/tree/d8ef15799470193f7c8176ef471245753a656626). It is **not** an integration with the separate VC Client v2 protocol. RVC is the supported model family; other families are rejected until their streaming behavior is verified.

The adapter is shared across Windows, Linux, and macOS. The installation recipe below is for **Bazzite/Linux**. On Windows or macOS, install a compatible v1 server using its [upstream instructions](https://github.com/w-okada/voice-changer/tree/d8ef15799470193f7c8176ef471245753a656626), then use the same endpoint and server settings here. The latest v2 client is not compatible with this pinned v1 protocol. Windows/macOS audio goes to the selected virtual cable; Linux audio goes to the managed Voice Warp source.

## Radeon Vega 64

Start with **CPU** inference. Vega 64 is not in AMD's current supported Radeon matrix, so this project does not promise ROCm acceleration on that card. A compatible community ROCm installation may be possible, but it is a separate experiment, not a requirement for Voice Warp. The standard voice effects and soundboard run on the CPU and do not require an AI runtime. See [AMD's compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

If an upstream server already works with your Vega 64, select its working device there; Voice Warp does not care which device performs inference. Upstream's [AMD/Linux instructions](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/tutorials/tutorial_anaconda_amd_rocm.md) describe a ROCm path, but do not establish Vega 64 or Bazzite compatibility. Do not install Ubuntu GPU drivers onto the Bazzite host.

## Install the optional server

Keep this separate from the Voice Warp app and its build environment. On Bazzite, an Ubuntu 22.04 Distrobox provides the Python 3.10 environment used by the upstream v1 dependencies:

```bash
distrobox create --name voice-warp-ai --image docker.io/library/ubuntu:22.04
distrobox enter voice-warp-ai
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-dev build-essential libportaudio2 libasound2-dev libsndfile1 ffmpeg
mkdir -p "$HOME/.local/share/voice-warp-ai"
cd "$HOME/.local/share/voice-warp-ai"
git clone https://github.com/w-okada/voice-changer.git server-source
cd server-source
git checkout d8ef15799470193f7c8176ef471245753a656626
python3 -m venv ../venv
source ../venv/bin/activate
python -m pip install --upgrade pip wheel 'setuptools<81'
cd server
```

Upstream's requirements pin an older inference stack. For CPU operation, install the CPU PyTorch wheels first and make a local requirements copy that substitutes CPU ONNX Runtime for its CUDA build:

```bash
python -m pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
python - <<'PY'
from pathlib import Path
source = Path('requirements.txt').read_text()
Path('requirements-cpu.txt').write_text(source.replace('onnxruntime-gpu==1.13.1', 'onnxruntime==1.13.1'))
PY
python -m pip install -r requirements-cpu.txt
python MMVCServerSIO.py --host 127.0.0.1 -p 18888 --https false
```

This is an optional setup recipe derived from the [upstream developer guide](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/README_dev_en.md), its [pinned dependencies](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/server/requirements.txt), and [PyTorch's CPU installation instructions](https://pytorch.org/get-started/previous-versions/#v201). The full inference stack and model performance have not been tested on your Bazzite machine. Leave it in its own environment, since upgrading its old dependencies independently can break model loading.

The server's first launch downloads its own pretrained assets and possibly sample models. This can use substantial disk space and bandwidth. Those downloads are behavior of the [upstream server](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/server/downloader/WeightDownloader.py), not Voice Warp. Use models whose license permits your intended use. Choose trusted model files: legacy PyTorch checkpoints may contain executable serialization.

## Connect it to Voice Warp

1. Open [the local server UI](http://127.0.0.1:18888), upload or select an **RVC** model, and select **CPU** as its device. If the console reports an uninitialized pipeline, reselect the device after loading the model.
2. Use client mode. Stop the server UI's own audio capture, disable Server Device audio, disable Pass Through, and disable input/output recording. Voice Warp supplies audio, so no second client should stream to the same server.
3. Ensure the server's input and output rates are both **48000 Hz**. The model's training rate may differ; the server performs that resampling.
4. In Voice Warp, select AI mode, keep the endpoint at `http://127.0.0.1:18888`, and check the connection. The reported model name comes from the server. Start the virtual microphone.
5. Select **Voice Warp** as the input device in your call, game, or recording app.

If a setting is difficult to locate in the upstream UI, run the following **after selecting your model and stopping its audio**. These requests explicitly configure the local server for Voice Warp; they do not upload any microphone audio:

```bash
curl --fail --show-error --silent -X POST http://127.0.0.1:18888/update_settings -d 'key=serverAudioStated&val=0'
curl --fail --show-error --silent -X POST http://127.0.0.1:18888/update_settings -d 'key=enableServerAudio&val=0'
curl --fail --show-error --silent -X POST http://127.0.0.1:18888/update_settings -d 'key=passThrough&val=false'
curl --fail --show-error --silent -X POST http://127.0.0.1:18888/update_settings -d 'key=recordIO&val=0'
curl --fail --show-error --silent -X POST http://127.0.0.1:18888/update_settings -d 'key=inputSampleRate&val=48000'
curl --fail --show-error --silent -X POST http://127.0.0.1:18888/update_settings -d 'key=outputSampleRate&val=48000'
curl --fail --show-error --silent -X POST http://127.0.0.1:18888/update_settings -d 'key=gpu&val=-1'
```

`gpu=-1` selects CPU. To use a separately verified GPU installation, select the desired device in the server UI instead of sending that last request. Changing model slots while streaming requires stopping and restarting Voice Warp's microphone.

## Latency and behavior

The default AI block contains 6144 samples, or **128 ms** of input at 48 kHz. That is the block duration, not the total latency: model inference, server crossfade/context, playback buffering, and capture latency add to it. The processing-time display is also not an end-to-end latency measurement. If inference is slower than each block's duration, audio cannot remain continuous in real time; the bounded queue prevents indefinitely increasing delay, but the result will have gaps. Try a larger AI block, less upstream extra context, or a less costly pitch detector/model. CPU performance depends on the processor and model.

The adapter only accepts loopback addresses. It ignores HTTP proxy environment variables and rejects redirects. It checks the loaded RVC pipeline, sample rates, recording setting, server audio state, and Pass Through before and after each conversion request. An invalid response, disconnect, timeout, or detected model change returns an error; it never substitutes the raw microphone as an AI result. The server remains a separate process under your control; these checks are not a security boundary against a malicious or concurrently manipulated server.

The wire format is 48 kHz mono little-endian signed 16-bit PCM wrapped in base64 JSON. `/test` is the actual upstream streaming conversion route despite its name; `/info` supplies settings and selected model metadata. Sources: [conversion route](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/server/restapi/MMVC_Rest_VoiceChanger.py), [settings routes](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/server/restapi/MMVC_Rest_Fileuploader.py), and [model manager](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/server/voice_changer/VoiceChangerManager.py).

## Verification

`cargo test ai::` runs the protocol tests with a local mock HTTP server, without microphone access or model downloads. It checks PCM conversion, clipping, malformed data, wrong response lengths, mid-request Pass Through changes, redirects, and oversized replies.

With a configured local RVC server running, the opt-in smoke test sends three blocks of **silence**:

```bash
cargo test ai::tests::local_server_integration -- --ignored --nocapture
```

That verifies connectivity and output shape; listen to actual speech through headphones to assess voice quality and latency. The real-server test is ignored in normal runs because it needs external weights and a running inference service.
