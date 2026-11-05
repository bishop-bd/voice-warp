# AI voices in Voice Warp

## Bundled engine (Linux x86-64)

1. Open **Local AI → Bundled engine → Install / repair AI engine**. Podman must be installed on the host. Bazzite already includes it. Setup runs off the UI thread and downloads several GB on its first run; progress is written to the displayed `install.log`.
2. Click **Choose voice model** and select a standard RVC `.pth` or metadata-bearing `.onnx` file. Optionally choose its matching `.index`. Files remain where you selected them, so keep them in a permanent folder. Selecting a new model clears the old index.
3. Set pitch in semitones and index strength. Start with 8192 samples on CPU; try 12288 if conversion cannot keep up.
4. Choose **AI voice**, then start your microphone from Studio. Select **Voice Warp** as the input in your call or game.

**Test voice engine** loads and warms up the model with a generated tone, without opening the microphone, then shuts the worker down. Starting the microphone loads a fresh worker. Stop the microphone before changing settings. Voice Warp saves the selected paths and controls with its usual settings; AI starts disabled on launch.

The runtime and shared inference weights are installed once, using a build recipe embedded in the desktop executable. There is no separate AI application, browser, HTTP server, or manual server startup. The engine is a Python subprocess in an ephemeral Podman container. Voice Warp owns its lifetime and communicates through private stdin/stdout pipes. The container has networking disabled, drops Linux capabilities, and mounts only the selected voice model and index read-only. Distrobox installations call host Podman through `distrobox-host-exec`.

Command-line setup:

```bash
voice-warp --install-ai
```

Model files are not bundled with releases. Use files from a trusted creator and check their usage terms. Legacy `.pth` checkpoints use Python serialization; loading them can execute code inside the worker. ONNX models must carry the RVC metadata supported by the pinned inference code. Arbitrary ONNX networks are not voice models.

## Performance and current scope

The first bundled runtime is CPU-only on Linux x86-64. Standard HuBERT RVC v1/v2 PyTorch models and compatible RVC ONNX exports are supported. Custom embedders, GPU acceleration, training, and exporting models are not included. An optional index must match the voice model's feature dimensions; matching dimensions alone do not prove it belongs to that voice.

Microphone audio is 48 kHz mono PCM. The worker performs resampling, HuBERT feature extraction, RMVPE pitch detection, RVC inference, and streaming overlap alignment. Startup warms up real inference before reporting ready. Audio responses must match the request sequence and sample count. Inference errors, malformed replies, process exits, or timeouts stop AI output; no raw microphone fallback is used. Stop cancels pending requests and removes the worker container.

An 8192-sample chunk contains about 171 ms of audio, before inference and other buffering. Throughput and speech quality still need listening tests on the user's chosen model and hardware. A passing synthetic test does not establish end-to-end speech quality.

## Logs and removal

The app displays its engine log directory in Local AI. On a default Linux installation it is `~/.config/voice-warp/rvc/`, containing `install.log` and `engine.log`. Setup failures often indicate unavailable Podman or a failed download. Invalid model errors are reported in English in the app, with detailed dependency diagnostics in the log.

Stop Voice Warp before removing the downloaded runtime:

```bash
podman image rm localhost/voice-warp-rvc:1
```

Your selected model files and app settings remain intact. The bundled engine is independent of any previously installed `voice-warp-ai` w-okada container.

## External server compatibility

The **External server** option retains the existing loopback-only w-okada v1 adapter, including on Windows/macOS. Its setup and protocol requirements are documented in [external AI setup](AI-EXTERNAL.md). The bundled engine does not need this server.

## Development and verification

The worker and build recipe live in `engine/` and are embedded with `include_str!`, so existing desktop package scripts carry them automatically. The worker reuses pinned upstream inference libraries without launching the upstream server. Attribution, weight origins, and licensing notes are in [engine notices](../engine/NOTICE.md).

```bash
cargo test --locked
podman build -t localhost/voice-warp-rvc:1 -f engine/Containerfile engine
VOICE_WARP_TEST_MODEL=/absolute/path/voice.onnx cargo test --locked rvc::tests::bundled_engine_integration -- --ignored --nocapture
```

The opt-in test loads a real model, converts generated tone blocks, checks nonzero output after warmup, and verifies container removal. Regular tests cover response bounds, invalid audio, latched failure, and cancellation of a stalled worker. Linux builds and local inference are tested separately from Windows/macOS; the bundled engine is not advertised as available on those platforms.
