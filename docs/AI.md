# AI voices in Voice Warp

Voice Warp can own a local RVC worker and communicate with it through private process pipes. It reuses pinned w-okada inference libraries without starting the upstream web server or browser UI. The optional **External server** mode remains available; its separate setup is described in [external AI setup](AI-EXTERNAL.md).

## What is available

The managed runtime now has platform launchers, versioned device reporting, cancellation, and backend selection. These implementation paths have different validation and distribution status:

| Platform | Runtime paths | Current limits |
| --- | --- | --- |
| Linux x86-64 | Podman CPU, NVIDIA CUDA, AMD ROCm | CPU is the existing inference baseline. GPU recipes and device access need qualification on the exact hardware/driver combination. |
| Windows x64 | Native CPU, CUDA, DirectML packages | Native package builder and installer are implemented. No qualified native download is in the trusted catalog yet. DirectML requires an ONNX voice and retains CPU stages. |
| macOS Apple Silicon | Native CPU and Metal/MPS packages | Candidate packages must be built and qualified on a Mac; no qualified download is in the catalog yet. |
| macOS Intel | Native CPU package candidate | Dependencies and packaging need independent qualification. AMD GPU support is not released; current NVIDIA CUDA is unavailable on macOS. |

Native Windows ROCm packaging, MIGraphX/Core ML integration, full ONNX export tooling, and Intel Mac AMD GPU qualification remain pending. GPU capability enumeration is not proof that a model works or meets real-time performance targets. The platform audio guides also retain their real-device verification limits.

## Linux setup

1. Open **Local AI → Bundled engine**. Choose **CPU**, **NVIDIA · CUDA**, or **AMD · ROCm** under **Processing**.
2. Choose **Install / repair AI engine**. Host Podman is required. Setup downloads the selected runtime and checksum-verified shared weights, potentially several GB. Logs appear at the displayed `install.log` path. **Cancel** stops pending setup or startup.
3. Choose **Check available devices** to inspect the installed runtime. Select a returned device ID or leave `0` for its first device. Explicit GPU selection fails if the requested device/provider is unavailable; it does not silently switch to CPU.
4. Choose a standard HuBERT RVC v1/v2 `.pth` or compatible metadata-bearing `.onnx` model, and optionally its matching `.index`. Files remain at their selected locations.
5. Set pitch, index strength, and chunk size. Start with 8192 samples on CPU. Choose **Test voice engine** to perform model warmup without opening the microphone. Check its reported device and stage placement.
6. Select **AI voice**, start the microphone, and select **Voice Warp** as the input in your call or game.

**Auto** currently uses the installed CPU runtime on Linux. Select and install a vendor backend explicitly to try GPU processing. It does not download or discover another runtime automatically. Changing backend resets the device selection; stop before changing model or processing settings.

Equivalent commands:

```bash
voice-warp --install-ai --ai-backend cpu
voice-warp --ai-devices --ai-backend cpu
voice-warp --ai-doctor --ai-backend cpu --ai-device 0
voice-warp --ai-benchmark --model /absolute/path/voice.pth --ai-backend cpu --blocks 64
```

For CUDA or ROCm, substitute `cuda` or `rocm` in the setup/probe commands. CUDA containers require a compatible host driver and NVIDIA Container Toolkit/CDI access. ROCm containers require a compatible host AMD stack, `/dev/kfd`, render devices, and permissions for rootless Podman. Distrobox installations continue to call host Podman through `distrobox-host-exec`. Installing user-space libraries in a container does not add host GPU support. [NVIDIA CDI](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html), [AMD container device access](https://rocm.docs.amd.com/en/docs-10.1.0/install/docker-containers.html)

The CUDA recipe pairs PyTorch 2.2.2/CUDA 11.8 with ONNX Runtime 1.18.0. The ROCm recipe uses PyTorch 2.5.1/ROCm 6.2 and CPU ONNX Runtime. These are explicit compatibility candidates for the pinned RVC code, not universal GPU support. In particular, the ROCm 6.2 recipe has not been qualified for Strix Radeon 880M/890M APUs; do not infer support from the AMD selector alone.

## Windows and macOS managed setup

The app launches a native worker rather than requiring WSL, Docker, Podman, system Python, or a separately installed w-okada server. The runtime itself is a separate platform-specific artifact.

**The trusted download catalog is currently empty.** The normal install button reports that no published runtime is available on native platforms. It will become a one-step download when qualified artifacts and their trusted manifest hashes are included in a desktop release. Until then, use an explicitly trusted local candidate package built with the [native runtime packaging workflow](AI-RUNTIME-PACKAGES.md):

```text
voice-warp --install-ai --ai-runtime-manifest /path/to/build.json --ai-manifest-sha256 TRUSTED_MANIFEST_SHA256
```

After installing the correct OS/architecture/backend package, choose its backend in **Local AI**, select a model, and use **Test voice engine**. A successful device probe alone does not qualify a package. Keep the existing external server mode if you already have a working setup.

Native runtime files live in the platform's per-user application data directory. Installation checks the trusted manifest digest, archive and per-file hashes, platform/protocol compatibility, safe extraction, and a staged worker probe before activation. The previous build is retained for rollback. See [native runtime packages](AI-RUNTIME-PACKAGES.md) for clean-machine checks, signing, artifact publication, and removal.

The Windows/macOS audio route is unchanged: install a virtual cable or loopback driver separately and select its playback endpoint in Voice Warp. See [Windows](WINDOWS.md) or [macOS](MACOS.md).

## Execution stages and model compatibility

GPU processing may be mixed with CPU processing. The loaded worker reports its actual selected backend, device, precision, runtime version, and stage placement after non-silent warmup. CPU FAISS/index retrieval, resampling, or pitch processing can remain present in a GPU session.

- PyTorch voices (`.pth`) use the selected CPU/CUDA/ROCm/MPS implementation when available.
- ONNX voice sessions select a compatible provider separately. CUDA and DirectML can accelerate supported ONNX graphs; provider partitioning can still place nodes on CPU.
- DirectML retains CPU HuBERT and requires a compatible ONNX voice. Installing DirectML does not convert an arbitrary `.pth` model.
- ROCm/MPS ONNX stages can remain CPU stages. Inspect the reported placement instead of treating the backend name as an all-GPU claim.

Model files are supplied by the user. Use trusted creators and check usage terms. Legacy `.pth` serialization can execute code while loading. A native worker runs with the user's permissions; private pipes are not an operating-system sandbox. Linux retains network-disabled containers, dropped capabilities, and read-only selected-model mounts; that isolation claim does not apply to native workers. Shared weights and dependencies have their own terms, described in [engine notices](../engine/NOTICE.md).

The worker supports standard HuBERT RVC v1/v2 models and checks index feature dimensions. A matching dimension does not prove that an index belongs to the selected voice. Custom embedders, training, and automatic model export are not implemented.

## Performance and failures

Microphone audio is 48 kHz mono PCM. The worker performs resampling, feature extraction, pitch detection, RVC synthesis, and streaming overlap alignment. Replies must have the correct sequence and exact sample count. Invalid audio, missing providers, inference errors, disconnects, and timeouts stop AI output; raw microphone audio is never substituted for failed conversion.

An 8192-sample chunk contains about 171 ms of audio before processing and other buffering. Larger chunks improve throughput at the cost of latency. GPU speedup does not eliminate chunk accumulation or playback buffering. Test real speech through the actual call/loopback route before relying on a model/device combination.

`--ai-benchmark` uses generated tones and reports startup/warmup plus p50/p95/p99 processing time. It does not capture microphone audio, measure end-to-end latency, or verify speech quality. Optional controls include `--block-size`, `--pitch`, `--index`, `--index-ratio`, `--blocks`, `--ai-device`, and `--report FILE`. Ctrl-C cancels AI setup, probes, and benchmarks.

Stop cancels pending work and releases the owned worker. Window close also cancels pending AI operations and waits for cleanup. Settings changes invalidate previous model-test results; a late result from a canceled operation cannot activate a new microphone session.

## Logs, runtime versions, and removal

The app displays its log directory in Local AI. On a default Linux installation this is `~/.config/voice-warp/rvc/`, containing `install.log` and `engine.log`. Logs report dependency/provider failures; microphone audio is not recorded by default.

Linux backend images are independently named:

```text
localhost/voice-warp-rvc:2-cpu
localhost/voice-warp-rvc:2-cuda
localhost/voice-warp-rvc:2-rocm
```

Stop the app before removing an image you no longer need, for example:

```bash
podman image rm localhost/voice-warp-rvc:2-cpu
```

The previous `:1` image is not overwritten or automatically upgraded. Reinstall the selected backend with the updated desktop application because the worker/recipe and protocol have changed. Your models, index files, and settings remain intact. The bundled worker remains independent of any `voice-warp-ai` external-server container.

## Development and verification

The worker and Linux recipes are embedded in the desktop executable. Native package build inputs are in `scripts/build-ai-runtime.py`; candidate artifact CI is manually dispatched through `.github/workflows/ai-runtime.yml`.

```bash
cargo test --locked
python3 -m unittest discover -s engine -p 'test_*.py'
podman build --build-arg AI_BACKEND=cpu -t localhost/voice-warp-rvc:2-cpu -f engine/Containerfile engine
```

The ignored real-engine integration test requires an installed runtime and a trusted model in `VOICE_WARP_TEST_MODEL`. Portable fake-worker tests exercise startup cancellation, protocol validation, and failure behavior without a model. Native artifact qualification additionally requires clean-machine installation, signing/quarantine checks, real speech, long-session memory/latency measurement, and process cleanup on each supported hardware/OS combination. Automated unit tests do not establish those release gates.
