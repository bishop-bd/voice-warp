# AI voices in Voice Warp

Voice Warp can own a local RVC worker and communicate with it through private process pipes. It reuses pinned w-okada inference libraries without starting the upstream web server or browser UI. The optional **External server** mode remains available; its separate setup is described in [external AI setup](AI-EXTERNAL.md).

## What is available

The managed runtime now has platform launchers, versioned device reporting, cancellation, and backend selection. These implementation paths have different validation and distribution status:

| Platform | Runtime paths | Current limits |
| --- | --- | --- |
| Linux x86-64 | Podman CPU, NVIDIA CUDA, AMD ROCm and MIGraphX | CPU is the existing inference baseline. GPU recipes and device access need qualification on the exact hardware/driver combination. |
| Windows x64 | Native CPU, CUDA, ROCm and DirectML packages | Native build profiles and installer are implemented. Native builds and GPU hardware qualification remain release gates. DirectML requires an ONNX voice. |
| macOS Apple Silicon | Native CPU, Metal/MPS and Core ML packages | Candidate packages must be built and qualified on a Mac; no qualified download is in the catalog yet. |
| macOS Intel | Native CPU and experimental Core ML candidates | Dependencies and packaging need independent qualification. AMD GPU support is not released; current NVIDIA CUDA is unavailable on macOS. |

The runtime profiles and ONNX export path are implemented. Intel Mac AMD GPU qualification remains open, and Core ML reports provider execution without claiming that its internal GPU placement was verified. See [the evidence matrix](AI-COMPATIBILITY.md) for exact tested and unverified combinations. GPU capability enumeration is not proof that a model works or meets real-time performance targets. The platform audio guides also retain their real-device verification limits.

## Linux setup

1. Open **Local AI → Bundled engine**. Choose the desired backend under **Processing**. Start with **CPU**, or choose **NVIDIA · CUDA**, **AMD · ROCm**, or **AMD · MIGraphX** for compatible hardware.
2. Choose **Install / repair AI engine**. Host Podman is required. Setup downloads the selected runtime and checksum-verified shared weights, potentially several GB. Logs appear at the displayed `install.log` path. **Cancel** stops pending setup or startup.
3. Choose **Check available devices** to inspect the installed runtime. Select a returned device ID or leave `0` for its first device. Explicit GPU selection fails if the requested device/provider is unavailable; it does not silently switch to CPU.
4. Choose a standard HuBERT RVC v1/v2 `.pth` or compatible metadata-bearing `.onnx` model, and optionally its matching `.index`. Files remain at their selected locations.
5. Set pitch, index strength, and chunk size. Start with 8192 samples on CPU. Choose **Test voice engine** to perform model warmup without opening the microphone. Check its reported device and stage placement.
6. Select **AI voice**, start the microphone, and select **Voice Warp** as the input in your call or game.

**Auto** tests installed GPU runtimes before CPU, loading and warming up the selected voice for each attempt. It measures eight additional warmup blocks, rejects a backend whose median processing time exceeds the audio block duration, shows why an earlier choice failed, and never changes backend during streaming. Install the desired GPU runtime explicitly first; Auto does not download additional runtimes. Installing Auto installs the CPU baseline. Changing backend resets the device selection; stop before changing model or processing settings.

Equivalent commands:

```bash
voice-warp --install-ai --ai-backend cpu
voice-warp --ai-devices --ai-backend cpu
voice-warp --ai-doctor --ai-backend cpu --ai-device 0
voice-warp --ai-benchmark --model /absolute/path/voice.pth --ai-backend cpu --blocks 64
```

For CUDA or ROCm, substitute `cuda` or `rocm` in the setup/probe commands. CUDA containers require a compatible host driver and NVIDIA Container Toolkit/CDI access. ROCm containers require a compatible host AMD stack, `/dev/kfd`, render devices, and permissions for rootless Podman. Distrobox installations continue to call host Podman through `distrobox-host-exec`. Installing user-space libraries in a container does not add host GPU support. [NVIDIA CDI](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html), [AMD container device access](https://rocm.docs.amd.com/en/docs-10.1.0/install/docker-containers.html)

Runtime version 3 uses Python 3.12 and a Fairseq-free HuBERT inference adapter. CUDA uses PyTorch 2.9.1/CUDA 12.8 and ONNX Runtime 1.23.2. ROCm uses AMD's pinned 7.2.1 image and PyTorch 2.9.1 with CPU ONNX stages; the separate MIGraphX profile supplies an AMD ONNX provider. Each profile has a complete dependency lock with hashes in `engine/profiles/` and `engine/locks/`. These are hardware-specific candidates; consult the evidence matrix before treating a combination as qualified.

## Windows and macOS managed setup

The app launches a native worker rather than requiring WSL, Docker, Podman, system Python, or a separately installed w-okada server. The runtime itself is a separate platform-specific artifact.

**The source catalog is empty until real artifacts are built.** The native workflow can assemble desktop packages with their CPU runtime and a compiled catalog that authenticates the included manifest. In that package, the normal install button installs offline without a separate download or manual hash. Additional backends can be included in the same way. A source-only desktop build needs an explicitly trusted candidate package from the [native runtime packaging workflow](AI-RUNTIME-PACKAGES.md):

```text
voice-warp --install-ai --ai-runtime-manifest /path/to/build.json --ai-manifest-sha256 TRUSTED_MANIFEST_SHA256
```

After installing the correct OS/architecture/backend package, choose its backend in **Local AI**, select a model, and use **Test voice engine**. A successful device probe alone does not qualify a package. Keep the existing external server mode if you already have a working setup.

Native runtime files live in the platform's per-user application data directory. Installation checks the trusted manifest digest, archive and per-file hashes, platform/protocol compatibility, safe extraction, and a staged worker probe before activation. The previous build is retained for rollback. See [native runtime packages](AI-RUNTIME-PACKAGES.md) for clean-machine checks, signing, artifact publication, and removal.

The Windows/macOS audio route is unchanged: install a virtual cable or loopback driver separately and select its playback endpoint in Voice Warp. See [Windows](WINDOWS.md) or [macOS](MACOS.md).

## Execution stages and model compatibility

GPU processing may be mixed with CPU processing. The loaded worker reports its actual selected backend, device, precision, runtime version, and stage placement after non-silent warmup. CPU FAISS/index retrieval, resampling, or pitch processing can remain present in a GPU session.

- PyTorch voices (`.pth`) use the selected CPU/CUDA/ROCm/MPS implementation when available.
- ONNX voices can use CUDA, DirectML, MIGraphX, or Core ML providers. Provider partitioning can still place nodes on CPU; the worker audits model execution during warmup.
- DirectML, MIGraphX and Core ML require an ONNX voice. A complete exported bundle includes HuBERT, RVC and the existing RMVPE graph. A standalone ONNX voice retains PyTorch HuBERT; this runs on CPU for DirectML/Core ML.
- Core ML requests its CPU-and-GPU compute policy, but its provider trace does not expose which internal compute unit ran each operation. The readiness report explicitly leaves GPU verification false.
- ROCm/MPS ONNX stages can remain CPU stages. Inspect the reported placement instead of treating the backend name as an all-GPU claim. First use of MIGraphX or Core ML can take several minutes to compile the voice; startup is cancellable and bounded to 15 minutes.

Model files are supplied by the user. Use trusted creators and check usage terms. Legacy `.pth` serialization can execute code while loading. A native worker runs with the user's permissions; private pipes are not an operating-system sandbox. Linux retains network-disabled containers, dropped capabilities, and read-only selected-model mounts; that isolation claim does not apply to native workers. Shared weights and dependencies have their own terms, described in [engine notices](../engine/NOTICE.md).

The worker supports standard HuBERT RVC v1/v2 models and checks index feature dimensions. A matching dimension does not prove that an index belongs to the selected voice. Custom embedders and training are not implemented.

## Export a voice for ONNX backends

Select a trusted `.pth` voice and choose **Export ONNX voice**, or run:

```bash
voice-warp --ai-export --model /absolute/path/voice.pth --ai-backend cpu --report export-result.json
```

Export runs in the installed managed worker. It preserves the original checkpoint and matching index, validates CPU outputs across dynamic input lengths, and caches a complete bundle under the app's `rvc/exports/` directory. The GUI selects its `voice.onnx`; CLI output names the same file. Keep `export.json`, `hubert.onnx`, and `rmvpe.onnx` beside it. The worker verifies every bundle hash before use. Cache keys include source hashes, exporter code and framework versions.

The parity report isolates random excitation for comparison while preserving randomness in the production voice graph. Passing export checks does not establish GPU compatibility or speech quality; run **Test voice engine** and a speech benchmark on the target backend. Cancel stops export along with other AI operations.

## Performance and failures

Microphone audio is 48 kHz mono PCM. The worker performs resampling, feature extraction, pitch detection, RVC synthesis, and streaming overlap alignment. Replies must have the correct sequence and exact sample count. Invalid audio, missing providers, inference errors, disconnects, and timeouts stop AI output; raw microphone audio is never substituted for failed conversion.

An 8192-sample chunk contains about 171 ms of audio before processing and other buffering. Larger chunks improve throughput at the cost of latency. GPU speedup does not eliminate chunk accumulation or playback buffering. Test real speech through the actual call/loopback route before relying on a model/device combination.

`--ai-benchmark` uses generated tones by default, or a 48 kHz mono WAV supplied with `--input-wav FILE`. It reports startup/warmup, p50/p95/p99 processing time and deadline misses. Use `--output-wav FILE` to retain converted audio for listening; the destination must not already exist. A WAV is processed once by default; explicit `--blocks` repeats it as needed. `--realtime` paces input blocks for sustained timing tests. It does not capture microphone audio, measure end-to-end latency, or verify speech quality. Optional controls include `--block-size`, `--pitch`, `--index`, `--index-ratio`, `--blocks`, `--ai-device`, and `--report FILE`. Ctrl-C cancels AI setup, probes, and benchmarks.

Stop cancels pending work and releases the owned worker. Window close also cancels pending AI operations and waits for cleanup. Settings changes invalidate previous model-test results; a late result from a canceled operation cannot activate a new microphone session.

## Logs, runtime versions, and removal

The app displays its log directory in Local AI. On a default Linux installation this is `~/.config/voice-warp/rvc/`, containing `install.log` and separate `engine-*.log` files for each worker. Errors and readiness reports identify the relevant log, so concurrent CLI operations do not overwrite each other. Logs report dependency/provider failures; microphone audio is not recorded by default.

Linux backend images are independently named:

```text
localhost/voice-warp-rvc:3-cpu
localhost/voice-warp-rvc:3-cuda
localhost/voice-warp-rvc:3-rocm
localhost/voice-warp-rvc:3-migraphx
```

Stop the app before removing an image you no longer need, for example:

```bash
podman image rm localhost/voice-warp-rvc:3-cpu
```

The previous `:1` and `:2-*` images are not overwritten or automatically upgraded. Reinstall the selected backend with the updated desktop application because the worker/recipe and protocol have changed. Your models, index files, and settings remain intact. The bundled worker remains independent of any `voice-warp-ai` external-server container.

## Development and verification

The worker and Linux recipes are embedded in the desktop executable. Native package build inputs are in `scripts/build-ai-runtime.py`; candidate artifact CI is manually dispatched through `.github/workflows/ai-runtime.yml`.

```bash
cargo test --locked
python3 -m unittest discover -s engine -p 'test_*.py'
podman build --build-arg AI_BACKEND=cpu -t localhost/voice-warp-rvc:3-cpu -f engine/Containerfile engine
```

The ignored real-engine integration test requires an installed runtime and a trusted model in `VOICE_WARP_TEST_MODEL`. Portable fake-worker tests exercise startup cancellation, protocol validation, and failure behavior without a model. Native artifact qualification additionally requires clean-machine installation, signing/quarantine checks, real speech, long-session memory/latency measurement, and process cleanup on each supported hardware/OS combination. Automated unit tests do not establish those release gates.
