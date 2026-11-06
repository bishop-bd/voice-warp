# AI runtime compatibility and qualification

Evidence recorded on 2026-11-05. This matrix distinguishes implemented inference paths, reproducible runtime profiles, and actual hardware qualification. A selectable backend or successful package build does not establish real-time speech quality. See [setup](AI.md) and [native packages](AI-RUNTIME-PACKAGES.md).

## Implemented paths and candidate versions

The worker is version `3.0.0`, using protocol 2. The streaming/synthesis code is pinned to w-okada commit `d8ef15799470193f7c8176ef471245753a656626`. The app owns the worker process and communicates over private pipes; users do not install a separate w-okada server. Voice models remain separate trusted inputs.

All modern profiles use Python 3.12, NumPy 1.26.4, SciPy 1.13.1, Numba 0.61.2 and resampy 0.4.3. Exact wheel URLs, hashes and dependency locks are in [`engine/profiles`](../engine/profiles) and [`engine/locks`](../engine/locks). Fairseq and its training/configuration dependencies are no longer needed for inference.

| Platform / backend | Candidate inference packages | Implemented execution path |
| --- | --- | --- |
| Linux x86-64 CPU | Torch/torchaudio 2.9.1 CPU; ORT 1.23.2 | Podman worker; Torch or exported ONNX models on CPU. |
| Linux x86-64 NVIDIA CUDA | Torch/torchaudio 2.9.1 CUDA 12.8; ORT GPU 1.23.2 | Torch CUDA and CUDA ONNX, with actual stage execution checks. |
| Linux x86-64 AMD ROCm | AMD Torch 2.9.1 / torchaudio 2.9.0 ROCm 7.2.1; CPU ORT 1.23.2 | Torch HuBERT/synthesis on ROCm; ONNX pitch on CPU. |
| Linux x86-64 AMD MIGraphX | Same AMD Torch stack; ORT MIGraphX 1.23.2 | Explicit MIGraphX HuBERT/synthesis, CPU RMVPE pitch; requires verified provider kernels. See graph compatibility findings below. |
| Windows x64 CPU | Torch/torchaudio 2.9.1 CPU; ORT 1.23.2 | Native managed worker; CPU Torch or ONNX. |
| Windows x64 NVIDIA CUDA | Torch/torchaudio 2.9.1 CUDA 12.8; ORT GPU 1.23.2 | Native Torch CUDA and CUDA ONNX. |
| Windows x64 AMD ROCm | AMD Torch/torchaudio 2.9.1 ROCm 7.2.1; CPU ORT 1.23.2 | Native Torch ROCm; CPU ONNX pitch. Exact supported GPU/driver combinations still require validation. |
| Windows x64 AMD/NVIDIA DirectML | CPU Torch 2.9.1; ORT DirectML 1.23.0 | ONNX HuBERT, RMVPE and RVC on the selected DXGI adapter; CPU partitions reported. |
| macOS arm64 CPU / MPS | Torch/torchaudio 2.9.1; ORT 1.23.2 | CPU or Torch MPS HuBERT/synthesis; ONNX pitch on CPU. Hidden MPS fallback is disabled. |
| macOS arm64 Core ML | Torch/torchaudio 2.9.1; ORT 1.23.2 | Exported ONNX stages with Core ML CPU-and-GPU compute policy. Physical GPU execution is not established by provider profiling. |
| macOS x86-64 CPU / Core ML | Torch/torchaudio 2.2.2; ORT 1.23.2 | Separate Intel native profiles, with CPU or exported Core ML stages. Intel AMD GPU hardware qualification remains pending. |

These are implemented candidate profiles, not a claim that each native artifact has been built, signed, installed on a clean machine, and tested on every listed GPU. No Windows/macOS hardware was available for the validation below. macOS NVIDIA CUDA, Windows/Linux ARM64, and Intel Mac MPS do not have managed runtime profiles. The trusted catalog must contain real published artifacts before automatic native downloads can be advertised.

## Modern HuBERT and trusted checkpoint loading

`voice_warp_hubert.py` uses torchaudio's HuBERT implementation and official Fairseq state-dictionary conversion. RVC v1 uses transformer layer 9 plus the checkpoint's 256-channel projection; v2 uses layer 12 with 768 channels. The shared checkpoint is checked against its pinned SHA-256 before a scoped deserializer handles its unused training dictionary. Arbitrary Fairseq classes are rejected. There is no global `torch.load` override or bypass of Python dataclass validation.

User-selected legacy voice `.pth` files remain trusted executable input and are loaded explicitly with `weights_only=False`. The worker validates standard RVC v1/v2 architecture/weights and removes the training-only posterior encoder. The inference adapter avoids the upstream managers that eagerly import Fairseq and unrelated pitch/training packages.

This removes the earlier Python 3.12 migration blocker: Fairseq 0.12.2's mutable dataclass defaults are not patched because Fairseq is absent from the inference path. AMD's Windows wheel/driver and exact-device requirements still apply; removing a Python dependency does not qualify a Windows GPU.

## ONNX export and execution reporting

`export_models.py` exports standard trusted RVC v1/v2 voice checkpoints and matching HuBERT graphs. It validates CPU numerical parity at multiple sequence lengths, preserves stochastic voice operators, records source/toolchain hashes, and writes an immutable content-keyed bundle containing `voice.onnx`, `hubert.onnx`, `rmvpe.onnx`, and `export.json`. The worker verifies bundle file hashes and HuBERT feature metadata before use. DirectML, Core ML and MIGraphX require an ONNX voice; the app's export action provides that route from supported `.pth` files.

Before ready, a voiced warmup must execute HuBERT and synthesis. Torch model parameters must reside on the selected device. GPU ONNX sessions must initialize their requested provider and execute kernels on it, verified from ONNX Runtime profiling; CPU partitions are reported. Explicit requests fail instead of silently falling back. A ROCm/MPS runtime with CPU-only ORT cannot advertise an all-ONNX bundle as GPU accelerated merely because an intermediate tensor was allocated on a GPU.

Core ML is deliberately different: `CPUAndGPU` is a compute policy, and Core ML provider events do not identify which physical compute unit executed each node. Its ready response reports `gpu_verified: false` and says physical GPU use was not measured. Actual Intel AMD/Apple GPU qualification requires platform profiling and model tests.

FAISS/index lookup, resampling and overlap alignment remain CPU work. Discovery returns `model_verified: false`. CUDA/ROCm UUIDs identify devices where available; ambiguous duplicate names fail. DirectML LUIDs are boot-scoped and may require reselection after reboot.

## Executed numerical and streaming tests

Host: Linux x86-64, AMD Ryzen AI 9 HX PRO 370, kernel 7.0.0-42-generic. Tests use generated audio and local trusted voice models; no microphone audio was captured and model weights are not redistributed.

| Test | Observed result |
| --- | --- |
| Fairseq-to-torchaudio HuBERT parity, legacy CPU runtime | All 12 cases matched exactly (`max_abs=0`): tones and seeded noise at 8192/16000/32160 input samples, both v1 layer9/projection and v2 layer12. Torch 2.0.1 CPU / torchaudio 2.0.2 / Fairseq 0.12.2 reference. |
| Modern CPU HuBERT parity against that reference | All 12 cases passed at `atol=5e-5, rtol=3e-4`; largest observed absolute difference `5.895e-5`. Python 3.12.3, Torch/torchaudio 2.9.1 CPU; Fairseq is absent. |
| Modern ROCm HuBERT parity on Radeon 890M | All 12 cases passed against the same Fairseq reference at `atol=5e-5, rtol=3e-4`; largest observed absolute difference `5.800e-5`. Torch `2.9.1+rocm7.2.1.gitff65f5bc`, device `cuda:0` (ROCm namespace). No experimental attention override enabled. |
| New inference adapters, trusted Dexter `.pth` | Warmup about 1784 ms; four 8192-sample blocks returned correctly; streaming history starts silent, later blocks nonzero. Legacy CPU package environment, with new Fairseq-free modules mounted read-only. |
| Modern CPU complete worker | Version-3 CPU image, Python 3.12.3 / Torch 2.9.1 / ORT 1.23.2. Real Dexter warmup 1989 ms; four valid 8192-sample blocks, later blocks nonzero; about 151–174 ms processing per block. Explicit shutdown and closed-stdin probe passed. |
| RVC ONNX export | Dexter and ClevelandBrown standard RVC v2, pitch enabled, 48 kHz, 109 speakers: CPU parity passed at 32/64/113/192 feature frames. |
| HuBERT ONNX export | Both v1 and v2 passed at four waveform lengths. Observed maximum absolute error about `3.2e-5` (v1) / `5.1e-5` (v2). |
| Modern CPU exporter integration | All five exporter tests passed on `:3-cpu` in about 13.3 seconds, including real voice/HuBERT exports at four dynamic lengths and immutable cache reuse. |
| Full ONNX worker pipeline | Real Dexter export bundle; HuBERT/RMVPE/RVC CPU; warmup 2286 ms, four valid 8192-sample blocks, later blocks nonzero; about 155–166 ms processing per block. |

The Dexter source SHA-256 is `1c0cc13dfb9d8c2ae3be9de74053a1e8cd90dadf2baf2392166d3052a423dbd0`. Shared HuBERT SHA-256 is `f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96`; RMVPE is `84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219`.

An earlier complete protocol-2 CPU image (`localhost/voice-warp-rvc:2-cpu`, image `9b41c514d7c083628fedba879d32b8c6904094d6d0aa93f7431b03c765a3abaa`) passed the real-model Rust processor integration test. A 64-block CLI run with the matching index enabled measured processing p50 152.47 ms, p95 158.56 ms, p99/max 164.09 ms at 8192 samples and 48 kHz. This reference image uses the legacy dependency set; its results do not qualify a new version-3 artifact. It also does not meet the proposed 20% processing headroom target at a 170.67 ms block duration.

An independent stdin reader terminates the worker on owner-pipe EOF even while imports/inference occupy its main thread. Actual local Podman/conmon tests removed the container in about 0.09 seconds on input EOF during initialization, and 0.15 seconds when the attached host Podman client was killed. Explicit shutdown was acknowledged, and one-shot probes still accepted immediately closed stdin. These establish local Linux ownership behavior, not Windows/macOS lifecycle qualification.

The same ownership tests passed again against the modern `:3-cpu` image: approximately 0.088 seconds for initialization EOF and 0.149 seconds for host-client termination. The modern CPU smoke ran with current worker modules mounted read-only; it validates the modern dependency/runtime combination and worker behavior, not every subsequent package artifact.

The complete modern ROCm image `localhost/voice-warp-rvc:3-rocm` (image prefix `d03c4c7e`) passed the real-Dexter Rust integration test in about 16.87 seconds, including model warmup, valid streaming output and owned-container cleanup. The test used the Radeon 890M (`gfx1150`) through `/dev/kfd` and `/dev/dri`, on the host's Ubuntu 26.04/kernel 7.0.0-42 environment. HuBERT and Torch synthesis ran on ROCm; RMVPE remained CPU. MIOpen emitted workspace fallback warnings, and the pipeline completed successfully. This records functional GPU conversion on this actual host; it does not establish official support for that host OS/kernel, sustained latency or speech quality.

Dependency-free tests cover protocol bounds, backend selection, file/bundle validation, CPU/MPS isolation, required ONNX kernels, honest Core ML reporting, acceleration checks, export configuration, and termination while initialization is blocked:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s engine -p 'test_*.py'
```

`engine/verify_hubert_parity.py` can write a synthetic reference in the old Fairseq runtime and compare it in a modern runtime without Fairseq. References contain generated input/features and asset metadata, not model weights. Use `--device cuda:0` for ROCm/CUDA comparison once the matching runtime is installed.

## Radeon 890M convolution policy

The initial ROCm speech run was functionally correct but too slow: p95 359.80 ms at a 170.67 ms block duration, with all 60 blocks missing their deadline. Synchronized stage measurements isolated approximately 309–321 ms in voice synthesis, 22–26 ms in HuBERT, and 11–14 ms in CPU pitch extraction. MIOpen logged that GEMM solvers required workspace but received zero bytes. Upstream reports describe the same `gfx1150` workspace problem and missing performant solver paths in ROCm 7.2.1; these reports support the diagnosis, although the measurements do not establish every internal solver choice. [PyTorch workspace issue](https://github.com/pytorch/pytorch/issues/178934), [ROCm gfx1150 solver issue](https://github.com/ROCm/rocm-libraries/issues/6045).

The worker applies a targeted convolution policy only for Linux, ROCm backend, Torch 2.9.1 with ROCm 7.2.1, and a device reporting the actual `gfx1150` architecture. It disables MIOpen convolution dispatch using `torch.backends.cudnn.enabled=False`; PyTorch's convolution dispatch checks that flag for MIOpen and selects native ATen GPU operations when disabled. GPU identity and float32 precision are unchanged. Other devices, operating systems and runtime versions retain their framework defaults. The ready response exposes `torch_policy` and an explicit warning. [PyTorch 2.9.1 convolution dispatch](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/Convolution.cpp).

With this policy, five measured steady blocks averaged 80.50 ms for synthesis and 119.20 ms for the complete pipeline, compared with roughly 313 ms and 353 ms using MIOpen. GPU profiling showed native GPU kernels. All 12 HuBERT comparisons against the Fairseq reference passed again, with maximum absolute error `5.800e-5`. Direct voice synthesis comparisons between MIOpen and native GPU convolutions passed at 32/64/113/192 feature frames, with model noise held at zero for numerical comparison: largest absolute difference `1.759e-6`, mean differences below `1e-7`. The opt-in reproducer is `engine/verify_convolution_policy.py`. These short measurements motivate the scoped policy; sustained speech benchmarks and listening checks remain separate qualification work.

Auto selection can request eight timed blocks after warmup and after ONNX profiling has ended. The worker returns their median and maximum processing times, then clears its streaming history before live input. This short measurement supports rejecting obviously unsuitable candidates; it is not a sustained latency guarantee.

## MIGraphX graph compatibility

The real exported Dexter bundle exposed two separate first-run issues on the Radeon 890M. All three ONNX sessions initialized in about four seconds, but the first RMVPE call remained inside GPU kernel compilation after three minutes. A debugger showed MIGraphX's compilation manager waiting for the AMD COMGR compiler thread. The worker therefore explicitly runs RMVPE on CPU for this backend; measured pitch processing was about 11 ms. This is a reported stage policy, not silent fallback. HuBERT completed its first MIGraphX compile/run in about 22.49 seconds.

Voice synthesis initially failed an internal `Concat` rank check. The original ONNX graph passed both ONNX shape inference and the MIGraphX parser with the actual input shapes. Reproducing ONNX Runtime's basic optimization on CPU isolated the problem: it folds an empty shape vector into a zero-length initializer, which MIGraphX 2.15 treats as a scalar. Removing the 24 statically empty concatenation operands from that diagnostic optimized graph made parsing succeed. The production worker preserves the original graph and disables ONNX Runtime graph optimization only for MIGraphX; MIGraphX retains its own compiler optimizations.

With this policy, the real voice graph compiled and ran successfully in 88.76 seconds. Three further 28-frame synthesis calls took 34.6, 29.3 and 30.9 ms, returning 13,440 finite, nonzero samples each. ONNX profiling verified `MIGraphXExecutionProvider` kernels. The full speech-worker result appears below; sustained MIGraphX latency remains a separate qualification.

The installed provider can compile at first inference and recompile when shapes change; its documented compiled-model cache is a separate facility from Voice Warp's validated ONNX export cache. Compiled-cache persistence is not enabled by this implementation, so these cold-start costs can recur. [ONNX Runtime MIGraphX provider and cache settings](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html), [ONNX Runtime MIGraphX compilation path](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc).

## Qualification still required

The implemented profiles and export paths are candidates until their actual artifact, OS, hardware, driver, model hashes, precision/stage placement and test results are recorded. Windows/macOS package signing, clean-machine installs and native audio/lifecycle tests require those operating systems. Core ML needs physical-compute evidence; CUDA/DirectML/MPS need successful complete-model runs on their target GPUs; the local AMD ROCm/MIGraphX results do not qualify other devices. Every advertised combination also needs speech listening/parity, a sustained run with p95/p99 latency and memory, device-loss/recovery, and audio-routing tests. Short generated-tone smoke runs do not establish those results.

## Speech timing checks on the local Radeon machine

The licensed VOiCES sample distributed in the [torchaudio speech tutorial](https://docs.pytorch.org/audio/2.5.0/tutorials/speech_recognition_pipeline_tutorial.html) supplies real speech (3.4 seconds, repeated explicitly). The original asset SHA-256 is `c65fcd726d6b08c82c1e5dc7558f863cd8d483e3ed2f4a7bcf271dc1865ada14`; the 48 kHz mono resampling, with one interpolation overshoot clipped, hashes to `aa76912ac9f661bd744e6cfd3d36fba0fc5246b3d442f4f319c6a589fa885c02`. No microphone was opened.

Short runs use the same trusted Dexter checkpoint, 8192-sample blocks, float32, pitch 0, and no index. Each processes 60 blocks / 10.24 seconds. Output WAVs are retained locally for listening, but no listening or end-to-end routing qualification is claimed.

| Runtime | Processing p50 / p95 / p99 (ms) | Blocks missing 170.67 ms deadline | Result |
| --- | --- | --- | --- |
| Modern CPU | 142.32 / 146.29 / 156.73 | 0 / 60 | Below block duration, fails proposed 20% headroom |
| Unmodified ROCm/MIOpen | 345.45 / 359.80 / 362.32 | 60 / 60 | Too slow for this configuration |
| Scoped ROCm ATen GPU convolution policy | 127.10 / 133.09 / 136.83 | 0 / 60 | Short timing thresholds pass; sustained run recorded separately |

The corrected short run used image `1378d11e727c6edb8e5bd18f59bbfe650b72de2e17abfc0a0be304a6551d0328`. Its HuBERT and synthesis stages ran on the Radeon GPU; pitch, resampling and overlap remained CPU. Startup including warmup was 4285 ms. This short measurement overlapped native packaging work, so it is a diagnostic comparison rather than a tightly isolated hardware benchmark.

The complete MIGraphX worker subsequently passed the same 60-block speech run using the exported voice bundle. HuBERT and synthesis reported actual `MIGraphXExecutionProvider` work; RMVPE, overlap and resampling reported CPU. Startup including first graph compilation took 115.08 seconds. Processing p50/p95/p99 was 130.56/148.36/160.97 ms with zero deadline misses. The 20% headroom criterion did not pass. This uses image `7aeb3b3fafa9e6d7798506b49b14772c922b90b7f85a7b5bfcc0ac7212c9d6ef`; it establishes functional exported-model streaming, not a general performance advantage over ROCm or native-platform qualification.

## Frozen packaging verification on Linux

The shared native builder was exercised with a Linux test target: a frozen worker passed actual HuBERT/RMVPE/synthetic RVC warmup, relocation, final ZIP extraction/hash verification, and voice ONNX export with numerical parity and clean exit. The exact extracted package then passed inference, an 8192-sample conversion block, shutdown and export in a bare Ubuntu image with no Python or external upstream source, a read-only payload, and only its private runtime on `PATH`. The deterministic random voice validates packaging; it is not a trained speech-quality fixture.

The local artifact is `target/frozen-smoke-output/linux-frozen-smoke-3b.zip`; its manifest SHA-256 is `c56015d4210dadf3a7479a1434cf472b262ff275fa234f761732078ae4893c6b`. This portable-freezer evidence does not replace Windows DLL, macOS signing, or native hardware checks. The Windows/macOS workflow now requires equivalent synthetic CPU inference/export checks for each candidate artifact. The Linux CUDA image also builds and probes successfully with PyTorch 2.9.1+cu128 and ONNX Runtime GPU 1.23.2; no NVIDIA GPU is present, so CUDA model execution remains unverified here.

## Sustained Radeon speech result

The same Dexter/PyTorch/ROCm configuration completed 3,516 paced blocks: **600.064 seconds of real speech input** (the disclosed 3.4-second reference repeated), float32, 8192 samples/block, no index, pitch 0. No other test/build GPU workloads ran during this measurement. Processing p50/p95/p99 was **126.24 / 131.91 / 141.10 ms** against 170.67 ms/block. The proposed p95 <80% of block and p99 <one block thresholds both passed. **One block** exceeded the deadline (maximum 172.43 ms); 62 blocks exceeded the 80% budget. There were no clipped output samples. This establishes the stated inference timing result, not zero-dropout live audio or listening quality.

Worker RSS measured every 10 seconds remained exactly 2,022,316 KiB (about 1.93 GiB) across all 60 observations; the reported high-water mark was 2,111,436 KiB. This is Linux process memory, not separate GPU-allocation accounting. The benchmark released its worker after completion. Reports and audio are retained locally under `target/ai-validation/rocm-speech-600s*`; source voice weights are not committed.

Additional final checks confirmed Auto selects Radeon with eight-block median 116.52 ms, and that an unavailable saved GPU UUID falls back to CPU with explicit reasons and a fresh warmup (median 136.60 ms). The second trusted Cleveland voice also passed eight real-speech blocks with its matching index and pitch +7; p95 was 128.11 ms, with GPU HuBERT/synthesis and CPU index lookup explicitly reported.
