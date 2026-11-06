# AI runtime compatibility and qualification

Evidence recorded on 2026-11-05. This matrix separates implemented code from built packages and hardware qualification. A selectable backend, successful package build, or listed GPU does not establish working real-time speech conversion. See [setup](AI.md) and [native package building](AI-RUNTIME-PACKAGES.md).

## Implemented paths

All paths use protocol 2 and the worker's `2.0.0` implementation. Inference code is pinned to w-okada commit `d8ef15799470193f7c8176ef471245753a656626`. User voice models are separate from the runtime.

| Platform / backend | Candidate runtime versions | Implementation and qualification |
| --- | --- | --- |
| Linux x86-64 CPU | Ubuntu 22.04/Python 3.10; torch 2.0.1+cpu; torchaudio 2.0.2; ONNX Runtime 1.13.1 | Worker and Podman path implemented. Real model inference tested as described below. This is the established reference path. |
| Linux x86-64 NVIDIA CUDA | Python 3.10; torch/torchaudio 2.2.2, CUDA 11.8; ONNX Runtime GPU 1.18.0 | Container recipe, CDI access, device selection and execution checks implemented. GPU container build, exact card/driver and speech qualification still required. |
| Linux x86-64 AMD ROCm | Python 3.10; torch/torchaudio 2.5.1, ROCm 6.2; CPU ONNX Runtime 1.18.0 | Container recipe and device access implemented. HuBERT/PyTorch synthesis target ROCm; ONNX stages remain CPU in this recipe. GPU hardware qualification still required. This does not establish support for Radeon 890M/Strix. |
| Windows x64 CPU | Python 3.10; torch/torchaudio 2.2.2 CPU; ONNX Runtime 1.17.3; FAISS CPU 1.8.0 | Native worker, builder, staged installer and supervision implemented. Native artifact and clean-machine qualification pending. |
| Windows x64 NVIDIA CUDA | Python 3.10; torch/torchaudio 2.2.2, CUDA 11.8; ONNX Runtime GPU 1.18.0; FAISS CPU 1.8.0 | Native candidate builder and GPU worker implemented. Native package build, driver/DLL loading and real GPU qualification pending. |
| Windows x64 AMD/NVIDIA DirectML | Python 3.10; torch/torchaudio 2.2.2 CPU; ONNX Runtime DirectML 1.17.3; FAISS CPU 1.8.0 | Native candidate builder and partial ONNX GPU path implemented. Requires an existing compatible RVC ONNX model. HuBERT remains CPU. DXGI/device/graph and speech qualification pending. |
| macOS arm64 CPU | Python 3.10; torch/torchaudio 2.2.2; ONNX Runtime 1.17.3; FAISS CPU 1.8.0 | Native worker, builder and installer implemented. Mac package and clean-machine qualification pending. |
| macOS arm64 MPS | Same framework versions as macOS CPU; candidate minimum macOS 12.3 | PyTorch GPU stages implemented, with ONNX stages on CPU. Hidden MPS CPU fallback is disabled before importing PyTorch. Actual Mac GPU/operator and speech qualification pending. |
| macOS x86-64 CPU | Python 3.10; torch/torchaudio 2.2.2; ONNX Runtime 1.17.3; FAISS CPU 1.8.0 | Separate native CPU build target implemented. Native dependency/package feasibility and speech qualification pending. |

The native builder preserves NumPy 1.23.5, SciPy 1.10.1, Numba 0.57.1 and Fairseq 0.12.2 from the baseline while changing the platform/provider packages listed above. Its resolved package inventory is written into each artifact. Candidate pins are deliberately explicit; they are not a recommendation to use these older frameworks for unrelated applications.

The trusted native download catalog is empty. No Windows/macOS native artifact is currently advertised as a published, qualified download. Explicitly trusted local candidate manifests can exercise the implemented installer. Publication requires actual package hashes, supported hardware/OS evidence, redistribution review, and signing checks; those cannot be replaced by invented URLs or manifests.

## What execution reporting establishes

The worker verifies a non-silent warmup before returning ready:

- HuBERT and synthesizer calls must execute, and PyTorch model parameters must reside on the selected device.
- GPU ONNX sessions must initialize their requested provider and execute kernels on it during warmup, verified from ONNX Runtime profiling. CPU graph partitions are reported.
- Explicit device requests fail when a device is missing or ambiguous. DirectML requires an ONNX voice; it does not convert `.pth` checkpoints.
- FAISS/index lookup, resampling and overlap alignment remain CPU stages. MPS and the current ROCm recipe retain CPU ONNX stages.

Discovery reports `model_verified: false`. Device names/UUIDs identify CUDA/ROCm candidates where available; duplicate names are rejected when ambiguous. DirectML uses DXGI adapter LUIDs, which are boot-scoped: a changed LUID requires reselection. These checks prevent misleading device labels; they do not prove speech quality or sustained performance.

## Executed CPU test

The protocol-2 worker and backend module were mounted read-only into the existing Linux CPU image, with a local trusted `.pth` model and networking disabled. This tested the new worker against the existing known runtime; it was not a fresh image-build or native-package test.

| Item | Observed value |
| --- | --- |
| CPU | AMD Ryzen AI 9 HX PRO 370 |
| Host | Linux x86-64, kernel 7.0.0-42-generic |
| Inference runtime | torch 2.0.1+cpu; ONNX Runtime 1.13.1; Python 3.10 environment |
| Model fixture SHA-256 | `1c0cc13dfb9d8c2ae3be9de74053a1e8cd90dadf2baf2392166d3052a423dbd0` |
| Input | Generated 180 Hz tone; 48 kHz mono PCM16; 8192 samples/block; pitch 0; index disabled |
| Warmup | About 2709 ms |
| Streaming | Four blocks returned exactly 8192 samples; initial streaming-history block silent, remaining three nonzero |
| Processing observations | Approximately 145–157 ms/block in this short smoke run |
| Reported placement | HuBERT CPU, RMVPE CPU, synthesis CPU, resampling/overlap CPU |

No microphone audio was captured. The model is not redistributed as a test fixture. This four-block observation is not a p95/p99 benchmark, ten-minute stability test, listening test, or end-to-end latency measurement. In particular, it does not establish the plan's sustained processing target of p95 below 80% of block duration.

The complete updated `engine/Containerfile` was subsequently built as `localhost/voice-warp-rvc:2-cpu` (image ID `9b41c514d7c083628fedba879d32b8c6904094d6d0aa93f7431b03c765a3abaa`). The Rust processor integration test passed against that image, including real inference, output shape/nonzero checks, and container removal.

The CLI also completed a 64-block generated-tone run with the same voice and its matching index enabled: 8192 samples/block, float32 CPU, approximately 3404 ms startup including 2699 ms warmup, processing p50 152.47 ms, p95 158.56 ms, p99/max 164.09 ms. This short run verifies the CLI and separate model/index paths through the container. It does **not** meet the plan's proposed 20% processing headroom target at a 170.67 ms block duration, and does not establish long-run real-time speech performance or listening quality.

After adding the independent input-pipe watchdog, an interactive repeat completed model warmup (about 2741 ms), four valid blocks, and acknowledged shutdown. Closing the input pipe during initialization removed the owned container in approximately 0.09 seconds. Killing the attached host Podman client during initialization removed its container in approximately 0.15 seconds. These tests exercised actual local Podman/conmon EOF forwarding while the worker was container PID 1; parent-PID polling alone would not establish that behavior. One-shot capability probes still succeeded with stdin closed immediately after the request. These are observations on this Linux host, not Windows/macOS lifecycle qualification.

Dependency-free Python tests cover framing, request validation, selection errors, CUDA-versus-ROCm identification, explicit CPU behavior on MPS systems, DirectML session requirements, actual-provider verification, worker error/EOF behavior, and pipe loss while initialization is blocked:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s engine -p 'test_*.py'
```

## Modern native Windows AMD ROCm: unresolved dependency migration

There is no native Windows ROCm package variant in the builder or catalog. The generic worker understands a HIP-backed PyTorch runtime, but this does not supply a compatible native runtime.

AMD's documented ROCm 7.2.1 Windows lane pairs selected Windows 11 GPUs with Python 3.12 and PyTorch 2.9; its installation page supplies CPython 3.12 wheels. The documentation also distinguishes PyTorch's included ROCm components from support for the entire ROCm stack. Recheck newer release-specific guidance when choosing a migration target. [AMD Windows matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html), [AMD Windows installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)

Two concrete conflicts prevent simply using those wheels with this project's current requirements:

1. Numba 0.57.1 explicitly supports Python versions below 3.12. Its pinned interpreter/NumPy/llvmlite set must be upgraded together. [Numba 0.57.1 version matrix](https://numba.readthedocs.io/en/0.57.1/user/installing.html#version-support-information)
2. Fairseq 0.12.2's `FairseqConfig` creates nested dataclass instances directly as field defaults. Python 3.11 and later reject unhashable defaults of this form. The installed baseline source was inspected at `fairseq/dataclass/configs.py`, including `common: CommonConfig = CommonConfig()`. [Fairseq 0.12.2 source](https://github.com/facebookresearch/fairseq/blob/v0.12.2/fairseq/dataclass/configs.py), [Python dataclass default rules](https://docs.python.org/3.12/library/dataclasses.html#mutable-default-values)

This small reproducer isolates the second conflict without installing any inference packages:

```python
from dataclasses import dataclass

@dataclass
class CommonConfig:
    seed: int = 1

@dataclass
class FairseqConfig:
    common: CommonConfig = CommonConfig()
```

On the available Python 3.14.4 interpreter this raises `ValueError` directing the caller to use `default_factory`; the Python 3.12 documentation specifies the same rule. This is a reproduced language-compatibility failure, not a claim that a complete Windows/Python 3.12 Fairseq environment was built and tested here.

A reviewed source patch replacing mutable defaults with factories is one necessary migration step, not a complete fix. Hydra/OmegaConf, native extensions, checkpoint loading defaults in newer PyTorch, and HuBERT feature parity also need validation. Do not disable the global dataclass checks, monkey-patch all `torch.load` calls, or claim success after a single import.

The next native ROCm implementation must choose an exact AMD-supported Windows/GPU release, port or replace the HuBERT/Fairseq loader with reference feature tests, update the numerical stack, build a clean CPython 3.12 worker, and test model loading plus streaming on that hardware. DirectML is the currently implemented experimental Windows AMD alternative, with the ONNX/CPU-stage limits above.

## Intel Macs with AMD graphics: no qualified GPU target

The native builder currently rejects MPS artifacts for x86-64 Macs. Although a generic MPS device adapter exists in the worker, there is no qualified Intel AMD runtime package, hardware result, or product support claim.

Current Apple guidance targets Apple Silicon for modern PyTorch/MPS. PyTorch also announced 2.2.x as the last officially supported macOS x86-64 release. Consequently, the project's pinned 2.2.2 Intel CPU candidate does not prove that AMD Metal inference will work or remain maintainable. [Apple MPS requirements](https://developer.apple.com/metal/pytorch/), [PyTorch macOS x86-64 deprecation](https://pytorch.org/blog/pytorch2-2/)

There are two possible research paths, neither qualified here:

- Test the legacy PyTorch/MPS lane on a named Intel/AMD Mac and exact macOS version, then run every RVC stage and speech/lifecycle tests. A successful tensor allocation alone is insufficient; a legacy framework maintenance decision is also required.
- Implement a Core ML route with a compatible native runtime and verified exported models. Core ML has a CPU-and-GPU compute policy, but provider presence does not prove GPU node placement or compatibility with RVC's changing shapes. This project currently has no Core ML backend or export pipeline. [Core ML execution options](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

No AMD-equipped Intel Mac is available in the implementation environment, so the plan's physical-device feasibility gate remains open. NVIDIA CUDA on macOS is a separate unsupported target; no modern CUDA package path is implemented.

## Remaining paths with no complete implementation

| Path | Actual status and missing work |
| --- | --- |
| Native Windows ROCm | No build variant. Requires the dependency migration and hardware qualification above. |
| MIGraphX ONNX | Worker contains provider-selection/audit hooks when an appropriate provider is installed. No MIGraphX runtime recipe, package or validated model path is supplied; the shipped ROCm recipe uses CPU ONNX. |
| Core ML | No worker backend, runtime variant, graph-export pipeline or hardware validation. |
| Full neural-stage DirectML | Only compatible ONNX RVC/RMVPE stages are implemented. HuBERT ONNX export/loader and broad `.pth` conversion tooling are absent. |
| Intel AMD Mac GPU | No qualified build target. Legacy MPS/Core ML feasibility and maintenance decision remain outstanding. |
| Windows/Linux ARM64 | No advertised managed runtime artifacts or hardware qualification. |
| GPU performance qualification | No GPU model inference measurements in this environment. Each advertised model/OS/device combination still needs speech parity, long-run latency/memory, failure and audio-routing tests. |

Promote a row from experimental only after recording the actual native artifact/hash, hardware, OS/kernel, driver, model hashes, precision/stage placement, test method and results. Keep unavailable combinations disabled rather than presenting an implementation hook as a working product feature.
