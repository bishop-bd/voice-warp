# Managed AI runtime packages

Native Windows and macOS workers include their own Python interpreter and inference libraries. End users do not need Python, w-okada, WSL, Docker, or compiler tools. Linux runs the worker in a rootless Podman container. Each runtime uses protocol 2 and reports its version, actual providers, device placement, CPU stages, and fallback reasons.

The repository contains buildable candidate profiles and release assembly tools. The checked-in trusted native catalog stays empty until real artifacts are assembled into a desktop release; it does not point at nonexistent downloads. Windows/macOS packages must be built and tested on those operating systems. A resolved dependency lock or successful device probe does not establish working GPU inference, speech quality, latency, routing, or long-session stability. See [AI.md](AI.md) for observed hardware results and limitations.

## Reviewed dependency profiles

`engine/profiles/*.json` selects an exact platform/backend stack. Its `lock_sha256` authenticates the complete transitive requirements in `engine/locks/*.txt`. `engine/install_profile.py` rejects a changed lock or mismatched Python/OS/architecture, installs with `--require-hashes`, and runs `pip check`. It does not resolve an unpinned set of dependencies at build time. Shared HuBERT/RMVPE assets and the upstream inference source revision are separately pinned.

| Profile group | Python / PyTorch | Inference package | Minimum native OS / qualification |
|---|---|---|---|
| Linux CPU, Windows CPU | 3.12 / 2.9.1 CPU | ONNX Runtime 1.23.2 | Windows 10 build 18362; Linux Ubuntu 24.04 container |
| Linux/Windows CUDA | 3.12 / 2.9.1 CUDA 12.8 | ONNX Runtime GPU 1.23.2, CUDA 12/cuDNN 9 | Compatible NVIDIA driver and tested GPU required |
| Linux ROCm | 3.12 / AMD 2.9.1 ROCm 7.2.1 | ONNX Runtime CPU 1.23.2 for pitch | Full ROCm userspace is in the pinned image; host driver/device access still required |
| Linux MIGraphX | Same AMD stack | ONNX Runtime MIGraphX 1.23.2, MIGraphX 2.15.0 | Separate provider build; requires actual graph/provider validation |
| Windows ROCm | 3.12 / AMD 2.9.1 ROCm 7.2.1 | AMD SDK wheels + ONNX Runtime CPU | Windows 11 build 22631; vendor-qualified device and AMD 26.2.2 driver prerequisite |
| Windows DirectML | 3.12 / 2.9.1 CPU | ONNX Runtime DirectML 1.23.0 | Exported ONNX voice; CPU stages remain explicit |
| Apple Silicon CPU/MPS/CoreML | 3.12 / 2.9.1 | ONNX Runtime 1.23.2 | macOS 14 minimum due to the FAISS wheel; MPS/CoreML require independent GPU qualification |
| Intel Mac CPU/CoreML | 3.12 / 2.2.2 | ONNX Runtime 1.23.2 | macOS 13; legacy final PyTorch Intel release, no promise of NVIDIA support |

AMD Linux pairs torch 2.9.1 with torchaudio 2.9.0 because those are the matched vendor 7.2.1 artifacts. The Windows profile includes the vendor SDK libraries needed by PyTorch's ROCm loader. Its small `rocm` metadata source distribution is built only on the build machine, using pinned setuptools/wheel with build isolation disabled. All other dependencies install as wheels. Users receive a frozen executable.

Sources: [AMD Linux 7.2.1 PyTorch](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installryz/native_linux/install-pytorch.html), [AMD Windows 7.2.1 PyTorch and driver prerequisites](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html), [AMD MIGraphX installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/native_linux/install-migraphx.html), [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

To intentionally regenerate dependency locks, install the recorded `uv==0.13.7` build tool and run `python scripts/lock-ai-profiles.py`. Supply profile IDs to update a subset and `--upgrade` only when deliberately changing transitive versions. Review both lock and profile hash changes. Dependency locks provide repeatable dependency selection; signing timestamps, compiler versions, and Python patch releases can still change archive bytes.

## Linux images

`engine/Containerfile` builds four separate variants, tagged by the app as `localhost/voice-warp-rvc:3-{cpu,cuda,rocm,migraphx}`. For example:

```sh
podman build --build-arg AI_BACKEND=rocm -t localhost/voice-warp-rvc:3-rocm engine
```

CPU/CUDA use an immutable Ubuntu 24.04 base. ROCm/MIGraphX use the full AMD image `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`, pinned to digest `sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191`. PyTorch wheels alone do not contain all required Linux ROCm libraries. MIGraphX additionally verifies the SHA-256 of its exact system package. The AMD base is approximately 10 GiB compressed before application layers; allow substantial download and disk space.

The installer reuses only the exact torch/torchaudio/triton versions supplied by that immutable AMD image. It verifies their versions and installs all remaining packages from the checked-in hash lock. Runtime assets are readable under rootless `--userns=keep-id`; writable scratch uses `/tmp` and model/export cache mounts are separate. No host-wide Python or ROCm library installation is performed by these build tools.

## Build a native candidate

Use Python 3.12, Git, and native compiler tools on the target Windows x64 or macOS arm64/x64 build machine. Run only commands matching that machine:

```text
python scripts/build-ai-runtime.py --backend cpu --build-id windows-x64-cpu-3-build1
python scripts/build-ai-runtime.py --backend cuda --build-id windows-x64-cuda-3-build1
python scripts/build-ai-runtime.py --backend rocm --build-id windows-x64-rocm-3-build1
python scripts/build-ai-runtime.py --backend directml --build-id windows-x64-directml-3-build1
python scripts/build-ai-runtime.py --backend mps --build-id macos-arm64-mps-3-build1
python scripts/build-ai-runtime.py --backend coreml --build-id macos-x64-coreml-3-build1
```

Add `--smoke-model /absolute/path/to/trusted-model.pth` for actual warmup; ONNX-only backends need the compatible exported model artifacts. `--wheel-cache PATH` can reuse downloaded vendor wheels while retaining all lock hash checks. The builder emits a ZIP, manifest, and manifest SHA-256 under `dist/ai-runtimes`.

The builder freezes the worker, portable HuBERT/RVC adapters, and ONNX exporter with PyInstaller. It preserves ROCm SDK library package paths on Windows, collects dynamic inference dependencies, and includes source/notices and distribution metadata. Every build generates a deterministic random RVC checkpoint and runs actual CPU warmup with the shared HuBERT/RMVPE assets. It probes after relocation, then extracts and hashes the final ZIP with the installer's actual file modes and repeats CPU warmup on those exact bytes. The extracted frozen worker also exports the synthetic voice to ONNX and validates CPU numerical parity; this catches missing frozen exporter modules. These are packaging checks, not trained voice or GPU-quality tests. If supplied, the trusted model additionally warms up using the requested backend from the final extracted package. Internal symlinks are materialized because the installer rejects archive symlinks. Signed macOS builds verify the extracted executable/framework signatures and fail if materialization invalidates them.

The manually dispatched `.github/workflows/ai-runtime.yml` builds nine native candidates, then assembles three desktop distributions containing the appropriate CPU and GPU variants. Hosted runners do not provide every relevant GPU. Artifacts remain build candidates; the workflow neither publishes a public release nor claims hardware qualification. The regular CI runs archive/profile/release trust-chain tests on all three operating systems.

## Assemble an offline desktop distribution

Pass actual built manifests for one OS/architecture. Include its CPU fallback. The assembly tool validates archive sizes/hashes, every member hash, path safety, extraction totals, target compatibility, and unique backends before copying anything into the final bundle.

```text
python scripts/assemble-ai-release.py dist/ai-runtimes/windows-x64-cpu-3-build1.json dist/ai-runtimes/windows-x64-cuda-3-build1.json --output dist/windows-runtime-bundle
```

This creates `dist/windows-runtime-bundle/ai-runtimes/` and writes the matching catalog into `engine/runtime-catalog.json`. Compile the desktop application **after** this step, so those manifest hashes are embedded in its executable. Then package with the bundle location:

```powershell
$env:VOICE_WARP_RUNTIME_BUNDLE = (Resolve-Path dist/windows-runtime-bundle)
./scripts/package-windows.ps1
```

```sh
VOICE_WARP_RUNTIME_BUNDLE="$PWD/dist/macos-runtime-bundle" bash scripts/package-macos.sh
```

Each packaging script verifies the source catalog before compiling. If that target references local manifests, omitting `VOICE_WARP_RUNTIME_BUNDLE` is an error; an empty source catalog continues to build without a bundle. When supplied, the bundle catalog must exactly match the source catalog and all artifacts must match it. Assembly also checks the complete Rust manifest schema, including required fields, optional profile metadata, and installer size limits. Catalog files use LF line endings on every OS so Windows checkout normalization cannot change the comparison.

Windows places `ai-runtimes` beside `voice-warp.exe`; macOS places it under `Voice Warp.app/Contents/Resources`. Desktop packaging needs Python 3.10 or newer on the build machine for these checks; building the runtime itself needs Python 3.12. Windows packaging uses ZIP64 to accommodate runtime archives larger than 2 GiB. The app's normal installation flow can install these bundled runtimes without downloading anything or requiring the user to type a manifest hash.

The catalog entry contains `os`, `arch`, `backend`, `manifest_file`, `manifest_sha256`, and `archive_size`. A downloaded distribution may instead use `manifest_url` with a direct immutable HTTPS URL. Exactly one location field is permitted. The catalog is compiled into the app: replacing a loose JSON file beside an existing executable does not authorize new runtime code. The ZIP/manifest sidecars are transport checks; the signed desktop executable containing the reviewed catalog is the trust anchor.

## Explicit local installation and maintenance

Keep the manifest and archive together. Obtain the manifest digest from an independently trusted build/release source; a hash supplied beside an untrusted download does not authenticate it.

```text
voice-warp --install-ai --ai-runtime-manifest /path/to/build.json --ai-manifest-sha256 THE_64_CHARACTER_SHA256
```

Installation verifies the trusted manifest digest, target/minimum OS, protocol, archive digest/size, every file hash, safe paths, and total extracted size. It rejects symlinks, special files, unexpected entries, duplicate/case-colliding names, and traversal. It probes the staged executable before selecting the new build. Cancellation or a failed probe leaves the active build unchanged.

Build IDs are immutable: changed bytes require a new ID. Repair restores the exact original package after all workers using it stop; it never replaces an in-use executable or library. The previous selection supports rollback. Workers hold shared runtime leases and maintenance takes exclusive locks. Removing a runtime never removes user voice models.

## Signing and qualification

Set `VOICE_WARP_CODESIGN_IDENTITY` to sign native macOS workers and the desktop app with an installed Developer ID identity. The worker receives narrowly scoped JIT/executable-memory entitlements for NumPy/Numba/Torch; the desktop receives the audio-input entitlement. `VOICE_WARP_NOTARY_PROFILE` invokes notarization. The desktop packaging script staples and validates the accepted app ticket, then recreates the archive and checksum. A worker ZIP contains standalone code and has no independently staplable app bundle; validate the final installed worker on a clean Mac as part of release qualification. [PyInstaller macOS signing](https://pyinstaller.org/en/stable/feature-notes.html)

On Windows, `VOICE_WARP_WINDOWS_CERT_SHA1` invokes installed `signtool` to sign and verify the worker and desktop executables. Certificate provisioning, signing-service access, and production credentials belong to the release environment. These optional hooks have not been exercised here with production certificates.

Before distributing a supported release, record exact OS/GPU/driver/model combinations and run the plan's native clean-machine, real speech, GPU-stage, long-session, shutdown, and virtual microphone checks. Review source, dependency, and shared-weight redistribution terms; retained notices do not alone settle all licensing obligations. Native workers run as the logged-in user, so trusted legacy `.pth` files retain that user's privileges. The private worker subprocess is not an operating-system security sandbox.

To uninstall a native backend, stop sessions using its current or rollback build, then run:

```text
voice-warp --ai-uninstall --ai-backend cpu
```

The backend name must be explicit (`auto` is rejected). This removes its active selection and current/rollback native builds after acquiring all runtime leases. It preserves settings, voice models, other backends, older inactive builds, and Linux container images. If file permissions or another program prevent complete deletion, the backend remains deactivated and the error identifies inactive build IDs to retry with `voice-warp --ai-remove-runtime BUILD_ID`. Use `voice-warp --ai-rollback --ai-backend cpu` to switch to the previous native build without uninstalling it.
