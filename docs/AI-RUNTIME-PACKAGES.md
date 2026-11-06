# Native AI runtime packages

The native installer uses immutable runtime builds under the application's local data directory. It does not require system Python, w-okada, WSL, or Docker at inference time. A native runtime must first be built and qualified on its target OS; this repository currently contains an empty trusted download catalog, so it does not advertise nonexistent downloadable artifacts.

## Building a candidate

Run `scripts/build-ai-runtime.py` on a Windows x64 or macOS build machine with Python 3.10, Git, and native compiler tools. The script creates its own build environment. Users of the resulting package do not need those build tools.

```text
python scripts/build-ai-runtime.py --backend cpu --build-id windows-x64-cpu-2-build1
python scripts/build-ai-runtime.py --backend cuda --build-id windows-x64-cuda-2-build1
python scripts/build-ai-runtime.py --backend directml --build-id windows-x64-directml-2-build1
python scripts/build-ai-runtime.py --backend mps --build-id macos-arm64-mps-2-build1
```

Run only commands matching the current OS. Add `--smoke-model /absolute/path/to/trusted-model.pth` to exercise real warmup. A successful build/probe does not qualify GPU execution, speech quality, real-time performance, or virtual microphone routing. Intel macOS CPU artifacts need independent dependency qualification; native Windows ROCm artifacts are not currently supplied by this builder. DirectML is an ONNX path with explicitly reported CPU stages.

The candidate builder uses Python 3.10, PyTorch/torchaudio 2.2.2, and backend-specific ONNX Runtime packages. The CUDA candidate pairs PyTorch CUDA 11.8 with the CUDA 11.8/cuDNN 8 ONNX Runtime 1.18.0 package. These are compatibility candidates for the legacy RVC source, not a claim of support for every current GPU architecture. The manifest captures resolved package versions; review and pin transitive dependencies before promoting a build. [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

The builder freezes an executable plus private libraries with PyInstaller, downloads shared weights with pinned hashes, retains upstream source/notices and dependency metadata, probes the executable, and repeats the probe after relocation. It then extracts its final ZIP with the installer's file modes and verifies/probes that exact payload, including real model warmup when requested. It dereferences in-package symlinks during archiving because the installer rejects archive symlinks; signed macOS builds verify the extracted executable/framework signatures and fail if that transformation breaks them. Only after these checks does it emit a JSON manifest with file/archive hashes and the manifest's SHA-256. [PyInstaller usage](https://pyinstaller.org/en/stable/usage.html)

The manually dispatched `ai-runtime.yml` workflow produces candidate artifacts. It does not publish releases or add artifacts to the catalog. Hosted runners may not have the GPUs needed for real inference tests; explicit GPU inference tests belong on qualified hardware.

## Installing an explicitly trusted local package

Keep the JSON manifest and its ZIP archive in the same directory. Obtain the manifest digest from an independently trusted build/release source; a digest beside an untrusted download does not authenticate it.

```text
voice-warp --install-ai --ai-runtime-manifest /path/to/build.json --ai-manifest-sha256 THE_64_CHARACTER_SHA256
```

Installation checks the manifest's trusted digest, platform/architecture, protocol, archive digest/size, every file digest, safe paths, and total extracted size. It rejects symlinks, special files, unexpected entries, duplicate/case-colliding names, and path traversal. It probes the staged executable before selecting the new build. Cancellation or a failed probe leaves the active build unchanged.

Build IDs are immutable: new contents require a new ID. Repair can restore the exact verified original package after all workers using that build stop; it never replaces an in-use executable or DLL. The installer retains the previous selection for rollback. Active workers hold shared runtime leases, and maintenance takes exclusive locks. User voice model files are never part of runtime removal.

## Publishing a qualified runtime

1. Run the plan's native clean-machine, GPU, speech, long-session, shutdown, and routing checks. Record exact supported OS/GPU/driver/model combinations.
2. Review dependency, source, and shared-weight redistribution terms; the existing engine notice identifies separate weight licenses. Including notices alone does not settle all redistribution obligations.
3. Sign applicable native components. `VOICE_WARP_CODESIGN_IDENTITY` passes a macOS signing identity to PyInstaller; `VOICE_WARP_NOTARY_PROFILE` requests notarization of the ZIP. `VOICE_WARP_WINDOWS_CERT_SHA1` invokes an installed `signtool` for the worker executable. Production validation must also inspect bundled libraries and the desktop distribution.
4. Host the immutable archive and manifest at adjacent direct HTTPS URLs. The installer rejects redirects, so catalog URLs must resolve directly to the bytes.
5. Add a release entry to `engine/runtime-catalog.json`: `os`, `arch`, `backend`, `manifest_url`, and `manifest_sha256`. OS values are `windows`/`macos`; architecture values are `x86_64`/`aarch64`. Review the exact manifest hash as part of the application change.
6. Rebuild and sign the desktop application. The trusted catalog is compiled into the app; replacing a loose JSON file beside an existing executable cannot authorize a new download.

Runtime minimum-OS metadata and probe failures should be considered together when qualifying packages. A private subprocess is not an OS security sandbox: trusted legacy `.pth` models execute with the native worker user's permissions. Linux container isolation claims do not extend to native workers.
