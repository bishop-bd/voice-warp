#!/usr/bin/env python3
"""Build a native, relocatable Voice Warp worker on its target OS.

Build machines need Python 3.12, Git, and native compiler tools. End users only
need the resulting package and Voice Warp. Artifacts are candidates until tested
on the target hardware; this script does not publish them or modify the catalog.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "d8ef15799470193f7c8176ef471245753a656626"
ASSETS = {
    "hubert_base.pt": (
        "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
        "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96",
    ),
    "rmvpe.onnx": (
        "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx",
        "84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219",
    ),
}


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_target():
    os_name = {"Windows": "windows", "Darwin": "macos"}.get(platform.system())
    architecture = {"AMD64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64"}.get(platform.machine())
    if os_name is None or architecture is None:
        raise ValueError("Build native runtime packages on Windows x64 or macOS arm64/x64.")
    if os_name == "windows" and architecture != "x86_64":
        raise ValueError("Windows ARM64 runtime packaging has not been qualified.")
    return os_name, architecture


def runtime_profile(backend, os_name, architecture):
    name = f"{os_name}-{architecture}-{backend}"
    path = ROOT / "engine/profiles" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"No reviewed native runtime profile exists for {name}.")
    return json.loads(path.read_text())


def fetch_asset(url, digest, destination):
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("xb") as output:
        shutil.copyfileobj(response, output, 1024 * 1024)
    if sha256(destination) != digest:
        raise ValueError(f"Shared asset checksum failed: {destination.name}")


def probe_environment(executable):
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    # A successful frozen smoke must not discover the build venv's interpreter
    # or compiler by PATH. Windows retains its OS DLL/tool directory only.
    paths = [str(executable.parent)]
    if os.name == "nt" and environment.get("SystemRoot"):
        paths.append(str(Path(environment["SystemRoot"]) / "System32"))
    environment["PATH"] = os.pathsep.join(paths)
    return environment


def probe(executable, backend, model=None, assets=None):
    request = {"protocol": 2, "action": "probe", "backend": backend, "device": "0"}
    if model:
        request.update(action="init", model_path=str(model.resolve()), index_path=None,
                       assets_dir=str(assets.resolve()), block_size=8192, pitch=0,
                       index_ratio=0.0)
    environment = probe_environment(executable)
    if model:
        # Init/stream sessions own the worker through an open stdin pipe. Unlike
        # a one-shot probe, communicate(input=...) would close that pipe during
        # model loading and correctly trigger the worker's orphan watchdog.
        with tempfile.TemporaryFile() as diagnostic:
            process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=diagnostic,
                                       text=True, env=environment, cwd=executable.parent)
            lines = queue.Queue(maxsize=1)

            def read_ready():
                try:
                    lines.put(process.stdout.readline(300_001))
                except Exception as error:
                    lines.put(error)

            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                threading.Thread(target=read_ready, daemon=True).start()
                line = lines.get(timeout=180)
                if isinstance(line, Exception):
                    raise line
                if len(line) > 300_000 or not line.endswith("\n"):
                    raise RuntimeError("Frozen worker closed or returned an invalid ready frame.")
                response = json.loads(line)
                if response.get("ready") is not True or response.get("error"):
                    raise RuntimeError(f"Frozen worker failed model warmup: {response}")
                process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                process.stdin.flush()
                remainder, _ = process.communicate(timeout=10)
                if process.returncode or json.loads(remainder).get("stopped") is not True:
                    raise RuntimeError("Frozen worker did not stop after model warmup.")
            except Exception as error:
                process.kill()
                process.wait()
                diagnostic.seek(0)
                detail = diagnostic.read().decode("utf-8", errors="replace")[-12000:]
                raise RuntimeError(f"Frozen worker failed: {error}\n{detail}") from error
            finally:
                process.stdin.close()
                process.stdout.close()
    else:
        result = subprocess.run([str(executable)], input=json.dumps(request) + "\n", text=True,
                                capture_output=True, timeout=180, env=environment,
                                cwd=executable.parent)
        if result.returncode:
            raise RuntimeError(f"Frozen worker failed: {result.stderr[-12000:]}")
        response = json.loads(result.stdout)
    if response.get("protocol") != 2 or response.get("error"):
        raise RuntimeError(f"Frozen worker protocol failed: {response}")
    if model and response.get("ready") is not True:
        raise RuntimeError("Frozen worker did not complete real model warmup.")
    if not model and not isinstance(response.get("devices"), list):
        raise RuntimeError("Frozen worker did not enumerate devices.")
    return response


def export_smoke(executable, model, assets, cache):
    """Exercise the frozen exporter and its PYZ module fingerprint on CPU."""
    request = {"protocol": 2, "action": "export", "backend": "cpu", "device": "0",
               "model_path": str(model.resolve()), "assets_dir": str(assets.resolve()),
               "cache_dir": str(cache.resolve()), "include_hubert": False}
    environment = probe_environment(executable)
    with tempfile.TemporaryFile() as diagnostic:
        process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=diagnostic, text=True, env=environment, cwd=executable.parent)
        lines = queue.Queue(maxsize=1)
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            threading.Thread(target=lambda: lines.put(process.stdout.readline(300_001)), daemon=True).start()
            line = lines.get(timeout=300)
            if len(line) > 300_000 or not line.endswith("\n"):
                raise RuntimeError("Frozen exporter returned an invalid frame")
            response = json.loads(line)
            if response.get("protocol") != 2 or response.get("exported") is not True or response.get("cpu_validated") is not True:
                raise RuntimeError(f"Frozen exporter failed CPU parity validation: {response}")
            process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError("Frozen exporter exited unsuccessfully")
            return response
        except Exception as error:
            process.kill()
            process.wait()
            diagnostic.seek(0)
            raise RuntimeError(f"Frozen export failed: {error}\n{diagnostic.read().decode('utf-8', errors='replace')[-12000:]}") from error
        finally:
            process.stdin.close()
            process.stdout.close()


def archive_payload(payload, archive_path):
    # PyInstaller uses symlinks on macOS. Materialize their contents in the ZIP;
    # the installer intentionally rejects all archive symlinks and special files.
    inventory = {}
    total = 0
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4, allowZip64=True) as archive:
        for directory, dirs, names in os.walk(payload, followlinks=True):
            current = Path(directory)
            if not current.resolve().is_relative_to(payload.resolve()):
                raise ValueError(f"Packaged directory links outside its runtime: {current}")
            ancestors = {current.resolve()}
            for ancestor in current.parents:
                ancestors.add(ancestor.resolve())
                if ancestor == payload:
                    break
            for child in dirs:
                if (current / child).resolve() in ancestors:
                    raise ValueError(f"Packaged dependency contains a directory symlink cycle: {current / child}")
            dirs.sort()
            for name in sorted(names):
                path = Path(directory) / name
                resolved = path.resolve()
                if not resolved.is_relative_to(payload.resolve()):
                    raise ValueError(f"Packaged dependency links outside its runtime: {path}")
                relative = path.relative_to(payload).as_posix()
                info = zipfile.ZipInfo.from_file(path, relative)
                info.compress_type = zipfile.ZIP_DEFLATED
                # Always encode a regular file, preserving executability only.
                executable = bool(path.stat().st_mode & 0o111)
                info.create_system = 3
                info.external_attr = (0o100755 if executable else 0o100644) << 16
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                inventory[relative] = sha256(path)
                total += path.stat().st_size
    return inventory, total


def extract_verified_payload(archive_path, destination, inventory):
    """Recreate the bytes/modes users install, including materialized symlinks."""
    destination.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(inventory):
            raise ValueError("Final runtime ZIP inventory differs from the manifest.")
        for info in archive.infolist():
            name = info.filename
            if (not name or name.startswith("/") or "\\" in name or ":" in name
                    or any(part in ("", ".", "..") for part in name.split("/"))):
                raise ValueError(f"Unsafe final runtime ZIP path: {name}")
            mode = info.external_attr >> 16
            if mode & 0o170000 != 0o100000:
                raise ValueError(f"Final runtime ZIP contains a non-regular file: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            target.chmod(0o755 if mode & 0o111 else 0o644)
            if sha256(target) != inventory[name]:
                raise ValueError(f"Final runtime ZIP file checksum failed: {name}")


def build(args):
    os_name, architecture = host_target()
    profile = runtime_profile(args.backend, os_name, architecture)
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Build with Python 3.12 to match the pinned runtime profile ABI.")
    if not args.build_id or len(args.build_id) > 160 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.build_id):
        raise ValueError("--build-id must contain 1–160 letters, digits, hyphens, or underscores.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_root = ROOT / "target/ai-runtime-build"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=work_root) as work_name:
        work = Path(work_name)
        source = work / "upstream"
        run("git", "init", source)
        run("git", "-C", source, "remote", "add", "origin", "https://github.com/w-okada/voice-changer.git")
        run("git", "-C", source, "fetch", "--depth", "1", "origin", UPSTREAM)
        run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os_name == "windows" else "bin/python")
        install_command = [python, ROOT / "engine/install_profile.py", profile["id"]]
        if args.wheel_cache:
            install_command += ["--wheel-cache", args.wheel_cache.resolve()]
        run(*install_command)
        command = [python, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
                   "--name", "voice-warp-worker", "--distpath", work / "frozen",
                   "--workpath", work / "pyinstaller", "--specpath", work,
                   "--paths", source / "server", "--paths", ROOT / "engine",
                   "--collect-submodules", "voice_changer.RVC.inferencer.rvc_models",
                   "--hidden-import", "voice_changer.RVC.RVCSettings",
                   "--hidden-import", "Exceptions", "--hidden-import", "export_models",
                   "--hidden-import", "voice_warp_hubert", "--hidden-import", "voice_warp_rvc",
                   "--hidden-import", "voice_warp_checkpoint"]
        for package in ("onnxruntime", "faiss", "scipy", "numba", "llvmlite", "resampy", "torchaudio"):
            command += ["--collect-all", package]
        if args.backend == "rocm":
            # AMD's loader discovers libraries relative to these package roots.
            # Preserve both SDK metadata and private native-library directories.
            for package in ("rocm_sdk", "rocm_sdk_core", "rocm_sdk_libraries_custom",
                            "_rocm_sdk_core", "_rocm_sdk_libraries_custom"):
                command += ["--collect-all", package]
        identity = os.environ.get("VOICE_WARP_CODESIGN_IDENTITY")
        if identity and os_name == "macos":
            command += ["--codesign-identity", identity, "--osx-entitlements-file", ROOT / "packaging/runtime-entitlements.plist"]
        command.append(ROOT / "engine/worker.py")
        run(*command)
        payload = work / "frozen/voice-warp-worker"
        executable_name = "voice-warp-worker.exe" if os_name == "windows" else "voice-warp-worker"
        executable = payload / executable_name
        assets = payload / "assets"
        assets.mkdir()
        for name, (url, digest) in ASSETS.items():
            fetch_asset(url, digest, assets / name)
        notices = payload / "licenses"
        notices.mkdir()
        for name in ("NOTICE.md", "UPSTREAM-LICENSE"):
            shutil.copy2(ROOT / "engine" / name, notices / name)
        shutil.copy2(ROOT / "LICENSE", notices / "VOICE-WARP-LICENSE")
        shutil.copytree(source / "server", notices / "upstream-server", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "engine/profiles" / f'{profile["id"]}.json', notices / "runtime-profile.json")
        shutil.copy2(ROOT / "engine/locks" / profile["lock"], notices / "requirements.lock")
        freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
        (notices / "requirements-resolved.txt").write_text(freeze)
        # Copy installed distribution metadata, including license files bundled by wheels.
        metadata_script = "import importlib.metadata,json; print(json.dumps([str(x._path) for x in importlib.metadata.distributions()]))"
        for metadata_dir in json.loads(subprocess.check_output([str(python), "-c", metadata_script], text=True)):
            source_metadata = Path(metadata_dir)
            if source_metadata.is_dir():
                shutil.copytree(source_metadata, notices / source_metadata.name, dirs_exist_ok=True)
        if os_name == "windows" and os.environ.get("VOICE_WARP_WINDOWS_CERT_SHA1"):
            run("signtool", "sign", "/sha1", os.environ["VOICE_WARP_WINDOWS_CERT_SHA1"],
                "/fd", "SHA256", "/tr", "http://timestamp.digicert.com", "/td", "SHA256", executable)
            run("signtool", "verify", "/pa", executable)
        fixture = work / "synthetic-packaging-test.pth"
        run(python, ROOT / "scripts/create-runtime-fixture.py", fixture, "--upstream-server", source / "server")
        packages = json.loads(subprocess.check_output([
            str(python), "-c", "import importlib.metadata,json; print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))"
        ], text=True))
        # Prove the frozen worker does not depend on its build environment, and
        # avoid retaining two large CUDA/ROCm library trees during ZIP checking.
        if environment.is_symlink():
            environment.unlink()
        else:
            shutil.rmtree(environment)
        shutil.rmtree(source)
        shutil.rmtree(work / "pyinstaller")
        response = probe(executable, args.backend)
        fixture_result = probe(executable, "cpu", fixture, assets)
        model_result = probe(executable, args.backend, args.smoke_model, assets) if args.smoke_model else None
        (payload / "build-verification.json").write_text(json.dumps({"probe": response, "synthetic_cpu_warmup": fixture_result, "model_warmup": model_result}, indent=2) + "\n")
        # Relocation is a required packaging check, even without a GPU/model fixture.
        moved = work / "relocated worker é"
        shutil.move(str(payload), moved)
        probe(moved / executable_name, args.backend)
        archive_name = f"{args.build_id}.zip"
        archive_path = output / archive_name
        inventory, total = archive_payload(moved, archive_path)
        # Test what users actually install. ZIP materialization may alter macOS
        # framework layout/signature validity even if the original bundle worked.
        unpacked = work / "archive verification é"
        extract_verified_payload(archive_path, unpacked, inventory)
        probe(unpacked / executable_name, args.backend)
        probe(unpacked / executable_name, "cpu", fixture, unpacked / "assets")
        export_result = export_smoke(unpacked / executable_name, fixture, unpacked / "assets", work / "frozen-export-cache")
        if not Path(export_result["voice_model"]).is_file():
            raise RuntimeError("Frozen export did not produce its validated ONNX voice")
        if args.smoke_model:
            probe(unpacked / executable_name, args.backend, args.smoke_model, unpacked / "assets")
        if os_name == "macos" and identity:
            run("codesign", "--verify", "--strict", "--verbose=2", unpacked / executable_name)
            for framework in sorted(unpacked.rglob("*.framework")):
                if framework.is_dir():
                    run("codesign", "--verify", "--strict", "--deep", "--verbose=2", framework)
        if os_name == "windows" and os.environ.get("VOICE_WARP_WINDOWS_CERT_SHA1"):
            run("signtool", "verify", "/pa", unpacked / executable_name)
        if os_name == "macos" and os.environ.get("VOICE_WARP_NOTARY_PROFILE"):
            run("xcrun", "notarytool", "submit", archive_path, "--keychain-profile",
                os.environ["VOICE_WARP_NOTARY_PROFILE"], "--wait")
        manifest = {
            "schema": 1, "protocol": 2, "id": args.build_id, "version": args.runtime_version,
            "os": os_name, "arch": architecture, "backend": args.backend,
            "minimum_os": profile["minimum_os"], "profile": profile["id"],
            "profile_lock_sha256": profile["lock_sha256"],
            "minimum_driver": profile.get("minimum_driver"),
            "executable": executable_name, "assets_dir": "assets", "archive": archive_name,
            "archive_sha256": sha256(archive_path), "archive_bytes": archive_path.stat().st_size,
            "unpacked_bytes": total, "files": inventory, "packages": packages,
            "upstream_revision": UPSTREAM,
            "qualification": "synthetic CPU inference/export and final ZIP verification only; hardware and speech qualification required" if not model_result else "build/probe/relocation/final ZIP model warmup only; speech qualification required",
        }
        manifest_path = output / f"{args.build_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = sha256(manifest_path)
        (output / f"{args.build_id}.json.sha256").write_text(f"{digest}  {manifest_path.name}\n")
        print(f"Runtime package: {archive_path}\nManifest: {manifest_path}\nManifest SHA-256: {digest}")
        print("Candidate package only. Qualify hardware/speech, review redistribution terms, then use assemble-ai-release.py to bundle its trusted manifest before rebuilding the signed app.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "mps", "coreml", "directml"), required=True)
    parser.add_argument("--build-id", required=True, help="Unique immutable package ID, e.g. windows-x64-cpu-2-20261105")
    parser.add_argument("--runtime-version", default="3.0.0")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/ai-runtimes")
    parser.add_argument("--wheel-cache", type=Path, help="Optional local vendor wheel cache; all bytes remain hash-verified")
    parser.add_argument("--smoke-model", type=Path, help="Trusted licensed model for actual inference warmup")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
