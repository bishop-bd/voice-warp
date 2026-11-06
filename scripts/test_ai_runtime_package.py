"""Packaging invariants that do not require an inference runtime or GPU."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("build_ai_runtime", Path(__file__).with_name("build-ai-runtime.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class RuntimePackageTests(unittest.TestCase):
    def test_archive_inventory_matches_relocated_payload_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "worker é"
            (payload / "assets").mkdir(parents=True)
            (payload / "assets/model").write_bytes(b"model bytes" * 128)
            worker = payload / "worker"
            worker.write_bytes(b"executable placeholder")
            worker.chmod(0o755)
            archive = root / "runtime.zip"
            files, total = builder.archive_payload(payload, archive)
            with zipfile.ZipFile(archive) as package:
                self.assertEqual(set(files), set(package.namelist()))
                self.assertEqual(total, sum(item.file_size for item in package.infolist()))
                for name, expected in files.items():
                    self.assertEqual(hashlib.sha256(package.read(name)).hexdigest(), expected)
                    self.assertEqual(package.getinfo(name).external_attr >> 16 & 0o170000, 0o100000)
                # Windows chmod only controls read-only status; execution uses
                # the PE file, not POSIX executable mode bits.
                if os.name != "nt":
                    self.assertTrue(package.getinfo("worker").external_attr >> 16 & 0o111)
            extracted = root / "installed payload é"
            builder.extract_verified_payload(archive, extracted, files)
            for name, digest in files.items():
                self.assertEqual(builder.sha256(extracted / name), digest)
            if os.name != "nt":
                self.assertTrue((extracted / "worker").stat().st_mode & 0o111)

    def test_materialized_internal_symlink_matches_installed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            (payload / "library").write_bytes(b"library contents")
            try:
                (payload / "alias").symlink_to("library")
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable on this test host")
            archive = root / "runtime.zip"
            files, _ = builder.archive_payload(payload, archive)
            installed = root / "installed"
            builder.extract_verified_payload(archive, installed, files)
            self.assertFalse((installed / "alias").is_symlink())
            self.assertEqual((installed / "alias").read_bytes(), (installed / "library").read_bytes())

    def test_final_payload_verification_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            (payload / "library").write_bytes(b"library contents")
            archive = root / "runtime.zip"
            files, _ = builder.archive_payload(payload, archive)
            files["library"] = "0" * 64
            with self.assertRaises(ValueError):
                builder.extract_verified_payload(archive, root / "installed", files)

    def test_external_symlinks_cannot_be_embedded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            external = root / "private"
            external.write_bytes(b"must not be packaged")
            try:
                (payload / "bad").symlink_to(external)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable on this test host")
            with self.assertRaises(ValueError):
                builder.archive_payload(payload, root / "runtime.zip")

    def test_native_gpu_packages_reject_unsupported_targets(self):
        for backend, system, architecture in [("mps", "windows", "x86_64"),
                                                ("cuda", "macos", "aarch64"),
                                                ("directml", "macos", "x86_64"),
                                                ("mps", "macos", "x86_64")]:
            with self.assertRaises(ValueError):
                builder.runtime_profile(backend, system, architecture)


if __name__ == "__main__":
    unittest.main()
