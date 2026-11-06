"""Export policy tests plus opt-in real checkpoint/ONNX parity integration."""
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest
import zipfile

import export_models as exporter


class ExportPolicyTests(unittest.TestCase):
    def test_packaged_code_can_be_fingerprinted_without_a_source_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "frozen.zip"
            with zipfile.ZipFile(archive, "w") as package:
                for name in ("export_models", "voice_warp_hubert", "voice_warp_checkpoint"):
                    source = Path(exporter.__file__).with_name(name + ".py")
                    bytecode = root / (name + ".pyc")
                    py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
                    package.write(bytecode, bytecode.name)
                    bytecode.unlink()
            # Exercise a real archive loader with no physical module source;
            # the HuBERT module's code is hashed without importing ML packages.
            script = ("import sys,pathlib;sys.path.insert(0,sys.argv[1]);import export_models as e;"
                      "assert not pathlib.Path(e.__file__).is_file();"
                      "a=e.module_digest('export_models',e.__file__);"
                      "b=e.module_digest('voice_warp_hubert',pathlib.Path(e.__file__).with_name('voice_warp_hubert.py'));"
                      "c=e.module_digest('voice_warp_checkpoint',pathlib.Path(e.__file__).with_name('voice_warp_checkpoint.py'));"
                      "assert len(a)==len(b)==len(c)==64 and len({a,b,c})==3;"
                      "assert a==e.module_digest('export_models',e.__file__)")
            subprocess.run([sys.executable, "-I", "-c", script, str(archive)], check=True)

    def test_abandoned_cleanup_keeps_other_keys_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            abandoned = root / ".key-old"
            abandoned.mkdir()
            (abandoned / "partial.onnx").write_bytes(b"partial")
            other = root / ".other-active"
            other.mkdir()
            exporter.remove_abandoned_exports(root, "key")
            self.assertFalse(abandoned.exists())
            self.assertTrue(other.exists())
            link = root / ".key-link"
            try:
                link.symlink_to(other, target_is_directory=True)
            except OSError:
                return  # Windows runners may lack symlink privileges.
            exporter.remove_abandoned_exports(root, "key")
            self.assertTrue(link.is_symlink())
            self.assertTrue(other.exists())

    def test_only_standard_rvc_metadata_is_exported(self):
        checkpoint = {"config": [0] * 17 + [40000], "f0": 1, "version": "v1"}
        v1 = exporter.voice_metadata(checkpoint)
        self.assertEqual((v1["embChannels"], v1["embOutputLayer"], v1["useFinalProj"]), (256, 9, True))
        v2 = exporter.voice_metadata({**checkpoint, "version": "v2", "f0": 0})
        self.assertEqual((v2["embChannels"], v2["embOutputLayer"], v2["useFinalProj"]), (768, 12, False))
        self.assertFalse(v2["f0"])
        for invalid in ({**checkpoint, "version": "custom"},
                        {**checkpoint, "config": [0] * 19},
                        {**checkpoint, "config": [0] * 17 + [12345]},
                        {**checkpoint, "f0": "yes"}):
            with self.assertRaises(ValueError):
                exporter.voice_metadata(invalid)

    def test_cache_key_changes_with_sources_framework_and_export_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "voice.pth"
            source.write_bytes(b"voice one")
            (root / "hubert_base.pt").write_bytes(b"shared hubert")
            (root / "rmvpe.onnx").write_bytes(b"shared pitch")
            key, payload = exporter.fingerprint(source, root, {"torch": "old"}, True)
            self.assertEqual(payload["checkpoint_adapter_sha256"],
                             exporter.module_digest("voice_warp_checkpoint", Path(exporter.__file__).with_name("voice_warp_checkpoint.py")))
            changed_framework, _ = exporter.fingerprint(source, root, {"torch": "new"}, True)
            changed_policy, _ = exporter.fingerprint(source, root, {"torch": "old"}, False)
            source.write_bytes(b"voice two")
            changed_voice, _ = exporter.fingerprint(source, root, {"torch": "old"}, True)
            self.assertEqual(len({key, changed_framework, changed_policy, changed_voice}), 4)

    def test_cache_hit_requires_complete_intact_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for filename in ("voice.onnx", "hubert.onnx", "rmvpe.onnx"):
                (root / filename).write_bytes(filename.encode())
                files[filename] = exporter.sha256(root / filename)
            manifest = {"cache_key": "key", "cpu_validated": True,
                        "sources": {"include_hubert": True}, "files": files}
            (root / "export.json").write_text(json.dumps(manifest))
            self.assertIsNotNone(exporter.cached_manifest(root, "key"))
            self.assertIsNone(exporter.cached_manifest(root, "another-key"))
            (root / "voice.onnx").write_bytes(b"corrupted")
            self.assertIsNone(exporter.cached_manifest(root, "key"))
            (root / "export.json").write_text(json.dumps({**manifest, "files": {}}))
            self.assertIsNone(exporter.cached_manifest(root, "key"))
            (root / "export.json").write_text("[]")
            self.assertIsNone(exporter.cached_manifest(root, "key"))


@unittest.skipUnless(os.environ.get("VOICE_WARP_EXPORT_TEST_MODEL"),
                     "requires an inference runtime and trusted VOICE_WARP_EXPORT_TEST_MODEL")
class RealExportTests(unittest.TestCase):
    def test_real_voice_and_hubert_exports_and_cache_reuse(self):
        assets = Path(os.environ.get("VOICE_WARP_EXPORT_TEST_ASSETS", "/assets"))
        with tempfile.TemporaryDirectory(dir=os.environ.get("VOICE_WARP_EXPORT_TEST_TEMP")) as temporary:
            first = exporter.export_bundle(os.environ["VOICE_WARP_EXPORT_TEST_MODEL"], assets, temporary)
            self.assertTrue(first["cpu_validated"])
            self.assertFalse(first["gpu_validated"])
            self.assertFalse(first["cached"])
            self.assertEqual(len(first["validation"]["voice"]["cases"]), len(exporter.VOICE_FRAMES))
            self.assertEqual(len(first["validation"]["hubert"]["cases"]), len(exporter.HUBERT_SAMPLES))
            second = exporter.export_bundle(os.environ["VOICE_WARP_EXPORT_TEST_MODEL"], assets, temporary)
            self.assertTrue(second["cached"])
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertEqual(first["files"], second["files"])


if __name__ == "__main__":
    unittest.main()
