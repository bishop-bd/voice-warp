"""Trust-chain and release assembly checks, requiring no model or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


release = module('assemble_ai_release', ROOT / 'scripts/assemble-ai-release.py')
builder = module('build_ai_runtime', ROOT / 'scripts/build-ai-runtime.py')
profiles = module('install_profile', ROOT / 'engine/install_profile.py')


class ReleaseTests(unittest.TestCase):
    def artifact(self, root, identifier='windows-cpu-test', backend='cpu'):
        payload = root / identifier
        (payload / 'assets').mkdir(parents=True)
        (payload / 'worker.exe').write_bytes(b'test executable bytes')
        (payload / 'assets/hubert_base.pt').write_bytes(b'authenticated shared embedding')
        (payload / 'assets/rmvpe.onnx').write_bytes(b'authenticated pitch graph')
        archive = root / (identifier + '.zip')
        files, total = builder.archive_payload(payload, archive)
        manifest = dict(schema=1, protocol=2, id=identifier, version='3.0.0', os='windows', arch='x86_64', backend=backend,
                        minimum_os='10.0.18362', packages={}, upstream_revision='test-fixture', qualification='test only',
                        executable='worker.exe', assets_dir='assets', archive=archive.name,
                        archive_sha256=release.digest_file(archive), archive_bytes=archive.stat().st_size,
                        unpacked_bytes=total, files=files)
        path = root / (identifier + '.json')
        path.write_text(json.dumps(manifest))
        return path

    def test_offline_catalog_authenticates_copied_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu = self.artifact(root)
            gpu = self.artifact(root, 'windows-cuda-test', 'cuda')
            catalog = root / 'catalog.json'
            entries = release.assemble([cpu, gpu], root / 'bundle', catalog)
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(e['archive_size'] > 0 and 'manifest_url' not in e for e in entries))
            self.assertNotIn(b'\r\n', catalog.read_bytes())
            self.assertEqual(catalog.read_bytes(), (root / 'bundle/runtime-catalog.json').read_bytes())
            release.verify_bundle(root / 'bundle', catalog, 'windows-x86_64')
            with self.assertRaisesRegex(ValueError, 'desktop target'):
                release.verify_bundle(root / 'bundle', catalog, 'macos-aarch64')
            copied = root / 'bundle' / entries[0]['manifest_file']
            copied.write_bytes(copied.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'trusted catalog'):
                release.verify_bundle(root / 'bundle', catalog, 'windows-x86_64')

    def test_tampered_archive_never_produces_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.artifact(root)
            archive = root / json.loads(manifest.read_text())['archive']
            archive.write_bytes(archive.read_bytes() + b'untrusted modification')
            with self.assertRaisesRegex(ValueError, 'archive size or SHA-256'):
                release.assemble([manifest], root / 'bundle', root / 'catalog.json')
            self.assertFalse((root / 'bundle').exists())
            self.assertFalse((root / 'catalog.json').exists())

    def test_bundle_must_contain_cpu_and_unique_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = self.artifact(root, 'gpu-only', 'cuda')
            with self.assertRaisesRegex(ValueError, 'CPU fallback'):
                release.assemble([gpu], root / 'bundle', root / 'catalog.json')
            cpu = self.artifact(root)
            other = self.artifact(root, 'cpu-other')
            with self.assertRaisesRegex(ValueError, 'Duplicate runtime backend'):
                release.assemble([cpu, other], root / 'bundle', root / 'catalog.json')

    def test_catalog_swap_fails_before_packaging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.artifact(root)
            release.assemble([path], root / 'bundle', root / 'catalog.json')
            (root / 'catalog.json').write_text('{"schema":1,"releases":[]}')
            with self.assertRaisesRegex(ValueError, 'about to be compiled'):
                release.verify_bundle(root / 'bundle', root / 'catalog.json', 'windows-x86_64')

    def test_archive_names_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.artifact(root)
            manifest = json.loads(path.read_text())
            manifest['archive'] = '../outside.zip'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'sibling file'):
                release.validate_artifact(path)

    def test_release_manifest_fields_match_the_rust_installer(self):
        source = (ROOT / 'src/rvc/runtime.rs').read_text()
        declaration = re.search(r'pub struct Manifest \{(.*?)\n\}', source, re.DOTALL).group(1)
        rust_fields = set(re.findall(r'pub\s+(\w+)\s*:', declaration))
        self.assertEqual(set(release.REQUIRED_FIELDS) | release.OPTIONAL_FIELDS, rust_fields)

    def test_manifest_schema_rejects_missing_unknown_and_wrong_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.artifact(Path(directory))
            original = json.loads(path.read_text())
            for field in release.REQUIRED_FIELDS:
                with self.subTest(missing=field):
                    changed = dict(original)
                    del changed[field]
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, 'Missing required'):
                        release.validate_artifact(path)
            changed = {**original, 'unrecognized_setting': True}
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Unknown runtime manifest'):
                release.validate_artifact(path)
            for field, value in [('schema', True), ('archive_bytes', True), ('packages', []),
                                 ('files', {'worker.exe': False}), ('minimum_driver', 123),
                                 ('minimum_os', '10.invalid'), ('unpacked_bytes', -1)]:
                with self.subTest(field=field, value=value):
                    path.write_text(json.dumps({**original, field: value}))
                    with self.assertRaises(ValueError):
                        release.validate_artifact(path)
            path.write_text(json.dumps(original)[:-1] + ', "protocol": 2}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON field'):
                release.validate_artifact(path)

    def test_optional_profile_metadata_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.artifact(Path(directory))
            original = json.loads(path.read_text())
            valid = {**original, 'profile': 'windows-x86_64-cpu',
                     'profile_lock_sha256': 'a' * 64, 'minimum_driver': None}
            path.write_text(json.dumps(valid))
            release.validate_artifact(path)
            for changes in ({'profile_lock_sha256': None}, {'profile': 'macos-aarch64-cpu'},
                            {'profile_lock_sha256': 'bad'}, {'minimum_driver': 'x' * 4097},
                            {'minimum_driver': 'driver\0override'}):
                with self.subTest(changes=changes):
                    path.write_text(json.dumps({**valid, **changes}))
                    with self.assertRaises(ValueError):
                        release.validate_artifact(path)
            path.write_text(json.dumps({**original, **dict.fromkeys(release.OPTIONAL_FIELDS)}))
            release.validate_artifact(path)

    def test_installer_size_limits_and_reserved_files_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.artifact(Path(directory))
            original = json.loads(path.read_text())
            for field, value in [('archive_bytes', release.MAX_ARCHIVE + 1),
                                 ('unpacked_bytes', release.MAX_UNPACKED + 1)]:
                path.write_text(json.dumps({**original, field: value}))
                with self.assertRaisesRegex(ValueError, 'installer limits'):
                    release.validate_artifact(path)
            original['files']['runtime.json'] = '0' * 64
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'installer-reserved'):
                release.validate_artifact(path)

    def test_desktop_packaging_requires_the_referenced_local_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / 'catalog.json'
            catalog.write_text('{"schema": 1, "releases": []}')
            release.check_desktop_catalog(catalog, None, 'windows-x86_64')
            remote = {'os': 'windows', 'arch': 'x86_64', 'backend': 'cpu',
                      'manifest_url': 'https://example.invalid/test-manifest.json', 'manifest_sha256': '0' * 64}
            catalog.write_text(json.dumps({'schema': 1, 'releases': [remote]}))
            release.check_desktop_catalog(catalog, None, 'windows-x86_64')
            path = self.artifact(root)
            release.assemble([path], root / 'bundle', catalog)
            with self.assertRaisesRegex(ValueError, 'VOICE_WARP_RUNTIME_BUNDLE'):
                release.check_desktop_catalog(catalog, None, 'windows-x86_64')
            release.check_desktop_catalog(catalog, root / 'bundle', 'windows-x86_64')

    def test_all_profiles_have_complete_authenticated_locks(self):
        paths = list((ROOT / 'engine/profiles').glob('*.json'))
        self.assertEqual(len(paths), 13)
        with patch.dict(os.environ, {'VOICE_WARP_BASE_IMAGE_DIGEST': ''}):
            for path in paths:
                profile, lock = profiles.load_profile(path.stem)
                blocks = profiles.checked_requirements(profile, lock)
                self.assertGreater(len(blocks), 20)
                self.assertTrue(all('--hash=sha256:' in block for block in blocks))
                self.assertFalse(any(block.startswith('fairseq') for block in blocks))
                if profile['os'] in ('windows', 'macos'):
                    self.assertEqual(builder.runtime_profile(profile['backend'], profile['os'], profile['arch']), profile)

    def test_modified_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'profiles').mkdir()
            (root / 'locks').mkdir()
            original = ROOT / 'engine/profiles/windows-x86_64-cpu.json'
            (root / 'profiles' / original.name).write_bytes(original.read_bytes())
            (root / 'locks/windows-x86_64-cpu.txt').write_text('malicious==1.0 --hash=sha256:' + '0' * 64)
            with patch.object(profiles, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    profiles.load_profile(original.stem)


if __name__ == '__main__':
    unittest.main()
