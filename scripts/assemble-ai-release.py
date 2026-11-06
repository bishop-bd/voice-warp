#!/usr/bin/env python3
"""Validate native runtime artifacts and assemble an offline desktop runtime bundle.

Run before compiling the desktop app: its catalog is embedded at compile time.
No network upload or publication occurs. Generated catalog hashes authenticate
bundled artifacts through the signed desktop application, not through sidecars.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_MANIFEST = 4 * 1024 * 1024
MAX_ARCHIVE = 16 * 1024**3
MAX_UNPACKED = 32 * 1024**3
REQUIRED_FIELDS = {
    'schema': int, 'protocol': int, 'id': str, 'version': str,
    'os': str, 'arch': str, 'backend': str, 'minimum_os': str,
    'executable': str, 'assets_dir': str, 'archive': str,
    'archive_sha256': str, 'archive_bytes': int, 'unpacked_bytes': int,
    'files': dict, 'packages': dict, 'upstream_revision': str,
    'qualification': str,
}
OPTIONAL_FIELDS = {'profile', 'profile_lock_sha256', 'minimum_driver'}


def strict_json(contents):
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f'Duplicate JSON field: {key}')
            value[key] = item
        return value
    return json.loads(contents, object_pairs_hook=unique_object)


def validate_manifest_schema(manifest):
    """Match the app's deny_unknown_fields Manifest wire contract."""
    if type(manifest) is not dict:
        raise ValueError('Runtime manifest must be an object')
    missing = REQUIRED_FIELDS.keys() - manifest.keys()
    unknown = manifest.keys() - REQUIRED_FIELDS.keys() - OPTIONAL_FIELDS
    if missing:
        raise ValueError(f'Missing required runtime manifest fields: {sorted(missing)}')
    if unknown:
        raise ValueError(f'Unknown runtime manifest fields: {sorted(unknown)}')
    for name, expected in REQUIRED_FIELDS.items():
        # bool is an int subclass in Python but is not a serde integer.
        if type(manifest[name]) is not expected:
            raise ValueError(f'Invalid runtime manifest field type: {name}')
    for name in OPTIONAL_FIELDS:
        if manifest.get(name) is not None and type(manifest[name]) is not str:
            raise ValueError(f'Invalid optional runtime manifest field type: {name}')
    for name in ('schema', 'protocol', 'archive_bytes', 'unpacked_bytes'):
        maximum = 2**32 - 1 if name in ('schema', 'protocol') else 2**64 - 1
        if not 0 <= manifest[name] <= maximum:
            raise ValueError(f'Runtime integer field is out of range: {name}')
    for name in ('files', 'packages'):
        if any(type(key) is not str or type(value) is not str for key, value in manifest[name].items()):
            raise ValueError(f'Runtime {name} must map strings to strings')
    minimum = manifest['minimum_os'].strip().split('.')
    if (not 1 <= len(minimum) <= 4 or any(not re.fullmatch(r'[0-9]+', part)
                                         or int(part) > 2**32 - 1 for part in minimum)):
        raise ValueError('Invalid runtime minimum OS version')
    profile, lock = manifest.get('profile'), manifest.get('profile_lock_sha256')
    if profile is not None or lock is not None:
        expected = f"{manifest['os']}-{manifest['arch']}-{manifest['backend']}"
        if profile != expected or lock is None or not re.fullmatch(r'[a-fA-F0-9]{64}', lock):
            raise ValueError('Invalid runtime dependency profile or lock checksum')
    driver = manifest.get('minimum_driver')
    if driver is not None and (len(driver.encode('utf-8')) > 4096 or '\0' in driver):
        raise ValueError('Invalid runtime driver requirement')
    if not 0 < manifest['archive_bytes'] <= MAX_ARCHIVE or manifest['unpacked_bytes'] > MAX_UNPACKED:
        raise ValueError('Runtime archive or extraction size exceeds installer limits')


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(name):
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    return (isinstance(name, str) and name and len(name.encode('utf-8')) <= 4096 and not any(c in name for c in '\\:\0')
            and all(p and p not in ('.', '..') and not p.endswith(('.', ' ')) and p.split('.')[0].upper() not in reserved for p in name.split('/')))


def validate_artifact(path):
    if path.stat().st_size > MAX_MANIFEST:
        raise ValueError('Runtime manifest exceeds the installer size limit')
    manifest = strict_json(path.read_text(encoding='utf-8'))
    validate_manifest_schema(manifest)
    identifier = manifest['id']
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}', identifier) or path.name != identifier + '.json':
        raise ValueError('Runtime manifest filename must match its immutable ID')
    if manifest['schema'] != 1 or manifest['protocol'] != 2:
        raise ValueError('Unsupported runtime manifest/protocol version')
    target = (manifest['os'], manifest['arch'])
    allowed = {('windows', 'x86_64'): {'cpu', 'cuda', 'rocm', 'directml'}, ('macos', 'aarch64'): {'cpu', 'mps', 'coreml'}, ('macos', 'x86_64'): {'cpu', 'coreml'}}
    if manifest['backend'] not in allowed.get(target, set()):
        raise ValueError('Runtime backend does not match a supported native target')
    archive_name = manifest['archive']
    if not safe_path(archive_name) or '/' in archive_name:
        raise ValueError('Runtime archive must be a sibling file')
    if not safe_path(manifest['executable']) or not safe_path(manifest['assets_dir']):
        raise ValueError('Unsafe runtime executable or assets directory')
    if not re.fullmatch(r'[a-f0-9]{64}', manifest['archive_sha256']):
        raise ValueError('Invalid runtime archive SHA-256')
    archive = path.parent / archive_name
    if (archive.stat().st_size != manifest['archive_bytes']
            or digest_file(archive) != manifest['archive_sha256']):
        raise ValueError('Runtime archive size or SHA-256 does not match its manifest')
    inventory = manifest['files']
    if not inventory or len(inventory) > 100_000 or len({p.casefold() for p in inventory}) != len(inventory):
        raise ValueError('Invalid or case-colliding runtime inventory')
    for name, digest in inventory.items():
        if not safe_path(name) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Invalid runtime inventory path or SHA-256')
        if name.lower() in ('.lease', 'runtime.json', 'probe.log'):
            raise ValueError('Runtime inventory contains an installer-reserved file')
    required = (manifest['executable'], manifest['assets_dir'] + '/hubert_base.pt', manifest['assets_dir'] + '/rmvpe.onnx')
    if any(name not in inventory for name in required):
        raise ValueError('Runtime executable or shared assets missing from inventory')
    with zipfile.ZipFile(archive) as package:
        if len(package.namelist()) != len(set(package.namelist())) or set(package.namelist()) != set(inventory):
            raise ValueError('Runtime ZIP inventory differs from its manifest')
        total = 0
        for entry in package.infolist():
            if entry.external_attr >> 16 & 0o170000 != 0o100000 or entry.flag_bits & 1:
                raise ValueError('Runtime ZIP contains a special or encrypted file')
            total += entry.file_size
            if total > manifest['unpacked_bytes'] or total > MAX_UNPACKED:
                raise ValueError('Runtime ZIP exceeds declared extraction size')
            digest = hashlib.sha256()
            with package.open(entry) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != inventory[entry.filename]:
                raise ValueError('Runtime ZIP member checksum mismatch')
        if total != manifest['unpacked_bytes']:
            raise ValueError('Runtime ZIP extraction size mismatch')
    return manifest, archive


def assemble(manifests, output, catalog_path):
    if output.exists():
        raise ValueError('Output bundle already exists; use a new release directory')
    records = [(*validate_artifact(path), path) for path in manifests]
    targets = {(m['os'], m['arch']) for m, _, _ in records}
    if len(targets) != 1:
        raise ValueError('Assemble one OS/architecture per desktop bundle')
    keys = [(m['os'], m['arch'], m['backend']) for m, _, _ in records]
    if len(set(keys)) != len(keys) or len({m['id'] for m, _, _ in records}) != len(records):
        raise ValueError('Duplicate runtime backend or immutable ID')
    if len({archive.name.casefold() for _, archive, _ in records}) != len(records):
        raise ValueError('Runtime archive filenames collide')
    if not any(m['backend'] == 'cpu' for m, _, _ in records):
        raise ValueError('An offline desktop bundle must include its CPU fallback runtime')
    releases = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.runtime-assembly-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'bundle'
        artifacts = staged / 'ai-runtimes'
        artifacts.mkdir(parents=True)
        for manifest, archive, path in records:
            shutil.copy2(path, artifacts / path.name)
            shutil.copy2(archive, artifacts / archive.name)
            releases.append({'os': manifest['os'], 'arch': manifest['arch'], 'backend': manifest['backend'],
                             'manifest_file': 'ai-runtimes/' + path.name,
                             'manifest_sha256': digest_file(artifacts / path.name), 'archive_size': archive.stat().st_size})
        catalog = json.dumps({'schema': 1, 'releases': sorted(releases, key=lambda r: r['backend'])}, indent=2) + '\n'
        (staged / 'runtime-catalog.json').write_bytes(catalog.encode('utf-8'))
        # Revalidate the copied payload before generating the trust anchor.
        system, architecture = next(iter(targets))
        verify_bundle(staged, staged / 'runtime-catalog.json', f'{system}-{architecture}')
        staged.rename(output)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_bytes(catalog.encode('utf-8'))
    return releases


def verify_bundle(bundle, catalog_path, target):
    if (bundle / 'runtime-catalog.json').read_bytes() != catalog_path.read_bytes():
        raise ValueError('Offline bundle catalog differs from the catalog about to be compiled')
    catalog = strict_json(catalog_path.read_text())
    if catalog['schema'] != 1 or not catalog['releases']:
        raise ValueError('Offline bundle catalog is empty or unsupported')
    keys = [(e['os'], e['arch'], e['backend']) for e in catalog['releases']]
    if len(keys) != len(set(keys)) or not any(e['backend'] == 'cpu' for e in catalog['releases']):
        raise ValueError('Offline catalog must have unique backends and a CPU fallback')
    for entry in catalog['releases']:
        name = entry.get('manifest_file')
        if not safe_path(name) or entry.get('manifest_url') or not name.startswith('ai-runtimes/'):
            raise ValueError('Expected a local offline catalog entry')
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()) or digest_file(path) != entry['manifest_sha256']:
            raise ValueError('Bundled manifest failed trusted catalog verification')
        manifest, archive = validate_artifact(path)
        if f"{manifest['os']}-{manifest['arch']}" != target:
            raise ValueError('Runtime bundle does not match this desktop target')
        if any(manifest[field] != entry[field] for field in ('os', 'arch', 'backend')) or archive.stat().st_size != entry['archive_size']:
            raise ValueError('Offline catalog metadata mismatch')


def check_desktop_catalog(catalog_path, bundle, target):
    if bundle is not None:
        verify_bundle(bundle, catalog_path, target)
        return
    catalog = strict_json(catalog_path.read_text(encoding='utf-8'))
    if (type(catalog) is not dict or type(catalog.get('schema')) is not int
            or catalog['schema'] != 1 or type(catalog.get('releases')) is not list):
        raise ValueError('Invalid desktop runtime catalog')
    for entry in catalog['releases']:
        if type(entry) is not dict or any(type(entry.get(key)) is not str
                                         for key in ('os', 'arch', 'backend', 'manifest_sha256')):
            raise ValueError('Invalid desktop runtime catalog release')
        if f"{entry['os']}-{entry['arch']}" == target and entry.get('manifest_file') is not None:
            raise ValueError('The compiled catalog requires local runtime artifacts; set VOICE_WARP_RUNTIME_BUNDLE before packaging')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', type=Path, nargs='*')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'engine/runtime-catalog.json')
    parser.add_argument('--verify-bundle', type=Path)
    parser.add_argument('--check-desktop', action='store_true',
                        help='Require a verified bundle when this target catalog references local artifacts')
    parser.add_argument('--target', choices=('windows-x86_64', 'macos-x86_64', 'macos-aarch64'))
    args = parser.parse_args()
    if args.verify_bundle or args.check_desktop:
        if not args.target or args.manifests or args.output:
            parser.error('Desktop verification requires --target and no manifests/output')
        check_desktop_catalog(args.catalog, args.verify_bundle, args.target)
        print('Desktop runtime catalog and required artifacts verified.')
    else:
        if not args.manifests or not args.output:
            parser.error('Provide native manifests and a new --output directory')
        releases = assemble(args.manifests, args.output, args.catalog)
        print(f'Assembled {len(releases)} verified runtimes. Compile the desktop app with this catalog, then package with VOICE_WARP_RUNTIME_BUNDLE={args.output.resolve()}')


if __name__ == '__main__':
    main()
