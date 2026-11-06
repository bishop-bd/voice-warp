#!/usr/bin/env python3
"""Regenerate reviewed runtime locks with uv 0.13.7; review resulting diffs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiles', nargs='*', help='Profile IDs; omit for all targets')
    parser.add_argument('--uv', default='uv')
    parser.add_argument('--upgrade', action='store_true', help='Intentionally update transitive dependency versions')
    args = parser.parse_args()
    version = subprocess.check_output([args.uv, '--version'], text=True).split()
    if version[:2] != ['uv', '0.13.7']:
        raise ValueError('Use the recorded lock generator version: uv 0.13.7')
    paths = sorted((ROOT / 'engine/profiles').glob('*.json'))
    if args.profiles:
        paths = [p for p in paths if p.stem in args.profiles]
        if len(paths) != len(set(args.profiles)):
            raise ValueError('Unknown profile name')
    for path in paths:
        profile = json.loads(path.read_text())
        target = {'linux': 'x86_64-unknown-linux-gnu', 'windows': 'x86_64-pc-windows-msvc',
                  'macos': 'aarch64-apple-darwin' if profile['arch'] == 'aarch64' else 'x86_64-apple-darwin'}[profile['os']]
        lock = ROOT / 'engine/locks' / profile['lock']
        with tempfile.TemporaryDirectory(prefix='voice-warp-lock-') as directory:
            requirements = Path(directory) / (profile['id'] + '.in')
            requirements.write_text('\n'.join(profile['requirements']) + '\n')
            command = [args.uv, 'pip', 'compile', str(requirements), '--python-version', profile['python'],
                       '--python-platform', target, '--generate-hashes', '--no-header', '--no-annotate', '--quiet', '--system-certs',
                       '--output-file', str(lock)]
            if args.upgrade:
                command.append('--upgrade')
            subprocess.run(command, check=True)
        profile['lock_sha256'] = hashlib.sha256(lock.read_bytes()).hexdigest()
        path.write_text(json.dumps(profile, indent=2) + '\n')


if __name__ == '__main__':
    main()
