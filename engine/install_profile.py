#!/usr/bin/env python3
"""Install one reviewed runtime profile with complete, hash-checked dependencies."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
ROCM_BASE = 'sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191'


def load_profile(name):
    if not re.fullmatch(r'[a-z0-9_-]+', name):
        raise ValueError('Invalid runtime profile name')
    profile = json.loads((ROOT / 'profiles' / (name + '.json')).read_text())
    lock = ROOT / 'locks' / profile['lock']
    if lock.parent != ROOT / 'locks' or lock.name != name + '.txt':
        raise ValueError('Runtime lock must belong to the selected profile')
    if hashlib.sha256(lock.read_bytes()).hexdigest() != profile['lock_sha256']:
        raise ValueError('Runtime dependency lock checksum differs from its profile')
    return profile, lock


def requirement_blocks(contents):
    # uv emits one requirement followed by indented hashes/comments.
    return re.findall(r'^[^\s#].*(?:\n[ \t].*)*', contents, re.MULTILINE)


def checked_requirements(profile, lock, wheel_cache=None):
    blocks = requirement_blocks(lock.read_text())
    result = []
    for block in blocks:
        if '--hash=sha256:' not in block or not ('==' in block.splitlines()[0] or ' @ https://' in block.splitlines()[0]):
            raise ValueError('Every dependency must be pinned and hashed')
        match = re.match(r'([\w-]+) @ (https://\S+)', block)
        if match:
            name, url = match.groups()
            filename = unquote(urlsplit(url).path.rsplit('/', 1)[-1])
            # This image is itself immutable and already contains the exact AMD
            # wheels. Reuse only these three distributions from that pinned base.
            if (os.environ.get('VOICE_WARP_BASE_IMAGE_DIGEST') == ROCM_BASE
                    and profile['os'] == 'linux' and profile['backend'] in ('rocm', 'migraphx')
                    and name in ('torch', 'torchaudio', 'triton')):
                installed = importlib.metadata.version(name)
                if installed != filename.split('-')[1]:
                    raise ValueError(f'Pinned ROCm image contains unexpected {name} {installed}')
                continue
            if wheel_cache:
                cached = wheel_cache / filename
                if cached.is_file():
                    # pip verifies the unchanged lock hash after the URL rewrite.
                    block = block.replace(url, cached.resolve().as_uri(), 1)
        result.append(block)
    return result


def install(name, wheel_cache=None):
    profile, lock = load_profile(name)
    host_os = {'Linux': 'linux', 'Windows': 'windows', 'Darwin': 'macos'}.get(platform.system())
    host_arch = {'AMD64': 'x86_64', 'x86_64': 'x86_64', 'arm64': 'aarch64', 'aarch64': 'aarch64'}.get(platform.machine())
    if (profile['os'], profile['arch'], profile['python']) != (host_os, host_arch, f'{sys.version_info.major}.{sys.version_info.minor}'):
        raise ValueError(f'Profile {name} does not match this OS, architecture and Python ABI')
    blocks = checked_requirements(profile, lock, wheel_cache)
    with tempfile.TemporaryDirectory(prefix='voice-warp-lock-') as directory:
        requirements = Path(directory) / 'requirements.txt'
        requirements.write_text('\n'.join(blocks) + '\n')
        # AMD's Windows rocm meta-package is an audited sdist. Its only build
        # backend is setuptools; bootstrap that exact hashed toolchain first and
        # disable build isolation so pip cannot fetch unpinned build dependencies.
        bootstrap = Path(directory) / 'bootstrap.txt'
        bootstrap.write_text('\n'.join(b for b in blocks if re.match(r'(pip|setuptools|wheel)==', b)) + '\n')
        common = [sys.executable, '-m', 'pip', 'install', '--require-hashes', '--no-deps', '--only-binary=:all:', '--disable-pip-version-check']
        subprocess.run(common + ['-r', str(bootstrap)], check=True)
        if profile['os'] == 'windows' and profile['backend'] == 'rocm':
            common += ['--no-binary=rocm', '--no-build-isolation']
        subprocess.run(common + ['-r', str(requirements)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile')
    parser.add_argument('--wheel-cache', type=Path)
    args = parser.parse_args()
    install(args.profile, args.wheel_cache)
