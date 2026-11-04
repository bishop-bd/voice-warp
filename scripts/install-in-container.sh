#!/usr/bin/env bash
# Called by install-bazzite.sh. Never install build dependencies on the host.
set -euo pipefail

if [[ ! -f /run/.containerenv && ! -f /.dockerenv ]]; then
    echo 'This script must run inside the Voice Warp Distrobox container.' >&2
    exit 1
fi
if ! command -v distrobox-export >/dev/null 2>&1; then
    echo 'This is not a configured Distrobox. Run scripts/install-bazzite.sh on the host.' >&2
    exit 1
fi
if [[ $EUID -eq 0 ]]; then
    echo 'Enter Distrobox as your normal user; this script uses sudo only for container packages.' >&2
    exit 1
fi
# This file belongs to the container image, not the project or a remote script.
# shellcheck source=/dev/null
source /etc/os-release
if [[ "${ID:-}" != fedora || "${VERSION_ID:-}" != 43 ]]; then
    echo 'Use a Fedora 43 Distrobox. Set VOICE_WARP_CONTAINER to a fresh name if needed.' >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
build_from_source="${VOICE_WARP_BUILD_FROM_SOURCE:-0}"
if [[ "$build_from_source" != 0 && "$build_from_source" != 1 ]]; then
    echo 'VOICE_WARP_BUILD_FROM_SOURCE must be 0 or 1.' >&2
    exit 1
fi

if [[ "$build_from_source" != 1 && "$(uname -m)" == x86_64 && -f "$project_root/bin/voice-warp" ]]; then
    if [[ ! -f "$project_root/bin/voice-warp.sha256" ]]; then
        echo 'The bundled binary is missing bin/voice-warp.sha256. Use the complete release archive or set VOICE_WARP_BUILD_FROM_SOURCE=1.' >&2
        exit 1
    fi
    printf 'Verifying bundled Linux x86_64 binary…\n'
    (
        cd -- "$project_root/bin"
        # Require an entry for the actual binary, rather than accepting a
        # manifest that successfully verifies only an unrelated file.
        if ! grep -Eq $'^[[:xdigit:]]{64} [ *]voice-warp\r?$' voice-warp.sha256; then
            echo 'The checksum manifest has no valid entry for voice-warp.' >&2
            exit 1
        fi
        if ! sha256sum --check --strict voice-warp.sha256; then
            echo 'Bundled binary checksum verification failed. Extract the release again or set VOICE_WARP_BUILD_FROM_SOURCE=1.' >&2
            exit 1
        fi
    )
    printf 'Installing runtime dependencies inside Distrobox…\n'
    sudo dnf install -y \
        libX11 libXcursor libXi libXrandr libxcb libxkbcommon libxkbcommon-x11 \
        wayland-libs mesa-libGL mesa-libEGL mesa-dri-drivers libstdc++ \
        fontconfig freetype pulseaudio-utils desktop-file-utils
    binary_path="$project_root/bin/voice-warp"
else
    printf 'Building Voice Warp from source inside Distrobox…\n'
    sudo dnf install -y \
        rust cargo gcc gcc-c++ make pkgconf-pkg-config \
        libX11-devel libXcursor-devel libXi-devel libXrandr-devel libxcb-devel \
        libxkbcommon-devel libxkbcommon-x11 wayland-devel \
        mesa-libGL-devel mesa-libEGL-devel mesa-dri-drivers \
        fontconfig freetype pulseaudio-utils desktop-file-utils

    # Prefer Fedora's compiler even if the shared home directory has rustup shims.
    cd -- "$project_root"
    RUSTC=/usr/bin/rustc /usr/bin/cargo build --release --locked --target-dir "$project_root/target"
    binary_path="$project_root/target/release/voice-warp"
fi

# Setting the mode here also handles archives extracted on NTFS or FAT.
sudo install -Dm755 "$binary_path" /usr/local/bin/voice-warp
sudo install -Dm644 "$project_root/packaging/voice-warp.desktop" \
    /usr/local/share/applications/voice-warp.desktop
sudo install -Dm644 "$project_root/packaging/voice-warp.svg" \
    /usr/local/share/icons/hicolor/scalable/apps/voice-warp.svg
desktop-file-validate /usr/local/share/applications/voice-warp.desktop
sudo update-desktop-database /usr/local/share/applications
distrobox-export --app /usr/local/share/applications/voice-warp.desktop --export-label none
