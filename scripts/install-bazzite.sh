#!/usr/bin/env bash
# Run this on the Bazzite host, as your normal desktop user.
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
    echo 'This installer runs on Linux. Copy this project to your Bazzite machine first.' >&2
    exit 1
fi
if [[ $EUID -eq 0 ]]; then
    echo 'Run this script as your normal desktop user, without sudo.' >&2
    exit 1
fi
if [[ -f /run/.containerenv || -f /.dockerenv ]]; then
    echo 'Run install-bazzite.sh from the Bazzite host terminal, outside any container.' >&2
    exit 1
fi
if ! command -v distrobox >/dev/null 2>&1; then
    echo 'Distrobox is required. Use the Distrobox installation provided by Bazzite.' >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
container_name="${VOICE_WARP_CONTAINER:-voice-warp}"
build_from_source="${VOICE_WARP_BUILD_FROM_SOURCE:-0}"
if [[ ! "$container_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo 'VOICE_WARP_CONTAINER must contain only letters, numbers, dots, underscores, or hyphens.' >&2
    exit 1
fi
if [[ "$build_from_source" != 0 && "$build_from_source" != 1 ]]; then
    echo 'VOICE_WARP_BUILD_FROM_SOURCE must be 0 or 1.' >&2
    exit 1
fi
if [[ ! -f "$project_root/Cargo.lock" ]]; then
    echo 'Cargo.lock is missing. Run this installer from a complete Voice Warp checkout.' >&2
    exit 1
fi

# Distrobox's table format is ID | NAME | STATUS | IMAGE. Match the whole name.
boxes="$(distrobox list --no-color)"
if awk -F '|' -v wanted="$container_name" '
    { name=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", name) }
    name == wanted { found=1 }
    END { exit !found }
' <<< "$boxes"; then
    printf 'Using existing Distrobox: %s\n' "$container_name"
else
    printf 'Creating Fedora 43 Distrobox: %s\n' "$container_name"
    distrobox create --yes --name "$container_name" \
        --image registry.fedoraproject.org/fedora:43 \
        --volume "$project_root:$project_root"
fi

printf '\nInstalling inside %s. The Bazzite host image stays unchanged.\n' "$container_name"
if [[ "$build_from_source" != 1 && "$(uname -m)" == x86_64 && -f "$project_root/bin/voice-warp" ]]; then
    printf 'Using the bundled Linux binary after checksum verification.\n'
else
    printf 'No matching bundled binary selected; compiling from source.\n'
fi
distrobox enter --name "$container_name" -- \
    env VOICE_WARP_BUILD_FROM_SOURCE="$build_from_source" \
    bash "$project_root/scripts/install-in-container.sh"

printf '\nInstalled. Open “Voice Warp” from your application launcher.\n'
printf 'Terminal launch: distrobox enter --name %s -- /usr/local/bin/voice-warp\n' "$container_name"
printf 'Select “Voice Warp” as the microphone in Discord, games, or OBS after starting the app.\n'
