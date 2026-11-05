#!/usr/bin/env bash
# Remove only the application installed by the Bazzite package.
set -euo pipefail

if [[ "$(uname -s)" != Linux || $EUID -eq 0 ]]; then
    echo 'Run this from a Linux host terminal as your normal user, without sudo.' >&2
    exit 1
fi
if [[ -f /run/.containerenv || -f /.dockerenv ]]; then
    echo 'Run the uninstaller outside the container, from the Bazzite desktop.' >&2
    exit 1
fi
if ! command -v distrobox >/dev/null 2>&1; then
    echo 'Distrobox is required to uninstall the container app.' >&2
    exit 1
fi
container_name="${VOICE_WARP_CONTAINER:-voice-warp}"
if [[ ! "$container_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo 'Invalid VOICE_WARP_CONTAINER name.' >&2
    exit 1
fi
boxes="$(distrobox list --no-color)"
if ! awk -F '|' -v wanted="$container_name" '
    { name=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", name) }
    name == wanted { found=1 }
    END { exit !found }
' <<< "$boxes"; then
    printf 'No container named %s was found.\n' "$container_name"
    exit 0
fi

# Expand these paths inside the container, not in the host shell.
# shellcheck disable=SC2016
distrobox enter --name "$container_name" -- bash -c '
    set -euo pipefail
    desktop=/usr/local/share/applications/voice-warp-bazzite.desktop
    if [[ ! -f "$desktop" ]]; then
        echo "The Bazzite package is not installed in this container."
        exit 0
    fi
    distrobox-export --app "$desktop" --delete
    sudo rm -f /usr/local/bin/voice-warp /usr/local/bin/voice-warp.new \
        "$desktop" /usr/share/icons/hicolor/scalable/apps/voice-warp-bazzite.svg
    sudo update-desktop-database /usr/local/share/applications
'
printf 'Voice Warp and its menu shortcut have been removed.\n'
printf 'Your settings, WAV files, and the reusable %s container remain.\n' "$container_name"
