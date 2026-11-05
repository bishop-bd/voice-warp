#!/usr/bin/env bash
# Bundle a Fedora-compatible build; never ship the newer Ubuntu host binary by accident.
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/package-bazzite.sh --binary /path/to/Fedora-built/voice-warp'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ $# -ne 2 || "$1" != --binary ]]; then
    usage >&2
    exit 2
fi

for program in awk readelf sort sha256sum tar gzip sed mktemp; do
    if ! command -v "$program" >/dev/null 2>&1; then
        printf 'Required packaging command is missing: %s\n' "$program" >&2
        exit 1
    fi
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
binary_path="$(realpath -- "$2")"
if [[ ! -f "$binary_path" || ! -x "$binary_path" ]]; then
    echo 'Pass an executable, prebuilt Voice Warp binary with --binary.' >&2
    exit 1
fi

version="$(awk '
    /^\[package\]$/ { package = 1; next }
    /^\[/ { package = 0 }
    package && /^version[[:space:]]*=/ {
        split($0, parts, "\""); print parts[2]; exit
    }
' "$project_root/Cargo.toml")"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][A-Za-z0-9.-]+)?$ ]]; then
    echo 'Could not read a valid package version from Cargo.toml.' >&2
    exit 1
fi

elf_header="$(LC_ALL=C readelf --file-header -- "$binary_path")"
if ! grep -Eq 'Class:[[:space:]]+ELF64' <<< "$elf_header" || \
    ! grep -Eq 'Machine:[[:space:]]+Advanced Micro Devices X86-64' <<< "$elf_header"; then
    echo 'The Bazzite package requires an x86_64 ELF executable.' >&2
    exit 1
fi
glibc_versions="$(LC_ALL=C readelf --version-info -- "$binary_path" | \
    grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sort -Vu || true)"
if [[ -z "$glibc_versions" ]]; then
    echo 'Could not determine the executable glibc requirements.' >&2
    exit 1
fi
maximum_glibc="$(tail -n 1 <<< "$glibc_versions")"
if [[ "$(printf '%s\n' GLIBC_2.42 "$maximum_glibc" | sort -V | tail -n 1)" != GLIBC_2.42 ]]; then
    printf 'This binary requires %s; Fedora 43 supplies GLIBC_2.42. Rebuild it in Fedora 43.\n' "$maximum_glibc" >&2
    exit 1
fi
binary_help="$("$binary_path" --help)"
if [[ "${binary_help%%$'\n'*}" != "Voice Warp $version" ]]; then
    printf 'The supplied binary does not report Voice Warp %s. Rebuild the current source.\n' "$version" >&2
    exit 1
fi

workdir="$(mktemp -d -t voice-warp-package.XXXXXX)"
trap 'rm -rf -- "$workdir"' EXIT
package="$workdir/package"
mkdir -p "$package/bin"
install -m755 "$binary_path" "$package/bin/voice-warp"
(
    cd -- "$package/bin"
    sha256sum voice-warp > voice-warp.sha256
)
for relative in \
    LICENSE \
    docs/BAZZITE.md \
    packaging/voice-warp.desktop \
    packaging/voice-warp.svg \
    scripts/install-bazzite.sh \
    scripts/install-in-container.sh \
    scripts/uninstall-bazzite.sh; do
    if [[ ! -f "$project_root/$relative" ]]; then
        printf 'Required package file is missing: %s\n' "$relative" >&2
        exit 1
    fi
    install -Dm644 "$project_root/$relative" "$package/$relative"
done

tar --create --sort=name --owner=0 --group=0 --numeric-owner \
    --directory "$package" . | gzip -n > "$workdir/payload.tar.gz"
payload_sha256="$(sha256sum < "$workdir/payload.tar.gz")"
payload_sha256=${payload_sha256%% *}
sed \
    -e "s/@VOICE_WARP_VERSION@/$version/g" \
    -e "s/@VOICE_WARP_PAYLOAD_SHA256@/$payload_sha256/g" \
    "$project_root/packaging/bazzite-installer.sh.in" > "$workdir/installer.run"
cat "$workdir/payload.tar.gz" >> "$workdir/installer.run"
chmod 755 "$workdir/installer.run"
bash "$workdir/installer.run" --check

dist="$project_root/dist"
mkdir -p -- "$dist"
filename="Voice-Warp-$version-Bazzite-x86_64.run"
install -m755 "$workdir/installer.run" "$dist/$filename"
(
    cd -- "$dist"
    sha256sum "$filename" > "$filename.sha256"
)
printf '\nCreated %s\n' "$dist/$filename"
printf 'Install on Bazzite: bash "%s"\n' "$filename"
