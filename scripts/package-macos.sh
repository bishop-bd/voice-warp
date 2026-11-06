#!/usr/bin/env bash
# Build on the Mac that will run the app; no virtual audio driver is bundled.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo 'Build the macOS app on a Mac with Xcode Command Line Tools and Rust installed.' >&2
    exit 1
fi
for program in cargo rustc xcrun plutil codesign ditto shasum python3; do
    if ! command -v "$program" >/dev/null 2>&1; then
        echo "Missing $program. Install Xcode Command Line Tools, Rust, and Python 3.10 or newer first." >&2
        exit 1
    fi
done
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo 'Desktop packaging needs Python 3.10 or newer; building AI runtimes needs Python 3.12.' >&2
    exit 1
fi
xcrun --find clang >/dev/null

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
output_dir="${1:-$project_root/dist}"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd -P)"
version="$(awk -F '"' '/^version[[:space:]]*=/ { print $2; exit }' "$project_root/Cargo.toml")"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo 'The app bundle requires a numeric major.minor.patch Cargo version.' >&2
    exit 1
fi
target="$(rustc -vV | awk '/^host:/ { print $2 }')"
case "$target" in
    aarch64-apple-darwin) architecture=arm64 ;;
    x86_64-apple-darwin) architecture=x86_64 ;;
    *) echo "Unsupported native macOS target: $target" >&2; exit 1 ;;
esac

cd -- "$project_root"
runtime_arch="$architecture"
if [[ "$runtime_arch" == arm64 ]]; then runtime_arch=aarch64; fi
catalog_args=(--check-desktop --target "macos-$runtime_arch")
if [[ -n "${VOICE_WARP_RUNTIME_BUNDLE:-}" ]]; then
    catalog_args+=(--verify-bundle "$VOICE_WARP_RUNTIME_BUNDLE")
fi
python3 scripts/assemble-ai-release.py "${catalog_args[@]}"
# Match LSMinimumSystemVersion in the bundle; the default is macOS 11 or newer.
MACOSX_DEPLOYMENT_TARGET=11.0 cargo build --release --locked --target "$target" --target-dir "$project_root/target"

package_name="voice-warp-$version-macos-$architecture"
staging_root="$(mktemp -d "$output_dir/.voice-warp-macos.XXXXXX")"
cleanup() {
    # The path is the exact mktemp result under the resolved output directory.
    case "$staging_root" in
        "$output_dir"/.voice-warp-macos.*) rm -rf -- "$staging_root" ;;
        *) echo 'Refusing to remove a staging path outside the packaging directory.' >&2 ;;
    esac
}
trap cleanup EXIT
payload_dir="$staging_root/$package_name"
app_dir="$payload_dir/Voice Warp.app"
mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources"
install -m755 "$project_root/target/$target/release/voice-warp" "$app_dir/Contents/MacOS/voice-warp"
install -m644 "$project_root/packaging/Info.plist" "$app_dir/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $version" "$app_dir/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $version" "$app_dir/Contents/Info.plist"
plutil -lint "$app_dir/Contents/Info.plist"

if [[ -n "${VOICE_WARP_RUNTIME_BUNDLE:-}" ]]; then
    cp -R "$VOICE_WARP_RUNTIME_BUNDLE/ai-runtimes" "$app_dir/Contents/Resources/ai-runtimes"
fi
if [[ -n "${VOICE_WARP_CODESIGN_IDENTITY:-}" ]]; then
    codesign --force --options runtime --timestamp --entitlements "$project_root/packaging/desktop-entitlements.plist" --sign "$VOICE_WARP_CODESIGN_IDENTITY" "$app_dir"
else
    # Ad-hoc signing is only a local development build.
    codesign --force --sign - "$app_dir"
fi
codesign --verify --strict "$app_dir"
cp "$project_root/LICENSE" "$project_root/README.md" "$payload_dir/"
cp -R "$project_root/docs" "$payload_dir/docs"
(
    cd -- "$payload_dir"
    shasum -a 256 'Voice Warp.app/Contents/MacOS/voice-warp' > SHA256SUMS.txt
)
ditto -c -k --sequesterRsrc --keepParent "$payload_dir" "$output_dir/$package_name.zip"
if [[ -n "${VOICE_WARP_NOTARY_PROFILE:-}" ]]; then
    xcrun notarytool submit "$output_dir/$package_name.zip" --keychain-profile "$VOICE_WARP_NOTARY_PROFILE" --wait
    xcrun stapler staple "$app_dir"
    xcrun stapler validate "$app_dir"
    # The stapled ticket changes the app; recreate its distribution archive and
    # compute the checksum only after the final bytes are in place.
    rm -- "$output_dir/$package_name.zip"
    ditto -c -k --sequesterRsrc --keepParent "$payload_dir" "$output_dir/$package_name.zip"
fi
(
    cd -- "$output_dir"
    shasum -a 256 "$package_name.zip" > "$package_name.zip.sha256"
)
printf 'Created %s/%s.zip\n' "$output_dir" "$package_name"
printf 'This is a native %s development bundle; verify microphone routing on this Mac.\n' "$architecture"
