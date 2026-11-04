#!/usr/bin/env bash
# Build on the Mac that will run the app; no virtual audio driver is bundled.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo 'Build the macOS app on a Mac with Xcode Command Line Tools and Rust installed.' >&2
    exit 1
fi
for program in cargo rustc xcrun plutil codesign ditto shasum; do
    if ! command -v "$program" >/dev/null 2>&1; then
        echo "Missing $program. Install Xcode Command Line Tools and Rust first." >&2
        exit 1
    fi
done
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

# Ad-hoc signing supports local development; this is not Developer ID signing
# or notarization. No hardened-runtime/sandbox entitlements are enabled here.
codesign --force --sign - "$app_dir"
codesign --verify --strict "$app_dir"
cp "$project_root/LICENSE" "$project_root/README.md" "$payload_dir/"
cp -R "$project_root/docs" "$payload_dir/docs"
(
    cd -- "$payload_dir"
    shasum -a 256 'Voice Warp.app/Contents/MacOS/voice-warp' > SHA256SUMS.txt
)
ditto -c -k --sequesterRsrc --keepParent "$payload_dir" "$output_dir/$package_name.zip"
(
    cd -- "$output_dir"
    shasum -a 256 "$package_name.zip" > "$package_name.zip.sha256"
)
printf 'Created %s/%s.zip\n' "$output_dir" "$package_name"
printf 'This is a native %s development bundle; verify microphone routing on this Mac.\n' "$architecture"
