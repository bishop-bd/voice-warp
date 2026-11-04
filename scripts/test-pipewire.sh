#!/usr/bin/env bash
# Integration test against an isolated audio server, without microphone hardware.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"

if [[ "${1:-}" != --inside ]]; then
    for program in pipewire pipewire-pulse wireplumber pactl parec pacat dbus-run-session cargo; do
        if ! command -v "$program" >/dev/null 2>&1; then
            echo "Missing $program. Install PipeWire, WirePlumber, pulseaudio-utils, D-Bus and Rust." >&2
            exit 1
        fi
    done
    test_runtime="$(mktemp -d -t voice-warp-pipewire.XXXXXX)"
    chmod 700 "$test_runtime"
    # Only the exact directory returned by mktemp is removed.
    trap 'rm -rf -- "$test_runtime"' EXIT
    mkdir -p "$test_runtime/config" "$test_runtime/state" "$test_runtime/cache"
    dbus-run-session -- env \
        XDG_RUNTIME_DIR="$test_runtime" \
        XDG_CONFIG_HOME="$test_runtime/config" \
        XDG_STATE_HOME="$test_runtime/state" \
        XDG_CACHE_HOME="$test_runtime/cache" \
        PIPEWIRE_RUNTIME_DIR="$test_runtime" \
        PIPEWIRE_REMOTE=pipewire-0 \
        PULSE_RUNTIME_PATH="$test_runtime/pulse" \
        PULSE_SERVER="unix:$test_runtime/pulse/native" \
        VOICE_WARP_TEST_ISOLATED=1 \
        bash "$script_dir/test-pipewire.sh" --inside
    exit
fi

if [[ "${VOICE_WARP_TEST_ISOLATED:-}" != 1 || "${XDG_RUNTIME_DIR:-}" != *voice-warp-pipewire.* ]]; then
    echo 'Refusing to run an integration test outside the isolated session.' >&2
    exit 1
fi

server_pids=()
cleanup() {
    result=$?
    trap - EXIT
    for ((i=${#server_pids[@]}-1; i>=0; i--)); do
        kill "${server_pids[i]}" 2>/dev/null || true
    done
    for pid in "${server_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    if [[ $result -ne 0 ]]; then
        for log in "$XDG_RUNTIME_DIR"/*.log; do
            [[ -f "$log" ]] || continue
            echo "Audio server log: $log" >&2
            tail -n 60 "$log" >&2
        done
    fi
    exit "$result"
}
trap cleanup EXIT

pipewire >"$XDG_RUNTIME_DIR/pipewire.log" 2>&1 &
server_pids+=("$!")
for ((attempt=0; attempt<100; attempt++)); do
    [[ -S "$XDG_RUNTIME_DIR/pipewire-0" ]] && break
    sleep 0.1
done
if [[ ! -S "$XDG_RUNTIME_DIR/pipewire-0" ]]; then
    echo 'Isolated PipeWire failed to start.' >&2
    exit 1
fi
wireplumber >"$XDG_RUNTIME_DIR/wireplumber.log" 2>&1 &
server_pids+=("$!")
pipewire-pulse >"$XDG_RUNTIME_DIR/pipewire-pulse.log" 2>&1 &
server_pids+=("$!")
for ((attempt=0; attempt<100; attempt++)); do
    pactl info >/dev/null 2>&1 && break
    sleep 0.1
done
pactl info >/dev/null
cd -- "$project_root"
cargo test --locked --lib audio::pulse::tests::pipewire_virtual_microphone_roundtrip -- \
    --ignored --exact --nocapture --test-threads=1
