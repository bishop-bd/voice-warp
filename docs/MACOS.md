# Voice Warp on macOS

Voice Warp includes a native CoreAudio path for microphone capture and selected-device playback, shared Rust voice effects, the soundboard, a managed native AI worker launcher, and the optional external AI adapter. **The macOS build and live routing have not been verified on a Mac during this implementation.** CI is configured to compile and test macOS; its results must be checked after the workflow actually runs.

## Build on your Mac

Install Xcode Command Line Tools and Rust on the Mac, then run from a source checkout:

```bash
bash scripts/package-macos.sh
```

The script builds for the native Rust host architecture: Apple Silicon (`arm64`) or Intel (`x86_64`). It creates a ZIP under `dist/` containing **Voice Warp.app**, documentation, the MIT license, and checksums. The bundle targets macOS 11 or newer and includes the microphone permission explanation required by [Apple's `NSMicrophoneUsageDescription` documentation](https://developer.apple.com/documentation/bundleresources/information-property-list/nsmicrophoneusagedescription).

Extract the ZIP, move **Voice Warp.app** to your Applications folder, and open that bundle. Allow microphone access when macOS asks. If permission is disabled, check **System Settings → Privacy & Security → Microphone**.

This is an ad-hoc-signed development bundle, not a notarized Developer ID release or a universal binary. The script does not install a driver, publish the app, or alter Gatekeeper settings. Distribution to other Macs requires separate signing/notarization and testing. Building through `cargo run` may associate microphone permission with the terminal instead of the app bundle; use the packaged app for normal use.

## Route your changed voice to another app

1. Install **BlackHole 2ch** separately using [the project's official installation instructions](https://github.com/ExistentialAudio/BlackHole#installation-instructions) or [its official download page](https://existential.audio/blackhole/).
2. Open Voice Warp and refresh the devices in **Audio setup**.
3. Select your physical microphone as Voice Warp's input and **BlackHole 2ch** as its main output.
4. Choose a preset, then start the microphone. Play soundboard clips as needed.
5. In your call, game, or recording app, select **BlackHole 2ch** as the microphone/input.

```text
Physical microphone
  -> Voice Warp effects / local AI + soundboard
  -> BlackHole 2ch output
  -> BlackHole 2ch input
  -> Call or recording app
```

BlackHole routes audio between applications; it is a separate driver with its own license. Voice Warp does not bundle or install it. Leave the Mac's ordinary playback output set to your headphones, and select BlackHole specifically inside Voice Warp. No Aggregate Device is required for this route because Voice Warp opens the microphone and output separately. BlackHole's [official support pages](https://existential.audio/blackhole/support/) explain its routing and additional device options.

Use Voice Warp's separate monitoring selection for physical headphones if you want to hear the result. Keep it different from the BlackHole main output. Do not select BlackHole as Voice Warp's microphone input while also sending audio into it; that creates a feedback route.

For AI conversion, the managed worker supports native CPU and Apple Silicon Metal/MPS execution paths. **The trusted native download catalog is currently empty.** A matching candidate runtime must first be built and installed using [native runtime packaging](AI-RUNTIME-PACKAGES.md), then tested on the target Mac. The resulting runtime does not require system Python or a separate w-okada server. The external adapter remains available for an existing working server.

Apple Silicon and Intel packages are separate artifacts, and the AI runtime's minimum macOS version may exceed the desktop app's macOS 11 minimum. Intel CPU dependencies need qualification; Intel Mac AMD GPU support is not released, and current NVIDIA CUDA is unavailable on macOS. Metal can retain CPU stages, which the worker reports after model warmup. Follow [the AI guide](AI.md) for model setup, device checks, and current validation limits.

## Checks on a real Mac

After compiling, verify microphone permission, device enumeration, live capture, monitoring, and BlackHole input in a second application. Test stopping/restarting, unplugging the microphone, soundboard playback, and mute controls. Hardware-free unit tests cannot establish that these operations work on the target Mac.

If a device does not open, try setting its format to 48 kHz in **Audio MIDI Setup**, then refresh/restart Voice Warp. If the destination app is silent, confirm it uses BlackHole 2ch as its input and that Voice Warp is running with the physical microphone selected.
