# Voice Warp

A native **Rust** voice changer and soundboard for **Windows, Linux/Bazzite, and macOS**. The desktop UI, voice effects, soundboard, and optional local AI adapter are shared. Audio routing uses each platform's native audio system.

## Platform support

| Platform | Audio backend | How other apps receive your voice |
| --- | --- | --- |
| Windows | WASAPI through CPAL | Send to an installed virtual cable's playback endpoint; select its recording endpoint in calls/games. |
| Linux / Bazzite | PipeWire's PulseAudio compatibility interface | Voice Warp creates a selectable **Voice Warp** microphone automatically. |
| macOS | CoreAudio through CPAL | Send to an installed loopback device such as BlackHole; select it as the microphone in other apps. |

Windows and macOS implement live audio, not only UI preview. Install a virtual audio driver separately on those platforms to expose processed audio as a system microphone. **No third-party driver is bundled, downloaded, or installed by Voice Warp.**

Windows and Linux builds can be produced on the current development host. macOS has a native backend and app-bundle build script but requires a Mac for build/runtime validation. Unit tests and device enumeration do not establish end-to-end microphone quality on every machine. See the platform guides for verification limits.

## Windows quick start

1. Extract the Windows x64 ZIP and open `voice-warp.exe`.
2. Install a virtual audio cable, such as [VB-CABLE from VB-Audio](https://vb-audio.com/Cable/), following its vendor's instructions. Restart Windows if requested.
3. In **Audio setup**, choose your physical microphone and **CABLE Input** as the playback destination.
4. Choose a preset and click **Start microphone**.
5. In Discord, OBS, your game, or browser, select **CABLE Output** as the microphone.

VB-CABLE's names are from the driver's perspective: audio sent to *CABLE Input* is received from *CABLE Output*. Other independently installed virtual cables can also be selected. Voice Warp never automatically chooses speakers or changes Windows' default microphone.

See [Windows setup and troubleshooting](docs/WINDOWS.md). Build a portable Windows release from source with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-windows.ps1
```

The script builds the MSVC x64 target with a statically linked C runtime and creates a ZIP in `dist`. Rust and Visual Studio C++ tools are needed to build, not to run the package.

## Bazzite quick start

For the prebuilt package, copy `Voice-Warp-0.3.0-Bazzite-x86_64.run` to your Bazzite machine and run this in Desktop Mode:

```bash
bash Voice-Warp-0.3.0-Bazzite-x86_64.run
```

Open **Voice Warp** from the applications menu when installation finishes. No compilation or reboot is needed. The first installation needs internet access. See [Bazzite installation, updates, and removal](docs/BAZZITE.md).

### Install from a source checkout

Copy or extract the project under your home directory. From a **host terminal**, run without `sudo`:

```bash
bash scripts/install-bazzite.sh
```

The installer creates a Fedora 43 Distrobox named `voice-warp`, installs dependencies **inside the container**, and exports a desktop launcher. A bundled Linux x86-64 binary is checksum-verified and installed directly; source checkouts build from the locked Rust dependencies. It may ask for a password for container packages. It does not layer packages onto the Bazzite host or require a host reboot.

Open **Voice Warp**, choose your physical mic, start processing, and select **Voice Warp** as the input in your other apps. Terminal launch:

```bash
distrobox enter --name voice-warp -- /usr/local/bin/voice-warp
```

Choose a fresh container name with `VOICE_WARP_CONTAINER=voice-warp-app bash scripts/install-bazzite.sh`. Force compilation with `VOICE_WARP_BUILD_FROM_SOURCE=1 bash scripts/install-bazzite.sh`.

Packaging follows Bazzite's [container guidance](https://docs.bazzite.gg/Installing_and_Managing_Software/Containers/) and Distrobox's [audio/display integration](https://distrobox.it/usage/distrobox-create/) and [desktop export](https://distrobox.it/usage/distrobox-export/) documentation.

## macOS

Install a loopback driver separately, such as [BlackHole 2ch](https://existential.audio/blackhole/). Build the app on a Mac:

```bash
bash scripts/package-macos.sh
```

Open the app, allow microphone access, select the physical mic and BlackHole playback destination, then choose BlackHole as the input in receiving apps. The bundle includes the microphone permission description. Development builds are not Apple-notarized. See [macOS setup and validation limits](docs/MACOS.md).

## Voice effects and soundboard

- Eleven presets: **Natural, Titan, Helium, Synth, Ghost, Radio, Demon, Space Marine, Alien, Cartoon, and Announcer**.
- The new voices add a gritty deep growl (Demon), armored communications (Space Marine), a wavering layered voice (Alien), bright playful character (Cartoon), and warm room presence (Announcer).
- Adjustable pitch, warm/bright tone, robot texture, distortion, chorus, echo, reverb, voice gain, noise gate, radio filtering, and bypass. These run locally on the CPU. Fine-tune any preset; tone at zero is neutral, and distortion, chorus, and reverb at zero are off.
- Four original synthesized sounds: **Chime, Airhorn, Level up, and Rimshot**. Any sound, including a built-in, can be removed from the library; removals are saved across restarts.
- No fixed limit on the number of sounds in the library. Up to 8 sounds play simultaneously.
- Select multiple WAVs in **Import WAVs** to import them together. Invalid files are reported without stopping the remaining imports; selecting an existing file reuses its sound.
- PCM 8/16/24/32-bit and float32 WAV import, multichannel downmix, and 48 kHz resampling. Imports are limited to 60 seconds and 128 MiB each. Stop the microphone before importing.
- Keys **1–9** play sounds; **Esc** stops them while the app window is focused. These are not global game hotkeys.
- **Mute voice** leaves the soundboard audible. **Mute all** silences both. Clips mix after conversion, so voice effects and AI do not alter them.
- Input/output meters, processing time, dropped-block indicators, and optional headphone monitoring.

Settings and imported paths are saved in the user's configuration directory. Original files stay in place. AI and monitoring start disabled. Closing the window stops audio rather than minimizing to a tray.

## Optional local AI

The shared Rust adapter connects to a separately installed **w-okada v1 RVC server** on the same computer. Load a model in that server, then select **Local AI** in Voice Warp. Models and the inference server are not included. See [AI setup, supported revision, and limitations](docs/AI.md).

For the Radeon Vega 64 system, start with CPU inference. GPU acceleration depends on a separately compatible runtime and is not promised. Standard effects/soundboard require no AI runtime. The default AI chunk is 6144 samples, or 128 ms before inference and other buffering. Continuous real-time output requires a sufficiently fast model and processor.

The adapter accepts loopback addresses only, validates model/server settings, and never substitutes raw input for a failed conversion. The separately installed server can download assets; Voice Warp itself does not download models.

## Audio routing and failure behavior

```text
Physical microphone
  -> native capture / format conversion
  -> Rust effects OR local RVC server
  -> Rust soundboard mixer + mute + limiter
  -> playback destination
       Linux: managed null sink -> Voice Warp source
       Windows: virtual cable playback -> cable recording device
       macOS: loopback playback -> loopback input
  -> calls / games / recording apps
```

Processing uses 48 kHz mono float audio. The Windows/macOS backend adapts device sample formats, channel counts, and sample rates. Input and playback destinations are explicit. Audio callbacks use bounded queues; processing runs off the UI thread. When processing falls behind, stale blocks are dropped instead of allowing latency to grow indefinitely. The timing display measures a processing block, not total latency.

On **Linux**, the app loads PipeWire's [Pulse-compatible modules](https://docs.pipewire.org/page_pulse_modules.html) without rewriting configuration. It removes only its own devices at Stop/close and restores a previous default only while the current default still points to Voice Warp. Processing failure retains a silent source until explicit restart/disconnect/close.

On **Windows/macOS**, Stop/close releases native audio streams; independently installed virtual drivers remain available. System defaults are unchanged. Audio failures do not automatically switch to another input or output. Receiving applications and the OS control their own fallback behavior, so select the virtual input explicitly in those apps.

Use headphones for monitoring to avoid feedback. The monitoring output must differ from the virtual-cable destination. Reconnects involving multiple devices with identical names may require refreshing and reselecting them.

## Development and checks

Rust 1.88 or newer is required; `Cargo.lock` pins dependencies.

```bash
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --all -- --check
cargo build --release --locked
```

Linux builds need OpenGL/EGL, X11/Wayland libraries, `libxkbcommon`, and `pulseaudio-utils`; the Distrobox installer supplies them. Windows needs the MSVC C++ build tools. macOS needs Xcode command-line tools.

CI checks Windows, Linux, and macOS and packages native Windows/macOS builds. It also runs an isolated PipeWire roundtrip on Linux. The workflow is included; a remote CI run has not been executed from this workspace.

```bash
# Linux only; isolated synthetic input, no physical microphone
bash scripts/test-pipewire.sh
```

Useful commands (`voice-warp.exe` on Windows):

```text
voice-warp --doctor --report diagnostics.json
voice-warp --list-inputs
voice-warp --list-outputs
voice-warp --headless --input INPUT_ID --output OUTPUT_ID --seconds 10 --preset 1
voice-warp --render-demo demo.wav
voice-warp --preview
```

`--doctor` lists devices without recording. Headless Windows/macOS requires `--output`; omit it on Linux. `--headless` records the chosen input for the specified duration. `--render-demo` writes a synthetic effects/soundboard WAV without hardware; it is not a speech-quality test. `--preview` deliberately disables live audio.

## Troubleshooting

**Other apps hear the original voice:** select the virtual recording device in each app. Voice Warp cannot override a specific hardware-mic selection. Some calls need input reselection or restart.

**Windows device missing:** install the virtual driver, restart if requested, and refresh devices. Enable microphone access for desktop apps in Windows Privacy settings. Select both input and output before starting.

**Bazzite connection fails:** run as the desktop user, not root. Check `distrobox enter --name voice-warp -- pactl info`. PipeWire and its PulseAudio compatibility server must be running in the desktop session.

**Audio gaps:** inspect dropped blocks; reduce workloads or choose a lighter AI model/larger chunk. Larger chunks trade latency for throughput. Wired audio usually avoids Bluetooth microphone profile limitations.

**Linux device left after a forced crash:** close all instances and inspect `pactl list short modules`. Unload only Voice Warp's `module-remap-source` (`source_name=voice_warp`), null sink (`sink_name=voice_warp_sink`), and any loopback whose source is Voice Warp, using `pactl unload-module MODULE_ID`.

**Remove the Bazzite app:** close it, then run the `.run` installer with `--uninstall`, or run `bash scripts/uninstall-bazzite.sh` from a source checkout. Settings and original WAVs stay in your home. Windows is portable; uninstall audio drivers separately with vendor instructions.

## Code layout

`src/audio.rs` defines the interface. `audio/pulse.rs` implements Linux routing, `audio/native.rs` implements WASAPI/CoreAudio, and `audio/resample.rs` handles streaming rate conversion. `dsp.rs`, `soundboard.rs`, and `ai.rs` implement shared processing; `app.rs` is the native egui UI.

MIT licensed. Third-party drivers, inference engines, and models retain their own licenses.
