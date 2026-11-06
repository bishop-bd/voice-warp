# Voice Warp on Windows

Voice Warp uses the native Windows audio backend to capture a microphone, apply Rust voice effects or local AI conversion, mix soundboard clips, and play the result into a selected output device. To make that output selectable as a microphone in other applications, install a virtual audio cable separately.

## Install and route audio

1. Extract the Windows x64 release ZIP and open `voice-warp.exe`.
2. Download **VB-CABLE** from [VB-Audio's official website](https://vb-audio.com/Cable/). Follow its installer instructions, including its requested administrator access and reboot. Voice Warp does not download, install, or include this driver. VB-CABLE has its own license terms.
3. Open **Audio setup** in Voice Warp and refresh devices after the driver is installed.
4. Select your **physical microphone** as the input.
5. Select **CABLE Input (VB-Audio Virtual Cable)** as Voice Warp's output. The exact label may vary by driver version or system language.
6. Choose a voice preset and start the microphone. Soundboard clips are mixed into the same output.
7. In Discord, OBS, your game, or your browser call, select **CABLE Output (VB-Audio Virtual Cable)** as the microphone.

```text
Your physical microphone
  -> Voice Warp effects / local AI + soundboard
  -> CABLE Input       [Windows playback device]
  -> CABLE Output      [Windows recording device]
  -> Your call or game
```

The names are from the cable's perspective: audio is played **into CABLE Input** and recorded **from CABLE Output**. VB-Audio documents that connection on its [product page](https://vb-audio.com/Cable/).

Keep your game's speakers/output set to your headphones. Select the cable explicitly in Voice Warp and your call app; making it the Windows default speaker would also send other desktop sounds into the cable. Voice Warp does not change the Windows default audio devices. The installed cable remains available after Voice Warp closes; it carries no Voice Warp audio while the app is stopped.

For monitoring, select a separate physical headphone output in Voice Warp. Monitoring and the main output must use different devices. Selecting ordinary headphones as the main output lets you hear the effects, but it does not expose a microphone for another application.

## Effects, soundboard, and AI

The voice presets and soundboard work with the native backend. Import your own WAV clips through **Soundboard → Import WAV** while stopped. Keys **1–9** play sounds and **Esc** stops them while the app window is focused. **Mute voice** leaves the soundboard audible; **Mute all** silences both.

AI is optional. The app can now own a native Windows x64 worker, with CPU, NVIDIA CUDA, and partial ONNX/DirectML execution paths. The runtime package contains its interpreter/libraries, so inference does not require a separate w-okada server, Python installation, WSL, or Docker. **The trusted native download catalog is currently empty:** build and install an explicitly trusted candidate package using [native runtime packaging](AI-RUNTIME-PACKAGES.md), or keep the existing external w-okada v1 setup. Candidate builds and GPU paths still require clean-machine and hardware qualification; native Windows ROCm packaging remains pending.

Follow [the AI setup guide](AI.md) for backend/device selection, model testing, diagnostics, and limitations. DirectML requires an ONNX voice and retains CPU stages. The audio route remains the same: converted audio goes to the selected cable output. Voice models and native runtime payloads are separate from the desktop ZIP.

## Build a Windows package

On Windows, install Rust's `x86_64-pc-windows-msvc` toolchain and Microsoft C++ Build Tools with the Windows SDK. From a source checkout, run in PowerShell:

```powershell
.\scripts\package-windows.ps1
```

The script builds the locked release with a statically linked MSVC C runtime, then writes `dist/voice-warp-VERSION-windows-x86_64.zip` and its SHA-256 file. The ZIP contains the executable, documentation, license, and an executable checksum. It does not require Rust or Visual Studio on the destination PC. Windows system libraries and a working graphics driver are still required. No virtual audio drivers are bundled or installed by the script.

To choose another artifact directory:

```powershell
.\scripts\package-windows.ps1 -OutputDirectory C:\Builds\VoiceWarp
```

The script restores any existing Rust build flags when it finishes. A normal `cargo build --release --locked` uses the toolchain's default runtime linkage instead; such a build may require the [Microsoft Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Troubleshooting

- **The cable is missing:** complete the driver's installation/reboot, then refresh devices. Keep input and output selections distinct: physical microphone in, CABLE Input out.
- **Other apps hear the original voice:** explicitly change that app's microphone to CABLE Output. Existing calls may need their input reselected.
- **No microphone audio:** check **Settings → Privacy & security → Microphone**, including access for desktop apps. Also confirm the physical microphone works in Windows and neither mute control is active.
- **A device cannot start:** close applications holding it in exclusive mode, select another device, or set the microphone and cable formats to 48 kHz in Windows sound settings, then restart Voice Warp.
- **Feedback or doubled speech:** use headphones, disable duplicate monitoring such as Windows **Listen to this device**, and avoid choosing CABLE Output as Voice Warp's microphone input.
- **Driver or device disconnected:** stop, reconnect or reinstall the affected device as appropriate, refresh, and start again. Check the displayed error before continuing a call.

The Windows backend still needs real-device and virtual-cable validation on the target PC. A successful build and unit tests do not verify a particular microphone, driver, or call application's routing.
