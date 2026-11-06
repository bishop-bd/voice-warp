# Install Voice Warp on Bazzite

Copy **Voice-Warp-0.3.0-Bazzite-x86_64.run** to your Bazzite computer. In Desktop Mode, open a terminal in the folder containing the file and run:

```bash
bash Voice-Warp-0.3.0-Bazzite-x86_64.run
```

Wait for **Installed. Open “Voice Warp” from your application launcher.** Then open **Voice Warp** from the applications menu.

The first installation needs an internet connection and may take several minutes to download its Fedora runtime and audio/graphics packages. Enter your password if the container's package installer asks for it. Run the installer as your normal user, without `sudo`. There is no compilation or reboot required.

The package is for x86-64 Bazzite desktops and handhelds, including Steam Deck. It installs into a Distrobox named `voice-warp` and adds a menu shortcut. Bazzite's host system image is not modified. You can delete the downloaded installer after installation.

## First use

1. In **Audio setup**, choose your physical microphone.
2. Choose a voice preset and click **Start microphone**.
3. Select **Voice Warp** as the input in Discord, OBS, or your calling app.

Use headphones if you enable monitoring. In **Soundboard**, **Import WAVs** accepts multiple files; **Remove** works on built-in sounds too. There is no library-count limit. Each WAV can be up to 60 seconds and 128 MiB, and eight sounds can play at once.

Optional local AI can use Voice Warp's managed Podman worker without a separately installed RVC server. Install the selected runtime in **Local AI** and supply a trusted voice model. CPU, CUDA, and ROCm recipes have different hardware requirements and validation limits; see [AI setup](AI.md). The normal voice effects and soundboard are included and need no AI runtime.

## Update or remove

To update, close Voice Warp and run the newer installer in the same way. Your settings and imported sound paths are retained.

To remove the app and menu shortcut, close Voice Warp and run:

```bash
bash Voice-Warp-0.3.0-Bazzite-x86_64.run --uninstall
```

The reusable container, settings, and original WAV files are retained. If this container was used only for Voice Warp, you can also remove it with:

```bash
distrobox rm --force voice-warp
```

## Check the download or troubleshoot

Version 0.2.2 fixes a pitch-buffer bug that could stop processing with Helium, Ghost, or Synth and show “The voice processor stopped unexpectedly.” Close Voice Warp and run the latest installer to update the existing installation.

Check the embedded archive without installing anything:

```bash
bash Voice-Warp-0.3.0-Bazzite-x86_64.run --check
```

The adjacent `.sha256` file can check the whole download with `sha256sum --check Voice-Warp-0.3.0-Bazzite-x86_64.run.sha256`. These checks detect corruption; the package is not digitally signed.

List audio devices without recording:

```bash
distrobox enter --name voice-warp -- /usr/local/bin/voice-warp --doctor
```

If a container named `voice-warp` already exists with a different Fedora version, choose a new name for installation. Use the same name for updates and uninstalling:

```bash
VOICE_WARP_CONTAINER=voice-warp-app bash Voice-Warp-0.3.0-Bazzite-x86_64.run
```

`--extract DIRECTORY` unpacks the files for inspection without installing. Choose a destination directory that does not already exist.

## Build a new installer

Build the binary against Fedora 43, the same runtime used by the installer. A binary compiled on a newer distribution may require a newer glibc and fail on the target machine.

From the source checkout:

```bash
podman build --file packaging/Containerfile.bazzite --tag localhost/voice-warp-bazzite-build:0.3.0 .
mkdir -p target/bazzite
podman run --rm --network=none \
  --volume "$PWD/target/bazzite:/out:Z" \
  localhost/voice-warp-bazzite-build:0.3.0 \
  install -m755 /build/target/release/voice-warp /out/voice-warp
bash scripts/package-bazzite.sh --binary target/bazzite/voice-warp
```

The result and its checksum are written to `dist/`. The packager rejects binaries requiring glibc newer than 2.42. No personal settings, imported WAVs, or AI models are included.

The installation follows Bazzite's [container guidance](https://docs.bazzite.gg/Installing_and_Managing_Software/Containers/) and Distrobox's [audio/display integration](https://distrobox.it/usage/distrobox-create/) and [menu export](https://distrobox.it/usage/distrobox-export/) support.
