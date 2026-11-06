#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use voice_warp::{
    audio::AudioEngine,
    config::{Settings, config_dir},
    dsp::{LiveControls, MixProcessor, PRESETS, VoiceDsp},
    rvc::{self, RvcConfig},
    soundboard::Soundboard,
};

fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn main() {
    let cli = std::env::args().skip(1).any(|arg| {
        arg.starts_with("--ai-")
            || matches!(
                arg.as_str(),
                "--help"
                    | "-h"
                    | "--doctor"
                    | "--list-inputs"
                    | "--list-outputs"
                    | "--headless"
                    | "--render-demo"
                    | "--install-ai"
                    | "--ai-doctor"
                    | "--ai-devices"
                    | "--ai-benchmark"
            )
    });
    #[cfg(target_os = "windows")]
    if cli {
        // GUI executables have no console when double-clicked. Preserve the GUI
        // launch experience while making command-line diagnostics visible.
        unsafe {
            windows_sys::Win32::System::Console::AttachConsole(
                windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS,
            );
        }
    }
    if let Err(error) = run() {
        eprintln!("Voice Warp: {error}");
        if !cli {
            rfd::MessageDialog::new()
                .set_title("Voice Warp could not start")
                .set_description(&error)
                .set_level(rfd::MessageLevel::Error)
                .show();
        }
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--preview") && args.iter().any(|a| a == "--headless") {
        return Err(
            "--preview cannot be combined with --headless; preview never opens audio streams."
                .into(),
        );
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Voice Warp {}\n\nvoice-warp                       Open the desktop app\nvoice-warp --doctor [--report FILE] Check audio devices without recording\nvoice-warp --list-inputs         Print available microphone IDs\nvoice-warp --list-outputs        Print available playback IDs\nvoice-warp --headless --input ID [--output ID] [--seconds 10] [--preset 0..{}]\nvoice-warp --render-demo FILE    Write an offline DSP + soundboard WAV\nvoice-warp --install-ai [--ai-backend cpu|cuda|rocm|migraphx|mps|coreml|directml]\nvoice-warp --ai-doctor [--ai-backend cpu] [--ai-device 0]\nvoice-warp --ai-devices [--ai-backend cpu]\nvoice-warp --ai-export --model FILE [--ai-backend cpu]    Create a validated ONNX bundle\nvoice-warp --ai-benchmark --model FILE [--ai-backend cpu] [--blocks 64]\n  AI diagnostics/benchmark never open the microphone. Benchmark uses a generated tone unless --input-wav FILE supplies 48 kHz mono audio.\n  Optional native package install: --install-ai --ai-runtime-manifest FILE --ai-manifest-sha256 HASH\nvoice-warp --ai-rollback --ai-backend cpu    Restore the previous native runtime\nvoice-warp --ai-uninstall --ai-backend NAME   Remove a native backend and its rollback build\nvoice-warp --ai-remove-runtime BUILD_ID     Remove an inactive native runtime build\n  Benchmark options: --input-wav FILE --output-wav NEW_FILE --realtime --index FILE --block-size 8192 --pitch 0 --index-ratio 0.5 --ai-device ID\n  WAV default: process the whole file and pad the final block; explicit --blocks repeats it.\n  --realtime paces blocks for sustained testing. WAV input limit: 20 minutes/256 MiB.\n  Output WAV files are never overwritten; timing does not establish speech quality.\nvoice-warp --preview             Inspect the UI without opening audio streams\n\nLinux creates a Voice Warp source automatically. Windows/macOS require a separately installed virtual cable; select its playback endpoint with --output.",
            env!("CARGO_PKG_VERSION"),
            PRESETS.len() - 1
        );
        return Ok(());
    }
    let shutdown = Arc::new(AtomicBool::new(false));
    let signal = shutdown.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed)).map_err(|e| e.to_string())?;
    if let Some(action) = AiCommand::parse(&args)? {
        return action.run(shutdown);
    }
    if args.iter().any(|a| a == "--doctor") {
        let report = serde_json::to_string_pretty(&AudioEngine::diagnostics()?)
            .map_err(|e| e.to_string())?;
        if let Some(path) = option(&args, "--report") {
            fs::write(path, &report).map_err(|e| e.to_string())?;
        } else {
            println!("{report}");
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-inputs") {
        for device in AudioEngine::devices()? {
            println!("{}\t{}", device.name, device.description);
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-outputs") {
        for device in AudioEngine::outputs()? {
            println!("{}\t{}", device.name, device.description);
        }
        return Ok(());
    }
    if let Some(path) = option(&args, "--render-demo") {
        return render_demo(&path);
    }
    let screenshot = option(&args, "--screenshot").map(PathBuf::from);
    let preview = args.iter().any(|a| a == "--preview");
    if !cfg!(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos"
    )) && !preview
    {
        return Err("Live audio is supported on Linux, Windows, and macOS.".into());
    }
    fs::create_dir_all(config_dir()).map_err(|e| e.to_string())?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(config_dir().join("instance.lock"))
        .map_err(|e| e.to_string())?;
    if !preview {
        fs2::FileExt::try_lock_exclusive(&lock).map_err(|_| {
            "Voice Warp is already open. Close the other instance first.".to_string()
        })?;
    }

    if args.iter().any(|a| a == "--headless") {
        let source = option(&args, "--input").ok_or("--headless requires --input NAME")?;
        let seconds: u64 = option(&args, "--seconds")
            .unwrap_or("10".into())
            .parse()
            .map_err(|_| "Invalid --seconds")?;
        let preset: usize = option(&args, "--preset")
            .unwrap_or("0".into())
            .parse()
            .map_err(|_| "Invalid --preset")?;
        let mut controls = LiveControls::default();
        PRESETS
            .get(preset)
            .ok_or_else(|| format!("Preset must be 0..{}", PRESETS.len() - 1))?
            .apply(&mut controls.effects);
        let processor = MixProcessor::new(
            Arc::new(Mutex::new(controls)),
            Arc::new(Mutex::new(Soundboard::with_defaults())),
            None,
        );
        let mut engine = AudioEngine::new();
        let output = option(&args, "--output");
        if AudioEngine::requires_output_device() && output.is_none() {
            return Err("--headless requires --output ID for the virtual cable playback endpoint on this platform. Use --list-outputs.".into());
        }
        engine.start(
            &source,
            Box::new(processor),
            960,
            false,
            None,
            output.as_deref(),
        )?;
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(seconds)
            && !shutdown.load(Ordering::Relaxed)
            && engine.running()
        {
            std::thread::sleep(Duration::from_millis(50));
        }
        let stats = engine.stats();
        engine.stop();
        if let Some(error) = stats.error {
            return Err(error);
        }
        println!(
            "Processed {} frames; {} dropped blocks",
            stats.frames, stats.dropped_blocks
        );
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 900.0])
            .with_min_inner_size([940.0, 700.0])
            .with_title("Voice Warp"),
        ..Default::default()
    };
    eframe::run_native(
        "Voice Warp",
        options,
        Box::new(move |context| {
            Ok(Box::new(app::VoiceWarpApp::new(
                context, preview, screenshot, shutdown,
            )))
        }),
    )
    .map_err(|e| e.to_string())
}

#[derive(Debug, PartialEq)]
enum AiAction {
    Install,
    Export,
    Doctor,
    Devices,
    Benchmark,
    Rollback,
    Uninstall,
    Remove,
}

#[derive(Debug)]
struct AiCommand {
    action: AiAction,
    backend: String,
    device: String,
    model: Option<PathBuf>,
    index: PathBuf,
    block_size: usize,
    pitch: i32,
    index_ratio: f32,
    blocks: Option<usize>,
    input_wav: Option<PathBuf>,
    output_wav: Option<PathBuf>,
    realtime: bool,
    remove_id: Option<String>,
    manifest: Option<PathBuf>,
    manifest_hash: Option<String>,
    report: Option<PathBuf>,
}

fn ai_option(args: &[String], name: &str) -> Result<Option<String>, String> {
    let indices: Vec<_> = args
        .iter()
        .enumerate()
        .filter(|(_, value)| *value == name)
        .map(|(index, _)| index)
        .collect();
    if indices.len() > 1 {
        return Err(format!("Specify {name} only once."));
    }
    let Some(index) = indices.first() else {
        return Ok(None);
    };
    let value = args.get(index + 1).filter(|value| !value.starts_with("--"));
    value
        .cloned()
        .map(Some)
        .ok_or_else(|| format!("{name} requires a value."))
}

impl AiCommand {
    fn parse(args: &[String]) -> Result<Option<Self>, String> {
        let actions: Vec<_> = [
            "--install-ai",
            "--ai-doctor",
            "--ai-devices",
            "--ai-benchmark",
            "--ai-export",
            "--ai-rollback",
            "--ai-uninstall",
            "--ai-remove-runtime",
        ]
        .into_iter()
        .filter(|flag| args.iter().any(|value| value == flag))
        .collect();
        if actions.is_empty() {
            if args.iter().any(|arg| arg.starts_with("--ai-")) {
                return Err("AI options require --install-ai, --ai-doctor, --ai-devices, --ai-benchmark, --ai-export, --ai-rollback, --ai-uninstall, or --ai-remove-runtime.".into());
            }
            return Ok(None);
        }
        if actions.len() != 1 {
            return Err("Choose only one AI command per invocation.".into());
        }
        let known_flags = [
            "--install-ai",
            "--ai-doctor",
            "--ai-devices",
            "--ai-benchmark",
            "--ai-export",
            "--ai-rollback",
            "--ai-uninstall",
            "--ai-remove-runtime",
            "--ai-backend",
            "--ai-device",
            "--ai-runtime-manifest",
            "--ai-manifest-sha256",
            "--model",
            "--index",
            "--block-size",
            "--pitch",
            "--index-ratio",
            "--blocks",
            "--input-wav",
            "--output-wav",
            "--realtime",
            "--report",
            "--preview",
        ];
        if let Some(flag) = args
            .iter()
            .find(|arg| arg.starts_with("--") && !known_flags.contains(&arg.as_str()))
        {
            return Err(format!(
                "Unknown or incompatible option for this AI command: {flag}"
            ));
        }
        if args.iter().any(|arg| arg == "--preview") {
            return Err(
                "AI installation, probing, and benchmarking are unavailable in preview mode."
                    .into(),
            );
        }
        let action = match actions[0] {
            "--install-ai" => AiAction::Install,
            "--ai-doctor" => AiAction::Doctor,
            "--ai-devices" => AiAction::Devices,
            "--ai-export" => AiAction::Export,
            "--ai-rollback" => AiAction::Rollback,
            "--ai-uninstall" => AiAction::Uninstall,
            "--ai-remove-runtime" => AiAction::Remove,
            _ => AiAction::Benchmark,
        };
        let explicit_backend = ai_option(args, "--ai-backend")?;
        if action == AiAction::Uninstall
            && (explicit_backend.is_none() || explicit_backend.as_deref() == Some("auto"))
        {
            return Err(
                "--ai-uninstall requires --ai-backend with an explicit native backend name.".into(),
            );
        }
        let backend = explicit_backend.unwrap_or_else(|| "cpu".into());
        if ![
            "cpu", "auto", "cuda", "rocm", "migraphx", "directml", "mps", "coreml",
        ]
        .contains(&backend.as_str())
        {
            return Err(
                "Unknown AI backend. Choose cpu, auto, cuda, rocm, migraphx, directml, mps, or coreml.".into(),
            );
        }
        let device = ai_option(args, "--ai-device")?.unwrap_or_else(|| "0".into());
        if device.is_empty() || device.len() > 256 {
            return Err("Invalid AI device ID.".into());
        }
        let block_size = ai_option(args, "--block-size")?
            .unwrap_or_else(|| "8192".into())
            .parse::<usize>()
            .map_err(|_| "Invalid --block-size")?;
        if ![4096, 6144, 8192, 12288].contains(&block_size) {
            return Err("--block-size must be 4096, 6144, 8192, or 12288.".into());
        }
        let pitch = ai_option(args, "--pitch")?
            .unwrap_or_else(|| "0".into())
            .parse::<i32>()
            .map_err(|_| "Invalid --pitch")?;
        if !(-24..=24).contains(&pitch) {
            return Err("--pitch must be between -24 and 24.".into());
        }
        let index_ratio = ai_option(args, "--index-ratio")?
            .unwrap_or_else(|| "0.5".into())
            .parse::<f32>()
            .map_err(|_| "Invalid --index-ratio")?;
        if !index_ratio.is_finite() || !(0.0..=1.0).contains(&index_ratio) {
            return Err("--index-ratio must be between 0 and 1.".into());
        }
        let blocks = ai_option(args, "--blocks")?
            .map(|value| value.parse::<usize>().map_err(|_| "Invalid --blocks"))
            .transpose()?;
        if blocks.is_some_and(|blocks| !(1..=100_000).contains(&blocks)) {
            return Err("--blocks must be between 1 and 100000.".into());
        }
        let input_wav = ai_option(args, "--input-wav")?.map(PathBuf::from);
        let output_wav = ai_option(args, "--output-wav")?.map(PathBuf::from);
        let realtime = args.iter().any(|arg| arg == "--realtime");
        if action != AiAction::Benchmark
            && (input_wav.is_some() || output_wav.is_some() || realtime || blocks.is_some())
        {
            return Err("WAV, block count, and realtime options require --ai-benchmark.".into());
        }
        let model = ai_option(args, "--model")?.map(PathBuf::from);
        if matches!(action, AiAction::Benchmark | AiAction::Export) && model.is_none() {
            return Err("--ai-benchmark and --ai-export require --model FILE.".into());
        }
        let manifest = ai_option(args, "--ai-runtime-manifest")?.map(PathBuf::from);
        let manifest_hash = ai_option(args, "--ai-manifest-sha256")?;
        if manifest.is_some() != manifest_hash.is_some() {
            return Err("A local runtime package requires both --ai-runtime-manifest and --ai-manifest-sha256 from a trusted source.".into());
        }
        if manifest.is_some() && action != AiAction::Install {
            return Err("Runtime package options require --install-ai.".into());
        }
        Ok(Some(Self {
            action,
            backend,
            device,
            model,
            index: ai_option(args, "--index")?
                .map(PathBuf::from)
                .unwrap_or_default(),
            block_size,
            pitch,
            index_ratio,
            blocks,
            input_wav,
            output_wav,
            realtime,
            manifest,
            manifest_hash,
            remove_id: ai_option(args, "--ai-remove-runtime")?,
            report: ai_option(args, "--report")?.map(PathBuf::from),
        }))
    }

    fn run(self, stop: Arc<AtomicBool>) -> Result<(), String> {
        if self
            .report
            .as_ref()
            .is_some_and(|path| path.symlink_metadata().is_ok())
        {
            return Err("The report path already exists. Choose a new file; AI reports never overwrite existing files.".into());
        }
        let result = match self.action {
            AiAction::Export => {
                rvc::export_model(self.model.as_deref().unwrap(), &self.backend, stop)?
            }
            AiAction::Install => {
                eprintln!(
                    "Installing AI runtime. Progress: {}",
                    rvc::runtime_dir().join("install.log").display()
                );
                if let Some(manifest) = &self.manifest {
                    rvc::runtime::install_package(
                        manifest,
                        self.manifest_hash.as_deref().unwrap(),
                        &stop,
                    )?;
                } else {
                    rvc::install_backend(&self.backend, stop)?;
                }
                serde_json::json!({"installed":true,"requested_backend":self.backend})
            }
            AiAction::Rollback => {
                rvc::runtime::rollback(&self.backend)?;
                serde_json::json!({"rolled_back":true,"backend":self.backend})
            }
            AiAction::Uninstall => {
                rvc::runtime::uninstall_backend(&self.backend)?;
                serde_json::json!({"uninstalled_native_backend":self.backend})
            }
            AiAction::Remove => {
                let id = self
                    .remove_id
                    .as_deref()
                    .ok_or("--ai-remove-runtime requires an inactive build ID.")?;
                rvc::runtime::remove_build(id)?;
                serde_json::json!({"removed_build":id})
            }
            AiAction::Doctor | AiAction::Devices => {
                let report = rvc::diagnostics(&self.backend, &self.device, stop)?;
                if self.action == AiAction::Devices {
                    serde_json::json!({"devices":report["devices"],"model_verified":false})
                } else {
                    report
                }
            }
            AiAction::Benchmark => rvc::benchmark::run(
                RvcConfig {
                    model: self.model.unwrap(),
                    index: self.index,
                    pitch: self.pitch,
                    index_ratio: self.index_ratio,
                    block_size: self.block_size,
                    backend: self.backend,
                    device: self.device,
                },
                rvc::benchmark::Options {
                    input_wav: self.input_wav,
                    output_wav: self.output_wav,
                    blocks: self.blocks,
                    realtime: self.realtime,
                },
                stop,
            )?,
        };
        let text = serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?;
        if let Some(path) = self.report {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .and_then(|mut file| file.write_all(text.as_bytes()))
                .map_err(|error| error.to_string())?;
        } else {
            println!("{text}");
        }
        Ok(())
    }
}

fn render_demo(path: &str) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    let mut dsp = VoiceDsp::default();
    let mut settings = Settings::default().effects;
    let mut board = Soundboard::with_defaults();
    let ids: Vec<_> = board.list().iter().map(|clip| clip.id).collect();
    for block in 0..PRESETS.len() * 80 {
        PRESETS[block / 80].apply(&mut settings);
        if block % 100 == 0
            && let Some(&id) = ids.get(block / 100)
        {
            board.trigger(id)?;
        }
        let input: Vec<f32> = (0..960)
            .map(|n| {
                ((block * 960 + n) as f32 * std::f32::consts::TAU * 180.0 / 48_000.0).sin() * 0.12
            })
            .collect();
        let mut output = dsp.process(&input, &settings);
        board.mix_into(&mut output);
        for sample in output {
            writer
                .write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
                .map_err(|e| e.to_string())?;
        }
    }
    writer.finalize().map_err(|e| e.to_string())
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn ai_cli_rejects_preview_ambiguity_and_missing_values_before_work() {
        for values in [
            vec!["--install-ai", "--preview"],
            vec!["--ai-doctor", "--ai-benchmark"],
            vec!["--ai-doctor", "--ai-backend"],
            vec!["--ai-devices", "--ai-device", "--report", "out.json"],
            vec!["--ai-benchmark"],
            vec!["--install-ai", "--ai-runtime-manifest", "runtime.json"],
            vec!["--ai-backend", "cuda"],
            vec!["--ai-doctor", "--ai-backned", "cuda"],
            vec!["--ai-devices", "--headless"],
            vec!["--ai-remove-runtime"],
            vec!["--ai-remove-runtime", "old-build", "--preview"],
        ] {
            assert!(AiCommand::parse(&args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn benchmark_options_are_validated_without_touching_model_or_microphone() {
        let command = AiCommand::parse(&args(&[
            "--ai-benchmark",
            "--model",
            "voice.pth",
            "--ai-backend",
            "cuda",
            "--ai-device",
            "uuid:GPU-test",
            "--pitch",
            "-12",
            "--blocks",
            "20",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(command.action, AiAction::Benchmark);
        assert_eq!(command.device, "uuid:GPU-test");
        assert_eq!(command.pitch, -12);
        assert_eq!(command.blocks, Some(20));
        for value in ["NaN", "inf", "-0.1", "1.1"] {
            assert!(
                AiCommand::parse(&args(&[
                    "--ai-benchmark",
                    "--model",
                    "voice.onnx",
                    "--index-ratio",
                    value
                ]))
                .is_err()
            );
        }
    }

    #[test]
    fn wav_options_require_benchmark_and_preserve_whole_file_default() {
        let command = AiCommand::parse(&args(&[
            "--ai-benchmark",
            "--model",
            "voice.pth",
            "--input-wav",
            "speech.wav",
            "--output-wav",
            "converted.wav",
            "--realtime",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(command.blocks, None);
        assert_eq!(command.input_wav, Some(PathBuf::from("speech.wav")));
        assert_eq!(command.output_wav, Some(PathBuf::from("converted.wav")));
        assert!(command.realtime);
        for values in [
            vec!["--ai-doctor", "--input-wav", "speech.wav"],
            vec!["--install-ai", "--output-wav", "output.wav"],
            vec!["--ai-doctor", "--realtime"],
            vec!["--ai-benchmark", "--model", "voice.pth", "--input-wav"],
            vec!["--ai-benchmark", "--model", "voice.pth", "--blocks", "0"],
            vec![
                "--ai-benchmark",
                "--model",
                "voice.pth",
                "--blocks",
                "100001",
            ],
        ] {
            assert!(AiCommand::parse(&args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn native_uninstall_requires_explicit_backend_and_exclusive_action() {
        let command = AiCommand::parse(&args(&["--ai-uninstall", "--ai-backend", "coreml"]))
            .unwrap()
            .unwrap();
        assert_eq!(command.action, AiAction::Uninstall);
        assert_eq!(command.backend, "coreml");
        for values in [
            vec!["--ai-uninstall"],
            vec!["--ai-uninstall", "--ai-backend", "auto"],
            vec!["--ai-uninstall", "--ai-backend", "cpu", "--install-ai"],
            vec!["--ai-uninstall", "--ai-backend", "cpu", "--preview"],
        ] {
            assert!(AiCommand::parse(&args(&values)).is_err(), "{values:?}");
        }
    }
}
