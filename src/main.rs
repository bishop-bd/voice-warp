#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;

use std::{
    fs::{self, OpenOptions},
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
    soundboard::Soundboard,
};

fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn main() {
    let cli = std::env::args().skip(1).any(|arg| {
        matches!(
            arg.as_str(),
            "--help"
                | "-h"
                | "--doctor"
                | "--list-inputs"
                | "--list-outputs"
                | "--headless"
                | "--render-demo"
                | "--install-ai"
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
    if args.iter().any(|a| a == "--install-ai") {
        if args.iter().any(|a| a == "--preview") {
            return Err("AI installation is unavailable in preview mode.".into());
        }
        println!(
            "Installing bundled RVC engine. Progress: {}",
            voice_warp::rvc::runtime_dir().join("install.log").display()
        );
        voice_warp::rvc::install()?;
        println!("AI engine installed. Choose your voice model in Local AI.");
        return Ok(());
    }
    if args.iter().any(|a| a == "--preview") && args.iter().any(|a| a == "--headless") {
        return Err(
            "--preview cannot be combined with --headless; preview never opens audio streams."
                .into(),
        );
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Voice Warp {}\n\nvoice-warp                       Open the desktop app\nvoice-warp --doctor [--report FILE] Check audio devices without recording\nvoice-warp --list-inputs         Print available microphone IDs\nvoice-warp --list-outputs        Print available playback IDs\nvoice-warp --headless --input ID [--output ID] [--seconds 10] [--preset 0..{}]\nvoice-warp --render-demo FILE    Write an offline DSP + soundboard WAV\nvoice-warp --install-ai          Install the bundled Linux RVC runtime\nvoice-warp --preview             Inspect the UI without opening audio streams\n\nLinux creates a Voice Warp source automatically. Windows/macOS require a separately installed virtual cable; select its playback endpoint with --output.",
            env!("CARGO_PKG_VERSION"),
            PRESETS.len() - 1
        );
        return Ok(());
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
    let shutdown = Arc::new(AtomicBool::new(false));
    let signal = shutdown.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed)).map_err(|e| e.to_string())?;

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
