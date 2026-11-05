//! Linux microphone routing through PipeWire's PulseAudio compatibility server.
//!
//! Audio stays on this machine. `parec` and `pacat` are deliberately separate,
//! killable processes so a disconnected audio server cannot trap the UI in a
//! blocking native audio call. Queues retain the newest two blocks only.

use super::{AudioDevice, AudioStats, Processor, SAMPLE_RATE, SOURCE_NAME};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SINK_NAME: &str = "voice_warp_sink";
const COMMAND_TIMEOUT: Duration = Duration::from_secs(4);
static INSTANCE_ACTIVE: AtomicBool = AtomicBool::new(false);

pub struct AudioEngine {
    session: Option<Session>,
    stats: Arc<Mutex<AudioStats>>,
    running: Arc<AtomicBool>,
}

struct Session {
    stop: Arc<AtomicBool>,
    supervisor: JoinHandle<()>,
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEngine {
    pub fn requires_output_device() -> bool {
        false
    }

    pub fn new() -> Self {
        Self {
            session: None,
            stats: Arc::new(Mutex::new(AudioStats::default())),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn devices() -> Result<Vec<AudioDevice>, String> {
        list_devices(&Pactl, "sources", true)
    }

    pub fn outputs() -> Result<Vec<AudioDevice>, String> {
        list_devices(&Pactl, "sinks", false)
    }

    pub fn diagnostics() -> Result<Value, String> {
        if !cfg!(target_os = "linux") {
            return Ok(json!({
                "platform": std::env::consts::OS,
                "supported": false,
                "message": "Audio routing requires Linux with PipeWire/PulseAudio."
            }));
        }
        let server = Pactl.command(&["--format=json", "info"])?;
        let server: Value = serde_json::from_str(&server)
            .map_err(|e| format!("Cannot read pactl server information: {e}"))?;
        Ok(json!({
            "platform": "linux",
            "version": env!("CARGO_PKG_VERSION"),
            "supported": true,
            "server": server,
            "inputs": Self::devices()?,
            "outputs": Self::outputs()?,
            "sample_rate": SAMPLE_RATE,
            "virtual_source": SOURCE_NAME,
            "parec": command_available("parec"),
            "pacat": command_available("pacat")
        }))
    }

    pub fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> AudioStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Creates the virtual microphone and starts processing. Run this outside
    /// the UI thread; server startup and rollback can take several seconds.
    pub fn start(
        &mut self,
        source: &str,
        processor: Box<dyn Processor>,
        block_size: usize,
        make_default: bool,
        monitor_sink: Option<&str>,
        output_device: Option<&str>,
    ) -> Result<(), String> {
        if output_device.is_some() {
            return Err("Linux creates the Voice Warp virtual microphone automatically; no cable output is needed.".into());
        }
        if !cfg!(target_os = "linux") {
            return Err("Live microphone routing requires Linux. Use this app on Bazzite.".into());
        }
        if self.running() {
            return Err("Voice Warp is already running.".into());
        }
        // Reap any session that stopped itself following an audio/AI failure.
        self.stop();
        if !(240..=48_000).contains(&block_size) {
            return Err("Audio block size must be between 240 and 48000 samples.".into());
        }
        if !Self::devices()?.iter().any(|device| device.name == source) {
            return Err("The selected microphone is unavailable. Refresh devices and select a physical input.".into());
        }
        if let Some(sink) = monitor_sink
            && !Self::outputs()?.iter().any(|device| device.name == sink)
        {
            return Err("The monitoring output is unavailable. Select a connected output.".into());
        }
        if INSTANCE_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("Another Voice Warp audio session is running or still stopping.".into());
        }
        self.reset_session_state();
        let result = self.start_inner(source, processor, block_size, make_default, monitor_sink);
        if let Err(error) = &result {
            INSTANCE_ACTIVE.store(false, Ordering::Release);
            self.stats.lock().unwrap_or_else(|e| e.into_inner()).error = Some(error.clone());
        }
        result
    }

    fn reset_session_state(&mut self) {
        // A timed-out third-party processor may outlive the session supervisor.
        // Never let its eventual completion mutate a newer session's state.
        self.stats = Arc::new(Mutex::new(AudioStats::default()));
        self.running = Arc::new(AtomicBool::new(false));
    }

    fn start_inner(
        &mut self,
        source: &str,
        mut processor: Box<dyn Processor>,
        block_size: usize,
        make_default: bool,
        monitor_sink: Option<&str>,
    ) -> Result<(), String> {
        let mut route = Routing::create(&Pactl, make_default, monitor_sink)?;
        let mut capture = match spawn_audio("parec", source) {
            Ok(child) => child,
            Err(error) => {
                route.close(&Pactl);
                return Err(error);
            }
        };
        let mut playback = match spawn_audio("pacat", SINK_NAME) {
            Ok(child) => child,
            Err(error) => {
                terminate(&mut capture);
                route.close(&Pactl);
                return Err(error);
            }
        };
        let capture_output = capture.stdout.take().expect("capture stdout was piped");
        let playback_input = playback.stdin.take().expect("playback stdin was piped");
        let capture_errors = capture.stderr.take();
        let playback_errors = playback.stderr.take();
        let capture = Arc::new(Mutex::new(capture));
        let playback = Arc::new(Mutex::new(playback));
        let stop = Arc::new(AtomicBool::new(false));
        processor.set_stop_signal(stop.clone());
        let shutdown = Arc::new(AtomicBool::new(false));
        let captured = Arc::new(BlockQueue::new(2));
        let processed = Arc::new(BlockQueue::new(2));
        let errors = Arc::new(Mutex::new(String::new()));
        let mut workers = Vec::new();
        for stream in [capture_errors, playback_errors].into_iter().flatten() {
            let errors = errors.clone();
            workers.push(thread::spawn(move || {
                let mut message = String::new();
                let _ = stream.take(16_384).read_to_string(&mut message);
                if !message.trim().is_empty() {
                    let mut text = errors.lock().unwrap_or_else(|e| e.into_inner());
                    text.push_str(message.trim());
                    text.push('\n');
                }
            }));
        }
        let shared = WorkerState {
            stop: stop.clone(),
            stats: self.stats.clone(),
            running: self.running.clone(),
        };
        self.running.store(true, Ordering::Release);
        {
            let state = shared.clone();
            let output = captured.clone();
            workers.push(thread::spawn(move || {
                capture_worker(capture_output, output, state, block_size)
            }));
        }
        {
            let state = shared.clone();
            let input = captured.clone();
            let output = processed.clone();
            workers.push(thread::spawn(move || {
                process_worker(processor, input, output, state)
            }));
        }
        {
            let state = shared.clone();
            let input = processed.clone();
            workers.push(thread::spawn(move || {
                playback_worker(playback_input, input, state)
            }));
        }
        let supervisor_stop = stop.clone();
        let supervisor_shutdown = shutdown.clone();
        let supervisor = thread::spawn(move || {
            while !supervisor_stop.load(Ordering::Acquire)
                && !supervisor_shutdown.load(Ordering::Acquire)
            {
                for child in [&capture, &playback] {
                    let status = child.lock().unwrap_or_else(|e| e.into_inner()).try_wait();
                    match status {
                        Ok(Some(status)) => shared.fail(format!("The audio stream exited ({status}). Check the selected microphone and PipeWire connection.")),
                        Err(error) => shared.fail(format!("Cannot check the audio stream: {error}")),
                        Ok(None) => {}
                    }
                }
                thread::sleep(Duration::from_millis(20));
            }
            supervisor_stop.store(true, Ordering::Release);
            shared.running.store(false, Ordering::Release);
            captured.wake_all();
            processed.wake_all();
            // Killing the subprocesses unblocks their Rust pipe readers/writers.
            terminate(&mut capture.lock().unwrap_or_else(|e| e.into_inner()));
            // An idle remap-source can still replay buffered output after its
            // writer disappears. Mute the public source before disconnecting
            // playback, and retain that muted device until explicit cleanup.
            if let Err(error) = route.mute(&Pactl) {
                shared.record_error(format!("Cannot mute the virtual microphone: {error}"));
            }
            terminate(&mut playback.lock().unwrap_or_else(|e| e.into_inner()));
            let deadline = Instant::now() + Duration::from_secs(2);
            for worker in workers {
                while !worker.is_finished() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(10));
                }
                if worker.is_finished() {
                    if worker.join().is_err() {
                        shared.record_error("An audio worker failed unexpectedly.".into());
                    }
                } else {
                    // A third-party processor must not prevent device cleanup.
                    // Detached workers only retain this already-stopped session.
                    shared.record_error(
                        "A voice processor did not stop promptly. Its session was disconnected."
                            .into(),
                    );
                }
            }
            {
                let child_error = errors.lock().unwrap_or_else(|e| e.into_inner());
                let mut stats = shared.stats.lock().unwrap_or_else(|e| e.into_inner());
                if !child_error.trim().is_empty()
                    && let Some(error) = stats.error.as_mut()
                {
                    error.push('\n');
                    error.push_str(child_error.trim());
                }
                stats.input_peak = 0.0;
                stats.output_peak = 0.0;
            }
            // Keep the muted virtual microphone after errors. Removing it here
            // could cause a chat application's automatic device fallback to
            // expose the physical microphone. Only explicit Stop/restart/Drop
            // releases the route. A crashed audio server is outside our control.
            while !supervisor_shutdown.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(20));
            }
            let cleanup_errors = route.close(&Pactl);
            if !cleanup_errors.is_empty() {
                shared.record_error(format!("Audio cleanup: {}", cleanup_errors.join("; ")));
            }
            INSTANCE_ACTIVE.store(false, Ordering::Release);
        });
        self.session = Some(Session {
            stop: shutdown,
            supervisor,
        });
        Ok(())
    }

    /// Stops recording and playback, removes this session's virtual devices,
    /// and restores defaults only if the user has not changed them meanwhile.
    pub fn stop(&mut self) {
        if let Some(session) = self.session.take() {
            session.stop.store(true, Ordering::Release);
            let _ = session.supervisor.join();
        }
        self.running.store(false, Ordering::Release);
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Clone)]
struct WorkerState {
    stop: Arc<AtomicBool>,
    stats: Arc<Mutex<AudioStats>>,
    running: Arc<AtomicBool>,
}

impl WorkerState {
    fn record_error(&self, message: String) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(error) = stats.error.as_mut() {
            if !error.contains(&message) {
                error.push('\n');
                error.push_str(&message);
            }
        } else {
            stats.error = Some(message);
        }
    }

    fn fail(&self, message: String) {
        if !self.stop.swap(true, Ordering::AcqRel) {
            self.record_error(message);
        }
        self.running.store(false, Ordering::Release);
    }

    fn dropped(&self, count: u64) {
        self.stats
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .dropped_blocks += count;
    }
}

/// A bounded queue with a newest-audio-wins policy. Slow AI conversion drops
/// stale microphone blocks instead of allowing conversation latency to grow.
struct BlockQueue {
    blocks: Mutex<VecDeque<Vec<f32>>>,
    ready: Condvar,
    capacity: usize,
}

impl BlockQueue {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            blocks: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            capacity,
        }
    }

    fn push(&self, block: Vec<f32>) -> u64 {
        let mut blocks = self.blocks.lock().unwrap_or_else(|e| e.into_inner());
        let dropped = if blocks.len() >= self.capacity {
            blocks.pop_front();
            1
        } else {
            0
        };
        blocks.push_back(block);
        self.ready.notify_one();
        dropped
    }

    fn pop(&self, stop: &AtomicBool) -> Option<Vec<f32>> {
        let mut blocks = self.blocks.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let Some(block) = blocks.pop_front() {
                return Some(block);
            }
            let (next, _) = self
                .ready
                .wait_timeout(blocks, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner());
            blocks = next;
        }
    }

    fn wake_all(&self) {
        self.ready.notify_all();
    }
}

fn capture_worker(
    mut reader: impl Read,
    output: Arc<BlockQueue>,
    state: WorkerState,
    block_size: usize,
) {
    let mut bytes = vec![0_u8; block_size * 4];
    while !state.stop.load(Ordering::Acquire) {
        if let Err(error) = reader.read_exact(&mut bytes) {
            state.fail(format!("Microphone capture stopped: {error}"));
            break;
        }
        let samples: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .map(|value| if value.is_finite() { value } else { 0.0 })
            .collect();
        state
            .stats
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .input_peak = peak(&samples);
        state.dropped(output.push(samples));
    }
}

fn process_worker(
    mut processor: Box<dyn Processor>,
    input: Arc<BlockQueue>,
    output: Arc<BlockQueue>,
    state: WorkerState,
) {
    while let Some(samples) = input.pop(&state.stop) {
        let started = Instant::now();
        // A processor panic must trigger the same fail-closed cleanup as a
        // returned error; otherwise capture could continue indefinitely.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| processor.process(&samples)));
        let result = match result {
            Ok(result) => result,
            Err(payload) => {
                let reason = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown processor panic");
                Err(format!(
                    "The voice processor stopped unexpectedly: {reason}"
                ))
            }
        };
        match result.and_then(|values| validate_output(values, samples.len())) {
            Ok(processed) => {
                if state.stop.load(Ordering::Acquire) {
                    break;
                }
                {
                    let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
                    stats.processing_ms = started.elapsed().as_secs_f32() * 1000.0;
                    stats.output_peak = peak(&processed);
                    stats.frames += samples.len() as u64;
                }
                state.dropped(output.push(processed));
            }
            Err(error) => {
                state.fail(format!("Voice processing stopped: {error}"));
                break;
            }
        }
    }
}

fn playback_worker(mut writer: impl Write, input: Arc<BlockQueue>, state: WorkerState) {
    while let Some(samples) = input.pop(&state.stop) {
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        if let Err(error) = writer.write_all(&bytes) {
            state.fail(format!("Virtual microphone playback stopped: {error}"));
            break;
        }
    }
}

fn validate_output(mut samples: Vec<f32>, expected: usize) -> Result<Vec<f32>, String> {
    if samples.len() != expected {
        return Err(format!(
            "The processor returned {} samples; expected {expected}.",
            samples.len()
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("The processor returned invalid audio samples.".into());
    }
    samples
        .iter_mut()
        .for_each(|sample| *sample = sample.clamp(-1.0, 1.0));
    Ok(samples)
}

fn peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max)
        .min(1.0)
}

fn spawn_audio(program: &str, device: &str) -> Result<Child, String> {
    let capture = program == "parec";
    Command::new(program)
        .args([
            "--raw",
            "--format=float32le",
            "--rate=48000",
            "--channels=1",
            "--channel-map=mono",
            "--latency-msec=20",
            "--process-time-msec=10",
            "--client-name=Voice Warp",
            "--stream-name=Voice Warp audio",
        ])
        .arg(format!("--device={device}"))
        .stdin(if capture {
            Stdio::null()
        } else {
            Stdio::piped()
        })
        .stdout(if capture {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stderr(Stdio::piped())
        .env("LC_ALL", "C")
        .spawn()
        .map_err(|error| {
            format!(
                "Cannot start {program}: {error}. Install pulseaudio-utils in the app's Distrobox."
            )
        })
}

fn terminate(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

trait PulseControl {
    fn command(&self, arguments: &[&str]) -> Result<String, String>;
}

struct Pactl;

impl PulseControl for Pactl {
    fn command(&self, arguments: &[&str]) -> Result<String, String> {
        if !cfg!(target_os = "linux") {
            return Err("Audio device discovery requires Linux with PipeWire/PulseAudio.".into());
        }
        run_command("pactl", arguments)
    }
}

fn command_available(program: &str) -> bool {
    run_command(program, &["--version"]).is_ok()
}

fn run_command(program: &str, arguments: &[&str]) -> Result<String, String> {
    let mut child = Command::new(program).args(arguments)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .env("LC_ALL", "C").spawn()
        .map_err(|error| format!("Cannot run {program}: {error}. Install pulseaudio-utils and check the PipeWire connection."))?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let output_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.take(4 * 1024 * 1024).read_to_end(&mut bytes);
        bytes
    });
    let error_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.take(64 * 1024).read_to_end(&mut bytes);
        bytes
    });
    let started = Instant::now();
    let result = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if started.elapsed() < COMMAND_TIMEOUT => {
                thread::sleep(Duration::from_millis(10))
            }
            Ok(None) => {
                terminate(&mut child);
                break Err(format!(
                    "{program} timed out. Check that PipeWire and pipewire-pulse are running."
                ));
            }
            Err(error) => {
                terminate(&mut child);
                break Err(format!("Cannot wait for {program}: {error}"));
            }
        }
    };
    let output = output_reader.join().unwrap_or_default();
    let error = error_reader.join().unwrap_or_default();
    let status = result?;
    if !status.success() {
        return Err(format!(
            "{program} {}: {}",
            arguments.first().unwrap_or(&""),
            String::from_utf8_lossy(&error).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output).trim().to_owned())
}

fn server_objects(control: &impl PulseControl, kind: &str) -> Result<Vec<Value>, String> {
    let text = control.command(&["--format=json", "list", kind])?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("Cannot read {kind} from pactl: {error}"))?;
    value
        .as_array()
        .cloned()
        .ok_or_else(|| format!("pactl returned an unexpected {kind} list."))
}

fn list_devices(
    control: &impl PulseControl,
    kind: &str,
    inputs: bool,
) -> Result<Vec<AudioDevice>, String> {
    Ok(server_objects(control, kind)?
        .iter()
        .filter_map(|entry| {
            let name = entry.get("name")?.as_str()?;
            if name == SOURCE_NAME || name.starts_with("voice_warp_") || name.ends_with(".monitor")
            {
                return None;
            }
            if inputs && is_monitor(entry) {
                return None;
            }
            let description = entry
                .get("description")
                .and_then(Value::as_str)
                .or_else(|| {
                    entry
                        .pointer("/properties/device.description")
                        .and_then(Value::as_str)
                })
                .unwrap_or(name);
            Some(AudioDevice {
                name: name.to_owned(),
                description: description.to_owned(),
            })
        })
        .collect())
}

fn is_monitor(entry: &Value) -> bool {
    if entry
        .pointer("/properties/device.class")
        .and_then(Value::as_str)
        == Some("monitor")
    {
        return true;
    }
    match entry.get("monitor_of_sink") {
        Some(Value::Number(index)) => index.as_u64().is_some_and(|index| index != u32::MAX as u64),
        Some(Value::String(value)) => !matches!(value.as_str(), "" | "n/a" | "4294967295"),
        _ => false,
    }
}

struct Routing {
    modules: Vec<u32>,
    source_module: Option<u32>,
    previous_source: Option<String>,
    previous_sink: Option<String>,
}

impl Routing {
    fn create(
        control: &impl PulseControl,
        make_default: bool,
        monitor: Option<&str>,
    ) -> Result<Self, String> {
        for (kind, name) in [("sinks", SINK_NAME), ("sources", SOURCE_NAME)] {
            if server_objects(control, kind)?
                .iter()
                .any(|entry| entry["name"] == name)
            {
                return Err("A Voice Warp virtual device already exists. Close the other Voice Warp instance. If it crashed, restart your user audio services or remove only its modules as described in the README.".into());
            }
        }
        let previous_source = control
            .command(&["get-default-source"])
            .ok()
            .filter(|value| !value.is_empty());
        let previous_sink = control
            .command(&["get-default-sink"])
            .ok()
            .filter(|value| !value.is_empty());
        if make_default && previous_source.is_none() {
            return Err("Cannot save the previous default microphone. Start without 'Make default' or repair your audio session.".into());
        }
        let mut route = Self {
            modules: Vec::new(),
            source_module: None,
            previous_source,
            previous_sink,
        };
        let result = (|| {
            let sink_id = route.load(control, &[
                "module-null-sink", "sink_name=voice_warp_sink", "format=float32le",
                "rate=48000", "channels=1", "channel_map=mono",
                "sink_properties=\"device.description='Voice Warp Processing' application.id=io.github.voicewarp.VoiceWarp\"",
            ])?;
            verify_owner(control, "sinks", SINK_NAME, sink_id)?;
            // Some session policies switch newly-created outputs automatically.
            route.restore_default(control, "sink", SINK_NAME, route.previous_sink.as_deref())?;
            let source_id = route.load(control, &[
                "module-remap-source", "master=voice_warp_sink.monitor", "source_name=voice_warp",
                "format=float32le", "rate=48000", "channels=1", "channel_map=mono",
                "master_channel_map=mono", "remix=no",
                "source_properties=\"device.description='Voice Warp' application.id=io.github.voicewarp.VoiceWarp\"",
            ])?;
            route.source_module = Some(source_id);
            verify_owner(control, "sources", SOURCE_NAME, source_id)?;
            // Session managers may restore the mute state of the previous
            // failed session when this named source is recreated.
            control.command(&["set-source-mute", SOURCE_NAME, "0"])?;
            if make_default {
                control.command(&["set-default-source", SOURCE_NAME])?;
            } else {
                route.restore_default(
                    control,
                    "source",
                    SOURCE_NAME,
                    route.previous_source.as_deref(),
                )?;
            }
            if let Some(sink) = monitor {
                let sink_arg = format!("sink={}", quote_module_value(sink));
                route.load(
                    control,
                    &[
                        "module-loopback",
                        "source=voice_warp",
                        &sink_arg,
                        "latency_msec=40",
                        "source_dont_move=true",
                        "sink_dont_move=true",
                        "channels=1",
                        "remix=yes",
                    ],
                )?;
            }
            Ok::<(), String>(())
        })();
        if let Err(error) = result {
            let cleanup_errors = route.close(control);
            return if cleanup_errors.is_empty() {
                Err(error)
            } else {
                Err(format!("{error}\nCleanup: {}", cleanup_errors.join("; ")))
            };
        }
        Ok(route)
    }

    fn mute(&self, control: &impl PulseControl) -> Result<(), String> {
        let module = self
            .source_module
            .ok_or("The session has no virtual microphone to mute.")?;
        verify_owner(control, "sources", SOURCE_NAME, module)?;
        control.command(&["set-source-mute", SOURCE_NAME, "1"])?;
        Ok(())
    }

    fn load(&mut self, control: &impl PulseControl, arguments: &[&str]) -> Result<u32, String> {
        let mut command = vec!["load-module"];
        command.extend_from_slice(arguments);
        let text = control.command(&command)?;
        let module: u32 = text
            .parse()
            .map_err(|_| format!("pactl returned an invalid module ID: {text}"))?;
        self.modules.push(module);
        Ok(module)
    }

    fn restore_default(
        &self,
        control: &impl PulseControl,
        kind: &str,
        owned_name: &str,
        previous: Option<&str>,
    ) -> Result<(), String> {
        if let Some(previous) = previous.filter(|name| *name != owned_name) {
            let query = format!("get-default-{kind}");
            if control.command(&[&query])? == owned_name {
                let change = format!("set-default-{kind}");
                control.command(&[&change, previous])?;
            }
        }
        Ok(())
    }

    fn close(&mut self, control: &impl PulseControl) -> Vec<String> {
        self.source_module = None;
        if self.modules.is_empty() {
            return Vec::new();
        }
        let mut errors = Vec::new();
        for (kind, owned, previous) in [
            ("source", SOURCE_NAME, self.previous_source.as_deref()),
            ("sink", SINK_NAME, self.previous_sink.as_deref()),
        ] {
            if let Err(error) = self.restore_default(control, kind, owned, previous) {
                errors.push(error);
            }
        }
        for module in self.modules.drain(..).rev() {
            if let Err(error) = control.command(&["unload-module", &module.to_string()]) {
                errors.push(error);
            }
        }
        errors
    }
}

fn verify_owner(
    control: &impl PulseControl,
    kind: &str,
    name: &str,
    module: u32,
) -> Result<(), String> {
    // A second process racing startup must never capture another instance's bus.
    // PipeWire's remap-source module returns before its async stream has been
    // advertised to the registry. Wait briefly for the newly-created device.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let objects = server_objects(control, kind)?;
        let matching: Vec<_> = objects
            .iter()
            .filter(|entry| entry["name"] == name)
            .collect();
        if matching.is_empty() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
            continue;
        }
        if matching.len() != 1 {
            return Err(format!(
                "Cannot reserve virtual audio device {name}; another instance may be running."
            ));
        }
        if let Some(owner) = matching[0].get("owner_module").and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        }) && owner != module as u64
        {
            return Err(format!(
                "Virtual audio device {name} belongs to another module."
            ));
        }
        return Ok(());
    }
}

fn quote_module_value(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    struct FakePulse {
        calls: RefCell<Vec<Vec<String>>>,
        source: RefCell<String>,
        sink: RefCell<String>,
        modules: RefCell<Vec<(u32, String)>>,
        source_muted: RefCell<bool>,
        fail_remap: bool,
        auto_switch_sink: bool,
    }

    impl FakePulse {
        fn ready() -> Self {
            Self {
                source: RefCell::new("hardware_mic".into()),
                sink: RefCell::new("headphones".into()),
                ..Self::default()
            }
        }
    }

    impl PulseControl for FakePulse {
        fn command(&self, arguments: &[&str]) -> Result<String, String> {
            self.calls.borrow_mut().push(
                arguments
                    .iter()
                    .map(|argument| (*argument).into())
                    .collect(),
            );
            match arguments {
                ["get-default-source"] => Ok(self.source.borrow().clone()),
                ["get-default-sink"] => Ok(self.sink.borrow().clone()),
                ["set-default-source", name] => {
                    *self.source.borrow_mut() = (*name).into();
                    Ok(String::new())
                }
                ["set-default-sink", name] => {
                    *self.sink.borrow_mut() = (*name).into();
                    Ok(String::new())
                }
                ["set-source-mute", SOURCE_NAME, mute] => {
                    *self.source_muted.borrow_mut() = *mute == "1";
                    Ok(String::new())
                }
                ["--format=json", "list", kind] => {
                    let modules = self.modules.borrow();
                    let mut values = if *kind == "sources" {
                        vec![
                            json!({"name":"hardware_mic", "description":"USB microphone", "monitor_of_sink":4294967295_u64}),
                            json!({"name":"headphones.monitor", "monitor_of_sink":4}),
                            json!({"name":"hidden_monitor", "monitor_of_sink":4}),
                        ]
                    } else {
                        vec![json!({"name":"headphones", "description":"Headphones"})]
                    };
                    for (id, module) in modules.iter() {
                        match (kind.to_owned(), module.as_str()) {
                            ("sources", "module-remap-source") => {
                                values.push(json!({"name":SOURCE_NAME,"owner_module":id}))
                            }
                            ("sinks", "module-null-sink") => {
                                values.push(json!({"name":SINK_NAME,"owner_module":id}))
                            }
                            _ => {}
                        }
                    }
                    Ok(serde_json::to_string(&values).unwrap())
                }
                ["load-module", module, ..] => {
                    if *module == "module-remap-source" && self.fail_remap {
                        return Err("remap failed".into());
                    }
                    let id = self.modules.borrow().len() as u32 + 10;
                    self.modules.borrow_mut().push((id, (*module).into()));
                    if *module == "module-null-sink" && self.auto_switch_sink {
                        *self.sink.borrow_mut() = SINK_NAME.into();
                    }
                    Ok(id.to_string())
                }
                ["unload-module", id] => {
                    let id = id.parse::<u32>().unwrap();
                    self.modules
                        .borrow_mut()
                        .retain(|(module, _)| *module != id);
                    Ok(String::new())
                }
                _ => panic!("Unexpected command: {arguments:?}"),
            }
        }
    }

    #[test]
    fn enumeration_excludes_monitor_and_own_devices() {
        let pulse = FakePulse::ready();
        pulse
            .modules
            .borrow_mut()
            .push((12, "module-remap-source".into()));
        assert_eq!(
            list_devices(&pulse, "sources", true).unwrap(),
            vec![AudioDevice {
                name: "hardware_mic".into(),
                description: "USB microphone".into(),
            }]
        );
    }

    #[test]
    fn failed_setup_rolls_back_only_its_loaded_modules() {
        let pulse = FakePulse {
            fail_remap: true,
            ..FakePulse::ready()
        };
        assert!(Routing::create(&pulse, true, None).is_err());
        assert!(pulse.modules.borrow().is_empty());
        let calls = pulse.calls.borrow();
        assert!(calls.iter().any(|call| call == &["unload-module", "10"]));
        assert!(!calls.iter().any(|call| call == &["unload-module", "11"]));
    }

    #[test]
    fn normal_stop_restores_defaults_and_unloads_in_reverse_order() {
        let pulse = FakePulse {
            auto_switch_sink: true,
            ..FakePulse::ready()
        };
        let mut route = Routing::create(&pulse, true, Some("headphones")).unwrap();
        assert_eq!(*pulse.source.borrow(), SOURCE_NAME);
        assert_eq!(*pulse.sink.borrow(), "headphones");
        assert!(route.close(&pulse).is_empty());
        assert_eq!(*pulse.source.borrow(), "hardware_mic");
        let unloaded: Vec<_> = pulse
            .calls
            .borrow()
            .iter()
            .filter(|call| call[0] == "unload-module")
            .cloned()
            .collect();
        assert_eq!(
            unloaded,
            vec![
                vec!["unload-module", "12"],
                vec!["unload-module", "11"],
                vec!["unload-module", "10"]
            ]
        );
        assert!(route.close(&pulse).is_empty());
    }

    #[test]
    fn cleanup_preserves_a_default_changed_by_the_user() {
        let pulse = FakePulse::ready();
        let mut route = Routing::create(&pulse, true, None).unwrap();
        *pulse.source.borrow_mut() = "new_headset".into();
        *pulse.sink.borrow_mut() = "new_output".into();
        route.close(&pulse);
        assert_eq!(*pulse.source.borrow(), "new_headset");
        assert_eq!(*pulse.sink.borrow(), "new_output");
    }

    #[test]
    fn muted_route_retains_devices_and_default_until_stop_and_restarts_unmuted() {
        let pulse = FakePulse::ready();
        let mut route = Routing::create(&pulse, true, None).unwrap();
        let modules = pulse.modules.borrow().clone();
        route.mute(&pulse).unwrap();
        assert!(*pulse.source_muted.borrow());
        assert_eq!(*pulse.modules.borrow(), modules);
        assert_eq!(*pulse.source.borrow(), SOURCE_NAME);
        assert_eq!(*pulse.sink.borrow(), "headphones");

        assert!(route.close(&pulse).is_empty());
        assert_eq!(*pulse.source.borrow(), "hardware_mic");
        assert!(pulse.modules.borrow().is_empty());
        // FakePulse retains the last mute state, like a session manager's
        // restored device settings. A new route must explicitly reset it.
        let mut restarted = Routing::create(&pulse, true, None).unwrap();
        assert!(!*pulse.source_muted.borrow());
        assert!(restarted.close(&pulse).is_empty());
    }

    #[test]
    fn mute_does_not_touch_a_replacement_source_owned_by_another_module() {
        let pulse = FakePulse::ready();
        let route = Routing::create(&pulse, true, None).unwrap();
        for (id, module) in pulse.modules.borrow_mut().iter_mut() {
            if module == "module-remap-source" {
                *id = 99;
            }
        }
        let modules = pulse.modules.borrow().clone();
        assert!(route.mute(&pulse).is_err());
        assert!(!*pulse.source_muted.borrow());
        assert_eq!(*pulse.modules.borrow(), modules);
        assert_eq!(*pulse.source.borrow(), SOURCE_NAME);
        assert!(!pulse.calls.borrow().iter().any(|call| {
            call == &["set-source-mute", SOURCE_NAME, "1"] || call[0] == "unload-module"
        }));
    }

    #[test]
    fn queue_discards_old_audio_and_stops_without_draining() {
        let queue = BlockQueue::new(2);
        assert_eq!(queue.push(vec![1.0]), 0);
        assert_eq!(queue.push(vec![2.0]), 0);
        assert_eq!(queue.push(vec![3.0]), 1);
        let stop = AtomicBool::new(false);
        assert_eq!(queue.pop(&stop), Some(vec![2.0]));
        stop.store(true, Ordering::Release);
        assert_eq!(queue.pop(&stop), None);
    }

    #[test]
    fn processor_output_is_length_checked_finite_and_limited() {
        assert!(validate_output(vec![0.1], 2).is_err());
        assert!(validate_output(vec![f32::NAN], 1).is_err());
        assert!(validate_output(vec![f32::INFINITY], 1).is_err());
        assert_eq!(
            validate_output(vec![-2.0, 0.5, 1.5], 3).unwrap(),
            vec![-1.0, 0.5, 1.0]
        );
    }

    #[test]
    fn processor_failure_stops_without_forwarding_raw_audio() {
        struct Failure;
        impl Processor for Failure {
            fn process(&mut self, _: &[f32]) -> Result<Vec<f32>, String> {
                Err("AI server disconnected".into())
            }
        }
        let input = Arc::new(BlockQueue::new(2));
        let output = Arc::new(BlockQueue::new(2));
        let state = WorkerState {
            stop: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(Mutex::new(AudioStats::default())),
            running: Arc::new(AtomicBool::new(true)),
        };
        input.push(vec![0.3, 0.2]);
        process_worker(Box::new(Failure), input, output.clone(), state.clone());
        assert!(state.stop.load(Ordering::Acquire));
        assert!(!state.running.load(Ordering::Acquire));
        assert!(output.blocks.lock().unwrap().is_empty());
        assert!(
            state
                .stats
                .lock()
                .unwrap()
                .error
                .as_ref()
                .unwrap()
                .contains("AI server disconnected")
        );
    }

    #[test]
    fn processor_panics_report_the_cause_without_forwarding_audio() {
        struct Panic(bool);
        impl Processor for Panic {
            fn process(&mut self, _: &[f32]) -> Result<Vec<f32>, String> {
                if self.0 {
                    std::panic::panic_any(String::from("processor failure detail"));
                }
                panic!("processor failure detail");
            }
        }
        for owned_message in [false, true] {
            let input = Arc::new(BlockQueue::new(2));
            let output = Arc::new(BlockQueue::new(2));
            let state = WorkerState {
                stop: Arc::new(AtomicBool::new(false)),
                stats: Arc::new(Mutex::new(AudioStats::default())),
                running: Arc::new(AtomicBool::new(true)),
            };
            input.push(vec![0.3, 0.2]);
            process_worker(
                Box::new(Panic(owned_message)),
                input,
                output.clone(),
                state.clone(),
            );
            assert!(state.stop.load(Ordering::Acquire));
            assert!(!state.running.load(Ordering::Acquire));
            assert!(output.blocks.lock().unwrap().is_empty());
            assert!(
                state
                    .stats
                    .lock()
                    .unwrap()
                    .error
                    .as_ref()
                    .unwrap()
                    .contains("processor failure detail")
            );
        }
    }

    #[test]
    fn late_failure_from_a_previous_session_cannot_stop_the_new_session() {
        let mut engine = AudioEngine::new();
        let old_state = WorkerState {
            stop: Arc::new(AtomicBool::new(true)),
            stats: engine.stats.clone(),
            running: engine.running.clone(),
        };
        engine.reset_session_state();
        engine.running.store(true, Ordering::Release);
        engine.stats.lock().unwrap().frames = 960;

        // Simulate a detached AI request completing after Stop and Start.
        old_state.fail("Old AI request failed after its session stopped".into());
        old_state.record_error("A late diagnostic from the old request".into());
        old_state.dropped(3);

        assert!(engine.running());
        let stats = engine.stats();
        assert_eq!(stats.frames, 960);
        assert_eq!(stats.dropped_blocks, 0);
        assert!(stats.error.is_none());
        assert!(!Arc::ptr_eq(&engine.stats, &old_state.stats));
        assert!(!Arc::ptr_eq(&engine.running, &old_state.running));
    }

    #[test]
    fn duplicate_virtual_device_is_not_adopted_or_unloaded() {
        let pulse = FakePulse::ready();
        pulse
            .modules
            .borrow_mut()
            .push((99, "module-null-sink".into()));
        assert!(Routing::create(&pulse, false, None).is_err());
        assert_eq!(pulse.modules.borrow().len(), 1);
        assert!(
            !pulse
                .calls
                .borrow()
                .iter()
                .any(|call| call[0] == "unload-module")
        );
    }

    #[test]
    fn module_values_quote_spaces_and_special_characters() {
        assert_eq!(quote_module_value("a\"b\\c d"), "\"a\\\"b\\\\c d\"");
    }

    /// Run through scripts/test-pipewire.sh, which supplies an isolated audio
    /// server and session manager. Never touches the desktop's microphones.
    #[test]
    #[ignore = "Requires an isolated Linux PipeWire session; use scripts/test-pipewire.sh"]
    fn pipewire_virtual_microphone_roundtrip() {
        assert_eq!(
            std::env::var("VOICE_WARP_TEST_ISOLATED").as_deref(),
            Ok("1"),
            "Run scripts/test-pipewire.sh to isolate the test from desktop audio."
        );

        struct SyntheticInput {
            modules: Vec<u32>,
            sink: String,
            source: String,
        }
        impl SyntheticInput {
            fn new() -> Self {
                let mut input = Self {
                    modules: Vec::new(),
                    sink: format!("vw_ci_sink_{}", std::process::id()),
                    source: format!("vw_ci_source_{}", std::process::id()),
                };
                let sink = format!("sink_name={}", input.sink);
                let id = Pactl
                    .command(&[
                        "load-module",
                        "module-null-sink",
                        &sink,
                        "channels=1",
                        "channel_map=mono",
                        "rate=48000",
                    ])
                    .unwrap();
                input.modules.push(id.parse().unwrap());
                verify_owner(&Pactl, "sinks", &input.sink, *input.modules.last().unwrap()).unwrap();
                let source = format!("source_name={}", input.source);
                let master = format!("master={}.monitor", input.sink);
                let id = Pactl
                    .command(&[
                        "load-module",
                        "module-remap-source",
                        &source,
                        &master,
                        "channels=1",
                        "channel_map=mono",
                        "master_channel_map=mono",
                        "rate=48000",
                    ])
                    .unwrap();
                input.modules.push(id.parse().unwrap());
                verify_owner(
                    &Pactl,
                    "sources",
                    &input.source,
                    *input.modules.last().unwrap(),
                )
                .unwrap();
                Pactl.command(&["set-default-sink", &input.sink]).unwrap();
                Pactl
                    .command(&["set-default-source", &input.source])
                    .unwrap();
                input
            }
        }
        impl Drop for SyntheticInput {
            fn drop(&mut self) {
                for id in self.modules.iter().rev() {
                    let _ = Pactl.command(&["unload-module", &id.to_string()]);
                }
            }
        }

        struct Tone {
            child: Child,
            writer: Option<JoinHandle<()>>,
        }
        impl Tone {
            fn start(sink: &str) -> Self {
                let mut child = spawn_audio("pacat", sink).unwrap();
                let mut stdin = child.stdin.take().unwrap();
                let writer = thread::spawn(move || {
                    let mut frame = 0_u64;
                    loop {
                        let bytes: Vec<u8> = (0..960)
                            .flat_map(|_| {
                                let phase = frame as f64 * 440.0 * std::f64::consts::TAU
                                    / SAMPLE_RATE as f64;
                                frame += 1;
                                (0.4 * phase.sin() as f32).to_le_bytes()
                            })
                            .collect();
                        if stdin.write_all(&bytes).is_err() {
                            break;
                        }
                    }
                });
                Self {
                    child,
                    writer: Some(writer),
                }
            }
        }
        impl Drop for Tone {
            fn drop(&mut self) {
                terminate(&mut self.child);
                if let Some(writer) = self.writer.take() {
                    let _ = writer.join();
                }
            }
        }

        fn record_virtual_source() -> Vec<f32> {
            let mut recorder = spawn_audio("parec", SOURCE_NAME).unwrap();
            let mut stdout = recorder.stdout.take().unwrap();
            let (sender, receiver) = std::sync::mpsc::channel();
            let reader = thread::spawn(move || {
                let mut bytes = vec![0; SAMPLE_RATE / 2 * 4];
                let result = stdout.read_exact(&mut bytes).map(|()| bytes);
                let _ = sender.send(result);
            });
            let result = receiver.recv_timeout(Duration::from_secs(5));
            terminate(&mut recorder);
            let _ = reader.join();
            result
                .expect("Timed out receiving the virtual microphone")
                .expect("Virtual microphone read failed")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect()
        }

        fn wait_for_source_mute(muted: bool) {
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                let sources = server_objects(&Pactl, "sources").unwrap();
                let source = sources.iter().find(|source| source["name"] == SOURCE_NAME);
                if source.and_then(|source| source["mute"].as_bool()) == Some(muted) {
                    return;
                }
                assert!(
                    Instant::now() < deadline,
                    "Virtual microphone did not reach mute={muted}: {source:?}"
                );
                thread::sleep(Duration::from_millis(20));
            }
        }

        fn assert_processed_tone(engine: &AudioEngine) {
            let signal = record_virtual_source();
            let rms = (signal.iter().map(|sample| sample * sample).sum::<f32>()
                / signal.len() as f32)
                .sqrt();
            assert!(
                rms > 0.08 && rms < 0.18,
                "Expected the processed tone (0.141 RMS), got {rms}; {:?}",
                engine.stats()
            );
            assert!(
                peak(&signal) > 0.15 && peak(&signal) < 0.25,
                "The selected processor must affect the public microphone."
            );
            assert!(engine.stats().frames > 0);
        }

        struct TestProcessor {
            fail: Arc<AtomicBool>,
        }
        impl Processor for TestProcessor {
            fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
                if self.fail.load(Ordering::Acquire) {
                    return Err("intentional test failure".into());
                }
                Ok(samples.iter().map(|sample| sample * 0.5).collect())
            }
        }

        let input = SyntheticInput::new();
        let _tone = Tone::start(&input.sink);
        let fail = Arc::new(AtomicBool::new(false));
        let mut engine = AudioEngine::new();
        engine
            .start(
                &input.source,
                Box::new(TestProcessor { fail: fail.clone() }),
                960,
                true,
                None,
                None,
            )
            .unwrap();
        assert_eq!(Pactl.command(&["get-default-source"]).unwrap(), SOURCE_NAME);
        assert_eq!(Pactl.command(&["get-default-sink"]).unwrap(), input.sink);
        wait_for_source_mute(false);
        thread::sleep(Duration::from_millis(400));
        assert_processed_tone(&engine);

        fail.store(true, Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(3);
        while engine.running() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(
            !engine.running(),
            "Processor errors must stop recording/playback."
        );
        assert!(
            engine
                .stats()
                .error
                .as_deref()
                .unwrap_or("")
                .contains("intentional test failure")
        );
        wait_for_source_mute(true);
        assert!(
            server_objects(&Pactl, "sources")
                .unwrap()
                .iter()
                .any(|entry| entry["name"] == SOURCE_NAME),
            "An error must retain the silent virtual microphone."
        );
        assert_eq!(Pactl.command(&["get-default-source"]).unwrap(), SOURCE_NAME);
        let silence = record_virtual_source();
        assert!(
            silence.iter().all(|sample| sample.is_finite()) && peak(&silence) < 0.001,
            "Failed processing must emit silence: peak={}, non-silent samples={}/{}; {:?}",
            peak(&silence),
            silence
                .iter()
                .filter(|sample| sample.abs() >= 0.001)
                .count(),
            silence.len(),
            engine.stats()
        );

        // Restart must reap the failed session and clear any mute setting
        // restored by the session manager for the newly-created source.
        fail.store(false, Ordering::Release);
        engine
            .start(
                &input.source,
                Box::new(TestProcessor { fail: fail.clone() }),
                960,
                true,
                None,
                None,
            )
            .unwrap();
        wait_for_source_mute(false);
        assert_eq!(Pactl.command(&["get-default-source"]).unwrap(), SOURCE_NAME);
        thread::sleep(Duration::from_millis(400));
        assert_processed_tone(&engine);

        engine.stop();
        assert_eq!(
            Pactl.command(&["get-default-source"]).unwrap(),
            input.source
        );
        assert_eq!(Pactl.command(&["get-default-sink"]).unwrap(), input.sink);
        assert!(
            !server_objects(&Pactl, "sources")
                .unwrap()
                .iter()
                .any(|entry| entry["name"] == SOURCE_NAME)
        );
        assert!(
            !server_objects(&Pactl, "sinks")
                .unwrap()
                .iter()
                .any(|entry| entry["name"] == SINK_NAME)
        );
    }
}
