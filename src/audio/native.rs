//! WASAPI/CoreAudio streams. Native virtual cables remain installed independently
//! of this process; errors stop our streams without changing system defaults.

use super::resample::LinearResampler;
use super::{AudioDevice, AudioStats, Processor, SAMPLE_RATE};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use crossbeam_queue::ArrayQueue;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

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
    pub fn new() -> Self {
        Self {
            session: None,
            stats: Arc::new(Mutex::new(AudioStats::default())),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn requires_output_device() -> bool {
        true
    }

    pub fn devices() -> Result<Vec<AudioDevice>, String> {
        Ok(enumerate(true)?.into_iter().map(|(info, _)| info).collect())
    }

    pub fn outputs() -> Result<Vec<AudioDevice>, String> {
        Ok(enumerate(false)?
            .into_iter()
            .map(|(info, _)| info)
            .collect())
    }

    pub fn diagnostics() -> Result<Value, String> {
        Ok(json!({
            "platform": std::env::consts::OS,
            "supported": true,
            "backend": cpal::default_host().id().name(),
            "sample_rate": SAMPLE_RATE,
            "inputs": Self::devices()?,
            "outputs": Self::outputs()?,
            "requires_output_device": true,
            "virtual_driver_bundled": false,
            "routing": "Select your virtual cable's playback endpoint. In other apps select its recording endpoint."
        }))
    }

    pub fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
    pub fn stats(&self) -> AudioStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &mut self,
        source: &str,
        processor: Box<dyn Processor>,
        block_size: usize,
        make_default: bool,
        monitor_sink: Option<&str>,
        output_device: Option<&str>,
    ) -> Result<(), String> {
        if self.running() {
            return Err("Voice Warp is already running.".into());
        }
        self.stop();
        if make_default {
            return Err("Set the virtual cable as your microphone in each app or in system settings. Automatic default-device changes are available on Linux only.".into());
        }
        let output = output_device.filter(|name| !name.is_empty()).ok_or_else(||
            "Select a virtual cable output first (CABLE Input on Windows, or BlackHole on macOS). Voice Warp never chooses your speakers automatically.".to_string())?;
        if !(240..=48_000).contains(&block_size) {
            return Err("Audio block size must be between 240 and 48000 samples.".into());
        }
        if monitor_sink == Some(output) {
            return Err("The monitoring output must differ from the virtual cable output.".into());
        }
        self.stats = Arc::new(Mutex::new(AudioStats::default()));
        self.running = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let state = WorkerState {
            stop: stop.clone(),
            stats: self.stats.clone(),
            running: self.running.clone(),
        };
        let (ready_sender, ready_receiver) = mpsc::channel();
        let source = source.to_owned();
        let output = output.to_owned();
        let monitor = monitor_sink.map(str::to_owned);
        // CPAL streams never cross this thread boundary (CoreAudio streams need
        // not be Send). Only Arc state and the join handle live in AudioEngine.
        let supervisor = thread::spawn(move || {
            run_session(
                source,
                output,
                monitor,
                processor,
                block_size,
                state,
                ready_sender,
            );
        });
        self.session = Some(Session { stop, supervisor });
        match ready_receiver.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(()),
            result => {
                let error = match result {
                    Ok(Err(error)) => error,
                    Err(_) => "Audio device initialization did not complete. Refresh devices and try again.".into(),
                    Ok(Ok(())) => unreachable!(),
                };
                self.stop();
                self.stats.lock().unwrap_or_else(|e| e.into_inner()).error = Some(error.clone());
                Err(error)
            }
        }
    }

    pub fn stop(&mut self) {
        if let Some(session) = self.session.take() {
            session.stop.store(true, Ordering::Release);
            let deadline = Instant::now() + Duration::from_secs(3);
            while !session.supervisor.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(10));
            }
            if session.supervisor.is_finished() {
                let _ = session.supervisor.join();
            }
            // A broken driver/slow inference may outlive Stop. Its callbacks
            // see cancellation, and all stats/running state belongs to it alone.
        }
        self.running.store(false, Ordering::Release);
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn enumerate(input: bool) -> Result<Vec<(AudioDevice, cpal::Device)>, String> {
    let host = cpal::default_host();
    let devices = if input {
        host.input_devices()
    } else {
        host.output_devices()
    }
    .map_err(|error| format!("Cannot enumerate audio devices: {error}"))?;
    let mut seen = HashMap::<String, usize>::new();
    let mut result = Vec::new();
    for device in devices {
        let label = match device.name() {
            Ok(name) if !name.is_empty() => name,
            _ => continue,
        };
        let ordinal = seen.entry(label.clone()).or_default();
        let name = device_token(input, &label, *ordinal);
        let description = if *ordinal == 0 {
            label
        } else {
            format!("{label} (device {})", *ordinal + 1)
        };
        *ordinal += 1;
        result.push((AudioDevice { name, description }, device));
    }
    Ok(result)
}

fn device_token(input: bool, name: &str, ordinal: usize) -> String {
    format!(
        "{}:{ordinal}:{name}",
        if input { "input" } else { "output" }
    )
}

fn selected_device(input: bool, selection: &str) -> Result<cpal::Device, String> {
    enumerate(input)?
        .into_iter()
        .find(|(info, _)| info.name == selection)
        .map(|(_, device)| device)
        .ok_or_else(|| {
            format!(
                "The selected {} is unavailable. Refresh devices and select it again.",
                if input { "microphone" } else { "output" }
            )
        })
}

fn same_device_family(input: &str, output: &str) -> bool {
    let input = input.to_lowercase();
    let output = output.to_lowercase();
    if input == output {
        return true;
    }
    // Windows gives the two ends of VB-CABLE complementary names. Feeding
    // CABLE Output back into CABLE Input creates a feedback loop.
    if input.contains("cable") && output.contains("cable") {
        return input.replace(" input", "").replace(" output", "")
            == output.replace(" input", "").replace(" output", "");
    }
    false
}

#[derive(Clone)]
struct WorkerState {
    stop: Arc<AtomicBool>,
    stats: Arc<Mutex<AudioStats>>,
    running: Arc<AtomicBool>,
}

impl WorkerState {
    fn fail(&self, error: String) {
        if !self.stop.swap(true, Ordering::AcqRel) {
            self.stats.lock().unwrap_or_else(|e| e.into_inner()).error = Some(error);
        }
        self.running.store(false, Ordering::Release);
    }
}

struct AudioRing {
    samples: ArrayQueue<f32>,
    dropped: AtomicU64,
}

impl AudioRing {
    fn new(capacity: usize) -> Self {
        Self {
            samples: ArrayQueue::new(capacity.max(1)),
            dropped: AtomicU64::new(0),
        }
    }
    fn push(&self, sample: f32) {
        if self.samples.force_push(sample).is_some() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn pop(&self) -> Option<f32> {
        self.samples.pop()
    }
}

struct OutputPath {
    queue: Arc<AudioRing>,
    rate: u32,
    resampler: LinearResampler,
}

fn ring_capacity(rate: u32, block_size: usize) -> usize {
    // At least two processing blocks, with a short cushion for device callbacks.
    ((rate as usize * block_size * 2) / SAMPLE_RATE).max(rate as usize / 10)
}

fn run_session(
    source: String,
    output: String,
    monitor: Option<String>,
    mut processor: Box<dyn Processor>,
    block_size: usize,
    state: WorkerState,
    ready: mpsc::Sender<Result<(), String>>,
) {
    let device_error = Arc::new(AtomicBool::new(false));
    let setup = (|| -> Result<_, String> {
        let input_device = selected_device(true, &source)?;
        let output_device = selected_device(false, &output)?;
        let input_label = input_device.name().map_err(|e| e.to_string())?;
        let output_label = output_device.name().map_err(|e| e.to_string())?;
        if same_device_family(&input_label, &output_label) {
            return Err("Input and output refer to the same audio device/cable. Select a physical microphone as input.".into());
        }
        let input_config = input_device
            .default_input_config()
            .map_err(|e| format!("Cannot open microphone format: {e}"))?;
        let output_config = output_device
            .default_output_config()
            .map_err(|e| format!("Cannot open cable output format: {e}"))?;
        validate_config(&input_config)?;
        validate_config(&output_config)?;
        let input_rate = input_config.sample_rate().0;
        let output_rate = output_config.sample_rate().0;
        let captured = Arc::new(AudioRing::new(ring_capacity(input_rate, block_size)));
        let output_queue = Arc::new(AudioRing::new(ring_capacity(output_rate, block_size)));
        let input_stream = input_stream(
            &input_device,
            &input_config,
            captured.clone(),
            state.stop.clone(),
            device_error.clone(),
        )?;
        let playback_stream = output_stream(
            &output_device,
            &output_config,
            output_queue.clone(),
            state.stop.clone(),
            device_error.clone(),
        )?;
        let mut paths = vec![OutputPath {
            queue: output_queue,
            rate: output_rate,
            resampler: LinearResampler::new(SAMPLE_RATE as u32, output_rate),
        }];
        let monitor_stream = if let Some(monitor) = &monitor {
            let device = selected_device(false, monitor)?;
            let config = device
                .default_output_config()
                .map_err(|e| format!("Cannot open monitor output: {e}"))?;
            validate_config(&config)?;
            let rate = config.sample_rate().0;
            let queue = Arc::new(AudioRing::new(ring_capacity(rate, block_size)));
            let stream = output_stream(
                &device,
                &config,
                queue.clone(),
                state.stop.clone(),
                device_error.clone(),
            )?;
            paths.push(OutputPath {
                queue,
                rate,
                resampler: LinearResampler::new(SAMPLE_RATE as u32, rate),
            });
            Some(stream)
        } else {
            None
        };
        if state.stop.load(Ordering::Acquire) {
            return Err("Audio startup was cancelled.".into());
        }
        processor.set_stop_signal(state.stop.clone());
        playback_stream
            .play()
            .map_err(|e| format!("Cannot start cable output: {e}"))?;
        if let Some(stream) = &monitor_stream {
            stream
                .play()
                .map_err(|e| format!("Cannot start monitoring: {e}"))?;
        }
        input_stream.play().map_err(|e| {
            format!("Cannot start microphone: {e}. Check microphone privacy permissions.")
        })?;
        Ok((
            input_stream,
            playback_stream,
            monitor_stream,
            captured,
            input_rate,
            paths,
        ))
    })();
    let (input_stream, output_stream, monitor_stream, captured, input_rate, paths) = match setup {
        Ok(setup) => setup,
        Err(error) => {
            state.fail(error.clone());
            let _ = ready.send(Err(error));
            return;
        }
    };
    state.running.store(true, Ordering::Release);
    let process_state = state.clone();
    let processing = thread::spawn(move || {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_audio(
                processor,
                captured,
                input_rate,
                paths,
                block_size,
                &process_state,
            );
        }));
        if caught.is_err() {
            process_state.fail("The voice processor stopped unexpectedly.".into());
        }
    });
    let _ = ready.send(Ok(()));
    while !state.stop.load(Ordering::Acquire) {
        if device_error.load(Ordering::Acquire) {
            state.fail("An audio device disconnected or its stream failed. Refresh devices and reconnect. No unprocessed audio was substituted.".into());
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    // Device streams stop even if an AI request is still finishing elsewhere.
    drop(input_stream);
    drop(output_stream);
    drop(monitor_stream);
    state.running.store(false, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(2);
    while !processing.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    if processing.is_finished() {
        let _ = processing.join();
    }
    let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
    stats.input_peak = 0.0;
    stats.output_peak = 0.0;
}

fn validate_config(config: &cpal::SupportedStreamConfig) -> Result<(), String> {
    if config.channels() == 0 || config.sample_rate().0 == 0 {
        return Err("The audio device returned an invalid channel count or sample rate.".into());
    }
    Ok(())
}

fn process_audio(
    mut processor: Box<dyn Processor>,
    captured: Arc<AudioRing>,
    input_rate: u32,
    mut paths: Vec<OutputPath>,
    block_size: usize,
    state: &WorkerState,
) {
    let mut resampler = LinearResampler::new(input_rate, SAMPLE_RATE as u32);
    let mut block = Vec::with_capacity(block_size);
    let mut resampled = Vec::with_capacity(16);
    while !state.stop.load(Ordering::Acquire) {
        let Some(sample) = captured.pop() else {
            thread::sleep(Duration::from_millis(1));
            continue;
        };
        resampled.clear();
        resampler.push(sample, &mut |sample| resampled.push(sample));
        for &sample in &resampled {
            block.push(sample);
            if block.len() != block_size {
                continue;
            }
            let started = Instant::now();
            let processed = match processor
                .process(&block)
                .and_then(|values| validated_audio(values, block_size))
            {
                Ok(values) => values,
                Err(error) => {
                    state.fail(format!("Voice processing stopped: {error}"));
                    return;
                }
            };
            if state.stop.load(Ordering::Acquire) {
                return;
            }
            let processing_ms = started.elapsed().as_secs_f32() * 1000.0;
            for path in &mut paths {
                for &value in &processed {
                    path.resampler
                        .push(value, &mut |sample| path.queue.push(sample));
                }
            }
            let dropped_input =
                captured.dropped.load(Ordering::Relaxed) * SAMPLE_RATE as u64 / input_rate as u64;
            let dropped_output: u64 = paths
                .iter()
                .map(|path| {
                    path.queue.dropped.load(Ordering::Relaxed) * SAMPLE_RATE as u64
                        / path.rate as u64
                })
                .sum();
            {
                let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
                stats.input_peak = peak(&block);
                stats.output_peak = peak(&processed);
                stats.processing_ms = processing_ms;
                stats.frames += block_size as u64;
                stats.dropped_blocks = (dropped_input + dropped_output) / block_size as u64;
            }
            block.clear();
        }
    }
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0, f32::max).min(1.0)
}
fn validated_audio(mut samples: Vec<f32>, expected: usize) -> Result<Vec<f32>, String> {
    if samples.len() != expected || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(
            "The voice processor returned invalid audio or a different block length.".into(),
        );
    }
    samples
        .iter_mut()
        .for_each(|sample| *sample = sample.clamp(-1.0, 1.0));
    Ok(samples)
}

// These helpers are shared by native callbacks and offline format/channel tests.
// The data callbacks do not allocate, lock mutexes, or perform inference/I/O.
fn capture_samples<T>(data: &[T], channels: usize, ring: &AudioRing)
where
    T: Sample + Copy,
    f32: FromSample<T>,
{
    for frame in data.chunks_exact(channels) {
        let sum: f32 = frame
            .iter()
            .map(|sample| {
                let value = f32::from_sample(*sample);
                if value.is_finite() { value } else { 0.0 }
            })
            .sum();
        ring.push((sum / channels as f32).clamp(-1.0, 1.0));
    }
}

fn render_samples<T>(data: &mut [T], channels: usize, ring: &AudioRing, stopped: bool)
where
    T: Sample + FromSample<f32>,
{
    if stopped {
        data.fill(T::EQUILIBRIUM);
        return;
    }
    for frame in data.chunks_exact_mut(channels) {
        let sample = T::from_sample(ring.pop().unwrap_or(0.0).clamp(-1.0, 1.0));
        frame.fill(sample);
    }
    // Defensively silence a malformed partial frame rather than leaving stale data.
    let remainder = data.len() % channels;
    if remainder != 0 {
        let start = data.len() - remainder;
        data[start..].fill(T::EQUILIBRIUM);
    }
}

fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    ring: Arc<AudioRing>,
    stop: Arc<AtomicBool>,
    error: Arc<AtomicBool>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                if !stop.load(Ordering::Acquire) {
                    capture_samples(data, channels, &ring);
                }
            },
            move |_| {
                error.store(true, Ordering::Release);
            },
            None,
        )
        .map_err(|e| format!("Cannot build microphone stream: {e}"))
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    ring: Arc<AudioRing>,
    stop: Arc<AtomicBool>,
    error: Arc<AtomicBool>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                render_samples(data, channels, &ring, stop.load(Ordering::Acquire));
            },
            move |_| {
                error.store(true, Ordering::Release);
            },
            None,
        )
        .map_err(|e| format!("Cannot build output stream: {e}"))
}

macro_rules! dispatch_format {
    ($function:ident, $device:expr, $config:expr, $ring:expr, $stop:expr, $error:expr) => {
        match $config.sample_format() {
            SampleFormat::I8 => $function::<i8>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::I16 => $function::<i16>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::I24 => {
                $function::<cpal::I24>($device, &$config.config(), $ring, $stop, $error)
            }
            SampleFormat::I32 => $function::<i32>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::I64 => $function::<i64>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::U8 => $function::<u8>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::U16 => $function::<u16>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::U32 => $function::<u32>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::U64 => $function::<u64>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::F32 => $function::<f32>($device, &$config.config(), $ring, $stop, $error),
            SampleFormat::F64 => $function::<f64>($device, &$config.config(), $ring, $stop, $error),
            _ => Err("This audio device uses an unsupported sample format.".into()),
        }
    };
}

fn input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    ring: Arc<AudioRing>,
    stop: Arc<AtomicBool>,
    error: Arc<AtomicBool>,
) -> Result<cpal::Stream, String> {
    dispatch_format!(build_input, device, config, ring, stop, error)
}
fn output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    ring: Arc<AudioRing>,
    stop: Arc<AtomicBool>,
    error: Arc<AtomicBool>,
) -> Result<cpal::Stream, String> {
    dispatch_format!(build_output, device, config, ring, stop, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_ring_retains_recent_audio() {
        let ring = AudioRing::new(2);
        ring.push(0.1);
        ring.push(0.2);
        ring.push(0.3);
        assert_eq!(ring.pop(), Some(0.2));
        assert_eq!(ring.pop(), Some(0.3));
        assert_eq!(ring.dropped.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn capture_downmixes_channels_and_converts_integer_formats() {
        let ring = AudioRing::new(8);
        capture_samples(&[i16::MAX, i16::MIN, 16_384, 16_384], 2, &ring);
        assert!(ring.pop().unwrap().abs() < 0.0001);
        assert!((ring.pop().unwrap() - 0.5).abs() < 0.0001);
        capture_samples(&[32_768_u16, 65_535], 1, &ring);
        assert_eq!(ring.pop(), Some(0.0));
        assert!(ring.pop().unwrap() > 0.99);
        capture_samples(&[1_073_741_824_i32], 1, &ring);
        assert_eq!(ring.pop(), Some(0.5));
    }

    #[test]
    fn output_duplicates_mono_and_uses_correct_unsigned_silence() {
        let ring = AudioRing::new(4);
        ring.push(0.5);
        let mut output = [0_i16; 4];
        render_samples(&mut output, 2, &ring, false);
        assert_eq!(output, [16_384, 16_384, 0, 0]);
        let mut unsigned = [0_u16; 3];
        render_samples(&mut unsigned, 2, &ring, true);
        assert_eq!(unsigned, [32_768; 3]);
    }

    #[test]
    fn formats_nan_and_stop_are_handled_without_stale_output() {
        let ring = AudioRing::new(8);
        capture_samples(&[f64::NAN, 0.25], 1, &ring);
        assert_eq!(ring.pop(), Some(0.0));
        assert_eq!(ring.pop(), Some(0.25));
        ring.push(0.8);
        let mut data = [1.0_f32; 4];
        render_samples(&mut data, 2, &ring, true);
        assert_eq!(data, [0.0; 4]);
        assert!(validated_audio(vec![f32::NAN], 1).is_err());
        assert!(validated_audio(vec![0.2], 2).is_err());
    }

    #[test]
    fn selections_do_not_silently_resolve_to_defaults() {
        assert_ne!(device_token(true, "Mic", 0), device_token(true, "Mic", 1));
        assert_ne!(device_token(true, "Mic", 0), device_token(false, "Mic", 0));
        assert!(same_device_family("BlackHole 2ch", "BlackHole 2ch"));
        assert!(same_device_family(
            "CABLE Output (VB-Audio Virtual Cable)",
            "CABLE Input (VB-Audio Virtual Cable)"
        ));
        assert!(!same_device_family(
            "Microphone (USB)",
            "CABLE Input (VB-Audio Virtual Cable)"
        ));
        let mut engine = AudioEngine::new();
        struct Identity;
        impl Processor for Identity {
            fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
                Ok(samples.to_vec())
            }
        }
        assert!(
            engine
                .start("unused", Box::new(Identity), 960, false, None, None)
                .is_err()
        );
        assert!(!engine.running());
    }

    #[test]
    fn late_worker_failure_cannot_modify_replacement_session() {
        let mut engine = AudioEngine::new();
        let old = WorkerState {
            stop: Arc::new(AtomicBool::new(true)),
            stats: engine.stats.clone(),
            running: engine.running.clone(),
        };
        engine.stats = Arc::new(Mutex::new(AudioStats::default()));
        engine.running = Arc::new(AtomicBool::new(true));
        old.fail("late request".into());
        assert!(engine.running());
        assert!(engine.stats().error.is_none());
    }

    #[test]
    #[ignore = "Records live audio only with explicit VOICE_WARP_TEST_INPUT and VOICE_WARP_TEST_OUTPUT selections"]
    fn native_hardware_selected_devices() {
        let input =
            std::env::var("VOICE_WARP_TEST_INPUT").expect("Explicit input device token required");
        let output = std::env::var("VOICE_WARP_TEST_OUTPUT")
            .expect("Explicit virtual cable output token required");
        struct Attenuated;
        impl Processor for Attenuated {
            fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
                Ok(samples.iter().map(|sample| sample * 0.5).collect())
            }
        }
        let mut engine = AudioEngine::new();
        engine
            .start(
                &input,
                Box::new(Attenuated),
                960,
                false,
                None,
                Some(&output),
            )
            .unwrap();
        thread::sleep(Duration::from_secs(2));
        assert!(engine.running(), "{:?}", engine.stats());
        assert!(engine.stats().frames > 0);
        engine.stop();
        assert!(!engine.running());
    }
}
