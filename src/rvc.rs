//! App-owned RVC worker. Private pipes carry audio; no web server is started.
use crate::audio::Processor;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
#[cfg(test)]
use std::{process::Command, sync::atomic::AtomicU64};

mod launch;
mod process;
pub mod runtime;

pub use launch::IMAGE;
const RESPONSE_LIMIT: u64 = 300_000;
#[cfg(test)]
static SESSION: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct RvcConfig {
    pub model: PathBuf,
    pub index: PathBuf,
    pub pitch: i32,
    pub index_ratio: f32,
    pub block_size: usize,
    pub backend: String,
    pub device: String,
}

pub fn runtime_dir() -> PathBuf {
    crate::config::config_dir().join("rvc")
}

pub fn supported() -> bool {
    (cfg!(any(target_os = "linux", target_os = "windows")) && cfg!(target_arch = "x86_64"))
        || (cfg!(target_os = "macos") && cfg!(any(target_arch = "x86_64", target_arch = "aarch64")))
}

pub fn install() -> Result<(), String> {
    install_backend("cpu", Arc::new(AtomicBool::new(false)))
}

pub fn install_backend(backend: &str, stop: Arc<AtomicBool>) -> Result<(), String> {
    if !supported() {
        return Err("No bundled AI runtime is available for this OS/architecture.".into());
    }
    if stop.load(Ordering::Relaxed) {
        return Err("AI installation canceled.".into());
    }
    launch::install(backend, &stop)
}

pub fn diagnostics(backend: &str, device: &str, stop: Arc<AtomicBool>) -> Result<Value, String> {
    let spec = launch::worker(backend, device, None, None, &stop)?;
    let mut worker = RvcProcessor::spawn_spec(spec, 0, stop)?;
    let report = worker.reply(Duration::from_secs(90))?;
    if report["protocol"] != 2 || !report["devices"].is_array() {
        return Err("Incompatible AI device probe. Repair the runtime.".into());
    }
    Ok(report)
}

pub struct RvcProcessor {
    child: process::ManagedChild,
    container: Option<String>,
    requests: mpsc::SyncSender<Value>,
    replies: mpsc::Receiver<Result<Value, String>>,
    pub info: Value,
    block_size: usize,
    sequence: u64,
    failure: Option<String>,
    log_path: PathBuf,
    stop: Arc<AtomicBool>,
}

fn model_file(path: &Path, extensions: &[&str]) -> Result<PathBuf, String> {
    if !path.is_file()
        || !path
            .extension()
            .and_then(|v| v.to_str())
            .is_some_and(|ext| extensions.contains(&ext))
    {
        return Err(format!(
            "Choose an existing {} file: {}",
            extensions.join(" or "),
            path.display()
        ));
    }
    fs::canonicalize(path).map_err(|e| e.to_string())
}

fn read_reply(reader: &mut impl BufRead) -> Result<Value, String> {
    let mut bytes = Vec::new();
    reader
        .take(RESPONSE_LIMIT + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| format!("Cannot read AI worker: {e}"))?;
    if bytes.len() as u64 > RESPONSE_LIMIT || bytes.last() != Some(&b'\n') {
        return Err("AI worker closed or returned an oversized response.".into());
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| "AI worker returned invalid JSON.".to_string())?;
    if let Some(error) = value.get("error").and_then(Value::as_str) {
        return Err(format!("AI engine: {error}"));
    }
    Ok(value)
}

impl RvcProcessor {
    pub fn new(config: RvcConfig) -> Result<Self, String> {
        Self::new_cancellable(config, Arc::new(AtomicBool::new(false)))
    }

    pub fn new_cancellable(config: RvcConfig, stop: Arc<AtomicBool>) -> Result<Self, String> {
        if !supported() {
            return Err("No bundled AI runtime is available for this OS/architecture.".into());
        }
        if stop.load(Ordering::Relaxed) {
            return Err("AI session stopped.".into());
        }
        if ![4096, 6144, 8192, 12288].contains(&config.block_size)
            || !(-24..=24).contains(&config.pitch)
            || !config.index_ratio.is_finite()
            || !(0.0..=1.0).contains(&config.index_ratio)
        {
            return Err("Invalid RVC audio, pitch, or index settings.".into());
        }
        let model = model_file(&config.model, &["pth", "onnx"])?;
        let index = if config.index.as_os_str().is_empty() {
            None
        } else {
            Some(model_file(&config.index, &["index"])?)
        };
        let mut spec = launch::worker(
            &config.backend,
            &config.device,
            Some(&model),
            index.as_deref(),
            &stop,
        )?;
        spec.init["block_size"] = json!(config.block_size);
        spec.init["pitch"] = json!(config.pitch);
        spec.init["index_ratio"] = json!(config.index_ratio);
        let mut worker = Self::spawn_spec(spec, config.block_size, stop)?;
        let ready = worker.reply(Duration::from_secs(180))?;
        validate_ready(&ready, &config.backend)?;
        worker.info = ready;
        worker.info["model_name"] = json!(model.file_stem().unwrap_or_default().to_string_lossy());
        worker.info["block_size"] = json!(config.block_size);
        Ok(worker)
    }

    fn spawn_spec(
        spec: launch::Launch,
        block_size: usize,
        stop: Arc<AtomicBool>,
    ) -> Result<Self, String> {
        let dir = runtime_dir();
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let log_path = dir.join("engine.log");
        let log = File::create(&log_path).map_err(|e| e.to_string())?;
        Self::spawn(spec, log, log_path, block_size, stop)
    }

    fn spawn(
        mut spec: launch::Launch,
        log: File,
        log_path: PathBuf,
        block_size: usize,
        stop: Arc<AtomicBool>,
    ) -> Result<Self, String> {
        if stop.load(Ordering::Relaxed) {
            return Err("AI session stopped.".into());
        }
        spec.command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log);
        let mut child = process::ManagedChild::spawn(&mut spec.command)?;
        child.retain(spec.lease.take());
        let mut input = child
            .child_mut()
            .stdin
            .take()
            .ok_or("Missing AI input pipe")?;
        let mut output = BufReader::new(
            child
                .child_mut()
                .stdout
                .take()
                .ok_or("Missing AI output pipe")?,
        );
        let (tx, rx) = mpsc::sync_channel::<Value>(1);
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            for request in std::iter::once(spec.init).chain(rx) {
                let response = (|| {
                    serde_json::to_writer(&mut input, &request).map_err(|e| e.to_string())?;
                    input
                        .write_all(b"\n")
                        .and_then(|_| input.flush())
                        .map_err(|e| e.to_string())?;
                    read_reply(&mut output)
                })();
                let failed = response.is_err();
                if reply_tx.send(response).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            container: spec.container,
            requests: tx,
            replies: reply_rx,
            info: Value::Null,
            block_size,
            sequence: 0,
            failure: None,
            log_path,
            stop,
        })
    }

    fn reply(&mut self, timeout: Duration) -> Result<Value, String> {
        let deadline = Instant::now() + timeout;
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return Err("AI session stopped.".into());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(format!(
                    "AI engine timed out. See {}",
                    self.log_path.display()
                ));
            }
            match self
                .replies
                .recv_timeout(remaining.min(Duration::from_millis(50)))
            {
                Ok(value) => {
                    return value.map_err(|e| format!("{e} See {}", self.log_path.display()));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => return Err("AI worker disconnected.".into()),
            }
        }
    }

    fn convert(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        if samples.len() != self.block_size || samples.iter().any(|v| !v.is_finite()) {
            return Err("Invalid RVC audio block.".into());
        }
        self.sequence += 1;
        let pcm: Vec<u8> = samples
            .iter()
            .flat_map(|s| {
                ((s.clamp(-1.0, 1.0) * 32768.0)
                    .round()
                    .clamp(-32768.0, 32767.0) as i16)
                    .to_le_bytes()
            })
            .collect();
        self.requests
            .send(json!({"sequence":self.sequence,"audio":STANDARD.encode(pcm)}))
            .map_err(|_| "AI worker disconnected.".to_string())?;
        let response = self.reply(Duration::from_secs(3))?;
        if response["sequence"].as_u64() != Some(self.sequence) {
            return Err("AI worker returned the wrong audio sequence.".into());
        }
        let audio = STANDARD
            .decode(
                response["audio"]
                    .as_str()
                    .ok_or("AI worker returned no audio.")?,
            )
            .map_err(|_| "AI worker returned invalid audio.".to_string())?;
        if audio.len() != self.block_size * 2 {
            return Err("AI worker returned the wrong audio length.".into());
        }
        Ok(audio
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| f32::from(i16::from_le_bytes(*p)) / 32768.0)
            .collect())
    }
}

impl Processor for RvcProcessor {
    fn set_stop_signal(&mut self, stop: Arc<AtomicBool>) {
        self.stop = stop;
    }
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result = self.convert(samples);
        if let Err(error) = &result {
            self.failure = Some(error.clone());
            self.terminate();
        }
        result
    }
}

impl RvcProcessor {
    fn terminate(&mut self) {
        if self.failure.is_none() && self.info["ready"] == true {
            // No conversion is in flight when a healthy processor is dropped.
            // Give the private worker a short opportunity to release resources;
            // the owned process/container cleanup remains authoritative.
            if self.requests.try_send(json!({"action":"shutdown"})).is_ok() {
                let _ = self.replies.recv_timeout(Duration::from_millis(100));
            }
            self.info["ready"] = json!(false);
        }
        if let Some(name) = self.container.take() {
            launch::remove_container(&name);
        }
        self.child.terminate();
    }
}

impl Drop for RvcProcessor {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn validate_ready(ready: &Value, requested: &str) -> Result<(), String> {
    if ready["protocol"] != 2
        || ready["ready"] != true
        || ready["sample_rate"] != 48000
        || ready["device"]
            .as_str()
            .is_none_or(|value| value.trim().is_empty())
        || ready["device_id"]
            .as_str()
            .is_none_or(|value| value.trim().is_empty())
        || !ready["stages"].is_object()
        || [
            "hubert",
            "pitch",
            "synthesizer",
            "index",
            "resample",
            "overlap",
        ]
        .iter()
        .any(|stage| {
            ready["stages"][stage]
                .as_str()
                .is_none_or(|value| value.trim().is_empty())
        })
        || ready["runtime_version"].as_str().is_none_or(str::is_empty)
    {
        return Err("Incompatible AI engine handshake. Repair the bundled runtime.".into());
    }
    let actual = ready["backend"]
        .as_str()
        .ok_or("Worker did not report its backend")?;
    if !["cpu", "cuda", "rocm", "mps", "directml"].contains(&actual)
        || (requested != "auto" && actual != requested)
    {
        return Err("AI worker selected a different backend than requested.".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture_command(mode: &str, response: Value) -> Command {
        static EXECUTABLE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
        let executable = EXECUTABLE.get_or_init(|| {
            let dir = std::env::temp_dir()
                .join(format!("voice-warp-worker-fixture-{}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("worker{}", std::env::consts::EXE_SUFFIX));
            let status = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
                .arg("--edition=2021")
                .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rvc_worker.rs"))
                .arg("-o")
                .arg(&path)
                .status()
                .unwrap();
            assert!(status.success());
            path
        });
        let mut command = Command::new(executable);
        command.env("VOICE_WARP_FIXTURE_MODE",mode)
            .env("VOICE_WARP_FIXTURE_READY",json!({"protocol":2,"ready":true,"sample_rate":48000,"device":"CPU","device_id":"cpu","backend":"cpu","stages":{"hubert":"CPU","pitch":"CPU","synthesizer":"CPU","index":"not used","resample":"CPU","overlap":"CPU"},"runtime_version":"test"}).to_string())
            .env("VOICE_WARP_FIXTURE_REPLY",response.to_string());
        command
    }

    fn fixture(mode: &str, response: Value, stop: Arc<AtomicBool>) -> RvcProcessor {
        let log_path = std::env::temp_dir().join(format!(
            "voice-warp-fixture-{}-{}.log",
            std::process::id(),
            SESSION.fetch_add(1, Ordering::Relaxed)
        ));
        let log = File::create(&log_path).unwrap();
        RvcProcessor::spawn(
            launch::Launch {
                command: fixture_command(mode, response),
                container: None,
                lease: None,
                init: json!({}),
            },
            log,
            log_path,
            4096,
            stop,
        )
        .unwrap()
    }

    fn fake_worker(mode: &str, response: Value) -> RvcProcessor {
        let mut worker = fixture(mode, response, Arc::new(AtomicBool::new(false)));
        validate_ready(&worker.reply(Duration::from_secs(5)).unwrap(), "cpu").unwrap();
        worker
    }

    #[test]
    fn rejects_bad_worker_audio_and_latches_failure() {
        for response in [
            json!({"sequence":99,"audio":""}),
            json!({"sequence":1,"audio":"!invalid"}),
            json!({"sequence":1,"audio":"AA=="}),
            json!({"error":"inference failed"}),
        ] {
            let mut worker = fake_worker("reply", response);
            let error = worker.process(&vec![0.1; 4096]).unwrap_err();
            assert_eq!(worker.process(&vec![0.1; 4096]).unwrap_err(), error);
            assert!(worker.child.try_wait().unwrap().is_some());
            let _ = fs::remove_file(&worker.log_path);
        }
    }

    #[test]
    fn cancellation_interrupts_a_stalled_worker() {
        let mut worker = fake_worker("stall_audio", Value::Null);
        let stop = Arc::new(AtomicBool::new(false));
        worker.set_stop_signal(stop.clone());
        let signal = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            stop.store(true, Ordering::Relaxed);
        });
        let started = Instant::now();
        assert!(
            worker
                .process(&vec![0.1; 4096])
                .unwrap_err()
                .contains("stopped")
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(worker.child.try_wait().unwrap().is_some());
        signal.join().unwrap();
        let _ = fs::remove_file(&worker.log_path);
    }

    #[test]
    fn cancellation_during_startup_terminates_worker() {
        // Compile the helper before measuring cancellation responsiveness.
        let _ = fixture_command("stall_start", Value::Null);
        let stop = Arc::new(AtomicBool::new(false));
        let mut worker = fixture("stall_start", Value::Null, stop.clone());
        let signal = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            stop.store(true, Ordering::Relaxed);
        });
        let start = Instant::now();
        assert!(
            worker
                .reply(Duration::from_secs(90))
                .unwrap_err()
                .contains("stopped")
        );
        worker.terminate();
        assert!(worker.child.try_wait().unwrap().is_some());
        assert!(start.elapsed() < Duration::from_secs(2));
        signal.join().unwrap();
        let _ = fs::remove_file(&worker.log_path);
    }

    #[test]
    fn handshake_rejects_silent_device_fallback_and_old_protocol() {
        let ready = json!({"protocol":2,"ready":true,"sample_rate":48000,"device":"CPU","device_id":"cpu","backend":"cpu","stages":{"hubert":"CPU","pitch":"CPU","synthesizer":"CPU","index":"not used","resample":"CPU","overlap":"CPU"},"runtime_version":"test"});
        assert!(validate_ready(&ready, "cpu").is_ok());
        assert!(validate_ready(&ready, "auto").is_ok());
        assert!(validate_ready(&ready, "cuda").is_err());
        let mut old = ready.clone();
        old["protocol"] = json!(1);
        assert!(validate_ready(&old, "cpu").is_err());
        let mut missing = ready;
        missing.as_object_mut().unwrap().remove("device");
        assert!(validate_ready(&missing, "cpu").is_err());
    }

    #[test]
    fn rejects_truncated_oversized_and_error_replies() {
        for bytes in [
            b"{}".to_vec(),
            vec![b'x'; RESPONSE_LIMIT as usize + 2],
            b"{\"error\":\"bad model\"}\n".to_vec(),
            b"not json\n".to_vec(),
        ] {
            assert!(read_reply(&mut bytes.as_slice()).is_err());
        }
    }

    #[test]
    #[ignore = "requires installed runtime and VOICE_WARP_TEST_MODEL"]
    fn bundled_engine_integration() {
        let model = std::env::var_os("VOICE_WARP_TEST_MODEL").expect("Set VOICE_WARP_TEST_MODEL");
        let mut engine = RvcProcessor::new(RvcConfig {
            model: model.into(),
            index: PathBuf::new(),
            pitch: 0,
            index_ratio: 0.0,
            block_size: 8192,
            backend: std::env::var("VOICE_WARP_TEST_BACKEND").unwrap_or_else(|_| "cpu".into()),
            device: "0".into(),
        })
        .unwrap();
        let name = engine.container.clone();
        for block in 0..4 {
            let input: Vec<f32> = (0..8192)
                .map(|i| {
                    ((i + block * 8192) as f32 * 180.0 * std::f32::consts::TAU / 48000.0).sin()
                        * 0.15
                })
                .collect();
            let output = engine.process(&input).unwrap();
            assert_eq!(output.len(), input.len());
            if block > 0 {
                assert!(output.iter().any(|v| v.abs() > 0.0001));
            }
        }
        drop(engine);
        if let Some(name) = name {
            assert!(
                !launch::podman()
                    .args(["container", "exists", &name])
                    .status()
                    .unwrap()
                    .success()
            );
        }
    }
}
