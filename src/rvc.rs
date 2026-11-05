//! App-owned RVC worker. Private pipes carry audio; no web server is started.
use crate::audio::Processor;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const IMAGE: &str = "localhost/voice-warp-rvc:1";
const RESPONSE_LIMIT: u64 = 300_000;
static SESSION: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct RvcConfig {
    pub model: PathBuf,
    pub index: PathBuf,
    pub pitch: i32,
    pub index_ratio: f32,
    pub block_size: usize,
}

pub fn runtime_dir() -> PathBuf {
    crate::config::config_dir().join("rvc")
}

pub fn supported() -> bool {
    cfg!(target_os = "linux") && cfg!(target_arch = "x86_64")
}

fn podman() -> Command {
    // Distrobox shares the user's home with the host. Build and run the worker
    // using host Podman rather than nesting containers inside the desktop app.
    if std::env::var_os("CONTAINER_ID").is_some() && Path::new("/run/.containerenv").exists() {
        let mut command = Command::new("distrobox-host-exec");
        command.arg("podman");
        command
    } else {
        Command::new("podman")
    }
}

pub fn install() -> Result<(), String> {
    if !supported() {
        return Err("The bundled RVC runtime currently supports Linux x86-64. External AI servers remain available on other platforms.".into());
    }
    let dir = runtime_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let lock = File::create(dir.join("install.lock")).map_err(|e| e.to_string())?;
    fs2::FileExt::try_lock_exclusive(&lock)
        .map_err(|_| "AI engine installation is already running.".to_string())?;
    for (name, contents) in [
        ("Containerfile", include_str!("../engine/Containerfile")),
        (
            "requirements.txt",
            include_str!("../engine/requirements.txt"),
        ),
        ("worker.py", include_str!("../engine/worker.py")),
        ("NOTICE.md", include_str!("../engine/NOTICE.md")),
        (
            "UPSTREAM-LICENSE",
            include_str!("../engine/UPSTREAM-LICENSE"),
        ),
    ] {
        fs::write(dir.join(name), contents).map_err(|e| e.to_string())?;
    }
    let log_path = dir.join("install.log");
    let log = File::create(&log_path).map_err(|e| e.to_string())?;
    let status = podman()
        .args(["build", "-t", IMAGE, "-f"])
        .arg(dir.join("Containerfile"))
        .arg(&dir)
        .stdin(Stdio::null())
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log)
        .status()
        .map_err(|e| format!("Cannot run Podman: {e}. Install Podman, then retry AI setup."))?;
    if !status.success() {
        return Err(format!(
            "AI engine installation failed. See {}",
            log_path.display()
        ));
    }
    Ok(())
}

pub struct RvcProcessor {
    child: Child,
    container: Option<String>,
    requests: mpsc::Sender<Value>,
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
        if !supported() {
            return Err("Bundled AI currently requires Linux x86-64.".into());
        }
        if ![4096, 6144, 8192, 12288].contains(&config.block_size)
            || !(-24..=24).contains(&config.pitch)
            || !config.index_ratio.is_finite()
            || !(0.0..=1.0).contains(&config.index_ratio)
        {
            return Err("Invalid RVC audio, pitch, or index settings.".into());
        }
        let model = model_file(&config.model, &["pth", "onnx"])?;
        let installed = podman()
            .args(["image", "exists", IMAGE])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Cannot run Podman: {e}. Install Podman, then run AI setup."))?;
        if !installed.success() {
            return Err("The bundled AI engine is not installed. Open Local AI and choose Install / repair AI engine.".into());
        }
        let index = if config.index.as_os_str().is_empty() {
            None
        } else {
            Some(model_file(&config.index, &["index"])?)
        };
        // Mount syntax uses commas, so reject ambiguous paths instead of mounting
        // a different location. Spaces and Unicode remain supported.
        if std::iter::once(&model)
            .chain(index.iter())
            .any(|p| p.to_string_lossy().contains(','))
        {
            return Err("Move the model/index to a folder without commas in its path.".into());
        }
        let extension = model.extension().unwrap().to_str().unwrap();
        let model_name = format!("model.{extension}");
        let id = format!(
            "voice-warp-rvc-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            SESSION.fetch_add(1, Ordering::Relaxed)
        );
        let dir = runtime_dir();
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let log_path = dir.join("engine.log");
        let log = File::create(&log_path).map_err(|e| e.to_string())?;
        let mut command = podman();
        command
            .args([
                "run",
                "--rm",
                "-i",
                "--pull=never",
                "--network=none",
                "--cap-drop=all",
                "--security-opt=no-new-privileges",
                "--security-opt=label=disable",
                "--name",
                &id,
                "--mount",
            ])
            .arg(format!(
                "type=bind,src={},dst=/models/0/{model_name},ro=true",
                model.display()
            ));
        if let Some(index) = index.as_ref() {
            command.arg("--mount").arg(format!(
                "type=bind,src={},dst=/models/0/voice.index,ro=true",
                index.display()
            ));
        }
        command.arg(IMAGE);
        let init = json!({"protocol":1,"model_file":model_name,"has_index":index.is_some(),
            "block_size":config.block_size,"pitch":config.pitch,"index_ratio":config.index_ratio});
        let mut processor = Self::spawn(command, log, log_path, Some(id), init, config.block_size)?;
        processor.info = json!({"backend":"Voice Warp RVC", "model_name":model.file_stem().unwrap_or_default().to_string_lossy(),
            "device":"CPU", "sample_rate":48000,"block_size":config.block_size});
        Ok(processor)
    }

    fn spawn(
        mut command: Command,
        log: File,
        log_path: PathBuf,
        container: Option<String>,
        init: Value,
        block_size: usize,
    ) -> Result<Self, String> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log)
            .spawn()
            .map_err(|e| {
                format!("Cannot start the AI engine: {e}. Run Install AI engine first.")
            })?;
        let mut input = child.stdin.take().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let (tx, rx) = mpsc::channel::<Value>();
        let (reply_tx, reply_rx) = mpsc::channel();
        std::thread::spawn(move || {
            for request in std::iter::once(init).chain(rx) {
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
        let mut result = Self {
            child,
            container,
            requests: tx,
            replies: reply_rx,
            info: Value::Null,
            block_size,
            sequence: 0,
            failure: None,
            log_path,
            stop: Arc::new(AtomicBool::new(false)),
        };
        let ready = result.reply(Duration::from_secs(90))?;
        if ready["protocol"] != 1 || ready["ready"] != true || ready["sample_rate"] != 48000 {
            return Err("Incompatible AI engine handshake. Reinstall the bundled engine.".into());
        }
        Ok(result)
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
        if let Some(name) = self.container.take() {
            let _ = podman()
                .args(["rm", "--force", "--time=0", "--ignore", &name])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for RvcProcessor {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    fn fake_worker(body: &str) -> RvcProcessor {
        let log_path = std::env::temp_dir().join(format!(
            "voice-warp-rvc-test-{}-{}.log",
            std::process::id(),
            SESSION.fetch_add(1, Ordering::Relaxed)
        ));
        let log = File::create(&log_path).unwrap();
        let mut command = Command::new("python3");
        command.args(["-u", "-c", &format!("import sys,json,time\njson.loads(sys.stdin.readline())\nprint(json.dumps(dict(protocol=1,ready=True,sample_rate=48000)),flush=True)\n{body}")]);
        RvcProcessor::spawn(command, log, log_path, None, json!({}), 4096).unwrap()
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rejects_bad_worker_audio_and_latches_failure() {
        for response in [
            json!({"sequence":99,"audio":""}),
            json!({"sequence":1,"audio":"!invalid"}),
            json!({"sequence":1,"audio":"AA=="}),
            json!({"error":"inference failed"}),
        ] {
            let mut worker = fake_worker(&format!(
                "sys.stdin.readline()\nprint({:?},flush=True)",
                response.to_string()
            ));
            let error = worker.process(&vec![0.1; 4096]).unwrap_err();
            assert_eq!(worker.process(&vec![0.1; 4096]).unwrap_err(), error);
            assert!(worker.child.try_wait().unwrap().is_some());
            let _ = fs::remove_file(&worker.log_path);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cancellation_interrupts_a_stalled_worker() {
        let mut worker = fake_worker("sys.stdin.readline()\ntime.sleep(60)");
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
        })
        .unwrap();
        let name = engine.container.clone().unwrap();
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
        assert!(
            !podman()
                .args(["container", "exists", &name])
                .status()
                .unwrap()
                .success()
        );
    }
}
