//! Local RVC inference using the w-okada v1 REST protocol.
//!
//! Protocol reference: w-okada/voice-changer at
//! d8ef15799470193f7c8176ef471245753a656626. No model weights are bundled.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::{Method, Url, blocking::Client, redirect::Policy};
use serde_json::{Value, json};
use std::{io::Read, net::IpAddr, time::Duration};

const SAMPLE_RATE: u64 = 48_000;
const INFO_LIMIT: usize = 2 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct AIConfig {
    pub endpoint: String,
    pub block_size: usize,
    pub timeout_secs: u64,
}

impl Default for AIConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:18888".into(),
            block_size: 6144,
            timeout_secs: 3,
        }
    }
}

pub struct AIProcessor {
    config: AIConfig,
    endpoint: Url,
    client: Client,
    sequence: u64,
    model_signature: Option<String>,
    failure: Option<String>,
}

#[derive(Debug)]
struct ModelInfo {
    slot: u64,
    name: String,
    signature: String,
    device: String,
}

impl AIProcessor {
    /// Construct without sending audio or changing the server's settings.
    pub fn new(config: AIConfig) -> Result<Self, String> {
        if !(2048..=48_000).contains(&config.block_size) {
            return Err("AI block size must be between 2048 and 48000 samples.".into());
        }
        if !(1..=15).contains(&config.timeout_secs) {
            return Err("AI request timeout must be between 1 and 15 seconds.".into());
        }
        let endpoint = local_endpoint(&config.endpoint)?;
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .connect_timeout(Duration::from_secs(config.timeout_secs))
            .redirect(Policy::none())
            .no_proxy()
            .tcp_nodelay(true)
            .user_agent("Voice-Warp/local-rvc")
            .build()
            .map_err(|e| format!("Cannot initialize the local AI client: {e}"))?;
        Ok(Self {
            config,
            endpoint,
            client,
            sequence: 0,
            model_signature: None,
            failure: None,
        })
    }

    /// Inspect the selected upstream model; this is not an inference benchmark.
    pub fn probe(&self) -> Result<Value, String> {
        let info = self.server_info()?;
        Ok(json!({
            "backend": "w-okada v1 / RVC",
            "model_name": info.name,
            "model_slot": info.slot,
            "device": info.device,
            "sample_rate": SAMPLE_RATE,
            "block_size": self.config.block_size,
            "block_ms": self.config.block_size as f64 / 48.0,
            "endpoint": self.endpoint.as_str(),
        }))
    }

    /// Convert exactly one mono 48 kHz block. Errors never return raw input.
    /// Once an error occurs, create a new processor to restart explicitly.
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result = self.process_inner(samples);
        if let Err(error) = &result {
            self.failure = Some(error.clone());
        }
        result
    }

    fn process_inner(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        if samples.len() != self.config.block_size || samples.iter().any(|s| !s.is_finite()) {
            return Err(format!(
                "AI expects {} finite mono samples per block.",
                self.config.block_size
            ));
        }
        let before = self.server_info()?;
        if let Some(signature) = &self.model_signature
            && signature != &before.signature
        {
            return Err("The AI model changed. Stop and restart Voice Warp's input.".into());
        }
        let pcm: Vec<u8> = samples
            .iter()
            .flat_map(|&sample| {
                ((sample.clamp(-1.0, 1.0) * 32768.0)
                    .round()
                    .clamp(-32768.0, 32767.0) as i16)
                    .to_le_bytes()
            })
            .collect();
        self.sequence += 1;
        let body = json!({"timestamp": self.sequence, "buffer": STANDARD.encode(&pcm)});
        let converted =
            self.request_json(Method::POST, "test", Some(&body), pcm.len() * 2 + 4096)?;
        if converted.get("timestamp").and_then(Value::as_u64) != Some(self.sequence) {
            return Err("The AI server returned an invalid request timestamp.".into());
        }
        let encoded = converted
            .get("changedVoiceBase64")
            .and_then(Value::as_str)
            .ok_or("The AI server returned no converted audio. Check its model and console.")?;
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|_| "The AI server returned invalid base64 audio.".to_string())?;
        if bytes.len() != pcm.len() {
            return Err(format!(
                "AI returned {} audio bytes; expected {}. Check the RVC model and 48000 Hz input/output settings.",
                bytes.len(),
                pcm.len()
            ));
        }
        // Recheck before releasing the result, catching a switch to pass-through
        // or a different model while the request was in flight.
        let after = self.server_info()?;
        if before.signature != after.signature {
            return Err(
                "The AI model changed during conversion. Stop and restart the input.".into(),
            );
        }
        self.model_signature = Some(before.signature);
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| f32::from(i16::from_le_bytes([p[0], p[1]])) / 32768.0)
            .collect())
    }

    fn server_info(&self) -> Result<ModelInfo, String> {
        validate_info(&self.request_json(Method::GET, "info", None, INFO_LIMIT)?)
    }

    fn request_json(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        limit: usize,
    ) -> Result<Value, String> {
        let url = self.endpoint.join(path).map_err(|e| e.to_string())?;
        let mut request = self.client.request(method, url);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().map_err(|e| {
            if e.is_timeout() {
                "The local AI server timed out. Reduce its model workload or increase the AI block size.".into()
            } else {
                format!("Cannot reach the local AI server at {}: {e}", self.endpoint)
            }
        })?;
        if !response.status().is_success() {
            return Err(format!(
                "The local AI server returned HTTP {} for /{path}. Use the supported w-okada v1 server; redirects are disabled.",
                response.status().as_u16()
            ));
        }
        if response.content_length().is_some_and(|n| n > limit as u64) {
            return Err("The AI server response exceeded the size limit.".into());
        }
        let mut bytes = Vec::new();
        response
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("Cannot read the AI server response: {e}"))?;
        if bytes.len() > limit {
            return Err("The AI server response exceeded the size limit.".into());
        }
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| {
            "The AI server returned invalid JSON. Check the endpoint and its console.".to_string()
        })?;
        if !value.is_object() {
            return Err("The AI server returned an error instead of a protocol response. Check its console.".into());
        }
        Ok(value)
    }
}

impl crate::audio::Processor for AIProcessor {
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        AIProcessor::process(self, samples)
    }
}

fn local_endpoint(endpoint: &str) -> Result<Url, String> {
    let mut url = Url::parse(endpoint.trim())
        .map_err(|_| "Enter a local AI URL such as http://127.0.0.1:18888.".to_string())?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(
            "Use an HTTP(S) loopback URL without credentials, a path, query, or fragment.".into(),
        );
    }
    let host = url
        .host_str()
        .ok_or("The AI URL needs a loopback address.")?;
    if host == "localhost" {
        // Pin localhost to the literal address; never use a proxy or external DNS.
        url.set_host(Some("127.0.0.1")).map_err(|e| e.to_string())?;
    } else if !host
        .trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
    {
        return Err("AI audio stays on this computer. Use localhost, 127.0.0.1, or [::1].".into());
    }
    url.set_path("/");
    Ok(url)
}

fn validate_info(info: &Value) -> Result<ModelInfo, String> {
    if info.get("status").and_then(Value::as_str) != Some("OK") {
        return Err("The server is not a ready w-okada v1 instance (missing OK status).".into());
    }
    if info.get("passThrough").and_then(Value::as_bool) != Some(false) {
        return Err("Turn OFF Pass Through in the AI server before starting conversion.".into());
    }
    if info.get("inputSampleRate").and_then(Value::as_u64) != Some(SAMPLE_RATE)
        || info.get("outputSampleRate").and_then(Value::as_u64) != Some(SAMPLE_RATE)
    {
        return Err("Set the AI server inputSampleRate and outputSampleRate to 48000 Hz.".into());
    }
    if info.get("enableServerAudio").and_then(Value::as_u64) != Some(0)
        || info.get("serverAudioStated").and_then(Value::as_u64) != Some(0)
    {
        return Err("Stop Server Device audio in the AI server and use client mode; Voice Warp supplies the microphone.".into());
    }
    if info.get("recordIO").and_then(Value::as_u64) != Some(0) {
        return Err("Turn OFF input/output recording in the AI server (recordIO = 0).".into());
    }
    let slot_id = info
        .get("modelSlotIndex")
        .and_then(Value::as_u64)
        .ok_or("Select and load an RVC model in the AI server first.")?;
    let slots = info.get("modelSlots").and_then(Value::as_array).ok_or(
        "The AI server did not report its model slots. Use the supported w-okada v1 server.",
    )?;
    let slot = slots
        .iter()
        .find(|s| s.get("slotIndex").and_then(Value::as_u64) == Some(slot_id))
        .ok_or("The selected AI model slot is missing. Load and select an RVC model.")?;
    if slot.get("voiceChangerType").and_then(Value::as_str) != Some("RVC") {
        return Err(
            "This adapter supports RVC models. Select an RVC model in the AI server.".into(),
        );
    }
    let model_file = slot
        .get("modelFile")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or("The selected RVC model has no model file.")?;
    if info
        .get("pipelineInfo")
        .and_then(|v| v.get("inferencer"))
        .and_then(Value::as_object)
        .is_none_or(|v| v.is_empty())
    {
        return Err("The RVC inference pipeline is not initialized. Select CPU or a supported GPU in the AI server, then inspect its console.".into());
    }
    let name = slot
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("RVC model {slot_id}"));
    let gpu = info.get("gpu").and_then(Value::as_i64).unwrap_or(-1);
    let device = if gpu < 0 {
        "CPU".to_string()
    } else {
        info.get("gpus")
            .and_then(Value::as_array)
            .and_then(|gpus| {
                gpus.iter()
                    .find(|g| g.get("id").and_then(Value::as_i64) == Some(gpu))
            })
            .and_then(|gpu| gpu.get("name"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| "Server-selected device".into())
    };
    Ok(ModelInfo {
        slot: slot_id,
        name,
        signature: json!([slot_id, model_file, slot.get("indexFile"), gpu]).to_string(),
        device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
        time::Instant,
    };

    fn ready_info() -> Value {
        json!({
            "status":"OK", "passThrough":false,
            "inputSampleRate":48000, "outputSampleRate":48000,
            "enableServerAudio":0, "serverAudioStated":0, "recordIO":0,
            "modelSlotIndex":0, "gpu":-1, "gpus":[],
            "modelSlots":[{"slotIndex":0,"voiceChangerType":"RVC","name":"Test voice","modelFile":"voice.pth","indexFile":""}],
            "pipelineInfo":{"inferencer":{"type":"pyTorchRVC"}}
        })
    }

    // Exercise real HTTP serialization without requiring any model downloads.
    fn server<F>(requests: usize, mut handle: F) -> (String, thread::JoinHandle<()>)
    where
        F: FnMut(usize, &str, Value) -> String + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let thread = thread::spawn(move || {
            for index in 0..requests {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "missing HTTP request {index}");
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(e) => panic!("HTTP accept: {e}"),
                    }
                };
                // Windows accepted sockets can inherit the listener's mode.
                // The mock reads full HTTP requests, so make each connection blocking.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse::<usize>().unwrap();
                    }
                    assert!(!line.is_empty(), "unexpected end of HTTP headers");
                }
                assert!(length <= 256 * 1024);
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let body = if body.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_slice(&body).unwrap()
                };
                let response = handle(index, first.trim_end(), body);
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (endpoint, thread)
    }

    fn response(body: Value) -> String {
        let body = body.to_string();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn processor(endpoint: String) -> AIProcessor {
        AIProcessor::new(AIConfig {
            endpoint,
            block_size: 2048,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn only_loopback_endpoints_are_allowed() {
        for bad in [
            "http://example.com:18888",
            "http://192.168.1.2:18888",
            "http://127.0.0.1@evil.test",
            "http://u:p@localhost",
            "file:///tmp/audio",
            "http://localhost/test",
            "http://localhost?url=x",
            "http://localhost#x",
        ] {
            assert!(local_endpoint(bad).is_err(), "accepted {bad}");
        }
        assert_eq!(
            local_endpoint("http://localhost:18888").unwrap().as_str(),
            "http://127.0.0.1:18888/"
        );
        assert!(local_endpoint("http://[::1]:18888").is_ok());
    }

    #[test]
    fn rejects_unloaded_or_unsafe_server_configuration() {
        for (key, value) in [
            ("passThrough", json!(true)),
            ("inputSampleRate", json!(44100)),
            ("outputSampleRate", json!(24000)),
            ("enableServerAudio", json!(1)),
            ("serverAudioStated", json!(1)),
            ("recordIO", json!(1)),
            ("modelSlotIndex", json!(-1)),
            ("pipelineInfo", json!("None")),
        ] {
            let mut info = ready_info();
            info[key] = value;
            assert!(validate_info(&info).is_err(), "accepted {key}");
        }
        let mut info = ready_info();
        info["modelSlots"][0]["voiceChangerType"] = json!("LLVC");
        assert!(validate_info(&info).is_err());
        assert_eq!(validate_info(&ready_info()).unwrap().name, "Test voice");
    }

    #[test]
    fn probe_reports_real_server_settings_without_audio() {
        let (url, worker) = server(1, |_, request, body| {
            assert!(request.starts_with("GET /info "));
            assert!(body.is_null());
            response(ready_info())
        });
        let info = processor(url).probe().unwrap();
        assert_eq!(info["model_name"], "Test voice");
        assert_eq!(info["device"], "CPU");
        worker.join().unwrap();
    }

    #[test]
    fn converts_int16_little_endian_and_clips_without_wraparound() {
        let (url, worker) = server(3, |index, request, body| {
            if index != 1 {
                return response(ready_info());
            }
            assert!(request.starts_with("POST /test "));
            let input = STANDARD.decode(body["buffer"].as_str().unwrap()).unwrap();
            let pcm: Vec<i16> = input
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| i16::from_le_bytes([p[0], p[1]]))
                .collect();
            assert_eq!(&pcm[..6], &[-32768, -32768, -16384, 0, 16384, 32767]);
            let output: Vec<u8> = vec![8192_i16; 2048]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect();
            response(
                json!({"timestamp":body["timestamp"], "changedVoiceBase64":STANDARD.encode(output)}),
            )
        });
        let mut input = vec![0.0; 2048];
        input[..6].copy_from_slice(&[-2.0, -1.0, -0.5, 0.0, 0.5, 2.0]);
        let output = processor(url).process(&input).unwrap();
        assert_eq!(output, vec![0.25; 2048]);
        worker.join().unwrap();
    }

    #[test]
    fn rejects_server_passthrough_enabled_during_conversion() {
        let (url, worker) = server(3, |index, _, body| match index {
            0 => response(ready_info()),
            1 => {
                response(json!({"timestamp":body["timestamp"],"changedVoiceBase64":body["buffer"]}))
            }
            _ => {
                let mut info = ready_info();
                info["passThrough"] = json!(true);
                response(info)
            }
        });
        let mut client = processor(url);
        let error = client.process(&vec![0.5; 2048]).unwrap_err();
        assert!(error.contains("Pass Through"));
        assert_eq!(client.process(&vec![0.5; 2048]).unwrap_err(), error);
        worker.join().unwrap();
    }

    #[test]
    fn rejects_malformed_and_wrong_length_audio() {
        for audio in ["not base64!".to_string(), STANDARD.encode([0, 0])] {
            let (url, worker) = server(2, move |index, _, body| {
                if index == 0 {
                    response(ready_info())
                } else {
                    response(json!({"timestamp":body["timestamp"],"changedVoiceBase64":audio}))
                }
            });
            assert!(processor(url).process(&vec![0.5; 2048]).is_err());
            worker.join().unwrap();
        }
    }

    #[test]
    fn refuses_http_redirects() {
        let (url, worker) = server(1, |_, _, _| {
            "HTTP/1.1 302 Found\r\nLocation: https://example.com/audio\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()
        });
        assert!(
            processor(url)
                .probe()
                .unwrap_err()
                .contains("redirects are disabled")
        );
        worker.join().unwrap();
    }

    #[test]
    fn bounds_server_response_sizes() {
        let (url, worker) = server(1, |_, _, _| {
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                INFO_LIMIT + 1
            )
        });
        assert!(processor(url).probe().unwrap_err().contains("size limit"));
        worker.join().unwrap();
    }

    #[test]
    #[ignore = "requires a configured local w-okada v1 RVC server"]
    fn local_server_integration() {
        let mut processor = AIProcessor::new(AIConfig::default()).unwrap();
        eprintln!("{}", processor.probe().unwrap());
        for _ in 0..3 {
            let converted = processor.process(&vec![0.0; 6144]).unwrap();
            assert_eq!(converted.len(), 6144);
            assert!(converted.iter().all(|s| s.is_finite()));
        }
    }
}
