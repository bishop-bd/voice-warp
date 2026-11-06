//! Repeatable file/tone inference measurements. This module never opens audio devices.
use super::{RvcConfig, RvcProcessor};
use crate::audio::Processor;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Cursor},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const SAMPLE_RATE: usize = 48_000;
const MAX_WAV_BYTES: u64 = 256 * 1024 * 1024;
const MAX_INPUT_SAMPLES: usize = SAMPLE_RATE * 20 * 60;

#[derive(Default)]
pub struct Options {
    pub input_wav: Option<PathBuf>,
    pub output_wav: Option<PathBuf>,
    /// Absent means one complete WAV or 64 generated-tone blocks.
    pub blocks: Option<usize>,
    pub realtime: bool,
}

struct Input {
    samples: Vec<f32>,
    sha256: String,
}

fn read_wav(path: &Path) -> Result<Input, String> {
    if fs::metadata(path)
        .map_err(|e| format!("Cannot read input WAV: {e}"))?
        .len()
        > MAX_WAV_BYTES
    {
        return Err("Input WAV must be at most 256 MiB.".into());
    }
    // Hash the exact bytes we decode, including the header and metadata.
    let bytes = fs::read(path).map_err(|e| format!("Cannot read input WAV: {e}"))?;
    if bytes.len() as u64 > MAX_WAV_BYTES {
        return Err("Input WAV must be at most 256 MiB.".into());
    }
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let mut reader =
        hound::WavReader::new(Cursor::new(bytes)).map_err(|e| format!("Invalid input WAV: {e}"))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE as u32 {
        return Err("Input WAV must be mono at 48000 Hz. Resample it before benchmarking.".into());
    }
    if reader.duration() == 0 || reader.duration() as usize > MAX_INPUT_SAMPLES {
        return Err("Input WAV must contain audio and be at most 20 minutes long.".into());
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int if [8, 16, 24, 32].contains(&spec.bits_per_sample) => {
            let scale = (1_u64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| (value as f64 / scale) as f32))
                .collect::<Result<_, _>>()
        }
        hound::SampleFormat::Float if spec.bits_per_sample == 32 => {
            reader.samples::<f32>().collect::<Result<_, _>>()
        }
        _ => {
            return Err(
                "Input WAV must use 8/16/24/32-bit integer PCM or 32-bit float samples.".into(),
            );
        }
    }
    .map_err(|e| format!("Invalid input WAV samples: {e}"))?;
    if samples.is_empty()
        || samples
            .iter()
            .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
    {
        return Err("Input WAV samples must be finite and between -1 and 1.".into());
    }
    Ok(Input { samples, sha256 })
}

/// Remove a partial recording on cancellation or inference failure. create_new
/// rejects existing files, including aliases of the input/checkpoint.
struct Output {
    path: PathBuf,
    writer: Option<hound::WavWriter<BufWriter<File>>>,
    complete: bool,
}

impl Output {
    fn create(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| {
                format!("Cannot create output WAV (existing files are never overwritten): {e}")
            })?;
        let mut output = Self {
            path: path.into(),
            writer: None,
            complete: false,
        };
        output.writer = Some(
            hound::WavWriter::new(
                BufWriter::new(file),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: SAMPLE_RATE as u32,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .map_err(|e| e.to_string())?,
        );
        Ok(output)
    }

    fn write(&mut self, samples: &[f32]) -> Result<(), String> {
        for sample in samples {
            self.writer
                .as_mut()
                .unwrap()
                .write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(), String> {
        self.writer
            .take()
            .unwrap()
            .finalize()
            .map_err(|e| e.to_string())?;
        self.complete = true;
        Ok(())
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if !self.complete {
            drop(self.writer.take());
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn block_input(source: Option<&Input>, block: usize, size: usize, repeat: bool) -> Vec<f32> {
    (0..size)
        .map(|sample| {
            let offset = block * size + sample;
            match source {
                Some(input) if repeat => input.samples[offset % input.samples.len()],
                Some(input) => input.samples.get(offset).copied().unwrap_or(0.0),
                None => {
                    ((offset as f64 * 180.0 * std::f64::consts::TAU / SAMPLE_RATE as f64).sin()
                        * 0.12) as f32
                }
            }
        })
        .collect()
}

fn summarize(times: &mut [f64], block_ms: f64, elapsed: f64, audio_seconds: f64) -> Value {
    times.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        times[((times.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(times.len() - 1)]
    };
    let p95 = percentile(0.95);
    let p99 = percentile(0.99);
    let sustained = elapsed >= 600.0 && audio_seconds >= 600.0;
    let p95_pass = p95 < block_ms * 0.8;
    let p99_pass = p99 < block_ms;
    json!({
        "processing_ms":{"p50":percentile(0.5),"p95":p95,"p99":p99,"max":times.last()},
        "deadline_misses":times.iter().filter(|&&time| time >= block_ms).count(),
        "blocks_over_80_percent_budget":times.iter().filter(|&&time| time >= block_ms * 0.8).count(),
        "inference_seconds":times.iter().sum::<f64>() / 1000.0,
        "timing_gate":{
            "p95_below_80_percent_block":p95_pass,"p99_below_block":p99_pass,
            "sustained_at_least_600_seconds":sustained,
            "passed":p95_pass && p99_pass && sustained,
            "scope":"inference timing only; listening and live audio qualification remain required"
        }
    })
}

pub fn run(config: RvcConfig, options: Options, stop: Arc<AtomicBool>) -> Result<Value, String> {
    if ![4096, 6144, 8192, 12288].contains(&config.block_size) {
        return Err("Unsupported benchmark block size.".into());
    }
    let input = options.input_wav.as_deref().map(read_wav).transpose()?;
    let blocks = options.blocks.unwrap_or_else(|| {
        input
            .as_ref()
            .map(|input| input.samples.len().div_ceil(config.block_size))
            .unwrap_or(64)
    });
    if !(1..=100_000).contains(&blocks) {
        return Err("Benchmark requires between 1 and 100000 blocks.".into());
    }
    // Reserve output before starting expensive model work. Any later failure
    // removes it; an existing input/model/output can never be truncated.
    let mut output = options
        .output_wav
        .as_deref()
        .map(Output::create)
        .transpose()?;
    let loading = Instant::now();
    let mut processor = RvcProcessor::new_cancellable(config.clone(), stop.clone())?;
    let startup_ms = loading.elapsed().as_secs_f64() * 1000.0;
    let samples = blocks * config.block_size;
    let audio_seconds = samples as f64 / SAMPLE_RATE as f64;
    let block_ms = config.block_size as f64 / 48.0;
    let mut times = Vec::with_capacity(blocks);
    let mut clipped_samples = 0_u64;
    let mut nonzero_samples = 0_u64;
    let measured = Instant::now();
    for block in 0..blocks {
        if stop.load(Ordering::Relaxed) {
            return Err("AI benchmark canceled.".into());
        }
        let waveform = block_input(
            input.as_ref(),
            block,
            config.block_size,
            options.blocks.is_some(),
        );
        let started = Instant::now();
        let converted = processor.process(&waveform)?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        if converted.len() != config.block_size || converted.iter().any(|v| !v.is_finite()) {
            return Err("AI benchmark returned an invalid audio block.".into());
        }
        clipped_samples += converted.iter().filter(|&&v| v.abs() >= 1.0).count() as u64;
        nonzero_samples += converted.iter().filter(|&&v| v != 0.0).count() as u64;
        if let Some(output) = &mut output {
            output.write(&converted)?;
        }
        if options.realtime {
            let due = Duration::from_secs_f64((block + 1) as f64 * block_ms / 1000.0);
            while let Some(wait) = due.checked_sub(measured.elapsed()) {
                if stop.load(Ordering::Relaxed) {
                    return Err("AI benchmark canceled.".into());
                }
                std::thread::sleep(wait.min(Duration::from_millis(20)));
            }
        }
    }
    let elapsed = measured.elapsed().as_secs_f64();
    let info = processor.info.clone();
    drop(processor);
    if let Some(output) = output {
        output.finish()?;
    }
    let mut report = summarize(&mut times, block_ms, elapsed, audio_seconds);
    let source_samples = input.as_ref().map(|input| input.samples.len());
    let details = json!({
        "kind":if input.is_some() {"wav_inference_benchmark"} else {"generated_tone_inference_benchmark"},
        "microphone_opened":false,"speech_quality_verified":false,"end_to_end_latency_measured":false,
        "runtime":info,"startup_and_warmup_ms":startup_ms,"wall_seconds":elapsed,
        "blocks":blocks,"block_samples":config.block_size,"block_duration_ms":block_ms,
        "processed_audio_seconds":audio_seconds,"realtime_pacing":options.realtime,
        "input_wav":options.input_wav,"input_sha256":input.as_ref().map(|input| &input.sha256),
        "input_audio_seconds":source_samples.map(|count| count as f64 / SAMPLE_RATE as f64),
        "input_repeated":source_samples.is_some_and(|count| options.blocks.is_some() && samples > count),
        "zero_padding_samples":source_samples.filter(|_| options.blocks.is_none()).map(|count| samples.saturating_sub(count)).unwrap_or(0),
        "output_wav":options.output_wav,"output_nonzero_samples":nonzero_samples,"output_clipped_samples":clipped_samples,
        "output_note":"48 kHz mono PCM16; includes pipeline delay; no tail flush or latency alignment"
    });
    report
        .as_object_mut()
        .unwrap()
        .extend(details.as_object().unwrap().clone());
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "voice-warp-benchmark-{}-{}-{label}.wav",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn wav_reads_integer_and_float_and_rejects_unsupported_audio() {
        for (channels, rate, float, values, valid) in [
            (1, 48000, false, vec![-1.0, 0.0, 0.5], true),
            (1, 48000, true, vec![-1.0, 0.0, 0.5], true),
            (2, 48000, false, vec![0.0, 0.0], false),
            (1, 16000, false, vec![0.0], false),
            (1, 48000, true, vec![f32::NAN], false),
            (1, 48000, true, vec![1.01], false),
            (1, 48000, false, vec![], false),
        ] {
            let path = path("input");
            let mut writer = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels,
                    sample_rate: rate,
                    bits_per_sample: if float { 32 } else { 16 },
                    sample_format: if float {
                        hound::SampleFormat::Float
                    } else {
                        hound::SampleFormat::Int
                    },
                },
            )
            .unwrap();
            for value in values {
                if float {
                    writer.write_sample(value).unwrap();
                } else {
                    writer.write_sample((value * 32767.0) as i16).unwrap();
                }
            }
            writer.finalize().unwrap();
            let result = read_wav(&path);
            assert_eq!(result.is_ok(), valid);
            if let Ok(input) = result {
                assert_eq!(
                    input.sha256,
                    format!("{:x}", Sha256::digest(fs::read(&path).unwrap()))
                );
                assert_eq!(input.samples.len(), 3);
            }
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn default_file_pads_but_explicit_block_count_repeats() {
        let input = Input {
            samples: vec![0.1, 0.2, 0.3],
            sha256: String::new(),
        };
        assert_eq!(block_input(Some(&input), 1, 2, false), vec![0.3, 0.0]);
        assert_eq!(block_input(Some(&input), 1, 2, true), vec![0.3, 0.1]);
    }

    #[test]
    fn output_never_overwrites_and_partial_output_is_removed() {
        let path = path("output");
        fs::write(&path, b"original model or input").unwrap();
        assert!(Output::create(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original model or input");
        fs::remove_file(&path).unwrap();
        {
            let mut output = Output::create(&path).unwrap();
            output.write(&[0.5]).unwrap();
        }
        assert!(!path.exists());
        let mut output = Output::create(&path).unwrap();
        output.write(&[0.5, -0.5]).unwrap();
        output.finish().unwrap();
        assert_eq!(read_wav(&path).unwrap().samples.len(), 2);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn timing_gate_requires_duration_and_both_percentiles() {
        let mut fast = vec![70.0; 100];
        assert_eq!(
            summarize(&mut fast, 100.0, 10.0, 600.0)["timing_gate"]["passed"],
            false
        );
        assert_eq!(
            summarize(&mut fast, 100.0, 600.0, 600.0)["timing_gate"]["passed"],
            true
        );
        fast[0] = 120.0;
        fast[1] = 100.0;
        let report = summarize(&mut fast, 100.0, 600.0, 600.0);
        assert_eq!(report["deadline_misses"], 2);
        assert_eq!(report["timing_gate"]["passed"], false);
    }
}
