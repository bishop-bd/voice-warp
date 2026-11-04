//! A small, offline soundboard mixed into the outgoing virtual microphone.
//!
//! Audio is decoded once at import time. `mix_into` only reads resident samples;
//! the caller can hold one mutex for the whole block, rather than per sample.

use std::f32::consts::TAU;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SAMPLE_RATE: u32 = 48_000;
pub const MAX_CLIPS: usize = 16;
pub const MAX_VOICES: usize = 8;
const MAX_SECONDS: u32 = 60;
const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;

pub type ClipId = u64;

#[derive(Clone, Debug)]
pub struct ClipInfo {
    pub id: ClipId,
    pub name: String,
    pub duration_secs: f32,
    /// `None` denotes an original synthesized sound bundled with Voice Warp.
    pub path: Option<PathBuf>,
    pub is_playing: bool,
}

/// A fully decoded import, ready for a short insertion under the board mutex.
/// Construct this with `Soundboard::prepare` before acquiring that mutex.
#[derive(Clone)]
pub struct PreparedClip {
    name: String,
    path: PathBuf,
    samples: Arc<[f32]>,
}

struct Clip {
    id: ClipId,
    name: String,
    path: Option<PathBuf>,
    samples: Arc<[f32]>,
}

struct Voice {
    id: ClipId,
    samples: Arc<[f32]>,
    position: usize,
}

pub struct Soundboard {
    clips: Vec<Clip>,
    voices: Vec<Voice>,
    next_id: ClipId,
    volume: f32,
}

impl Default for Soundboard {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Soundboard {
    pub fn new() -> Self {
        Self {
            clips: Vec::new(),
            voices: Vec::new(),
            next_id: 1,
            volume: 0.65,
        }
    }

    pub fn with_defaults() -> Self {
        let mut board = Self::new();
        for (name, samples) in [
            ("Chime", synth_chime()),
            ("Air horn", synth_airhorn()),
            ("Level up", synth_arcade()),
            ("Rimshot", synth_rimshot()),
        ] {
            board.insert_samples(name.into(), None, samples);
        }
        board
    }

    /// Import a PCM or IEEE float WAV, downmix it, and resample to 48 kHz.
    /// Reimporting the same canonical path selects the existing clip.
    /// For a live shared board, use `prepare` outside the mutex, then `insert`.
    pub fn load(&mut self, path: &Path) -> Result<ClipId, String> {
        self.insert(Self::prepare(path)?)
    }

    /// Decode and allocate all audio without borrowing a soundboard. Run this
    /// in a background worker without holding the shared soundboard mutex.
    pub fn prepare(path: &Path) -> Result<PreparedClip, String> {
        let canonical = path
            .canonicalize()
            .map_err(|e| format!("Cannot open sound: {e}"))?;
        let metadata = canonical
            .metadata()
            .map_err(|e| format!("Cannot read sound: {e}"))?;
        if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES {
            return Err("Choose a WAV file smaller than 128 MiB.".into());
        }
        let reader = hound::WavReader::open(&canonical).map_err(|e| {
            format!("Cannot decode WAV: {e}. Import an uncompressed PCM or float WAV file.")
        })?;
        let samples = decode_wav(reader)?;
        let name = canonical
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "Imported sound".into());
        Ok(PreparedClip {
            name,
            path: canonical,
            samples: samples.into(),
        })
    }

    /// Insert a prepared clip without disk I/O, decoding, or audio allocation.
    /// Check capacity here because other imports may have arrived meanwhile.
    pub fn insert(&mut self, clip: PreparedClip) -> Result<ClipId, String> {
        if let Some(existing) = self
            .clips
            .iter()
            .find(|existing| existing.path.as_ref() == Some(&clip.path))
        {
            return Ok(existing.id);
        }
        if self.clips.len() >= MAX_CLIPS {
            return Err(format!(
                "The soundboard holds {MAX_CLIPS} sounds. Remove one before importing another."
            ));
        }
        Ok(self.insert_samples(clip.name, Some(clip.path), clip.samples))
    }

    fn insert_samples(
        &mut self,
        name: String,
        path: Option<PathBuf>,
        samples: impl Into<Arc<[f32]>>,
    ) -> ClipId {
        let id = self.next_id;
        self.next_id += 1;
        self.clips.push(Clip {
            id,
            name,
            path,
            samples: samples.into(),
        });
        id
    }

    pub fn list(&self) -> Vec<ClipInfo> {
        self.clips
            .iter()
            .map(|clip| ClipInfo {
                id: clip.id,
                name: clip.name.clone(),
                duration_secs: clip.samples.len() as f32 / SAMPLE_RATE as f32,
                path: clip.path.clone(),
                is_playing: self.voices.iter().any(|voice| voice.id == clip.id),
            })
            .collect()
    }

    /// Play a sound from its beginning. Retriggering restarts the same sound.
    /// At most eight distinct sounds overlap; the oldest is evicted if full.
    pub fn trigger(&mut self, id: ClipId) -> Result<(), String> {
        let clip = self
            .clips
            .iter()
            .find(|clip| clip.id == id)
            .ok_or_else(|| "This sound has been removed.".to_string())?;
        self.voices.retain(|voice| voice.id != id);
        if self.voices.len() == MAX_VOICES {
            self.voices.remove(0);
        }
        self.voices.push(Voice {
            id,
            samples: Arc::clone(&clip.samples),
            position: 0,
        });
        Ok(())
    }

    pub fn stop_all(&mut self) {
        self.voices.clear();
    }

    pub fn remove(&mut self, id: ClipId) -> bool {
        let old_len = self.clips.len();
        self.voices.retain(|voice| voice.id != id);
        self.clips.retain(|clip| clip.id != id);
        old_len != self.clips.len()
    }

    pub fn set_volume(&mut self, volume: f32) {
        if volume.is_finite() {
            self.volume = volume.clamp(0.0, 1.5);
        }
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    pub fn active_count(&self) -> usize {
        self.voices.len()
    }

    /// Add the soundboard to the outgoing block. The caller applies its master
    /// mute and limiter *after* this function, so those also affect the sounds.
    pub fn mix_into(&mut self, output: &mut [f32]) {
        for voice in &mut self.voices {
            let count = output.len().min(voice.samples.len() - voice.position);
            for (out, sample) in output[..count]
                .iter_mut()
                .zip(&voice.samples[voice.position..voice.position + count])
            {
                *out += sample * self.volume;
            }
            voice.position += count;
        }
        self.voices
            .retain(|voice| voice.position < voice.samples.len());
    }
}

fn decode_wav<R: Read + Seek>(mut reader: hound::WavReader<R>) -> Result<Vec<f32>, String> {
    let spec = reader.spec();
    if !(1..=8).contains(&spec.channels) {
        return Err("WAV files must have between 1 and 8 channels.".into());
    }
    if !(8_000..=192_000).contains(&spec.sample_rate) {
        return Err("WAV sample rates must be between 8 kHz and 192 kHz.".into());
    }
    let channels = spec.channels as usize;
    if !(reader.len() as usize).is_multiple_of(channels) {
        return Err("This WAV ends partway through an audio frame.".into());
    }
    let frame_count = reader.duration() as usize;
    if frame_count == 0 {
        return Err("This WAV contains no audio.".into());
    }
    if frame_count > (spec.sample_rate * MAX_SECONDS) as usize {
        return Err(format!("Sounds must be {MAX_SECONDS} seconds or shorter."));
    }
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 8 | 16 | 24 | 32) | (hound::SampleFormat::Float, 32) => {}
        _ => return Err("Use 8/16/24/32-bit PCM or 32-bit float WAV audio.".into()),
    }

    // Stream channel samples into a mono buffer: an eight-channel source does
    // not require an eight-channel float allocation.
    let mut mono = Vec::with_capacity(frame_count);
    let mut sum = 0.0_f64;
    let mut channel = 0;
    let mut push = |sample: Result<f32, hound::Error>| -> Result<(), String> {
        let value = sample.map_err(|e| format!("Corrupt or truncated WAV: {e}"))?;
        if !value.is_finite() {
            return Err("This WAV contains invalid floating-point samples.".into());
        }
        sum += f64::from(value.clamp(-1.0, 1.0));
        channel += 1;
        if channel == channels {
            mono.push((sum / channels as f64) as f32);
            channel = 0;
            sum = 0.0;
        }
        Ok(())
    };
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1_u64 << (spec.bits_per_sample - 1)) as f64;
            for sample in reader.samples::<i32>() {
                push(sample.map(|sample| (f64::from(sample) / scale) as f32))?;
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                push(sample)?;
            }
        }
    }
    if mono.len() != frame_count {
        return Err("This WAV contains an incomplete audio stream.".into());
    }
    Ok(resample_linear(&mono, spec.sample_rate))
}

/// Linear interpolation is deliberately inexpensive for one-shot effects.
/// For best fidelity, import 48 kHz WAV files so no resampling is required.
fn resample_linear(input: &[f32], source_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == SAMPLE_RATE {
        return input.to_vec();
    }
    let output_len = ((input.len() as u64 * u64::from(SAMPLE_RATE) + u64::from(source_rate) / 2)
        / u64::from(source_rate)) as usize;
    let ratio = f64::from(source_rate) / f64::from(SAMPLE_RATE);
    (0..output_len.max(1))
        .map(|index| {
            let position = index as f64 * ratio;
            let left = (position as usize).min(input.len() - 1);
            let right = (left + 1).min(input.len() - 1);
            let fraction = (position - position.floor()) as f32;
            input[left] + (input[right] - input[left]) * fraction
        })
        .collect()
}

fn envelope(time: f32, duration: f32, attack: f32, release: f32) -> f32 {
    (time / attack).clamp(0.0, 1.0) * ((duration - time) / release).clamp(0.0, 1.0)
}

fn synth_chime() -> Vec<f32> {
    let duration = 1.5;
    (0..(duration * SAMPLE_RATE as f32) as usize)
        .map(|index| {
            let t = index as f32 / SAMPLE_RATE as f32;
            let first = (TAU * 880.0 * t).sin() * (-3.4 * t).exp();
            let second_t = (t - 0.18).max(0.0);
            let second = if t >= 0.18 {
                (TAU * 1320.0 * second_t).sin() * (-4.0 * second_t).exp()
            } else {
                0.0
            };
            (first + second * 0.6) * 0.38 * envelope(t, duration, 0.004, 0.06)
        })
        .collect()
}

fn synth_airhorn() -> Vec<f32> {
    let duration = 0.85;
    (0..(duration * SAMPLE_RATE as f32) as usize)
        .map(|index| {
            let t = index as f32 / SAMPLE_RATE as f32;
            let mut sample = 0.0;
            for base in [233.08, 311.13] {
                for harmonic in 1..=7 {
                    sample += (TAU * base * harmonic as f32 * t).sin() / harmonic as f32;
                }
            }
            sample * 0.18 * envelope(t, duration, 0.03, 0.16)
        })
        .collect()
}

fn synth_arcade() -> Vec<f32> {
    let duration = 0.72;
    let notes = [523.25, 659.25, 783.99, 1046.5];
    (0..(duration * SAMPLE_RATE as f32) as usize)
        .map(|index| {
            let t = index as f32 / SAMPLE_RATE as f32;
            let note_index = ((t / 0.14) as usize).min(notes.len() - 1);
            let note_t = t - note_index as f32 * 0.14;
            let note_duration = if note_index == 3 {
                duration - 0.42
            } else {
                0.14
            };
            let phase = TAU * notes[note_index] * note_t;
            // A few harmonics keep the arcade character without a hard square
            // wave's out-of-band discontinuities.
            ((phase.sin() + (3.0 * phase).sin() / 3.0 + (5.0 * phase).sin() / 5.0)
                * 0.34
                * envelope(note_t, note_duration, 0.006, 0.035))
            .clamp(-1.0, 1.0)
        })
        .collect()
}

fn synth_rimshot() -> Vec<f32> {
    let duration = 0.5;
    let mut random = 0x6d2b_79f5_u32;
    (0..(duration * SAMPLE_RATE as f32) as usize)
        .map(|index| {
            let t = index as f32 / SAMPLE_RATE as f32;
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            let noise = random as f32 / u32::MAX as f32 * 2.0 - 1.0;
            let click = noise * (-75.0 * t).exp();
            let ring =
                ((TAU * 1780.0 * t).sin() + (TAU * 2340.0 * t).sin() * 0.5) * (-28.0 * t).exp();
            (click * 0.45 + ring * 0.3) * envelope(t, duration, 0.0003, 0.04)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn wav_bytes(spec: hound::WavSpec, integers: &[i32], floats: &[f32]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for sample in integers {
                writer.write_sample(*sample).unwrap();
            }
            for sample in floats {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn decodes_every_supported_pcm_depth_and_downmixes() {
        for bits in [8, 16, 24, 32] {
            let quarter = 1_i32 << (bits - 3);
            let bytes = wav_bytes(
                hound::WavSpec {
                    channels: 2,
                    sample_rate: SAMPLE_RATE,
                    bits_per_sample: bits,
                    sample_format: hound::SampleFormat::Int,
                },
                &[quarter, -quarter, quarter, quarter],
                &[],
            );
            let decoded = decode_wav(hound::WavReader::new(Cursor::new(bytes)).unwrap()).unwrap();
            assert_eq!(decoded, [0.0, 0.25], "PCM {bits}");
        }
    }

    #[test]
    fn floats_are_decoded_and_nonfinite_values_rejected() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let bytes = wav_bytes(spec, &[], &[-0.5, 0.0, 0.25, 3.0]);
        let decoded = decode_wav(hound::WavReader::new(Cursor::new(bytes)).unwrap()).unwrap();
        assert_eq!(decoded, [-0.5, 0.0, 0.25, 1.0]);
        let invalid = wav_bytes(spec, &[], &[f32::NAN]);
        assert!(decode_wav(hound::WavReader::new(Cursor::new(invalid)).unwrap()).is_err());
    }

    #[test]
    fn resampling_preserves_duration_and_interpolates() {
        assert_eq!(
            resample_linear(&[0.0, 1.0, 0.0, -1.0], 24_000),
            [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0]
        );
        let source = vec![0.25; 44_100];
        let result = resample_linear(&source, 44_100);
        assert_eq!(result.len(), SAMPLE_RATE as usize);
        assert!(result.iter().all(|x| (*x - 0.25).abs() < 1e-6));
    }

    #[test]
    fn clips_overlap_advance_across_blocks_and_end_without_altering_tail() {
        let mut board = Soundboard::new();
        board.set_volume(0.5);
        let first = board.insert_samples("One".into(), None, vec![0.2, 0.4, 0.6]);
        let second = board.insert_samples("Two".into(), None, vec![0.4]);
        board.trigger(first).unwrap();
        board.trigger(second).unwrap();
        let mut frame = [0.1; 2];
        board.mix_into(&mut frame);
        assert!((frame[0] - 0.4).abs() < 1e-6);
        assert!((frame[1] - 0.3).abs() < 1e-6);
        assert_eq!(board.active_count(), 1);
        let mut tail = [0.1; 2];
        board.mix_into(&mut tail);
        assert!((tail[0] - 0.4).abs() < 1e-6);
        assert_eq!(tail[1], 0.1);
        assert_eq!(board.active_count(), 0);
    }

    #[test]
    fn retrigger_restarts_and_stop_removes_all_audio() {
        let mut board = Soundboard::new();
        board.set_volume(1.0);
        let id = board.insert_samples("One".into(), None, vec![0.2, 0.8]);
        board.trigger(id).unwrap();
        board.mix_into(&mut [0.0]);
        board.trigger(id).unwrap();
        assert_eq!(board.active_count(), 1);
        let mut frame = [0.0];
        board.mix_into(&mut frame);
        assert_eq!(frame, [0.2]);
        board.stop_all();
        board.mix_into(&mut frame);
        assert_eq!(frame, [0.2]);
        assert!(!board.list()[0].is_playing);
    }

    #[test]
    fn voice_count_is_bounded_and_removing_a_clip_stops_it() {
        let mut board = Soundboard::new();
        for _ in 0..10 {
            let id = board.insert_samples("One".into(), None, vec![0.1; 2]);
            board.trigger(id).unwrap();
        }
        assert_eq!(board.active_count(), MAX_VOICES);
        assert!(!board.list()[0].is_playing);
        assert!(board.remove(10));
        assert_eq!(board.active_count(), MAX_VOICES - 1);
        assert!(board.trigger(10).is_err());
    }

    #[test]
    fn generated_sounds_are_original_finite_and_not_silent() {
        let board = Soundboard::with_defaults();
        assert_eq!(board.list().len(), 4);
        for clip in &board.clips {
            assert!(clip.path.is_none());
            assert!(
                clip.samples
                    .iter()
                    .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
            );
            assert!(clip.samples.iter().any(|sample| sample.abs() > 0.1));
        }
    }

    #[test]
    fn prepared_import_contains_all_audio_before_board_insertion() {
        use std::io::Write;

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "voice-warp-soundboard-{}-{nonce}.wav",
            std::process::id()
        ));
        let bytes = wav_bytes(
            hound::WavSpec {
                channels: 2,
                sample_rate: 24_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
            &[8192, 8192, 16384, 16384],
            &[],
        );
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        let prepared = Soundboard::prepare(&path);
        // Insertion must not reopen or decode the original file.
        std::fs::remove_file(&path).unwrap();
        let prepared = prepared.unwrap();
        let mut board = Soundboard::new();
        board.set_volume(1.0);
        let id = board.insert(prepared).unwrap();
        board.trigger(id).unwrap();
        let mut output = [0.0; 4];
        board.mix_into(&mut output);
        assert_eq!(output, [0.25, 0.375, 0.5, 0.5]);
    }

    #[test]
    fn prepared_import_reuses_paths_and_checks_current_capacity() {
        let prepared = |number: usize| PreparedClip {
            name: format!("Sound {number}"),
            path: PathBuf::from(format!("sound-{number}.wav")),
            samples: vec![0.1; 4].into(),
        };
        let mut board = Soundboard::new();
        let first = board.insert(prepared(0)).unwrap();
        let waiting_import = prepared(MAX_CLIPS);
        for number in 1..MAX_CLIPS {
            board.insert(prepared(number)).unwrap();
        }
        assert!(board.insert(waiting_import.clone()).is_err());
        assert_eq!(board.list().len(), MAX_CLIPS);
        // Reimporting an existing path does not consume another slot.
        assert_eq!(board.insert(prepared(0)).unwrap(), first);
        assert_eq!(board.list().len(), MAX_CLIPS);
        board.remove(first);
        assert!(board.insert(waiting_import).is_ok());
        assert_eq!(board.list().len(), MAX_CLIPS);
    }

    #[test]
    fn malformed_truncated_empty_and_long_audio_are_rejected() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let empty = wav_bytes(spec, &[], &[]);
        assert!(decode_wav(hound::WavReader::new(Cursor::new(empty)).unwrap()).is_err());
        let mut truncated = wav_bytes(spec, &[100, 200, 300], &[]);
        truncated.pop();
        assert!(decode_wav(hound::WavReader::new(Cursor::new(truncated)).unwrap()).is_err());
        let long = wav_bytes(spec, &vec![0; 8_000 * 60 + 1], &[]);
        assert!(decode_wav(hound::WavReader::new(Cursor::new(long)).unwrap()).is_err());
    }
}
