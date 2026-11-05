//! Stateful, allocation-bounded voice effects. Audio is mono, 48 kHz.
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::{audio::Processor, soundboard::Soundboard};

pub const SAMPLE_RATE: f32 = 48_000.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EffectSettings {
    pub pitch: f32,
    pub robot: f32,
    pub echo: f32,
    pub tone: f32,
    pub distortion: f32,
    pub chorus: f32,
    pub reverb: f32,
    pub gain_db: f32,
    pub gate_db: f32,
    pub radio: bool,
    pub bypass: bool,
}

impl Default for EffectSettings {
    fn default() -> Self {
        Self {
            pitch: 0.0,
            robot: 0.0,
            echo: 0.0,
            tone: 0.0,
            distortion: 0.0,
            chorus: 0.0,
            reverb: 0.0,
            gain_db: 0.0,
            gate_db: -50.0,
            radio: false,
            bypass: false,
        }
    }
}

impl EffectSettings {
    pub fn sanitize(&mut self) {
        fn bounded(v: f32, min: f32, max: f32, default: f32) -> f32 {
            if v.is_finite() {
                v.clamp(min, max)
            } else {
                default
            }
        }
        self.pitch = bounded(self.pitch, -12.0, 12.0, 0.0);
        self.robot = bounded(self.robot, 0.0, 1.0, 0.0);
        self.echo = bounded(self.echo, 0.0, 0.65, 0.0);
        self.tone = bounded(self.tone, -1.0, 1.0, 0.0);
        self.distortion = bounded(self.distortion, 0.0, 1.0, 0.0);
        self.chorus = bounded(self.chorus, 0.0, 1.0, 0.0);
        self.reverb = bounded(self.reverb, 0.0, 0.65, 0.0);
        self.gain_db = bounded(self.gain_db, -24.0, 12.0, 0.0);
        self.gate_db = bounded(self.gate_db, -80.0, -20.0, -50.0);
    }
}

pub struct Preset {
    pub name: &'static str,
    pub subtitle: &'static str,
    pub mark: &'static str,
    pub pitch: f32,
    pub robot: f32,
    pub echo: f32,
    pub tone: f32,
    pub distortion: f32,
    pub chorus: f32,
    pub reverb: f32,
    pub radio: bool,
}

pub const PRESETS: [Preset; 11] = [
    Preset {
        name: "Natural",
        subtitle: "Your voice, cleaned up",
        mark: "01",
        pitch: 0.0,
        robot: 0.0,
        echo: 0.0,
        tone: 0.0,
        distortion: 0.0,
        chorus: 0.0,
        reverb: 0.0,
        radio: false,
    },
    Preset {
        name: "Titan",
        subtitle: "Low, larger than life",
        mark: "02",
        pitch: -5.0,
        robot: 0.0,
        echo: 0.0,
        tone: 0.0,
        distortion: 0.0,
        chorus: 0.0,
        reverb: 0.0,
        radio: false,
    },
    Preset {
        name: "Helium",
        subtitle: "A little above it all",
        mark: "03",
        pitch: 7.0,
        robot: 0.0,
        echo: 0.0,
        tone: 0.0,
        distortion: 0.0,
        chorus: 0.0,
        reverb: 0.0,
        radio: false,
    },
    Preset {
        name: "Synth",
        subtitle: "From another circuit",
        mark: "04",
        pitch: -2.0,
        robot: 0.85,
        echo: 0.0,
        tone: 0.0,
        distortion: 0.0,
        chorus: 0.0,
        reverb: 0.0,
        radio: false,
    },
    Preset {
        name: "Ghost",
        subtitle: "Leave an echo behind",
        mark: "05",
        pitch: 3.0,
        robot: 0.15,
        echo: 0.4,
        tone: 0.0,
        distortion: 0.0,
        chorus: 0.0,
        reverb: 0.0,
        radio: false,
    },
    Preset {
        name: "Radio",
        subtitle: "Over and out",
        mark: "06",
        pitch: 0.0,
        robot: 0.0,
        echo: 0.0,
        tone: 0.0,
        distortion: 0.0,
        chorus: 0.0,
        reverb: 0.0,
        radio: true,
    },
    Preset {
        name: "Demon",
        subtitle: "Deep growl, dark presence",
        mark: "07",
        pitch: -9.0,
        robot: 0.12,
        echo: 0.1,
        tone: -0.6,
        distortion: 0.65,
        chorus: 0.2,
        reverb: 0.3,
        radio: false,
    },
    Preset {
        name: "Space Marine",
        subtitle: "Armored helmet comms",
        mark: "08",
        pitch: -3.5,
        robot: 0.22,
        echo: 0.0,
        tone: 0.15,
        distortion: 0.5,
        chorus: 0.0,
        reverb: 0.12,
        radio: true,
    },
    Preset {
        name: "Alien",
        subtitle: "A signal from deep space",
        mark: "09",
        pitch: 4.5,
        robot: 0.3,
        echo: 0.16,
        tone: 0.35,
        distortion: 0.5,
        chorus: 0.5,
        reverb: 0.16,
        radio: false,
    },
    Preset {
        name: "Cartoon",
        subtitle: "Bright, bouncy character",
        mark: "10",
        pitch: 4.0,
        robot: 0.0,
        echo: 0.0,
        tone: 0.55,
        distortion: 0.08,
        chorus: 0.22,
        reverb: 0.06,
        radio: false,
    },
    Preset {
        name: "Announcer",
        subtitle: "Warm broadcast presence",
        mark: "11",
        pitch: -1.5,
        robot: 0.0,
        echo: 0.0,
        tone: -0.3,
        distortion: 0.18,
        chorus: 0.0,
        reverb: 0.14,
        radio: false,
    },
];

impl Preset {
    pub fn apply(&self, settings: &mut EffectSettings) {
        settings.pitch = self.pitch;
        settings.robot = self.robot;
        settings.echo = self.echo;
        settings.tone = self.tone;
        settings.distortion = self.distortion;
        settings.chorus = self.chorus;
        settings.reverb = self.reverb;
        settings.radio = self.radio;
        settings.bypass = false;
    }
}

#[derive(Clone, Default)]
pub struct LiveControls {
    pub effects: EffectSettings,
    pub mic_muted: bool,
    pub master_muted: bool,
}

struct RoomComb {
    samples: Vec<f32>,
    write: usize,
    damped: f32,
}

impl RoomComb {
    fn new(length: usize) -> Self {
        Self {
            samples: vec![0.0; length],
            write: 0,
            damped: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.samples[self.write];
        // Absorb highs on each reflection. Feedback remains below unity,
        // independent of the wet level, so a full reverb cannot self-oscillate.
        self.damped = flush_tiny(self.damped + 0.28 * (delayed - self.damped));
        self.samples[self.write] = flush_tiny(input * 0.26 + self.damped * 0.74);
        self.write = (self.write + 1) % self.samples.len();
        delayed
    }
}

struct RoomAllpass {
    samples: Vec<f32>,
    write: usize,
}

impl RoomAllpass {
    fn new(length: usize) -> Self {
        Self {
            samples: vec![0.0; length],
            write: 0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.samples[self.write] - 0.5 * input;
        self.samples[self.write] = flush_tiny(input + 0.5 * output);
        self.write = (self.write + 1) % self.samples.len();
        output
    }
}

struct RoomReverb {
    combs: [RoomComb; 4],
    diffusers: [RoomAllpass; 2],
}

impl Default for RoomReverb {
    fn default() -> Self {
        Self {
            // Incommensurate reflection lengths avoid a single ringing pitch.
            combs: [1423, 1789, 1999, 2333].map(RoomComb::new),
            diffusers: [241, 557].map(RoomAllpass::new),
        }
    }
}

impl RoomReverb {
    fn process(&mut self, input: f32) -> f32 {
        let mut wet = 0.0;
        for comb in &mut self.combs {
            wet += comb.process(input) * 0.25;
        }
        for diffuser in &mut self.diffusers {
            wet = diffuser.process(wet);
        }
        wet
    }
}

fn flush_tiny(sample: f32) -> f32 {
    if sample.abs() < 1.0e-20 { 0.0 } else { sample }
}

fn smooth_effect(current: &mut f32, target: f32) {
    *current += 0.0015 * (target - *current);
    if (*current - target).abs() < 1.0e-6 {
        *current = target;
    }
}

pub struct VoiceDsp {
    pitch_ring: Vec<f32>,
    pitch_write: usize,
    phase: f32,
    oscillator: f32,
    delay: Vec<f32>,
    delay_write: usize,
    envelope: f32,
    gate: f32,
    radio_prev_in: f32,
    radio_hp: f32,
    radio_lp: f32,
    smoothed_pitch: f32,
    smoothed_gain: f32,
    smoothed_tone: f32,
    smoothed_distortion: f32,
    smoothed_chorus: f32,
    smoothed_reverb: f32,
    tone_low: f32,
    chorus_ring: Vec<f32>,
    chorus_write: usize,
    chorus_phases: [f32; 2],
    room: RoomReverb,
}

impl Default for VoiceDsp {
    fn default() -> Self {
        Self {
            pitch_ring: vec![0.0; 8192],
            pitch_write: 0,
            phase: 0.0,
            oscillator: 0.0,
            delay: vec![0.0; 11_520],
            delay_write: 0,
            envelope: 0.0,
            gate: 0.0,
            radio_prev_in: 0.0,
            radio_hp: 0.0,
            radio_lp: 0.0,
            smoothed_pitch: 0.0,
            smoothed_gain: 1.0,
            smoothed_tone: 0.0,
            smoothed_distortion: 0.0,
            smoothed_chorus: 0.0,
            smoothed_reverb: 0.0,
            tone_low: 0.0,
            chorus_ring: vec![0.0; 2048],
            chorus_write: 0,
            chorus_phases: [0.0, 0.25],
            room: RoomReverb::default(),
        }
    }
}

impl VoiceDsp {
    fn tap(&self, phase: f32) -> f32 {
        let delay = 64.0 + phase * 2048.0;
        let position = (self.pitch_write as f32 - delay).rem_euclid(self.pitch_ring.len() as f32);
        let left = position.floor() as usize;
        let fraction = position - left as f32;
        // Floating-point rem_euclid can round a tiny negative remainder up to
        // the divisor. Wrap the integer index too, including that exact edge.
        let left = left % self.pitch_ring.len();
        self.pitch_ring[left] * (1.0 - fraction)
            + self.pitch_ring[(left + 1) % self.pitch_ring.len()] * fraction
    }

    fn chorus_tap(&self, delay: f32) -> f32 {
        let position = (self.chorus_write as f32 - delay).rem_euclid(self.chorus_ring.len() as f32);
        let index = position.floor() as usize;
        let fraction = position - index as f32;
        let left = index % self.chorus_ring.len();
        self.chorus_ring[left] * (1.0 - fraction)
            + self.chorus_ring[(left + 1) % self.chorus_ring.len()] * fraction
    }

    pub fn process(&mut self, input: &[f32], settings: &EffectSettings) -> Vec<f32> {
        let mut settings = settings.clone();
        settings.sanitize();
        let threshold = 10.0_f32.powf(settings.gate_db / 20.0);
        let gain = 10.0_f32.powf(settings.gain_db / 20.0);
        let mut output = Vec::with_capacity(input.len());
        for &sample in input {
            let raw = if sample.is_finite() {
                sample.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let env_coeff = if raw.abs() > self.envelope {
                0.02
            } else {
                0.0005
            };
            self.envelope += env_coeff * (raw.abs() - self.envelope);
            let target = if self.envelope > threshold { 1.0 } else { 0.0 };
            self.gate += (if target > self.gate { 0.02 } else { 0.002 }) * (target - self.gate);
            let gated = raw * self.gate;
            self.pitch_ring[self.pitch_write] = gated;
            self.smoothed_pitch += 0.001 * (settings.pitch - self.smoothed_pitch);
            let ratio = 2.0_f32.powf(self.smoothed_pitch / 12.0);
            let weight = 0.5 - 0.5 * (TAU * self.phase).cos();
            let pitched = self.tap(self.phase) * weight
                + self.tap((self.phase + 0.5).fract()) * (1.0 - weight);
            self.phase = (self.phase + (1.0 - ratio) / 2048.0).rem_euclid(1.0);
            self.pitch_write = (self.pitch_write + 1) % self.pitch_ring.len();
            let mut value = if self.smoothed_pitch.abs() < 0.01 {
                gated
            } else {
                pitched
            };
            let carrier = (TAU * self.oscillator).sin();
            self.oscillator = (self.oscillator + 85.0 / SAMPLE_RATE).fract();
            value *= 1.0 - settings.robot + settings.robot * carrier;
            smooth_effect(&mut self.smoothed_tone, settings.tone);
            smooth_effect(&mut self.smoothed_distortion, settings.distortion);
            smooth_effect(&mut self.smoothed_chorus, settings.chorus);
            smooth_effect(&mut self.smoothed_reverb, settings.reverb);
            self.tone_low = flush_tiny(self.tone_low + 0.0994 * (value - self.tone_low));
            if self.smoothed_tone != 0.0 {
                let tone = self.smoothed_tone;
                let (low_gain, high_gain) = if tone > 0.0 {
                    (1.0 - 0.45 * tone, 1.0 + 0.6 * tone)
                } else {
                    (1.0 - 0.3 * tone, 1.0 + 0.65 * tone)
                };
                value = self.tone_low * low_gain + (value - self.tone_low) * high_gain;
            }
            if self.smoothed_distortion != 0.0 {
                let amount = self.smoothed_distortion;
                // Compensated soft saturation adds harmonics without making
                // the drive control a large, unexpected volume boost.
                let saturated = (value * (1.0 + 10.0 * amount)).tanh() / (1.0 + 2.0 * amount);
                value += amount * (saturated - value);
            }
            self.radio_hp = 0.962 * (self.radio_hp + value - self.radio_prev_in);
            self.radio_prev_in = value;
            self.radio_lp += 0.36 * (self.radio_hp - self.radio_lp);
            if settings.radio {
                value = (self.radio_lp * 64.0).round() / 64.0;
            }
            self.chorus_ring[self.chorus_write] = value;
            if self.smoothed_chorus != 0.0 {
                let first = self.chorus_tap(624.0 + 144.0 * (TAU * self.chorus_phases[0]).sin());
                let second = self.chorus_tap(912.0 + 192.0 * (TAU * self.chorus_phases[1]).sin());
                let doubled = 0.5 * (first + second);
                // Keep a dry center for intelligibility even at maximum width.
                value += 0.55 * self.smoothed_chorus * (doubled - value);
            }
            self.chorus_phases[0] = (self.chorus_phases[0] + 0.73 / SAMPLE_RATE).fract();
            self.chorus_phases[1] = (self.chorus_phases[1] + 1.07 / SAMPLE_RATE).fract();
            self.chorus_write = (self.chorus_write + 1) % self.chorus_ring.len();
            let echo = self.delay[self.delay_write];
            self.delay[self.delay_write] = value + echo * settings.echo * 0.7;
            self.delay_write = (self.delay_write + 1) % self.delay.len();
            value += echo * settings.echo;
            let room = self.room.process(value);
            if self.smoothed_reverb != 0.0 {
                value += self.smoothed_reverb * (room - value);
            }
            self.smoothed_gain += 0.002 * (gain - self.smoothed_gain);
            output.push((if settings.bypass { raw } else { value }) * self.smoothed_gain);
        }
        output
    }
}

/// The soundboard is mixed after conversion so AI never transforms a sound clip.
pub struct MixProcessor {
    pub controls: Arc<Mutex<LiveControls>>,
    pub soundboard: Arc<Mutex<Soundboard>>,
    pub ai: Option<Box<dyn Processor>>,
    dsp: VoiceDsp,
    stop_signal: Option<Arc<AtomicBool>>,
}

impl MixProcessor {
    pub fn new(
        controls: Arc<Mutex<LiveControls>>,
        soundboard: Arc<Mutex<Soundboard>>,
        ai: Option<Box<dyn Processor>>,
    ) -> Self {
        Self {
            controls,
            soundboard,
            ai,
            dsp: VoiceDsp::default(),
            stop_signal: None,
        }
    }
}

impl Processor for MixProcessor {
    fn set_stop_signal(&mut self, stop: Arc<AtomicBool>) {
        self.stop_signal = Some(stop);
    }

    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if self
            .stop_signal
            .as_ref()
            .is_some_and(|stop| stop.load(Ordering::Acquire))
        {
            return Ok(vec![0.0; input.len()]);
        }
        let controls = self
            .controls
            .lock()
            .map_err(|_| "Voice controls unavailable")?
            .clone();
        // Muting does not bypass AI errors or route an unconverted voice to output.
        let silence;
        let voice_input = if controls.mic_muted || controls.master_muted {
            silence = vec![0.0; input.len()];
            &silence
        } else {
            input
        };
        let mut output = if let Some(ai) = &mut self.ai {
            ai.process(voice_input)?
        } else {
            self.dsp.process(voice_input, &controls.effects)
        };
        if output.len() != input.len() {
            return Err("Voice engine returned the wrong audio length".into());
        }
        // A timed-out worker may finish after an explicit stop or restart.
        // It must not consume clips from the next session's soundboard.
        if self
            .stop_signal
            .as_ref()
            .is_some_and(|stop| stop.load(Ordering::Acquire))
        {
            return Ok(vec![0.0; input.len()]);
        }
        // Inference may block. Re-read mute state so a click during inference
        // also silences the result that is about to leave this process.
        let controls = self
            .controls
            .lock()
            .map_err(|_| "Voice controls unavailable")?
            .clone();
        if controls.mic_muted {
            output.fill(0.0);
        }
        self.soundboard
            .lock()
            .map_err(|_| "Soundboard unavailable")?
            .mix_into(&mut output);
        if controls.master_muted {
            output.fill(0.0);
        }
        for sample in &mut output {
            if !sample.is_finite() {
                return Err("Voice engine returned invalid audio".into());
            }
            // Transparent below -0.9 dBFS, smoothly bounded above it.
            if sample.abs() > 0.9 {
                *sample = sample.signum() * (0.9 + 0.1 * ((sample.abs() - 0.9) * 10.0).tanh());
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|n| (n as f32 * TAU * frequency / SAMPLE_RATE).sin() * 0.2)
            .collect()
    }

    fn mean_square(samples: &[f32]) -> f32 {
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32
    }

    #[test]
    fn new_effects_return_to_a_dry_voice_after_switching_off() {
        let input = sine(220.0, 48_000);
        let mut dry = VoiceDsp::default();
        let mut wet = VoiceDsp::default();
        let enabled = EffectSettings {
            tone: -0.8,
            distortion: 0.8,
            chorus: 0.8,
            reverb: 0.6,
            ..Default::default()
        };
        let plain = EffectSettings::default();
        dry.process(&input, &plain);
        wet.process(&input, &enabled);
        let expected = dry.process(&input, &plain);
        let actual = wet.process(&input, &plain);
        // Switching back to Natural must not leave chorus or reverb audible.
        // Allow the deliberate control fade before comparing the dry stream.
        assert_eq!(&actual[24_000..], &expected[24_000..]);
    }

    #[test]
    fn new_tone_control_changes_spectral_balance() {
        let energy = |tone, frequency| {
            let output = VoiceDsp::default().process(
                &sine(frequency, 24_000),
                &EffectSettings {
                    tone,
                    ..Default::default()
                },
            );
            mean_square(&output[12_000..])
        };
        assert!(energy(-1.0, 180.0) > energy(1.0, 180.0) * 2.0);
        assert!(energy(1.0, 6000.0) > energy(-1.0, 6000.0) * 4.0);
    }

    #[test]
    fn new_distortion_adds_harmonics_without_unbounded_gain() {
        let output = VoiceDsp::default().process(
            &sine(240.0, 48_000),
            &EffectSettings {
                distortion: 1.0,
                ..Default::default()
            },
        );
        let tail = &output[24_000..];
        let amplitude = |frequency: f32| {
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (n, sample) in tail.iter().enumerate() {
                let angle = n as f32 * TAU * frequency / SAMPLE_RATE;
                real += sample * angle.cos();
                imaginary += sample * angle.sin();
            }
            real.hypot(imaginary) / tail.len() as f32
        };
        assert!(amplitude(720.0) > amplitude(240.0) * 0.05);
        assert!(tail.iter().all(|sample| sample.abs() < 0.34));
    }

    #[test]
    fn new_chorus_produces_a_short_layer_without_feedback() {
        let mut dsp = VoiceDsp::default();
        let settings = EffectSettings {
            chorus: 1.0,
            ..Default::default()
        };
        dsp.process(&sine(220.0, 24_000), &settings);
        let tail = dsp.process(&[0.0; 3000], &settings);
        assert!(mean_square(&tail[..960]) > 1.0e-5);
        assert!(tail[1500..].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn new_room_reverb_has_a_decaying_tail_and_no_self_noise() {
        let settings = EffectSettings {
            reverb: 0.65,
            ..Default::default()
        };
        let mut dsp = VoiceDsp::default();
        assert!(
            dsp.process(&[0.0; 4096], &settings)
                .iter()
                .all(|s| *s == 0.0)
        );
        dsp.process(&sine(220.0, 24_000), &settings);
        let tail = dsp.process(&vec![0.0; 96_000], &settings);
        let early = mean_square(&tail[4800..24_000]);
        let late = mean_square(&tail[72_000..]);
        assert!(
            early > 1.0e-8,
            "room tail should survive the first reflection"
        );
        assert!(
            late < early * 1.0e-4,
            "room tail should decay: {early} -> {late}"
        );
    }

    #[test]
    fn new_effects_remain_finite_at_extremes_and_bypass_is_exact() {
        for tone in [-1.0, 1.0] {
            let mut dsp = VoiceDsp::default();
            let mut settings = EffectSettings {
                tone,
                distortion: 1.0,
                chorus: 1.0,
                reverb: 0.65,
                echo: 0.65,
                ..Default::default()
            };
            for value in [1.0, -1.0, 0.0] {
                let output = dsp.process(&vec![value; 96_000], &settings);
                assert!(
                    output
                        .iter()
                        .all(|sample| sample.is_finite() && sample.abs() < 2.0)
                );
            }
            settings.bypass = true;
            assert_eq!(
                dsp.process(&[0.25, -0.25, 0.0], &settings),
                [0.25, -0.25, 0.0]
            );
        }
    }

    #[test]
    fn new_presets_reset_effects_without_changing_microphone_calibration() {
        let mut settings = EffectSettings {
            gain_db: -7.0,
            gate_db: -43.0,
            ..Default::default()
        };
        for preset in &PRESETS[6..] {
            preset.apply(&mut settings);
            PRESETS[0].apply(&mut settings);
            let expected = EffectSettings {
                gain_db: -7.0,
                gate_db: -43.0,
                ..Default::default()
            };
            assert_eq!(
                serde_json::to_value(&settings).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        }
    }

    #[test]
    fn pitch_tap_wraps_when_float_remainder_rounds_to_the_ring_length() {
        let mut dsp = VoiceDsp {
            pitch_write: 64,
            ..VoiceDsp::default()
        };
        dsp.pitch_ring[0] = 1.0;

        // 64 - (64 + phase * 2048) is just below zero. Its f32 Euclidean
        // remainder rounds to 8192, which previously indexed past the buffer.
        assert_eq!(dsp.tap(0.000_000_1), 1.0);
        assert_eq!(dsp.tap(0.0), 1.0);
        // Interpolation across the end must still blend the last and first cells.
        assert_eq!(dsp.tap(0.5 / 2048.0), 0.5);
    }

    #[test]
    fn pitched_presets_keep_processing_for_five_minutes() {
        let input = vec![0.2; 960];
        for preset in PRESETS.iter().filter(|preset| preset.pitch != 0.0) {
            let mut dsp = VoiceDsp::default();
            let mut settings = EffectSettings::default();
            preset.apply(&mut settings);
            for block in 0..15_000 {
                let output = dsp.process(&input, &settings);
                assert_eq!(output.len(), input.len());
                assert!(
                    output.iter().all(|sample| sample.is_finite()),
                    "{} produced invalid audio at block {block}",
                    preset.name
                );
            }
        }
    }

    #[test]
    fn every_preset_is_finite_and_preserves_length_over_streaming_blocks() {
        for preset in PRESETS {
            let mut dsp = VoiceDsp::default();
            let mut settings = EffectSettings::default();
            preset.apply(&mut settings);
            for block in 0..100 {
                let input: Vec<_> = (0..960)
                    .map(|n| ((block * 960 + n) as f32 * TAU * 220.0 / SAMPLE_RATE).sin() * 0.3)
                    .collect();
                let out = dsp.process(&input, &settings);
                assert_eq!(out.len(), input.len());
                assert!(out.iter().all(|x| x.is_finite() && x.abs() < 1.0));
            }
        }
    }
    #[test]
    fn pitch_shift_moves_a_sine_up_one_octave() {
        let mut dsp = VoiceDsp::default();
        let settings = EffectSettings {
            pitch: 12.0,
            gate_db: -80.0,
            ..Default::default()
        };
        let input: Vec<_> = (0..96_000)
            .map(|n| (n as f32 * TAU * 220.0 / SAMPLE_RATE).sin() * 0.3)
            .collect();
        let out = dsp.process(&input, &settings);
        let tail = &out[48_000..];
        let crossings = tail
            .windows(2)
            .filter(|pair| pair[0] < 0.0 && pair[1] >= 0.0)
            .count();
        assert!(
            (420..460).contains(&crossings),
            "Expected about 440 Hz; got {crossings}"
        );
    }
    #[test]
    fn bypass_and_silence_are_predictable() {
        let mut dsp = VoiceDsp::default();
        let settings = EffectSettings {
            bypass: true,
            ..Default::default()
        };
        assert_eq!(
            dsp.process(&[0.25, -0.25, 0.0], &settings),
            vec![0.25, -0.25, 0.0]
        );
        let mut dsp = VoiceDsp::default();
        assert!(
            dsp.process(&vec![0.0; 4096], &EffectSettings::default())
                .iter()
                .all(|x| *x == 0.0)
        );
    }
    #[test]
    fn nan_settings_and_samples_do_not_poison_stream() {
        let mut dsp = VoiceDsp::default();
        let settings = EffectSettings {
            pitch: f32::NAN,
            tone: f32::INFINITY,
            distortion: f32::NAN,
            chorus: f32::NEG_INFINITY,
            reverb: f32::NAN,
            gain_db: f32::INFINITY,
            ..Default::default()
        };
        assert!(
            dsp.process(&[f32::NAN, f32::INFINITY, 0.2], &settings)
                .iter()
                .all(|s| s.is_finite())
        );
    }
    #[test]
    fn master_mute_silences_soundboard_too() {
        let controls = Arc::new(Mutex::new(LiveControls {
            master_muted: true,
            ..Default::default()
        }));
        let mut board = Soundboard::with_defaults();
        let first = board.list()[0].id;
        board.trigger(first).unwrap();
        let mut mixer = MixProcessor::new(controls, Arc::new(Mutex::new(board)), None);
        assert!(
            mixer
                .process(&[0.2; 960])
                .unwrap()
                .iter()
                .all(|x| *x == 0.0)
        );
    }

    #[test]
    fn mute_changed_during_inference_silences_the_result() {
        struct MutingAi(Arc<Mutex<LiveControls>>);
        impl Processor for MutingAi {
            fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
                self.0.lock().unwrap().mic_muted = true;
                Ok(vec![0.3; samples.len()])
            }
        }
        let controls = Arc::new(Mutex::new(LiveControls::default()));
        let ai = MutingAi(controls.clone());
        let mut mixer = MixProcessor::new(
            controls,
            Arc::new(Mutex::new(Soundboard::with_defaults())),
            Some(Box::new(ai)),
        );
        assert!(
            mixer
                .process(&[0.2; 960])
                .unwrap()
                .iter()
                .all(|x| *x == 0.0)
        );
    }

    #[test]
    fn stopped_inference_does_not_advance_soundboard() {
        struct StoppingAi(Arc<AtomicBool>);
        impl Processor for StoppingAi {
            fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
                self.0.store(true, Ordering::Release);
                Ok(vec![0.3; samples.len()])
            }
        }
        let stop = Arc::new(AtomicBool::new(false));
        let board = Arc::new(Mutex::new(Soundboard::with_defaults()));
        let id = board.lock().unwrap().list()[0].id;
        board.lock().unwrap().trigger(id).unwrap();
        let mut mixer = MixProcessor::new(
            Arc::new(Mutex::new(LiveControls::default())),
            board.clone(),
            Some(Box::new(StoppingAi(stop.clone()))),
        );
        mixer.set_stop_signal(stop);
        assert!(
            mixer
                .process(&[0.2; 960])
                .unwrap()
                .iter()
                .all(|x| *x == 0.0)
        );
        let mut actual = vec![0.0; 960];
        board.lock().unwrap().mix_into(&mut actual);
        let mut fresh_board = Soundboard::with_defaults();
        fresh_board.trigger(id).unwrap();
        let mut expected = vec![0.0; 960];
        fresh_board.mix_into(&mut expected);
        assert_eq!(actual, expected);
    }
}
