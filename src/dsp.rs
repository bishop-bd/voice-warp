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
    pub radio: bool,
}

pub const PRESETS: [Preset; 6] = [
    Preset {
        name: "Natural",
        subtitle: "Your voice, cleaned up",
        mark: "01",
        pitch: 0.0,
        robot: 0.0,
        echo: 0.0,
        radio: false,
    },
    Preset {
        name: "Titan",
        subtitle: "Low, larger than life",
        mark: "02",
        pitch: -5.0,
        robot: 0.0,
        echo: 0.0,
        radio: false,
    },
    Preset {
        name: "Helium",
        subtitle: "A little above it all",
        mark: "03",
        pitch: 7.0,
        robot: 0.0,
        echo: 0.0,
        radio: false,
    },
    Preset {
        name: "Synth",
        subtitle: "From another circuit",
        mark: "04",
        pitch: -2.0,
        robot: 0.85,
        echo: 0.0,
        radio: false,
    },
    Preset {
        name: "Ghost",
        subtitle: "Leave an echo behind",
        mark: "05",
        pitch: 3.0,
        robot: 0.15,
        echo: 0.4,
        radio: false,
    },
    Preset {
        name: "Radio",
        subtitle: "Over and out",
        mark: "06",
        pitch: 0.0,
        robot: 0.0,
        echo: 0.0,
        radio: true,
    },
];

impl Preset {
    pub fn apply(&self, settings: &mut EffectSettings) {
        settings.pitch = self.pitch;
        settings.robot = self.robot;
        settings.echo = self.echo;
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
        }
    }
}

impl VoiceDsp {
    fn tap(&self, phase: f32) -> f32 {
        let delay = 64.0 + phase * 2048.0;
        let position = (self.pitch_write as f32 - delay).rem_euclid(self.pitch_ring.len() as f32);
        let left = position.floor() as usize;
        let fraction = position - left as f32;
        self.pitch_ring[left] * (1.0 - fraction)
            + self.pitch_ring[(left + 1) % self.pitch_ring.len()] * fraction
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
            self.radio_hp = 0.962 * (self.radio_hp + value - self.radio_prev_in);
            self.radio_prev_in = value;
            self.radio_lp += 0.36 * (self.radio_hp - self.radio_lp);
            if settings.radio {
                value = (self.radio_lp * 64.0).round() / 64.0;
            }
            let echo = self.delay[self.delay_write];
            self.delay[self.delay_write] = value + echo * settings.echo * 0.7;
            self.delay_write = (self.delay_write + 1) % self.delay.len();
            value += echo * settings.echo;
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
