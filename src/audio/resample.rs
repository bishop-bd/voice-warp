//! Small streaming sample-rate adapter for microphone and output clocks.
//!
//! Linear interpolation is inexpensive and has no block-boundary discontinuity.
//! It is not a band-limited resampler: content above the destination Nyquist
//! frequency can alias when downsampling. Voice effects should use their own
//! filtering if that distinction is material to the effect.

/// A mono resampler with constant storage and no allocation in `push`.
///
/// Output starts at the first input sample. A fractional output position is
/// emitted once both adjacent input samples are available. The rational phase
/// stays bounded, so long-running streams do not accumulate timing drift.
#[derive(Debug, Clone)]
pub struct LinearResampler {
    input_rate: u64,
    output_rate: u64,
    phase: u64,
    previous: Option<f32>,
}

impl LinearResampler {
    /// Both sample rates must be nonzero.
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        assert!(input_rate != 0, "input sample rate must be nonzero");
        assert!(output_rate != 0, "output sample rate must be nonzero");
        Self {
            input_rate: u64::from(input_rate),
            output_rate: u64::from(output_rate),
            phase: u64::from(input_rate),
            previous: None,
        }
    }

    /// Consume one input sample and emit zero or more output samples.
    ///
    /// Non-finite samples become silence. Finite values are left unclipped so
    /// the caller can apply gain and limiting after resampling.
    pub fn push(&mut self, sample: f32, output: &mut impl FnMut(f32)) {
        let sample = if sample.is_finite() { sample } else { 0.0 };
        let Some(previous) = self.previous.replace(sample) else {
            output(sample);
            return;
        };
        if self.input_rate == self.output_rate {
            output(sample);
            return;
        }
        while self.phase <= self.output_rate {
            if self.phase == self.output_rate {
                output(sample);
            } else {
                let fraction = self.phase as f64 / self.output_rate as f64;
                // f64 intermediate also prevents subtraction overflow for
                // finite f32 values near opposite extremes of their range.
                output(
                    (f64::from(previous) * (1.0 - fraction) + f64::from(sample) * fraction) as f32,
                );
            }
            self.phase += self.input_rate;
        }
        self.phase -= self.output_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(samples: &[f32], from: u32, to: u32, chunk_size: usize) -> Vec<f32> {
        let mut resampler = LinearResampler::new(from, to);
        let mut output = Vec::new();
        for chunk in samples.chunks(chunk_size) {
            for sample in chunk {
                resampler.push(*sample, &mut |sample| output.push(sample));
            }
        }
        output
    }

    #[test]
    fn identity_preserves_samples_exactly_without_delay() {
        let input = vec![0.0, 0.3, -0.7, 1.0, -1.0, f32::MIN_POSITIVE, 0.12345678];
        assert_eq!(convert(&input, 48_000, 48_000, 3), input);
    }

    #[test]
    fn upsampling_interpolates_between_adjacent_samples() {
        assert_eq!(
            convert(&[0.0, 1.0, 0.0, -1.0], 2, 4, 2),
            vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0]
        );
    }

    #[test]
    fn downsampling_uses_the_correct_sample_positions() {
        assert_eq!(
            convert(&[0.0, 1.0, 2.0, 3.0, 4.0], 4, 2, 3),
            vec![0.0, 2.0, 4.0]
        );
    }

    #[test]
    fn fractional_sample_counts_follow_the_exact_ratio() {
        for (input_rate, output_rate) in [
            (44_100, 48_000),
            (48_000, 44_100),
            (96_000, 48_000),
            (8_000, 48_000),
        ] {
            let input_count = input_rate as usize * 3 + 19;
            let output = convert(&vec![0.5; input_count], input_rate, output_rate, 960);
            let expected = (input_count - 1) * output_rate as usize / input_rate as usize + 1;
            assert_eq!(output.len(), expected);
            assert!(output.iter().all(|value| *value == 0.5));
        }
    }

    #[test]
    fn block_boundaries_do_not_reset_phase_or_change_samples() {
        let input: Vec<f32> = (0..44_137)
            .map(|index| (index as f32 * 0.031).sin())
            .collect();
        let reference = convert(&input, 44_100, 48_000, input.len());
        for chunk in [1, 17, 128, 960, 6144] {
            assert_eq!(convert(&input, 44_100, 48_000, chunk), reference);
        }
    }

    #[test]
    fn conversion_preserves_a_voice_band_sines_frequency_and_amplitude() {
        for (from, to) in [(44_100, 48_000), (48_000, 44_100)] {
            let tone = |index: usize, rate: u32| {
                0.7 * (index as f64 * 440.0 * std::f64::consts::TAU / f64::from(rate)).sin() as f32
            };
            let input: Vec<f32> = (0..from as usize).map(|index| tone(index, from)).collect();
            let output = convert(&input, from, to, 127);
            let rms_error = (output
                .iter()
                .enumerate()
                .map(|(index, value)| (*value - tone(index, to)).powi(2))
                .sum::<f32>()
                / output.len() as f32)
                .sqrt();
            assert!(
                rms_error < 0.001,
                "rate {from} -> {to}, RMS error {rms_error}"
            );
            let crossings = output
                .windows(2)
                .filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0)
                .count();
            assert!(
                (439..=441).contains(&crossings),
                "rate {from} -> {to}, crossings {crossings}"
            );
        }
    }

    #[test]
    fn invalid_samples_become_silence_and_extreme_finite_samples_stay_finite() {
        assert_eq!(
            convert(
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
                48_000,
                48_000,
                1
            ),
            vec![0.0; 3]
        );
        let output = convert(&[f32::MAX, -f32::MAX], 24_000, 48_000, 1);
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(output[1], 0.0);
    }

    #[test]
    #[should_panic(expected = "input sample rate must be nonzero")]
    fn zero_input_rate_is_rejected() {
        LinearResampler::new(0, 48_000);
    }

    #[test]
    #[should_panic(expected = "output sample rate must be nonzero")]
    fn zero_output_rate_is_rejected() {
        LinearResampler::new(48_000, 0);
    }
}
