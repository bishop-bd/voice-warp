//! Platform audio: managed PipeWire microphone on Linux; explicit virtual
//! cable playback through WASAPI/CoreAudio on Windows and macOS.

use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub const SAMPLE_RATE: usize = 48_000;
pub const SOURCE_NAME: &str = "voice_warp";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioDevice {
    /// Platform-specific selection token. Persist this, rather than description.
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AudioStats {
    pub input_peak: f32,
    pub output_peak: f32,
    /// Most recent block's processing time, not end-to-end latency.
    pub processing_ms: f32,
    pub dropped_blocks: u64,
    pub error: Option<String>,
    pub frames: u64,
}

/// Process mono, 48 kHz audio. Return the same number of finite samples.
pub trait Processor: Send {
    /// Cancellation applies to this session only, including late AI responses.
    fn set_stop_signal(&mut self, _stop: Arc<AtomicBool>) {}
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String>;
}

#[cfg(any(not(any(target_os = "windows", target_os = "macos")), test))]
#[allow(dead_code)]
mod pulse;
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
pub use pulse::AudioEngine;

#[cfg(any(target_os = "windows", target_os = "macos"))]
mod native;
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub use native::AudioEngine;

#[cfg(any(target_os = "windows", target_os = "macos", test))]
mod resample;
