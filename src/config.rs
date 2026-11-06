use crate::dsp::{EffectSettings, PRESETS};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub source: String,
    pub monitor_sink: String,
    /// Playback endpoint of an independently installed virtual cable on Windows/macOS.
    pub output_device: String,
    pub make_default: bool,
    pub effects: EffectSettings,
    pub preset: usize,
    pub ai_endpoint: String,
    pub ai_block_size: usize,
    pub ai_bundled: bool,
    pub rvc_model: PathBuf,
    pub rvc_index: PathBuf,
    pub rvc_pitch: i32,
    pub rvc_index_ratio: f32,
    /// Requested runtime. The worker reports the effective backend separately.
    pub rvc_backend: String,
    /// Device identifier returned by the runtime probe, or its default index.
    pub rvc_device: String,
    pub sound_paths: Vec<PathBuf>,
    /// Original bundled-sound names hidden from the user's library.
    pub removed_builtin_sounds: Vec<String>,
    pub sound_volume: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            source: String::new(),
            monitor_sink: String::new(),
            output_device: String::new(),
            make_default: false,
            effects: EffectSettings::default(),
            preset: 0,
            ai_endpoint: "http://127.0.0.1:18888".into(),
            ai_block_size: 6144,
            ai_bundled: crate::rvc::supported(),
            rvc_model: PathBuf::new(),
            rvc_index: PathBuf::new(),
            rvc_pitch: 0,
            rvc_index_ratio: 0.5,
            rvc_backend: "cpu".into(),
            rvc_device: "0".into(),
            sound_paths: Vec::new(),
            removed_builtin_sounds: Vec::new(),
            sound_volume: 0.75,
        }
    }
}

pub fn config_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "voice-warp")
        .map(|d| d.config_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".voice-warp"))
}

impl Settings {
    pub fn load() -> Result<Self, String> {
        let path = config_dir().join("settings.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&path).map_err(|e| format!("Cannot read settings: {e}"))?;
        Self::from_json(&bytes)
    }

    fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let mut settings: Self =
            serde_json::from_slice(bytes).map_err(|e| format!("Cannot parse settings: {e}"))?;
        settings.effects.sanitize();
        settings.rvc_pitch = settings.rvc_pitch.clamp(-24, 24);
        settings.rvc_index_ratio = if settings.rvc_index_ratio.is_finite() {
            settings.rvc_index_ratio.clamp(0.0, 1.0)
        } else {
            0.5
        };
        settings.preset = settings.preset.min(PRESETS.len() - 1);
        settings.ai_block_size = match settings.ai_block_size {
            4096 | 6144 | 8192 | 12288 => settings.ai_block_size,
            _ => 6144,
        };
        settings.sound_volume = if settings.sound_volume.is_finite() {
            settings.sound_volume.clamp(0.0, 1.5)
        } else {
            0.75
        };
        Ok(settings)
    }

    pub fn save(&self) -> Result<(), String> {
        let dir = config_dir();
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join("settings.json");
        let temporary = dir.join("settings.json.tmp");
        let data = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        fs::write(&temporary, data).map_err(|e| e.to_string())?;
        fs::rename(&temporary, &path).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settings_accept_older_partial_files() {
        let parsed = Settings::from_json(br#"{"source":"mic","effects":{"pitch":-5.0}}"#).unwrap();
        assert_eq!(parsed.effects.pitch, -5.0);
        assert_eq!(parsed.effects.gate_db, -50.0);
        assert_eq!(parsed.effects.tone, 0.0);
        assert_eq!(parsed.effects.distortion, 0.0);
        assert_eq!(parsed.effects.chorus, 0.0);
        assert_eq!(parsed.effects.reverb, 0.0);
        assert_eq!(parsed.ai_block_size, 6144);
        assert_eq!(parsed.rvc_backend, "cpu");
        assert_eq!(parsed.rvc_device, "0");
        assert!(parsed.removed_builtin_sounds.is_empty());
    }

    #[test]
    fn new_presets_and_custom_effects_survive_settings_reload() {
        for (preset, voice) in PRESETS.iter().enumerate().skip(6) {
            let mut settings = Settings {
                preset,
                ..Settings::default()
            };
            voice.apply(&mut settings.effects);
            settings.effects.tone = -0.35;
            settings.effects.distortion = 0.25;
            settings.effects.chorus = 0.4;
            settings.effects.reverb = 0.3;
            let restored = Settings::from_json(&serde_json::to_vec(&settings).unwrap()).unwrap();
            assert_eq!(restored.preset, preset);
            assert_eq!(restored.effects.pitch, settings.effects.pitch);
            assert_eq!(restored.effects.tone, settings.effects.tone);
            assert_eq!(restored.effects.distortion, settings.effects.distortion);
            assert_eq!(restored.effects.chorus, settings.effects.chorus);
            assert_eq!(restored.effects.reverb, settings.effects.reverb);
        }
    }

    #[test]
    fn invalid_preset_is_clamped_to_the_current_library() {
        let restored = Settings::from_json(br#"{"preset":999}"#).unwrap();
        assert_eq!(restored.preset, PRESETS.len() - 1);
    }

    #[test]
    fn rvc_settings_round_trip_and_invalid_values_are_clamped() {
        let parsed = Settings::from_json(br#"{"rvc_model":"voice.onnx","rvc_index":"voice.index","rvc_pitch":99,"rvc_index_ratio":-3}"#).unwrap();
        assert_eq!(parsed.ai_bundled, crate::rvc::supported());
        assert_eq!(parsed.rvc_pitch, 24);
        assert_eq!(parsed.rvc_index_ratio, 0.0);
        let restored = Settings::from_json(&serde_json::to_vec(&parsed).unwrap()).unwrap();
        assert_eq!(restored.rvc_model, PathBuf::from("voice.onnx"));
        assert_eq!(restored.rvc_index, PathBuf::from("voice.index"));
    }

    #[test]
    fn large_sound_libraries_and_builtin_removals_survive_settings_reload() {
        let settings = Settings {
            sound_paths: (0..10_000)
                .map(|number| PathBuf::from(format!("{}/sound-{number}.wav", "a".repeat(100))))
                .collect(),
            removed_builtin_sounds: vec!["Chime".into(), "Rimshot".into()],
            ..Settings::default()
        };
        let bytes = serde_json::to_vec_pretty(&settings).unwrap();
        assert!(bytes.len() > 1_048_576);
        let restored = Settings::from_json(&bytes).unwrap();
        assert_eq!(restored.sound_paths, settings.sound_paths);
        assert_eq!(
            restored.removed_builtin_sounds,
            settings.removed_builtin_sounds
        );
    }

    #[test]
    fn device_preferences_preserve_explicit_external_server_choice() {
        let parsed = Settings::from_json(
            br#"{"ai_bundled":false,"rvc_backend":"cuda","rvc_device":"GPU-123"}"#,
        )
        .unwrap();
        let restored = Settings::from_json(&serde_json::to_vec(&parsed).unwrap()).unwrap();
        assert!(!restored.ai_bundled);
        assert_eq!(restored.rvc_backend, "cuda");
        assert_eq!(restored.rvc_device, "GPU-123");
    }
}
