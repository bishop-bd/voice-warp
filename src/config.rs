use crate::dsp::EffectSettings;
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
    pub sound_paths: Vec<PathBuf>,
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
            sound_paths: Vec::new(),
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
        if bytes.len() > 1_048_576 {
            return Err("Settings file exceeds 1 MB".into());
        }
        let mut settings: Self =
            serde_json::from_slice(&bytes).map_err(|e| format!("Cannot parse settings: {e}"))?;
        settings.effects.sanitize();
        settings.preset = settings.preset.min(5);
        settings.ai_block_size = match settings.ai_block_size {
            4096 | 6144 | 8192 | 12288 => settings.ai_block_size,
            _ => 6144,
        };
        settings.sound_volume = if settings.sound_volume.is_finite() {
            settings.sound_volume.clamp(0.0, 1.5)
        } else {
            0.75
        };
        settings.sound_paths.truncate(12);
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
        let parsed: Settings =
            serde_json::from_str(r#"{"source":"mic","effects":{"pitch":-5.0}}"#).unwrap();
        assert_eq!(parsed.effects.pitch, -5.0);
        assert_eq!(parsed.effects.gate_db, -50.0);
        assert_eq!(parsed.ai_block_size, 6144);
    }
}
