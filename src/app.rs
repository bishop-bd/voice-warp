use eframe::egui::{self, Color32, CornerRadius, FontId, Margin, RichText, Stroke, Vec2};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use voice_warp::{
    ai::{AIConfig, AIProcessor},
    audio::{AudioDevice, AudioEngine, AudioStats, Processor},
    config::Settings,
    dsp::{LiveControls, MixProcessor, PRESETS},
    rvc::{self, RvcConfig, RvcProcessor},
    soundboard::Soundboard,
};

const BG: Color32 = Color32::from_rgb(13, 17, 25);
const PANEL: Color32 = Color32::from_rgb(19, 25, 36);
const CARD: Color32 = Color32::from_rgb(25, 33, 46);
const BORDER: Color32 = Color32::from_rgb(41, 53, 70);
const TEXT: Color32 = Color32::from_rgb(233, 240, 250);
const MUTED: Color32 = Color32::from_rgb(144, 159, 181);
const MINT: Color32 = Color32::from_rgb(111, 235, 206);
const LILAC: Color32 = Color32::from_rgb(184, 165, 255);
const RED: Color32 = Color32::from_rgb(255, 137, 147);

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Studio,
    Soundboard,
    Ai,
    Setup,
}

enum Event {
    Devices(Result<(Vec<AudioDevice>, Vec<AudioDevice>), String>),
    Started {
        generation: u64,
        result: Result<AudioEngine, String>,
    },
    Stopped,
    Probed {
        generation: u64,
        result: Result<serde_json::Value, String>,
    },
    RvcInstalled {
        generation: u64,
        result: Result<(), String>,
    },
    AiDevices {
        generation: u64,
        result: Result<serde_json::Value, String>,
    },
    RvcFile {
        index: bool,
        path: Option<PathBuf>,
    },
    ImportProgress {
        current: usize,
        total: usize,
    },
    Imported(Option<ImportReport>),
    Diagnostic(Result<serde_json::Value, String>),
}

#[derive(PartialEq)]
struct AiSettingsKey {
    bundled: bool,
    endpoint: String,
    backend: String,
    device: String,
    model: PathBuf,
    index: PathBuf,
    pitch: i32,
    index_ratio: u32,
    block_size: usize,
}

/// Keep canceled operations owned until their completion arrives. A generation
/// prevents an old result from clearing a later operation or starting audio.
#[derive(Default)]
struct OperationControl {
    generation: u64,
    active: Option<(u64, Arc<AtomicBool>)>,
}

impl OperationControl {
    fn begin(&mut self) -> (u64, Arc<AtomicBool>) {
        self.cancel();
        self.generation = self.generation.wrapping_add(1);
        let stop = Arc::new(AtomicBool::new(false));
        self.active = Some((self.generation, stop.clone()));
        (self.generation, stop)
    }

    fn cancel(&self) {
        if let Some((_, stop)) = &self.active {
            stop.store(true, Ordering::Relaxed);
        }
    }

    // None denotes an unrelated/stale result; false denotes canceled work.
    fn finish(&mut self, generation: u64) -> Option<bool> {
        if self.active.as_ref().map(|(id, _)| *id) != Some(generation) {
            return None;
        }
        self.active
            .take()
            .map(|(_, stop)| !stop.load(Ordering::Relaxed))
    }
}

#[derive(Default)]
struct ImportReport {
    loaded: usize,
    errors: Vec<String>,
}

impl ImportReport {
    fn message(&self) -> String {
        let mut message = format!(
            "{} WAV {} loaded.",
            self.loaded,
            if self.loaded == 1 { "file" } else { "files" }
        );
        if !self.errors.is_empty() {
            message.push('\n');
            message.push_str(&self.errors.join("\n"));
        }
        message
    }
}

fn import_sounds(
    board: &Mutex<Soundboard>,
    paths: &[PathBuf],
    mut progress: impl FnMut(usize, usize),
) -> ImportReport {
    let mut report = ImportReport::default();
    for (index, path) in paths.iter().enumerate() {
        progress(index + 1, paths.len());
        // Decode outside the lock so the UI remains responsive during large imports.
        let result = Soundboard::prepare(path).and_then(|clip| {
            board
                .lock()
                .map_err(|_| "Soundboard unavailable".to_string())?
                .insert(clip)
        });
        match result {
            Ok(_) => report.loaded += 1,
            Err(error) => report.errors.push(format!("{}: {error}", path.display())),
        }
    }
    report
}

pub struct VoiceWarpApp {
    settings: Settings,
    controls: Arc<Mutex<LiveControls>>,
    board: Arc<Mutex<Soundboard>>,
    engine: Option<AudioEngine>,
    sources: Vec<AudioDevice>,
    outputs: Vec<AudioDevice>,
    tab: Tab,
    use_ai: bool,
    monitor: bool,
    busy: Option<String>,
    message: Option<(String, bool)>,
    ai_info: Option<serde_json::Value>,
    ai_devices: Option<serde_json::Value>,
    operation: OperationControl,
    diagnostics: Option<String>,
    tx: mpsc::Sender<Event>,
    rx: mpsc::Receiver<Event>,
    preview: bool,
    screenshot: Option<PathBuf>,
    screenshot_requested: bool,
    opened: Instant,
    shutdown: Arc<AtomicBool>,
    previous_live: bool,
    closing: bool,
}

impl VoiceWarpApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        preview: bool,
        screenshot: Option<PathBuf>,
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = BG;
        style.visuals.window_fill = PANEL;
        style.visuals.override_text_color = Some(TEXT);
        style.visuals.selection.bg_fill = Color32::from_rgb(39, 77, 75);
        style.visuals.selection.stroke = Stroke::new(1.0_f32, MINT);
        style.visuals.widgets.inactive.bg_fill = CARD;
        style.visuals.widgets.inactive.weak_bg_fill = CARD;
        style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
        style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(37, 52, 68);
        style.visuals.widgets.active.bg_fill = Color32::from_rgb(39, 77, 75);
        style.visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, MINT);
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        style.spacing.button_padding = Vec2::new(14.0, 10.0);
        style.spacing.slider_width = 190.0;
        style
            .text_styles
            .insert(egui::TextStyle::Body, FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Button, FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Small, FontId::proportional(12.0));
        cc.egui_ctx.set_style(style);
        let (settings, mut message) = if preview {
            (Settings::default(), None)
        } else {
            match Settings::load() {
                Ok(settings) => (settings, None),
                Err(e) => (Settings::default(), Some((e, true))),
            }
        };
        let mut board = Soundboard::with_removed_builtins(&settings.removed_builtin_sounds);
        board.set_volume(settings.sound_volume);
        for path in &settings.sound_paths {
            if let Err(e) = board.load(path) {
                message = Some((format!("Could not restore {}: {e}", path.display()), true));
            }
        }
        let controls = Arc::new(Mutex::new(LiveControls {
            effects: settings.effects.clone(),
            ..Default::default()
        }));
        let (tx, rx) = mpsc::channel();
        let initial_tab = if AudioEngine::requires_output_device()
            && settings.output_device.is_empty()
            && !preview
        {
            Tab::Setup
        } else {
            Tab::Studio
        };
        let mut app = Self {
            settings,
            controls,
            board: Arc::new(Mutex::new(board)),
            engine: None,
            sources: Vec::new(),
            outputs: Vec::new(),
            tab: initial_tab,
            use_ai: false,
            monitor: false,
            busy: None,
            message,
            ai_info: None,
            ai_devices: None,
            operation: OperationControl::default(),
            diagnostics: None,
            tx,
            rx,
            preview,
            screenshot,
            screenshot_requested: false,
            opened: Instant::now(),
            shutdown,
            previous_live: false,
            closing: false,
        };
        if preview {
            app.sources.push(AudioDevice {
                name: "preview".into(),
                description: "Microphone · UI preview".into(),
            });
            app.settings.source = "preview".into();
        } else {
            app.refresh();
        }
        app
    }

    fn live(&self) -> bool {
        self.engine.as_ref().is_some_and(AudioEngine::running)
    }
    fn routing_ready(&self) -> bool {
        !self.settings.source.is_empty()
            && (!AudioEngine::requires_output_device() || !self.settings.output_device.is_empty())
    }
    fn destination_label(&self) -> String {
        if AudioEngine::requires_output_device() {
            self.outputs
                .iter()
                .find(|d| d.name == self.settings.output_device)
                .map(|d| d.description.clone())
                .unwrap_or_else(|| "Choose a virtual cable".into())
        } else {
            "Voice Warp".into()
        }
    }
    fn destination_help(&self) -> &'static str {
        if cfg!(target_os = "windows") {
            "In your call or game, select the recording end of your virtual cable. For VB-CABLE, this is CABLE Output."
        } else if cfg!(target_os = "macos") {
            "In your call or game, select your loopback device, such as BlackHole 2ch, as the microphone."
        } else {
            "In your call or game, select Voice Warp as the microphone."
        }
    }
    fn stats(&self) -> AudioStats {
        self.engine
            .as_ref()
            .map(AudioEngine::stats)
            .unwrap_or_default()
    }
    fn refresh(&mut self) {
        if self.busy.is_some() || self.live() {
            return;
        }
        self.busy = Some("Finding audio devices…".into());
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let result = AudioEngine::devices()
                .and_then(|sources| AudioEngine::outputs().map(|outputs| (sources, outputs)));
            let _ = tx.send(Event::Devices(result));
        });
    }
    fn start(&mut self) {
        if self.preview || self.busy.is_some() || self.live() || !self.routing_ready() {
            return;
        }
        self.push_controls();
        self.board.lock().unwrap().stop_all();
        let controls = self.controls.clone();
        let board = self.board.clone();
        let source = self.settings.source.clone();
        let make_default = !AudioEngine::requires_output_device() && self.settings.make_default;
        let output_device = if AudioEngine::requires_output_device() {
            Some(self.settings.output_device.clone())
        } else {
            None
        };
        let monitor_sink = if self.monitor {
            Some(self.settings.monitor_sink.clone())
        } else {
            None
        };
        let ai_config = if self.use_ai {
            Some(self.ai_config())
        } else {
            None
        };
        let rvc_config = (self.use_ai && self.settings.ai_bundled).then(|| self.rvc_config());
        let mut engine = self.engine.take().unwrap_or_default();
        let tx = self.tx.clone();
        let (generation, stop) = self.operation.begin();
        self.busy = Some("Connecting virtual microphone…".into());
        self.message = None;
        std::thread::spawn(move || {
            let result = (|| {
                let block_size = ai_config
                    .as_ref()
                    .map(|config| config.block_size)
                    .unwrap_or(960);
                let ai: Option<Box<dyn Processor>> = if let Some(config) = rvc_config {
                    Some(Box::new(RvcProcessor::new_cancellable(
                        config,
                        stop.clone(),
                    )?))
                } else if let Some(config) = ai_config {
                    let processor = AIProcessor::new(config)?;
                    processor.probe()?;
                    Some(Box::new(processor))
                } else {
                    None
                };
                if stop.load(Ordering::Relaxed) {
                    return Err("Startup canceled.".into());
                }
                engine.start(
                    &source,
                    Box::new(MixProcessor::new(controls, board, ai)),
                    block_size,
                    make_default,
                    monitor_sink.as_deref(),
                    output_device.as_deref(),
                )?;
                if stop.load(Ordering::Relaxed) {
                    engine.stop();
                    return Err("Startup canceled.".into());
                }
                Ok(engine)
            })();
            let _ = tx.send(Event::Started { generation, result });
        });
    }
    fn stop(&mut self) {
        self.cancel_operation();
        self.board.lock().unwrap().stop_all();
        if let Some(mut engine) = self.engine.take() {
            self.busy = Some("Disconnecting…".into());
            let tx = self.tx.clone();
            std::thread::spawn(move || {
                engine.stop();
                let _ = tx.send(Event::Stopped);
            });
        }
    }
    fn ai_config(&self) -> AIConfig {
        AIConfig {
            endpoint: self.settings.ai_endpoint.clone(),
            block_size: self.settings.ai_block_size,
            timeout_secs: 3,
        }
    }
    fn cancel_operation(&mut self) {
        if self.operation.active.is_some() {
            self.operation.cancel();
            self.busy = Some("Canceling and releasing the AI worker…".into());
        }
    }

    fn probe(&mut self) {
        if self.preview || self.live() || self.busy.is_some() {
            return;
        }
        let config = self.ai_config();
        let rvc_config = self.settings.ai_bundled.then(|| self.rvc_config());
        let tx = self.tx.clone();
        let (generation, stop) = self.operation.begin();
        self.ai_info = None;
        self.busy = Some("Loading and testing the selected voice…".into());
        std::thread::spawn(move || {
            let result = if let Some(config) = rvc_config {
                RvcProcessor::new_cancellable(config, stop).map(|p| p.info.clone())
            } else {
                AIProcessor::new(config).and_then(|p| p.probe())
            };
            let _ = tx.send(Event::Probed { generation, result });
        });
    }

    fn probe_devices(&mut self) {
        if self.preview || self.live() || self.busy.is_some() {
            return;
        }
        let backend = self.settings.rvc_backend.clone();
        let device = self.settings.rvc_device.clone();
        let tx = self.tx.clone();
        let (generation, stop) = self.operation.begin();
        self.ai_devices = None;
        self.busy = Some("Checking the installed AI runtime and devices…".into());
        std::thread::spawn(move || {
            let result = rvc::diagnostics(&backend, &device, stop);
            let _ = tx.send(Event::AiDevices { generation, result });
        });
    }
    fn rvc_config(&self) -> RvcConfig {
        RvcConfig {
            model: self.settings.rvc_model.clone(),
            index: self.settings.rvc_index.clone(),
            pitch: self.settings.rvc_pitch,
            index_ratio: self.settings.rvc_index_ratio,
            block_size: self.settings.ai_block_size,
            backend: self.settings.rvc_backend.clone(),
            device: self.settings.rvc_device.clone(),
        }
    }
    fn choose_rvc_file(&mut self, index: bool) {
        let tx = self.tx.clone();
        let directory = self.settings.rvc_model.parent().map(PathBuf::from);
        self.busy = Some("Choosing voice model…".into());
        std::thread::spawn(move || {
            let mut dialog = rfd::FileDialog::new();
            if let Some(directory) = directory {
                dialog = dialog.set_directory(directory);
            }
            let path = if index {
                dialog.add_filter("RVC index", &["index"]).pick_file()
            } else {
                dialog
                    .add_filter("RVC voice model", &["pth", "onnx"])
                    .pick_file()
            };
            let _ = tx.send(Event::RvcFile { index, path });
        });
    }
    fn install_rvc(&mut self) {
        if self.preview || self.live() || self.busy.is_some() {
            return;
        }
        let tx = self.tx.clone();
        let backend = self.settings.rvc_backend.clone();
        let (generation, stop) = self.operation.begin();
        self.ai_info = None;
        self.ai_devices = None;
        self.busy =
            Some("Installing the selected AI runtime. Setup can download several GB…".into());
        std::thread::spawn(move || {
            let result = rvc::install_backend(&backend, stop);
            let _ = tx.send(Event::RvcInstalled { generation, result });
        });
    }
    fn push_controls(&self) {
        if let Ok(mut controls) = self.controls.lock() {
            controls.effects = self.settings.effects.clone();
        }
    }
    fn persist_settings(&mut self) -> Result<(), String> {
        if self.preview {
            return Ok(());
        }
        let board = self.board.lock().unwrap();
        self.settings.sound_volume = board.volume();
        self.settings.sound_paths = board
            .list()
            .into_iter()
            .filter_map(|clip| clip.path)
            .collect();
        self.settings.removed_builtin_sounds = board.removed_builtin_sounds();
        drop(board);
        self.settings.save()
    }
    fn save(&mut self) {
        self.message = Some(match self.persist_settings() {
            Ok(()) => ("Settings saved.".into(), false),
            Err(e) => (format!("Could not save settings: {e}"), true),
        });
    }
    fn events(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            let generation = match &event {
                Event::Started { generation, .. }
                | Event::Probed { generation, .. }
                | Event::RvcInstalled { generation, .. }
                | Event::AiDevices { generation, .. } => Some(*generation),
                _ => None,
            };
            if let Some(generation) = generation {
                let accepted = self.operation.finish(generation);
                if accepted == Some(false) || (accepted == Some(true) && self.closing) {
                    // The final cancellation may race with successful audio startup.
                    if let Event::Started {
                        result: Ok(mut engine),
                        ..
                    } = event
                    {
                        self.busy = Some("Disconnecting canceled startup…".into());
                        let tx = self.tx.clone();
                        std::thread::spawn(move || {
                            engine.stop();
                            let _ = tx.send(Event::Stopped);
                        });
                    } else {
                        self.busy = None;
                        self.message = Some(("Operation canceled.".into(), false));
                    }
                    continue;
                }
                if accepted.is_none() {
                    if let Event::Started {
                        result: Ok(mut engine),
                        ..
                    } = event
                    {
                        std::thread::spawn(move || engine.stop());
                    }
                    continue;
                }
            }
            self.busy = None;
            match event {
                Event::RvcInstalled { result, .. } => {
                    self.message = Some(match result {
                        Ok(()) => ("AI engine installed. Choose a voice model, then start your microphone.".into(), false),
                        Err(error) => (error, true),
                    });
                }
                Event::RvcFile { index, path } => {
                    if let Some(path) = path {
                        if index {
                            self.settings.rvc_index = path;
                        } else {
                            self.settings.rvc_model = path;
                            self.settings.rvc_index.clear();
                        }
                        self.ai_info = None;
                        if let Err(error) = self.persist_settings() {
                            self.message = Some((error, true));
                        }
                    }
                }
                Event::Devices(Ok((sources, outputs))) => {
                    self.sources = sources;
                    self.outputs = outputs;
                    if !self.sources.iter().any(|d| d.name == self.settings.source) {
                        self.settings.source = if AudioEngine::requires_output_device() {
                            String::new()
                        } else {
                            self.sources
                                .first()
                                .map(|d| d.name.clone())
                                .unwrap_or_default()
                        };
                    }
                    if !self
                        .outputs
                        .iter()
                        .any(|d| d.name == self.settings.output_device)
                    {
                        self.settings.output_device.clear();
                    }
                    if !self
                        .outputs
                        .iter()
                        .any(|d| d.name == self.settings.monitor_sink)
                    {
                        self.settings.monitor_sink = self
                            .outputs
                            .first()
                            .map(|d| d.name.clone())
                            .unwrap_or_default();
                    }
                    if self.sources.is_empty() {
                        self.message = Some((
                            "No microphones found. Connect one, then refresh devices.".into(),
                            true,
                        ));
                    }
                }
                Event::Started {
                    result: Ok(engine), ..
                } => {
                    self.engine = Some(engine);
                    self.message = if let Some(error) = self.stats().error {
                        Some((error, true))
                    } else if !self.live() {
                        Some((
                            "Audio stopped during startup. Check the selected devices and restart."
                                .into(),
                            true,
                        ))
                    } else {
                        Some((format!("You're live. {}", self.destination_help()), false))
                    };
                }
                Event::Stopped => {
                    self.message = Some((
                        if AudioEngine::requires_output_device() {
                            "Stopped. Audio capture and playback are disconnected; your virtual cable remains installed.".into()
                        } else {
                            "Stopped. The virtual microphone has been removed.".into()
                        },
                        false,
                    ));
                }
                Event::Probed {
                    result: Ok(info), ..
                } => {
                    self.ai_info = Some(info);
                    self.message = Some((
                        "Local AI engine is ready. Choose AI mode, then start your microphone."
                            .into(),
                        false,
                    ));
                }
                Event::AiDevices {
                    result: Ok(info), ..
                } => {
                    self.ai_devices = Some(info);
                    self.message = Some((
                        "Device check finished. Test your selected voice before starting audio."
                            .into(),
                        false,
                    ));
                }
                Event::ImportProgress { current, total } => {
                    self.busy = Some(format!("Importing WAV {current} of {total}…"));
                }
                Event::Imported(Some(mut report)) => {
                    if report.loaded > 0
                        && let Err(error) = self.persist_settings()
                    {
                        report
                            .errors
                            .push(format!("Could not save library: {error}"));
                    }
                    self.message = Some((report.message(), !report.errors.is_empty()));
                }
                Event::Imported(None) => {
                    self.message = Some(("Import cancelled.".into(), false));
                }
                Event::Diagnostic(Ok(info)) => {
                    self.diagnostics = serde_json::to_string_pretty(&info).ok();
                }
                Event::Devices(Err(e))
                | Event::Started { result: Err(e), .. }
                | Event::Probed { result: Err(e), .. }
                | Event::AiDevices { result: Err(e), .. }
                | Event::Diagnostic(Err(e)) => {
                    self.message = Some((e, true));
                }
            }
        }
        let live = self.live();
        if self.previous_live && !live && self.busy.is_none() {
            if let Some(error) = self.stats().error {
                self.message = Some((error, true));
            }
            self.board.lock().unwrap().stop_all();
        }
        self.previous_live = live;
    }

    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sidebar")
            .exact_width(204.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::same(22)),
            )
            .show(ctx, |ui| {
                ui.add_space(13.0);
                ui.horizontal(|ui| {
                    logo(ui, 34.0);
                    ui.label(RichText::new("voice\nwarp").size(22.0).strong());
                });
                ui.add_space(35.0);
                eyebrow(ui, "WORKSPACE");
                ui.add_space(8.0);
                for (tab, label) in [
                    (Tab::Studio, "Voice studio"),
                    (Tab::Soundboard, "Soundboard"),
                    (Tab::Ai, "Local AI"),
                    (Tab::Setup, "Audio setup"),
                ] {
                    let selected = self.tab == tab;
                    let button = egui::Button::new((
                        RichText::new(label).color(if selected { MINT } else { MUTED }),
                        egui::Atom::grow(),
                    ))
                    .fill(if selected {
                        Color32::from_rgb(30, 55, 55)
                    } else {
                        Color32::TRANSPARENT
                    })
                    .stroke(Stroke::NONE)
                    .corner_radius(8);
                    if ui.add_sized([160.0, 44.0], button).clicked() {
                        self.tab = tab;
                    }
                }
                ui.add_space(30.0);
                egui::Frame::new()
                    .fill(CARD)
                    .corner_radius(10)
                    .inner_margin(14)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("ONE MIC. EVERY APP.")
                                .size(10.0)
                                .strong()
                                .color(MINT),
                        );
                        ui.add_space(5.0);
                        ui.label(
                            RichText::new(
                                if AudioEngine::requires_output_device() { "Send your voice to a virtual cable, then use it in calls and games." } else { "Choose Voice Warp in Discord, OBS, games, or your next call." },
                            )
                            .size(12.0)
                            .color(MUTED),
                        );
                    });
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.label(
                        RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                            .size(10.0)
                            .color(MUTED),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(if self.preview {
                            "UI preview · no audio"
                        } else {
                            "Local audio processing"
                        })
                        .size(12.0)
                        .color(MINT),
                    );
                    ui.separator();
                });
            });
    }

    fn header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(match self.tab {
                        Tab::Studio => "Your voice. Reimagined.",
                        Tab::Soundboard => "Make an entrance.",
                        Tab::Ai => "A different kind of voice.",
                        Tab::Setup => "Connect your world.",
                    })
                    .size(29.0)
                    .strong(),
                );
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                pill(
                    ui,
                    if self.preview {
                        "UI PREVIEW"
                    } else if cfg!(target_os = "windows") {
                        "WINDOWS / WASAPI"
                    } else if cfg!(target_os = "macos") {
                        "MACOS / COREAUDIO"
                    } else {
                        "LINUX / PIPEWIRE"
                    },
                    MUTED,
                );
            });
        });
        ui.add_space(14.0);
    }

    fn transport(&mut self, ui: &mut egui::Ui) {
        let live = self.live();
        let stats = self.stats();
        card().show(ui, |ui| {
            ui.horizontal(|ui| {
                let enabled =
                    self.busy.is_none() && !self.preview && (live || self.routing_ready());
                let text = if live {
                    "■  Stop microphone"
                } else {
                    "▶  Start microphone"
                };
                let button = egui::Button::new(RichText::new(text).strong().color(BG))
                    .fill(if live { LILAC } else { MINT })
                    .corner_radius(8);
                if ui
                    .add_enabled(enabled, button.min_size(Vec2::new(190.0, 44.0)))
                    .clicked()
                {
                    if live {
                        self.stop();
                    } else {
                        self.start();
                    }
                }
                if !live
                    && self.engine.is_some()
                    && self.busy.is_none()
                    && ui.small_button("Disconnect silent mic").clicked()
                {
                    self.stop();
                }
                if self.operation.active.is_some() && ui.button("Cancel").clicked() {
                    self.cancel_operation();
                }
                ui.add_space(5.0);
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(if let Some(busy) = &self.busy {
                            busy.as_str()
                        } else if live {
                            "VOICE PROCESSING IS LIVE"
                        } else if self.engine.is_some() {
                            "PROCESSING STOPPED"
                        } else {
                            "READY WHEN YOU ARE"
                        })
                        .size(11.0)
                        .strong()
                        .color(if live { MINT } else { MUTED }),
                    );
                    ui.label(
                        RichText::new(if live {
                            format!("Output: {}", self.destination_label())
                        } else if self.engine.is_some() {
                            "Restart or disconnect the silent microphone".into()
                        } else if !self.routing_ready() && !self.preview {
                            "Choose your devices in Audio setup".into()
                        } else {
                            "Start to send your processed voice to your apps".into()
                        })
                        .size(12.0)
                        .color(MUTED),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut controls = self.controls.lock().unwrap();
                    if ui
                        .selectable_label(
                            controls.master_muted,
                            RichText::new("Mute all").color(if controls.master_muted {
                                RED
                            } else {
                                MUTED
                            }),
                        )
                        .on_hover_text("Silence both your microphone and the soundboard")
                        .clicked()
                    {
                        controls.master_muted = !controls.master_muted;
                    }
                    if ui
                        .selectable_label(
                            controls.mic_muted,
                            RichText::new("Mute voice").color(if controls.mic_muted {
                                RED
                            } else {
                                MUTED
                            }),
                        )
                        .on_hover_text("Keep soundboard clips audible")
                        .clicked()
                    {
                        controls.mic_muted = !controls.mic_muted;
                    }
                });
            });
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(5.0);
            ui.columns(3, |columns| {
                meter(&mut columns[0], "MIC INPUT", stats.input_peak, MINT);
                meter(
                    &mut columns[1],
                    "PROCESSED OUTPUT",
                    stats.output_peak,
                    LILAC,
                );
                columns[2].horizontal(|ui| {
                    ui.label(
                        RichText::new(if self.use_ai { "AI" } else { "DSP" })
                            .strong()
                            .color(MINT),
                    );
                    ui.label(
                        RichText::new(if live {
                            format!(
                                "{:.1} ms processing · {} drops",
                                stats.processing_ms, stats.dropped_blocks
                            )
                        } else {
                            "48 kHz · mono · local".into()
                        })
                        .size(12.0)
                        .color(MUTED),
                    );
                });
            });
        });
    }

    fn studio(&mut self, ui: &mut egui::Ui) {
        ui.add_space(18.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Choose your character").size(19.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                pill(ui, &format!("{} VOICE PRESETS", PRESETS.len()), MUTED);
            });
        });
        ui.add_space(6.0);
        ui.add_enabled_ui(!self.use_ai, |ui| {
            let width = (ui.available_width() - 20.0) / 3.0;
            egui::Grid::new("presets")
                .num_columns(3)
                .spacing([10.0, 10.0])
                .show(ui, |ui| {
                    for (index, preset) in PRESETS.iter().enumerate() {
                        let selected = self.settings.preset == index;
                        let mut text = egui::text::LayoutJob::simple(
                            format!("{}\n{}", preset.name, preset.subtitle),
                            FontId::proportional(14.0),
                            if selected { MINT } else { TEXT },
                            width - 28.0,
                        );
                        text.halign = egui::Align::Min;
                        let button =
                            egui::Button::new((egui::WidgetText::from(text), egui::Atom::grow()))
                                .fill(if selected {
                                    Color32::from_rgb(27, 53, 54)
                                } else {
                                    CARD
                                })
                                .stroke(Stroke::new(
                                    1.0_f32,
                                    if selected {
                                        MINT.gamma_multiply(0.65)
                                    } else {
                                        BORDER
                                    },
                                ))
                                .corner_radius(10);
                        if ui.add_sized([width, 72.0], button).clicked() {
                            self.settings.preset = index;
                            preset.apply(&mut self.settings.effects);
                        }
                        if index % 3 == 2 {
                            ui.end_row();
                        }
                    }
                });
        });
        if self.use_ai {
            ui.label(
                RichText::new("AI mode is selected. Choose your model and pitch in Local AI.")
                    .color(LILAC),
            );
        }
        ui.add_space(16.0);
        card().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Fine-tune your voice").size(16.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_enabled(
                        !self.use_ai,
                        egui::Checkbox::new(&mut self.settings.effects.bypass, "Bypass effects"),
                    );
                });
            });
            ui.add_space(8.0);
            ui.add_enabled_ui(!self.use_ai, |ui| {
                ui.columns(2, |columns| {
                    slider(
                        &mut columns[0],
                        "Pitch",
                        &mut self.settings.effects.pitch,
                        -12.0..=12.0,
                        " st",
                    );
                    slider(
                        &mut columns[0],
                        "Tone",
                        &mut self.settings.effects.tone,
                        -1.0..=1.0,
                        "",
                    )
                    .on_hover_text("Negative values add warmth; positive values brighten your voice. Zero is neutral.");
                    slider(
                        &mut columns[0],
                        "Robot texture",
                        &mut self.settings.effects.robot,
                        0.0..=1.0,
                        "",
                    );
                    slider(
                        &mut columns[0],
                        "Distortion",
                        &mut self.settings.effects.distortion,
                        0.0..=1.0,
                        "",
                    )
                    .on_hover_text("Adds soft saturation and grit. Zero turns distortion off.");
                    slider(
                        &mut columns[0],
                        "Chorus",
                        &mut self.settings.effects.chorus,
                        0.0..=1.0,
                        "",
                    )
                    .on_hover_text("Blends gently shifting copies of your voice for a layered, wavering sound.");
                    slider(
                        &mut columns[1],
                        "Echo",
                        &mut self.settings.effects.echo,
                        0.0..=0.65,
                        "",
                    );
                    slider(
                        &mut columns[1],
                        "Reverb",
                        &mut self.settings.effects.reverb,
                        0.0..=0.65,
                        "",
                    )
                    .on_hover_text("Adds room reflections and a fading tail. Zero turns reverb off.");
                    slider(
                        &mut columns[1],
                        "Voice gain",
                        &mut self.settings.effects.gain_db,
                        -24.0..=12.0,
                        " dB",
                    );
                    slider(
                        &mut columns[1],
                        "Noise gate",
                        &mut self.settings.effects.gate_db,
                        -80.0..=-20.0,
                        " dB",
                    );
                    columns[1].checkbox(&mut self.settings.effects.radio, "Radio filter");
                });
            });
        });
        ui.add_space(16.0);
        self.soundboard(ui, true);
    }

    fn soundboard(&mut self, ui: &mut egui::Ui, compact: bool) {
        let live = self.live();
        let clips = self.board.lock().unwrap().list();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Soundboard").size(19.0).strong());
            ui.label(
                RichText::new(format!("{} sounds", clips.len()))
                    .size(12.0)
                    .color(MUTED),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if compact {
                    if ui.small_button("Open library >").clicked() {
                        self.tab = Tab::Soundboard;
                    }
                } else if ui
                    .add_enabled(
                        !live && self.busy.is_none(),
                        egui::Button::new("+ Import WAVs"),
                    )
                    .on_hover_text(
                        "Select one or more WAV files. Stop the microphone before importing.",
                    )
                    .clicked()
                {
                    self.import();
                }
                if ui.small_button("Stop sounds").clicked() {
                    self.board.lock().unwrap().stop_all();
                }
            });
        });
        if !compact {
            ui.label(RichText::new("Clips play into your virtual microphone, alongside your voice. Use headphones to monitor.").color(MUTED));
            ui.horizontal(|ui| {
                let mut volume = self.board.lock().unwrap().volume();
                ui.label("Sound volume");
                if ui
                    .add(egui::Slider::new(&mut volume, 0.0..=1.5).show_value(true))
                    .changed()
                {
                    self.board.lock().unwrap().set_volume(volume);
                }
                ui.label(
                    RichText::new(
                        "1–9 to play · Esc to stop · shortcuts work while this window is focused",
                    )
                    .size(11.0)
                    .color(MUTED),
                );
            });
            ui.add_space(12.0);
        }
        let width = (ui.available_width() - 30.0) / 4.0;
        let mut remove = None;
        egui::Grid::new(if compact {
            "quick_sounds"
        } else {
            "all_sounds"
        })
        .num_columns(4)
        .spacing([10.0, 10.0])
        .show(ui, |ui| {
            for (index, clip) in clips
                .iter()
                .take(if compact { 4 } else { clips.len() })
                .enumerate()
            {
                ui.vertical(|ui| {
                    let shortcut = if index < 9 {
                        format!("{}", index + 1)
                    } else {
                        "•".into()
                    };
                    let label = format!(
                        "{}   {}\n{:.1}s   {}",
                        shortcut,
                        clip.name,
                        clip.duration_secs,
                        if clip.is_playing {
                            "PLAYING"
                        } else {
                            "ONE SHOT"
                        }
                    );
                    let button = egui::Button::new(
                        RichText::new(label).size(13.0).color(if clip.is_playing {
                            LILAC
                        } else {
                            TEXT
                        }),
                    )
                    .fill(if clip.is_playing {
                        Color32::from_rgb(48, 39, 70)
                    } else {
                        CARD
                    })
                    .stroke(Stroke::new(
                        1.0_f32,
                        if clip.is_playing { LILAC } else { BORDER },
                    ))
                    .corner_radius(9)
                    .min_size(Vec2::new(width, 66.0));
                    if ui
                        .add_enabled(live, button)
                        .on_hover_text("Start the microphone to play sounds into Voice Warp")
                        .clicked()
                    {
                        let _ = self.board.lock().unwrap().trigger(clip.id);
                    }
                    if !compact && ui.small_button("Remove").clicked() {
                        remove = Some(clip.id);
                    }
                });
                if index % 4 == 3 {
                    ui.end_row();
                }
            }
        });
        if let Some(id) = remove {
            self.board.lock().unwrap().remove(id);
            self.message = Some(match self.persist_settings() {
                Ok(()) => ("Sound removed from your library.".into(), false),
                Err(error) => (
                    format!("Sound removed, but could not save library: {error}"),
                    true,
                ),
            });
        }
        if clips.is_empty() {
            ui.label(
                RichText::new("Your soundboard is empty. Import WAV files to add sounds.")
                    .color(MUTED),
            );
        }
        if !compact {
            ui.add_space(18.0);
            card().show(ui, |ui| {
                eyebrow(ui, "YOUR CLIPS, YOUR LIBRARY");
                ui.label(RichText::new("Select multiple WAV files to import them together. There is no limit on the number of sounds in your library. Each PCM or float WAV can be up to 60 seconds. Stereo files are mixed to mono and resampled automatically. Removing a sound leaves its original file in place.").color(MUTED));
            });
        }
    }

    fn import(&mut self) {
        let board = self.board.clone();
        let tx = self.tx.clone();
        self.busy = Some("Choose WAV sounds…".into());
        std::thread::spawn(move || {
            let result = rfd::FileDialog::new()
                .add_filter("WAV audio", &["wav"])
                .pick_files()
                .map(|paths| {
                    import_sounds(&board, &paths, |current, total| {
                        let _ = tx.send(Event::ImportProgress { current, total });
                    })
                });
            let _ = tx.send(Event::Imported(result));
        });
    }

    fn ai_page(&mut self, ui: &mut egui::Ui) {
        let previous = self.ai_settings_signature();
        ui.add_space(18.0);
        card().show(ui, |ui| {
            ui.horizontal(|ui| {
                pill(ui, "LOCAL RVC", LILAC);
                ui.label(RichText::new("Your AI voice").size(18.0).strong());
            });
            ui.label("Choose a voice model. Voice Warp starts the engine with your microphone and stops it when you disconnect.");
            ui.add_space(10.0);
            ui.add_enabled_ui(!self.preview && !self.live() && self.busy.is_none(), |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.use_ai, false, "Built-in effects");
                    ui.selectable_value(&mut self.use_ai, true, "AI voice");
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.settings.ai_bundled, true, "Bundled engine");
                    ui.selectable_value(&mut self.settings.ai_bundled, false, "External server");
                });
                if self.settings.ai_bundled {
                    if rvc::supported() {
                        ui.horizontal(|ui| {
                            ui.label("Processing");
                            egui::ComboBox::from_id_salt("rvc_backend")
                                .selected_text(backend_label(&self.settings.rvc_backend))
                                .show_ui(ui, |ui| {
                                    for backend in ["cpu", "auto", "cuda", "rocm", "directml", "mps"] {
                                        ui.add_enabled_ui(backend_for_platform(backend), |ui| {
                                            if ui.selectable_value(&mut self.settings.rvc_backend, backend.into(), backend_label(backend)).changed() {
                                                self.settings.rvc_device = "0".into();
                                            }
                                        });
                                    }
                                });
                        });
                        ui.label(RichText::new("GPU choices require a compatible runtime and driver. Auto selects within an installed runtime; a device check does not verify your voice model.").size(12.0).color(MUTED));
                        if cfg!(target_os = "macos") {
                            ui.label(RichText::new("Apple GPU acceleration uses Metal. Current NVIDIA CUDA is unavailable on macOS; Intel Mac GPU support is not qualified.").size(12.0).color(MUTED));
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Install / repair AI engine").clicked() { self.install_rvc(); }
                            if ui.button("Check available devices").clicked() { self.probe_devices(); }
                        });
                        ui.label(RichText::new(if cfg!(target_os = "linux") {
                            "Setup requires Podman and downloads the selected runtime and shared weights. Voice models are selected separately."
                        } else {
                            "The managed runtime runs locally without a separate server or Python installation. A matching runtime package must be available for this platform."
                        }).size(12.0).color(MUTED));
                        ui.horizontal(|ui| {
                            ui.label("Device ID");
                            ui.add(egui::TextEdit::singleline(&mut self.settings.rvc_device).desired_width(200.0));
                            if let Some(devices) = self.ai_devices.as_ref().and_then(|info| info["devices"].as_array()) {
                                egui::ComboBox::from_id_salt("rvc_device")
                                    .selected_text("Choose detected device")
                                    .show_ui(ui, |ui| {
                                        for device in devices {
                                            if self.settings.rvc_backend != "auto"
                                                && device["backend"].as_str().is_some_and(|backend| backend != self.settings.rvc_backend) {
                                                continue;
                                            }
                                            if let Some(id) = device["id"].as_str() {
                                                let name = device["name"].as_str().unwrap_or(id);
                                                ui.selectable_value(&mut self.settings.rvc_device, id.to_string(), name);
                                            }
                                        }
                                    });
                            }
                        });
                        ui.label(RichText::new("Use a device ID from the runtime check; 0 selects its first device.").size(12.0).color(MUTED));
                        ui.horizontal(|ui| {
                            if ui.button("Choose voice model…").clicked() { self.choose_rvc_file(false); }
                            ui.label(self.settings.rvc_model.file_name().unwrap_or_default().to_string_lossy());
                        });
                        ui.label(RichText::new("RVC .pth or compatible .onnx. Choose files from a trusted model creator.").size(12.0).color(MUTED));
                        ui.horizontal(|ui| {
                            if ui.button("Choose matching index…").clicked() { self.choose_rvc_file(true); }
                            if self.settings.rvc_index.as_os_str().is_empty() {
                                ui.label("Optional");
                            } else {
                                ui.label(self.settings.rvc_index.file_name().unwrap_or_default().to_string_lossy());
                                if ui.small_button("Clear").clicked() { self.settings.rvc_index.clear(); }
                            }
                        });
                        ui.add(egui::Slider::new(&mut self.settings.rvc_pitch, -24..=24).text("Pitch (semitones)"));
                        ui.add_enabled(!self.settings.rvc_index.as_os_str().is_empty(),
                            egui::Slider::new(&mut self.settings.rvc_index_ratio, 0.0..=1.0).text("Index strength"));
                    } else {
                        ui.label("A managed AI runtime is not available for this operating system or architecture. External server compatibility remains available.");
                    }
                } else {
                    ui.label("Compatible w-okada v1 server address (loopback only)");
                    ui.add(egui::TextEdit::singleline(&mut self.settings.ai_endpoint).desired_width(450.0));
                }
                ui.horizontal(|ui| {
                    ui.label("Audio chunk");
                    egui::ComboBox::from_id_salt("ai_chunk").selected_text(format!("{} samples / {:.0} ms", self.settings.ai_block_size, self.settings.ai_block_size as f32 / 48.0)).show_ui(ui, |ui| {
                        for size in [4096, 6144, 8192, 12288] { ui.selectable_value(&mut self.settings.ai_block_size, size, format!("{} samples / {:.0} ms", size, size as f32 / 48.0)); }
                    });
                });
                if ui.button("Test voice engine").clicked() { self.probe(); }
            });
            if let Some(info) = &self.ai_info {
                ui.add_space(8.0);
                ui.label(RichText::new(format!("Last successful test: {}", info["model_name"].as_str().unwrap_or("RVC model"))).color(MINT));
                ui.label(RichText::new(format!("Device: {} · 48 kHz · local inference", info["device"].as_str().unwrap_or("Reported by server"))).size(12.0).color(MUTED));
                if let Some(backend) = info["backend"].as_str() {
                    ui.label(RichText::new(format!("Active processing: {backend}")).size(12.0).color(MUTED));
                }
                if let Some(version) = info["runtime_version"].as_str() {
                    ui.label(RichText::new(format!("Runtime: {version}")).size(12.0).color(MUTED));
                }
                if let Some(stages) = info["stages"].as_object() {
                    for (stage, placement) in stages {
                        let placement = placement.as_str().map(str::to_owned).unwrap_or_else(|| placement.to_string());
                        ui.label(RichText::new(format!("{stage}: {placement}")).size(12.0).color(MUTED));
                    }
                }
                if let Some(reason) = info["fallback_reason"].as_str() {
                    ui.label(RichText::new(reason).size(12.0).color(MUTED));
                }
            }
            if let Some(info) = &self.ai_devices {
                egui::CollapsingHeader::new("Runtime and device diagnostics").show(ui, |ui| {
                    ui.monospace(serde_json::to_string_pretty(info).unwrap_or_default());
                });
            }
        });
        if previous != self.ai_settings_signature() {
            self.ai_info = None;
            // A backend change invalidates enumeration from the previous runtime.
            if previous.backend != self.settings.rvc_backend {
                self.ai_devices = None;
            }
        }
        ui.add_space(16.0);
        card().show(ui, |ui| {
            ui.label(RichText::new("Performance").size(17.0).strong());
            ui.label("Larger audio chunks improve throughput but add latency. GPU processing can still include CPU stages. Test the selected voice and check its actual device before starting a call.");
            ui.label("Your microphone stays on this computer. Conversion errors stop AI audio. Explicit GPU selection never silently switches to CPU.");
            ui.label(RichText::new(format!("Setup and engine logs: {}", rvc::runtime_dir().display())).size(12.0).color(MUTED));
        });
    }

    fn ai_settings_signature(&self) -> AiSettingsKey {
        AiSettingsKey {
            bundled: self.settings.ai_bundled,
            endpoint: self.settings.ai_endpoint.clone(),
            backend: self.settings.rvc_backend.clone(),
            device: self.settings.rvc_device.clone(),
            model: self.settings.rvc_model.clone(),
            index: self.settings.rvc_index.clone(),
            pitch: self.settings.rvc_pitch,
            index_ratio: self.settings.rvc_index_ratio.to_bits(),
            block_size: self.settings.ai_block_size,
        }
    }

    fn setup(&mut self, ui: &mut egui::Ui) {
        ui.add_space(18.0);
        if AudioEngine::requires_output_device() {
            card().show(ui, |ui| {
                ui.label(RichText::new("Connect a virtual audio cable").size(17.0).strong());
                if cfg!(target_os = "windows") {
                    ui.label(RichText::new("Install a virtual audio driver once, then choose its playback endpoint below. With VB-CABLE, send to CABLE Input here and select CABLE Output as the microphone in your other apps.").color(MUTED));
                    ui.hyperlink_to("Get VB-CABLE from VB-Audio", "https://vb-audio.com/Cable/");
                } else {
                    ui.label(RichText::new("Install a loopback audio device such as BlackHole 2ch. Choose it as the output here and as the microphone in your other apps. Allow microphone access when macOS asks.").color(MUTED));
                    ui.hyperlink_to("Get BlackHole from Existential Audio", "https://existential.audio/blackhole/");
                }
                ui.label(RichText::new("The driver is installed separately. Voice Warp never changes your default devices on this platform.").size(12.0).color(MUTED));
            });
            ui.add_space(14.0);
        }
        card().show(ui, |ui| {
            ui.label(RichText::new("Audio routing").size(18.0).strong());
            ui.add_space(8.0);
            ui.add_enabled_ui(!self.live() && self.busy.is_none(), |ui| {
                ui.label(RichText::new("PHYSICAL MICROPHONE").size(11.0).color(MUTED));
                let selected = self.sources.iter().find(|d| d.name == self.settings.source).map(|d| d.description.as_str()).unwrap_or("Choose your physical microphone");
                egui::ComboBox::from_id_salt("input").width(500.0).selected_text(selected).show_ui(ui, |ui| { for device in &self.sources { ui.selectable_value(&mut self.settings.source, device.name.clone(), &device.description); } });
                if AudioEngine::requires_output_device() {
                    ui.add_space(5.0);
                    ui.label(RichText::new("VIRTUAL CABLE / PLAYBACK DESTINATION").size(11.0).color(MUTED));
                    let selected = self.outputs.iter().find(|d| d.name == self.settings.output_device).map(|d| d.description.as_str()).unwrap_or("Choose the virtual cable playback endpoint");
                    egui::ComboBox::from_id_salt("virtual_output").width(500.0).selected_text(selected).show_ui(ui, |ui| {
                        for device in &self.outputs { ui.selectable_value(&mut self.settings.output_device, device.name.clone(), &device.description); }
                    });
                    if self.settings.monitor_sink == self.settings.output_device { self.settings.monitor_sink.clear(); }
                    ui.label(RichText::new("This destination receives both your changed voice and soundboard. A physical speaker output is audible locally; a virtual cable delivers audio to other apps.").size(12.0).color(MUTED));
                } else {
                    ui.checkbox(&mut self.settings.make_default, "Make Voice Warp the default microphone while running");
                    ui.label(RichText::new("You can also select Voice Warp directly in each app. Existing calls may need their input reselected.").size(12.0).color(MUTED));
                }
                ui.add_space(6.0);
                ui.checkbox(&mut self.monitor, "Hear myself through headphones");
                if self.monitor {
                    let selected = self.outputs.iter().find(|d| d.name == self.settings.monitor_sink).map(|d| d.description.as_str()).unwrap_or("Choose headphones");
                    egui::ComboBox::from_id_salt("output").width(500.0).selected_text(selected).show_ui(ui, |ui| { for device in &self.outputs {
                        if !AudioEngine::requires_output_device() || device.name != self.settings.output_device {
                            ui.selectable_value(&mut self.settings.monitor_sink, device.name.clone(), &device.description);
                        }
                    } });
                    ui.label(RichText::new("Use headphones to avoid microphone feedback.").size(12.0).color(LILAC));
                }
                ui.add_space(8.0);
                if ui.button("Refresh audio devices").clicked() && !self.preview { self.refresh(); }
            });
        });
        ui.add_space(16.0);
        card().show(ui, |ui| {
            ui.label(RichText::new("Works where you talk").size(17.0).strong());
            ui.label(RichText::new("1. Pick your physical microphone and routing above.").color(MUTED));
            ui.label(RichText::new("2. Choose a voice preset, then start the microphone.").color(MUTED));
            ui.label(RichText::new(format!("3. {}", self.destination_help())).color(MUTED));
            ui.add_space(6.0);
            ui.label(RichText::new(if AudioEngine::requires_output_device() {
                "Closing the app stops recording and playback. Your independently installed virtual cable remains available. If a device disconnects or processing fails, the audio stops; it never falls back to your raw microphone."
            } else { "Closing the app removes its virtual microphone and restores the previous default if Voice Warp is still selected. It does not rewrite PipeWire configuration." }).size(12.0).color(MUTED));
        });
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            if ui.button("Save settings").clicked() {
                self.save();
            }
            if ui
                .add_enabled(
                    self.busy.is_none() && !self.preview && !self.live(),
                    egui::Button::new("Run diagnostics"),
                )
                .clicked()
            {
                self.busy = Some("Checking audio…".into());
                let tx = self.tx.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(Event::Diagnostic(AudioEngine::diagnostics()));
                });
            }
        });
        if let Some(diagnostics) = &self.diagnostics {
            ui.add_space(8.0);
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    ui.monospace(diagnostics);
                });
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.board.lock().unwrap().stop_all();
        }
        if !self.live() {
            return;
        }
        let keys = [
            egui::Key::Num1,
            egui::Key::Num2,
            egui::Key::Num3,
            egui::Key::Num4,
            egui::Key::Num5,
            egui::Key::Num6,
            egui::Key::Num7,
            egui::Key::Num8,
            egui::Key::Num9,
        ];
        let mut board = self.board.lock().unwrap();
        let clips = board.list();
        for (index, key) in keys.iter().enumerate() {
            if ctx.input(|input| !input.modifiers.any() && input.key_pressed(*key))
                && let Some(clip) = clips.get(index)
            {
                let _ = board.trigger(clip.id);
            }
        }
    }
}

impl eframe::App for VoiceWarpApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.events();
        self.shortcuts(ctx);
        if self.shutdown.load(Ordering::Relaxed)
            || ctx.input(|input| input.viewport().close_requested())
        {
            self.closing = true;
        }
        if self.closing {
            self.cancel_operation();
            if self.busy.is_some() || self.engine.is_some() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                if self.busy.is_none() {
                    self.stop();
                }
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if let Some(path) = &self.screenshot {
            for event in ctx.input(|input| input.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = event {
                    let pixels: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                    if let Err(error) = image::save_buffer(
                        path,
                        &pixels,
                        image.width() as u32,
                        image.height() as u32,
                        image::ColorType::Rgba8,
                    ) {
                        eprintln!("Screenshot: {error}");
                    }
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            if !self.screenshot_requested && self.opened.elapsed() > Duration::from_millis(600) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                self.screenshot_requested = true;
            }
        }
        self.sidebar(ctx);
        egui::TopBottomPanel::bottom("footer")
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(24, 9)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let source = self
                        .sources
                        .iter()
                        .find(|d| d.name == self.settings.source)
                        .map(|d| d.description.as_str())
                        .unwrap_or("Choose a microphone in Audio setup");
                    ui.label(RichText::new(source).size(11.0).color(MUTED));
                    ui.label(
                        RichText::new("  >  Voice Warp  >  Your apps")
                            .size(11.0)
                            .color(MINT),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(if self.use_ai {
                                "LOCAL AI"
                            } else {
                                "LIVE EFFECTS"
                            })
                            .size(10.0)
                            .color(MUTED),
                        );
                    });
                });
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG).inner_margin(Margin::same(26)))
            .show(ctx, |ui| {
                self.header(ui);
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.transport(ui);
                        if let Some((message, error)) = self.message.clone() {
                            ui.add_space(8.0);
                            egui::Frame::new()
                                .fill(if error {
                                    Color32::from_rgb(54, 30, 38)
                                } else {
                                    Color32::from_rgb(25, 43, 43)
                                })
                                .corner_radius(7)
                                .inner_margin(10)
                                .show(ui, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            RichText::new(message).size(12.0).color(if error {
                                                RED
                                            } else {
                                                MINT
                                            }),
                                        );
                                        if ui.small_button("Dismiss").clicked() {
                                            self.message = None;
                                        }
                                    });
                                });
                        }
                        match self.tab {
                            Tab::Studio => self.studio(ui),
                            Tab::Soundboard => {
                                ui.add_space(18.0);
                                self.soundboard(ui, false);
                            }
                            Tab::Ai => self.ai_page(ui),
                            Tab::Setup => self.setup(ui),
                        }
                        ui.add_space(8.0);
                    });
            });
        self.push_controls();
        ctx.request_repaint_after(Duration::from_millis(if self.live() { 33 } else { 100 }));
    }
}

impl Drop for VoiceWarpApp {
    fn drop(&mut self) {
        self.operation.cancel();
        if let Some(engine) = &mut self.engine {
            engine.stop();
        }
        self.save();
    }
}

fn backend_label(backend: &str) -> &str {
    match backend {
        "cpu" => "CPU",
        "auto" => "Auto (installed runtime)",
        "cuda" => "NVIDIA · CUDA",
        "rocm" => "AMD · ROCm",
        "directml" => "Windows GPU · DirectML",
        "mps" => "Apple GPU · Metal",
        other => other,
    }
}

fn backend_for_platform(backend: &str) -> bool {
    match backend {
        "cpu" | "auto" => true,
        "cuda" | "rocm" => cfg!(any(target_os = "linux", target_os = "windows")),
        "directml" => cfg!(target_os = "windows"),
        "mps" => cfg!(all(target_os = "macos", target_arch = "aarch64")),
        _ => false,
    }
}

fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(12)
        .inner_margin(18)
}
fn eyebrow(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).size(10.0).color(MUTED).strong());
}
fn pill(ui: &mut egui::Ui, text: &str, color: Color32) {
    egui::Frame::new()
        .fill(CARD)
        .corner_radius(5)
        .inner_margin(Margin::symmetric(9, 5))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(10.0).color(color).strong());
        });
}
fn slider(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
) -> egui::Response {
    // Keep the label and numeric editor inside each column at narrow widths.
    let track_width =
        (ui.available_width() - 98.0 - 90.0 - 2.0 * ui.spacing().item_spacing.x).clamp(60.0, 190.0);
    ui.scope(|ui| {
        ui.spacing_mut().slider_width = track_width;
        ui.horizontal(|ui| {
            ui.add_sized(
                [98.0, 20.0],
                egui::Label::new(RichText::new(label).size(12.0).color(MUTED)),
            );
            ui.add(
                egui::Slider::new(value, range)
                    .suffix(suffix)
                    .max_decimals(1),
            );
        })
        .response
    })
    .inner
}
fn meter(ui: &mut egui::Ui, label: &str, peak: f32, color: Color32) {
    ui.horizontal(|ui| {
        eyebrow(ui, label);
        ui.label(
            RichText::new(if peak > 0.00001 {
                format!("{:.0} dB", 20.0 * peak.log10())
            } else {
                "−∞ dB".into()
            })
            .size(10.0)
            .color(MUTED),
        );
    });
    let level = if peak > 0.0 {
        ((20.0 * peak.log10() + 60.0) / 60.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    ui.add(
        egui::ProgressBar::new(level)
            .desired_height(4.0)
            .fill(if peak >= 0.98 { RED } else { color }),
    );
}
fn logo(ui: &mut egui::Ui, size: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(9), MINT);
    for (index, height) in [10.0, 22.0, 16.0, 8.0].iter().enumerate() {
        let x = rect.left() + 9.0 + index as f32 * 5.5;
        ui.painter().line_segment(
            [
                egui::pos2(x, rect.center().y - height / 2.0),
                egui::pos2(x, rect.center().y + height / 2.0),
            ],
            Stroke::new(2.5_f32, BG),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canceled_or_stale_background_results_cannot_activate_a_session() {
        let mut operation = OperationControl::default();
        let (first, first_stop) = operation.begin();
        operation.cancel();
        assert!(first_stop.load(Ordering::Relaxed));
        assert_eq!(operation.finish(first), Some(false));
        assert!(operation.active.is_none());

        let (second, second_stop) = operation.begin();
        assert_ne!(first, second);
        assert_eq!(operation.finish(first), None);
        assert!(operation.active.is_some());
        assert!(!second_stop.load(Ordering::Relaxed));
        assert_eq!(operation.finish(second), Some(true));
        assert_eq!(operation.finish(second), None);
    }

    #[test]
    fn batch_import_continues_after_errors_and_reuses_duplicate_paths() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("voice-warp-batch-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let mut paths = Vec::new();
        for index in 0..24 {
            let path = directory.join(format!("sound-{index}.wav"));
            let mut writer = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            writer.write_sample(8192_i16).unwrap();
            writer.finalize().unwrap();
            paths.push(path);
        }
        let invalid = directory.join("broken.wav");
        std::fs::write(&invalid, b"not WAV audio").unwrap();
        paths.insert(1, invalid);
        paths.push(paths[0].clone());
        let board = Mutex::new(Soundboard::new());
        let mut progress = Vec::new();
        let report = import_sounds(&board, &paths, |current, total| {
            progress.push((current, total));
        });
        std::fs::remove_dir_all(directory).unwrap();

        assert_eq!(report.loaded, 25);
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].contains("broken.wav"));
        assert_eq!(
            progress,
            (1..=26).map(|current| (current, 26)).collect::<Vec<_>>()
        );
        let mut board = board.lock().unwrap();
        let clips = board.list();
        assert_eq!(clips.len(), 24);
        assert_eq!(clips.first().unwrap().name, "sound-0");
        assert_eq!(clips.last().unwrap().name, "sound-23");
        board.trigger(clips.last().unwrap().id).unwrap();
        let mut output = [0.0];
        board.mix_into(&mut output);
        assert!(output[0] > 0.0);
    }
}
