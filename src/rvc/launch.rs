use super::{process::ManagedChild, runtime, runtime_dir};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

static SERIAL: AtomicU64 = AtomicU64::new(0);
pub const IMAGE: &str = "localhost/voice-warp-rvc:2-cpu";

pub fn podman() -> Command {
    if std::env::var_os("CONTAINER_ID").is_some() && Path::new("/run/.containerenv").exists() {
        let mut command = Command::new("distrobox-host-exec");
        command.arg("podman");
        command
    } else {
        Command::new("podman")
    }
}

pub fn validate_backend(backend: &str) -> Result<(), String> {
    if !["auto", "cpu", "cuda", "rocm", "mps", "directml"].contains(&backend) {
        return Err("Unknown AI backend.".into());
    }
    let allowed = match std::env::consts::OS {
        "linux" => ["auto", "cpu", "cuda", "rocm"].contains(&backend),
        "windows" => ["auto", "cpu", "cuda", "rocm", "directml"].contains(&backend),
        "macos" => ["auto", "cpu", "mps"].contains(&backend),
        _ => false,
    };
    if !allowed {
        return Err(format!(
            "The {backend} backend is unavailable on this operating system."
        ));
    }
    Ok(())
}

pub fn image(backend: &str) -> String {
    format!("localhost/voice-warp-rvc:2-{backend}")
}

fn image_exists(backend: &str, stop: &AtomicBool) -> Result<bool, String> {
    let mut command = podman();
    command
        .args(["image", "exists", &image(backend)])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    Ok(ManagedChild::spawn(&mut command)?
        .wait_cancellable(stop, Duration::from_secs(15))?
        .success())
}

pub fn install(backend: &str, stop: &AtomicBool) -> Result<(), String> {
    validate_backend(backend)?;
    if !cfg!(target_os = "linux") {
        return runtime::install_bundled(backend, stop);
    }
    let backend = if backend == "auto" { "cpu" } else { backend };
    let dir = runtime_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let lock = File::create(dir.join("install.lock")).map_err(|e| e.to_string())?;
    fs2::FileExt::try_lock_exclusive(&lock).map_err(|_| "AI installation is already running.")?;
    for (name, content) in [
        ("Containerfile", include_str!("../../engine/Containerfile")),
        (
            "requirements.txt",
            include_str!("../../engine/requirements.txt"),
        ),
        ("worker.py", include_str!("../../engine/worker.py")),
        (
            "voice_warp_backend.py",
            include_str!("../../engine/voice_warp_backend.py"),
        ),
        ("NOTICE.md", include_str!("../../engine/NOTICE.md")),
        (
            "UPSTREAM-LICENSE",
            include_str!("../../engine/UPSTREAM-LICENSE"),
        ),
    ] {
        fs::write(dir.join(name), content).map_err(|e| e.to_string())?;
    }
    let log = File::create(dir.join("install.log")).map_err(|e| e.to_string())?;
    let mut command = podman();
    command
        .args([
            "build",
            "--build-arg",
            &format!("AI_BACKEND={backend}"),
            "-t",
            &image(backend),
            "-f",
        ])
        .arg(dir.join("Containerfile"))
        .arg(&dir)
        .stdin(Stdio::null())
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log);
    let status =
        ManagedChild::spawn(&mut command)?.wait_cancellable(stop, Duration::from_secs(7200))?;
    if !status.success() {
        return Err(format!(
            "AI engine installation failed. See {}",
            dir.join("install.log").display()
        ));
    }
    Ok(())
}

pub struct Launch {
    pub command: Command,
    pub container: Option<String>,
    pub lease: Option<runtime::InstalledRuntime>,
    pub init: Value,
}

pub fn worker(
    backend: &str,
    device: &str,
    model: Option<&Path>,
    index: Option<&Path>,
    stop: &AtomicBool,
) -> Result<Launch, String> {
    validate_backend(backend)?;
    if device.len() > 256 || device.contains(['\0', '\n', '\r']) {
        return Err("Invalid AI device identifier.".into());
    }
    if !cfg!(target_os = "linux") {
        let runtime = runtime::locate(backend)?;
        let mut command = Command::new(&runtime.executable);
        command.current_dir(&runtime.root);
        clean_environment(&mut command);
        let init = json!({"protocol":2,"action":if model.is_some(){"init"}else{"probe"},
            "backend":backend,"device":device,"model_path":model,"index_path":index,"assets_dir":runtime.assets_dir});
        return Ok(Launch {
            command,
            container: None,
            init,
            lease: Some(runtime),
        });
    }
    // Auto uses a known installed CPU baseline. It never downloads or guesses
    // a vendor runtime; choosing a GPU backend is an explicit installation.
    let selected = if backend == "auto" { "cpu" } else { backend };
    if !image_exists(selected, stop)? {
        return Err(format!(
            "The {selected} AI runtime is not installed. Open Local AI and install / repair this backend."
        ));
    }
    let id = format!(
        "voice-warp-rvc-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    );
    let mut command = podman();
    command.args([
        "run",
        "--rm",
        "-i",
        "--pull=never",
        "--network=none",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--security-opt=label=disable",
        "--name",
        &id,
    ]);
    gpu_args(&mut command, selected)?;
    let mut model_path = Value::Null;
    let mut index_path = Value::Null;
    if let Some(path) = model {
        let target = format!(
            "/models/0/model.{}",
            path.extension().unwrap().to_string_lossy()
        );
        mount(&mut command, path, &target)?;
        model_path = json!(target);
    }
    if let Some(path) = index {
        mount(&mut command, path, "/models/0/voice.index")?;
        index_path = json!("/models/0/voice.index");
    }
    command.arg(image(selected));
    Ok(Launch {
        command,
        container: Some(id),
        lease: None,
        init: json!({"protocol":2,
        "action":if model.is_some(){"init"}else{"probe"},"backend":selected,"device":device,
        "model_path":model_path,"index_path":index_path,"assets_dir":"/assets"}),
    })
}

fn mount(command: &mut Command, path: &Path, target: &str) -> Result<(), String> {
    if path.to_string_lossy().contains(',') {
        return Err(
            "Move the model/index to a path without commas for the Linux container mount.".into(),
        );
    }
    command.arg("--mount").arg(format!(
        "type=bind,src={},dst={target},ro=true",
        path.display()
    ));
    Ok(())
}

fn gpu_args(command: &mut Command, backend: &str) -> Result<(), String> {
    match backend {
        // Enumerate inside the same container used for inference. Device
        // selection is verified by stable identifier in the worker.
        "cuda" => {
            command.args(["--device=nvidia.com/gpu=all"]);
        }
        "rocm" => {
            command.args([
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--group-add=keep-groups",
            ]);
        }
        "cpu" => {}
        _ => return Err("This backend has no Linux container variant.".into()),
    }
    Ok(())
}

pub fn clean_environment(command: &mut Command) {
    for name in [
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONINSPECT",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ] {
        command.env_remove(name);
    }
    command
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "0");
}

pub fn remove_container(name: &str) {
    let mut command = podman();
    command
        .args(["rm", "--force", "--time=0", "--ignore", name])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Ok(mut process) = ManagedChild::spawn(&mut command) {
        let _ = process.wait_cancellable(&AtomicBool::new(false), Duration::from_secs(3));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_launch_does_not_grant_privileged_access_or_use_a_shell() {
        for backend in ["cpu", "cuda", "rocm"] {
            let mut command = Command::new("podman");
            gpu_args(&mut command, backend).unwrap();
            let args: Vec<_> = command
                .get_args()
                .map(|v| v.to_string_lossy().into_owned())
                .collect();
            assert!(!args.iter().any(|v| v == "--privileged"));
            assert_eq!(args.is_empty(), backend == "cpu");
        }
    }
    #[test]
    fn mount_rejects_container_option_injection_and_preserves_spaces() {
        let mut command = Command::new("podman");
        assert!(
            mount(
                &mut command,
                Path::new("/tmp/model,ro=false.pth"),
                "/model.pth"
            )
            .is_err()
        );
        mount(&mut command, Path::new("/tmp/my model.pth"), "/model.pth").unwrap();
        assert_eq!(command.get_args().count(), 2);
    }
}
