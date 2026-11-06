//! Versioned native runtime installation. Catalog hashes are part of the application;
//! local developer packages require an explicitly supplied manifest digest.
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

const MAX_MANIFEST: u64 = 4 * 1024 * 1024;
const MAX_ARCHIVE: u64 = 16 * 1024 * 1024 * 1024;
const MAX_UNPACKED: u64 = 32 * 1024 * 1024 * 1024;
static SERIAL: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub protocol: u32,
    pub id: String,
    pub version: String,
    pub os: String,
    pub arch: String,
    pub backend: String,
    pub minimum_os: String,
    pub executable: String,
    pub assets_dir: String,
    pub archive: String,
    pub archive_sha256: String,
    pub archive_bytes: u64,
    pub unpacked_bytes: u64,
    pub files: BTreeMap<String, String>,
    pub packages: BTreeMap<String, String>,
    pub upstream_revision: String,
    pub qualification: String,
}

#[derive(Deserialize)]
struct Catalog {
    schema: u32,
    releases: Vec<Release>,
}

#[derive(Deserialize)]
struct Release {
    os: String,
    arch: String,
    backend: String,
    manifest_url: String,
    manifest_sha256: String,
}

#[derive(Default, Deserialize, Serialize)]
struct Active {
    current: String,
    previous: Option<String>,
}

pub struct InstalledRuntime {
    pub root: PathBuf,
    pub executable: PathBuf,
    pub assets_dir: PathBuf,
    pub manifest: Manifest,
    _lease: LockGuard,
}

/// A lock belongs to this guard, not to every incidental copy of its descriptor.
/// Explicit unlock matters on Unix: fork/dup can keep the same open file
/// description alive after the owning File is closed, even with CLOEXEC set.
struct LockGuard(File);

impl LockGuard {
    fn shared(file: File) -> std::io::Result<Self> {
        FileExt::try_lock_shared(&file)?;
        Ok(Self(file))
    }

    fn exclusive(file: File) -> std::io::Result<Self> {
        file.try_lock_exclusive()?;
        Ok(Self(file))
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

pub fn data_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "voice-warp")
        .map(|d| d.data_local_dir().join("ai-runtimes"))
        .unwrap_or_else(|| crate::config::config_dir().join("ai-runtimes"))
}

fn check_stop(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Relaxed) {
        Err("AI runtime installation canceled.".into())
    } else {
        Ok(())
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 160
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
}

fn normalized_backend(backend: &str) -> Result<&str, String> {
    match backend {
        "auto" => Ok("cpu"),
        "cpu" | "cuda" | "rocm" | "mps" | "directml" => Ok(backend),
        _ => Err("Unknown AI runtime backend.".into()),
    }
}

fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 4096
        && !path.contains(['\\', ':', '\0'])
        && path.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && !part.ends_with(['.', ' '])
                && !matches!(
                    part.split('.')
                        .next()
                        .unwrap_or("")
                        .to_ascii_uppercase()
                        .as_str(),
                    "CON"
                        | "PRN"
                        | "AUX"
                        | "NUL"
                        | "COM1"
                        | "COM2"
                        | "COM3"
                        | "COM4"
                        | "COM5"
                        | "COM6"
                        | "COM7"
                        | "COM8"
                        | "COM9"
                        | "LPT1"
                        | "LPT2"
                        | "LPT3"
                        | "LPT4"
                        | "LPT5"
                        | "LPT6"
                        | "LPT7"
                        | "LPT8"
                        | "LPT9"
                )
        })
}

fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|c| c.is_ascii_hexdigit())
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(file: &mut File, stop: &AtomicBool) -> Result<String, String> {
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        check_stop(stop)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("Cannot open {}: {e}", path.display()))?
        .take(MAX_MANIFEST + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("AI runtime manifest is too large.".into());
    }
    Ok(bytes)
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.schema != 1 || manifest.protocol != 2 {
        return Err("Incompatible AI runtime package protocol. Update the application and package together.".into());
    }
    if manifest.os != std::env::consts::OS || manifest.arch != std::env::consts::ARCH {
        return Err(format!(
            "This runtime is for {} {}, but this app is {} {}.",
            manifest.os,
            manifest.arch,
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
    }
    let backend_valid = matches!(
        (
            manifest.os.as_str(),
            manifest.arch.as_str(),
            manifest.backend.as_str()
        ),
        ("windows", "x86_64", "cpu" | "cuda" | "rocm" | "directml")
            | ("macos", "aarch64", "cpu" | "mps")
            | ("macos", "x86_64", "cpu")
            | ("linux", "x86_64", "cpu" | "cuda" | "rocm")
    );
    if !backend_valid {
        return Err(
            "This native runtime backend/architecture is not supported on this platform.".into(),
        );
    }
    verify_minimum_os(&manifest.minimum_os)?;
    if !valid_id(&manifest.id)
        || normalized_backend(&manifest.backend)? != manifest.backend
        || !safe_relative(&manifest.executable)
        || !safe_relative(&manifest.assets_dir)
        || !safe_relative(&manifest.archive)
        || manifest.archive.contains('/')
        || !is_digest(&manifest.archive_sha256)
        || manifest.archive_bytes == 0
        || manifest.archive_bytes > MAX_ARCHIVE
        || manifest.unpacked_bytes > MAX_UNPACKED
        || manifest.files.is_empty()
        || manifest.files.len() > 100_000
    {
        return Err("Invalid AI runtime package manifest.".into());
    }
    let mut folded = HashSet::new();
    for (name, digest) in &manifest.files {
        if !safe_relative(name)
            || !is_digest(digest)
            || matches!(
                name.to_ascii_lowercase().as_str(),
                ".lease" | "runtime.json" | "probe.log"
            )
            || !folded.insert(name.to_lowercase())
        {
            return Err(format!("Unsafe or duplicate runtime file: {name}"));
        }
    }
    if !manifest.files.contains_key(&manifest.executable) {
        return Err("Runtime executable is missing from the package manifest.".into());
    }
    for name in ["hubert_base.pt", "rmvpe.onnx"] {
        if !manifest
            .files
            .contains_key(&format!("{}/{name}", manifest.assets_dir))
        {
            return Err(format!("Runtime shared asset is missing: {name}"));
        }
    }
    Ok(())
}

fn version_parts(value: &str) -> Result<Vec<u32>, String> {
    let parts = value
        .trim()
        .split('.')
        .map(|part| part.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "Invalid runtime minimum OS version.".to_string())?;
    if parts.is_empty() || parts.len() > 4 {
        return Err("Invalid runtime minimum OS version.".into());
    }
    Ok(parts)
}

fn version_at_least(actual: &[u32], required: &[u32]) -> bool {
    for index in 0..actual.len().max(required.len()) {
        match actual
            .get(index)
            .unwrap_or(&0)
            .cmp(required.get(index).unwrap_or(&0))
        {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    true
}

fn verify_minimum_os(required: &str) -> Result<(), String> {
    let required_parts = version_parts(required)?;
    #[cfg(target_os = "macos")]
    let actual = {
        let mut command = Command::new("/usr/bin/sw_vers");
        command
            .arg("-productVersion")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut process = super::process::ManagedChild::spawn(&mut command)?;
        let status = process.wait_cancellable(&AtomicBool::new(false), Duration::from_secs(5))?;
        if !status.success() {
            return Err("Cannot determine the macOS version for this runtime.".into());
        }
        let mut version = String::new();
        process
            .child_mut()
            .stdout
            .take()
            .unwrap()
            .take(100)
            .read_to_string(&mut version)
            .map_err(|e| e.to_string())?;
        version_parts(&version)?
    };
    #[cfg(target_os = "windows")]
    let actual = unsafe {
        use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
        use windows_sys::Win32::System::SystemInformation::OSVERSIONINFOW;
        let mut version: OSVERSIONINFOW = std::mem::zeroed();
        version.dwOSVersionInfoSize = std::mem::size_of::<OSVERSIONINFOW>() as u32;
        if RtlGetVersion(&mut version) < 0 {
            return Err("Cannot determine the Windows version for this runtime.".into());
        }
        vec![
            version.dwMajorVersion,
            version.dwMinorVersion,
            version.dwBuildNumber,
        ]
    };
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let actual = required_parts.clone(); // Linux native imports are developer-only; containers handle their own user space.
    if !version_at_least(&actual, &required_parts) {
        return Err(format!(
            "This AI runtime requires {} {required} or newer.",
            std::env::consts::OS
        ));
    }
    Ok(())
}

fn global_lock(root: &Path, exclusive: bool) -> Result<LockGuard, String> {
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join("install.lock"))
        .map_err(|e| e.to_string())?;
    let result = if exclusive {
        LockGuard::exclusive(lock)
    } else {
        LockGuard::shared(lock)
    };
    result.map_err(|_| {
        "Another AI runtime installation or maintenance operation is running.".to_string()
    })
}

fn active_path(root: &Path, backend: &str) -> PathBuf {
    root.join(format!("active-{backend}.json"))
}

fn read_active(root: &Path, backend: &str) -> Result<Active, String> {
    serde_json::from_slice(&read_bounded(&active_path(root, backend))?)
        .map_err(|_| "Invalid installed AI runtime selection. Repair the engine.".into())
}

fn write_active(root: &Path, backend: &str, active: &Active) -> Result<(), String> {
    let temporary = root.join(format!("active-{backend}.tmp"));
    let mut file = File::create(&temporary).map_err(|e| e.to_string())?;
    file.write_all(&serde_json::to_vec_pretty(active).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);
    fs::rename(temporary, active_path(root, backend)).map_err(|e| e.to_string())
}

pub fn locate(backend: &str) -> Result<InstalledRuntime, String> {
    locate_at(&data_dir(), normalized_backend(backend)?)
}

fn locate_at(root: &Path, backend: &str) -> Result<InstalledRuntime, String> {
    let _lock = global_lock(root, false)?;
    let active = read_active(root, backend)
        .map_err(|_| "This native AI runtime is not installed. Install a published runtime or import a verified runtime package.".to_string())?;
    if !valid_id(&active.current) {
        return Err("Invalid installed runtime ID.".into());
    }
    let dir = root.join(&active.current);
    let lease = OpenOptions::new()
        .read(true)
        .write(true)
        .open(dir.join(".lease"))
        .map_err(|e| e.to_string())?;
    let lease =
        LockGuard::shared(lease).map_err(|_| "This AI runtime is being removed.".to_string())?;
    let manifest: Manifest = serde_json::from_slice(&read_bounded(&dir.join("runtime.json"))?)
        .map_err(|e| e.to_string())?;
    validate_manifest(&manifest)?;
    if manifest.id != active.current || manifest.backend != backend {
        return Err("Installed runtime identity mismatch. Repair the engine.".into());
    }
    let executable = dir.join(&manifest.executable);
    let assets_dir = dir.join(&manifest.assets_dir);
    if !executable.is_file() || !assets_dir.is_dir() {
        return Err("Installed AI runtime is incomplete. Repair the engine.".into());
    }
    Ok(InstalledRuntime {
        root: dir,
        executable,
        assets_dir,
        manifest,
        _lease: lease,
    })
}

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn staging(root: &Path) -> Result<Staging, String> {
    let path = root.join(format!(
        ".staging-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).map_err(|e| e.to_string())?;
    Ok(Staging(path))
}

fn extract(
    archive: &mut File,
    destination: &Path,
    manifest: &Manifest,
    stop: &AtomicBool,
) -> Result<(), String> {
    let mut archive = zip::ZipArchive::new(archive).map_err(|e| e.to_string())?;
    if archive.len() > manifest.files.len() + 10_000 {
        return Err("Runtime archive contains too many entries.".into());
    }
    let mut seen = HashSet::new();
    let mut total = 0_u64;
    for index in 0..archive.len() {
        check_stop(stop)?;
        let mut entry = archive.by_index(index).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        let is_directory = entry.is_dir();
        let path = name.strip_suffix('/').unwrap_or(&name);
        if !safe_relative(path) {
            return Err(format!("Unsafe runtime archive path: {name}"));
        }
        let kind = entry.unix_mode().unwrap_or(0) & 0o170000;
        if ![0, 0o100000, 0o040000].contains(&kind) {
            return Err("Runtime archives may not contain symlinks or special files.".into());
        }
        if is_directory {
            continue;
        }
        let expected = manifest
            .files
            .get(&name)
            .ok_or_else(|| format!("Unlisted runtime archive entry: {name}"))?;
        if !seen.insert(name.clone()) {
            return Err(format!("Duplicate runtime archive entry: {name}"));
        }
        total = total
            .checked_add(entry.size())
            .ok_or("Runtime archive size overflow")?;
        if total > manifest.unpacked_bytes || total > MAX_UNPACKED {
            return Err("Runtime archive exceeds its declared size.".into());
        }
        let target = destination.join(&name);
        fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&target)
            .map_err(|e| e.to_string())?;
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 128 * 1024];
        let mut count = 0_u64;
        loop {
            check_stop(stop)?;
            let read = entry.read(&mut buffer).map_err(|e| e.to_string())?;
            if read == 0 {
                break;
            }
            count += read as u64;
            if count > entry.size() {
                return Err("Runtime entry exceeds its declared size.".into());
            }
            hash.update(&buffer[..read]);
            output
                .write_all(&buffer[..read])
                .map_err(|e| e.to_string())?;
        }
        if count != entry.size()
            || format!("{:x}", hash.finalize()) != expected.to_ascii_lowercase()
        {
            return Err(format!("Runtime file checksum failed: {name}"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // Preserve executable bits only, never archive ownership or setuid bits.
            let mode = if entry.unix_mode().unwrap_or(0) & 0o111 != 0 || name == manifest.executable
            {
                0o755
            } else {
                0o644
            };
            fs::set_permissions(&target, fs::Permissions::from_mode(mode))
                .map_err(|e| e.to_string())?;
        }
    }
    if seen.len() != manifest.files.len() || total != manifest.unpacked_bytes {
        return Err("Runtime archive does not match its manifest inventory.".into());
    }
    Ok(())
}

fn smoke_probe(root: &Path, manifest: &Manifest, stop: &AtomicBool) -> Result<(), String> {
    let log = File::create(root.join("probe.log")).map_err(|e| e.to_string())?;
    let mut command = Command::new(root.join(&manifest.executable));
    command
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(log);
    super::launch::clean_environment(&mut command);
    let mut process = super::process::ManagedChild::spawn(&mut command)?;
    let result = (|| {
        let mut stdin = process.child_mut().stdin.take().unwrap();
        writeln!(stdin, "{}", serde_json::json!({"protocol":2,"action":"probe","backend":manifest.backend,"device":"0"})).map_err(|e| e.to_string())?;
        drop(stdin);
        let stdout = process.child_mut().stdout.take().unwrap();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut data = Vec::new();
            let read = stdout
                .take(300_001)
                .read_to_end(&mut data)
                .map(|_| data)
                .map_err(|e| e.to_string());
            let _ = tx.send(read);
        });
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            check_stop(stop)?;
            if let Some(status) = process.try_wait().map_err(|e| e.to_string())? {
                if !status.success() {
                    return Err("Packaged worker probe failed. Inspect the runtime dependency build before distribution.".into());
                }
                let data = rx
                    .recv_timeout(Duration::from_secs(1))
                    .map_err(|_| "Packaged worker did not close its output pipe")??;
                if data.len() > 300_000 {
                    return Err("Packaged worker probe response is too large.".into());
                }
                let response: serde_json::Value = serde_json::from_slice(&data)
                    .map_err(|_| "Packaged worker returned invalid probe data")?;
                if response["protocol"] != 2
                    || !response["devices"].is_array()
                    || response.get("error").is_some()
                {
                    return Err("Packaged worker is incompatible with this app.".into());
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err("Packaged worker probe timed out.".into());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    })();
    process.terminate();
    result
}

pub fn install_package(
    manifest_path: &Path,
    expected_manifest_sha256: &str,
    stop: &AtomicBool,
) -> Result<(), String> {
    install_at(
        &data_dir(),
        manifest_path,
        expected_manifest_sha256,
        stop,
        smoke_probe,
    )
}

fn install_at(
    root: &Path,
    manifest_path: &Path,
    expected: &str,
    stop: &AtomicBool,
    probe: fn(&Path, &Manifest, &AtomicBool) -> Result<(), String>,
) -> Result<(), String> {
    check_stop(stop)?;
    if !is_digest(expected) {
        return Err("Supply the runtime manifest SHA-256 from a trusted release source.".into());
    }
    let bytes = read_bounded(manifest_path)?;
    if digest_bytes(&bytes) != expected.to_ascii_lowercase() {
        return Err("AI runtime manifest checksum failed.".into());
    }
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("Invalid runtime manifest: {e}"))?;
    validate_manifest(&manifest)?;
    let _lock = global_lock(root, true)?;
    let destination = root.join(&manifest.id);
    let old = read_active(root, &manifest.backend).ok();
    let repair_lease = if destination.exists() {
        // Repair may restore precisely the same build, never change an immutable
        // ID's contents or replace libraries while another process is using them.
        if read_bounded(&destination.join("runtime.json"))? != bytes {
            return Err("This build ID already has a different manifest. Use a unique ID for a new runtime build.".into());
        }
        let lease = OpenOptions::new()
            .read(true)
            .write(true)
            .open(destination.join(".lease"))
            .map_err(|e| e.to_string())?;
        let lease = LockGuard::exclusive(lease)
            .map_err(|_| "Stop all workers using this runtime before repairing it.".to_string())?;
        Some(lease)
    } else {
        None
    };
    let archive_path = manifest_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(&manifest.archive);
    let mut archive = File::open(&archive_path).map_err(|e| {
        format!(
            "Cannot open runtime archive {}: {e}",
            archive_path.display()
        )
    })?;
    if archive.metadata().map_err(|e| e.to_string())?.len() != manifest.archive_bytes {
        return Err("Runtime archive size does not match its manifest.".into());
    }
    if hash_file(&mut archive, stop)? != manifest.archive_sha256.to_ascii_lowercase() {
        return Err("Runtime archive checksum failed.".into());
    }
    archive.rewind().map_err(|e| e.to_string())?;
    let temporary = staging(root)?;
    extract(&mut archive, &temporary.0, &manifest, stop)?;
    fs::write(temporary.0.join("runtime.json"), bytes).map_err(|e| e.to_string())?;
    File::create(temporary.0.join(".lease")).map_err(|e| e.to_string())?;
    if let Err(error) = probe(&temporary.0, &manifest, stop) {
        let log_path = root.join("install-probe.log");
        if fs::copy(temporary.0.join("probe.log"), &log_path).is_ok() {
            return Err(format!("{error} See {}", log_path.display()));
        }
        return Err(error);
    }
    check_stop(stop)?;
    let backup = if repair_lease.is_some() {
        let backup = staging(root)?;
        fs::remove_dir(&backup.0).map_err(|e| e.to_string())?;
        // Windows cannot rename a directory with the lease descriptor open.
        // The global exclusive lock still excludes new users after closing it.
        drop(repair_lease);
        fs::rename(&destination, &backup.0).map_err(|e| e.to_string())?;
        Some(backup)
    } else {
        None
    };
    if let Err(error) = fs::rename(&temporary.0, &destination) {
        if let Some(backup) = &backup {
            let _ = fs::rename(&backup.0, &destination);
        }
        return Err(error.to_string());
    }
    let previous = old.and_then(|active| {
        if active.current == manifest.id {
            active.previous
        } else {
            Some(active.current)
        }
    });
    write_active(
        root,
        &manifest.backend,
        &Active {
            current: manifest.id,
            previous,
        },
    )
}

fn download(url: &str, destination: &Path, limit: u64, stop: &AtomicBool) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err("Runtime downloads require HTTPS.".into());
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(async {
        let operation = async {
            let client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(15))
                .read_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(1800))
                .build()
                .map_err(|e| e.to_string())?;
            let mut response = client
                .get(url)
                .send()
                .await
                .map_err(|e| e.to_string())?
                .error_for_status()
                .map_err(|e| e.to_string())?;
            if !response.status().is_success()
                || response.content_length().is_some_and(|size| size > limit)
            {
                return Err("Invalid or oversized runtime download response.".into());
            }
            let mut output = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(destination)
                .map_err(|e| e.to_string())?;
            let mut size = 0_u64;
            while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
                size += chunk.len() as u64;
                if size > limit {
                    return Err("Runtime download exceeds its size limit.".into());
                }
                output.write_all(&chunk).map_err(|e| e.to_string())?;
            }
            Ok(())
        };
        let mut operation = std::pin::pin!(operation);
        loop {
            check_stop(stop)?;
            // Dropping the in-flight future on cancellation also drops the HTTP
            // response/file, so stalled network reads do not block Stop.
            if let Ok(result) =
                tokio::time::timeout(Duration::from_millis(50), operation.as_mut()).await
            {
                return result;
            }
        }
    })
}

pub fn install_bundled(backend: &str, stop: &AtomicBool) -> Result<(), String> {
    let backend = normalized_backend(backend)?;
    let catalog: Catalog = serde_json::from_str(include_str!("../../engine/runtime-catalog.json"))
        .map_err(|e| e.to_string())?;
    if catalog.schema != 1 {
        return Err("Unsupported runtime catalog. Update Voice Warp.".into());
    }
    let release = catalog.releases.iter().find(|entry| entry.os == std::env::consts::OS && entry.arch == std::env::consts::ARCH && entry.backend == backend)
        .ok_or_else(|| format!("No published {backend} native runtime is included in this app's trusted catalog for {} {}. Build a runtime with scripts/build-ai-runtime.py and import its manifest with --ai-runtime-manifest and --ai-manifest-sha256, or use a release with a qualified runtime catalog.", std::env::consts::OS, std::env::consts::ARCH))?;
    let root = data_dir();
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let temporary = staging(&root)?;
    let manifest_path = temporary.0.join("manifest.json");
    download(&release.manifest_url, &manifest_path, MAX_MANIFEST, stop)?;
    let bytes = read_bounded(&manifest_path)?;
    if digest_bytes(&bytes) != release.manifest_sha256.to_ascii_lowercase() {
        return Err("Published runtime manifest checksum failed.".into());
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate_manifest(&manifest)?;
    if manifest.backend != backend {
        return Err("Published runtime backend mismatch.".into());
    }
    let base = reqwest::Url::parse(&release.manifest_url).map_err(|e| e.to_string())?;
    let archive_url = base.join(&manifest.archive).map_err(|e| e.to_string())?;
    download(
        archive_url.as_str(),
        &temporary.0.join(&manifest.archive),
        manifest.archive_bytes,
        stop,
    )?;
    install_package(&manifest_path, &release.manifest_sha256, stop)
}

pub fn rollback(backend: &str) -> Result<(), String> {
    let backend = normalized_backend(backend)?;
    let root = data_dir();
    let _lock = global_lock(&root, true)?;
    let active = read_active(&root, backend)?;
    let previous = active.previous.ok_or("No previous runtime is available.")?;
    if !valid_id(&previous) || !root.join(&previous).join("runtime.json").is_file() {
        return Err("Previous runtime is missing.".into());
    }
    let manifest: Manifest =
        serde_json::from_slice(&read_bounded(&root.join(&previous).join("runtime.json"))?)
            .map_err(|e| e.to_string())?;
    validate_manifest(&manifest)?;
    if manifest.id != previous || manifest.backend != backend {
        return Err("Previous runtime identity mismatch.".into());
    }
    write_active(
        &root,
        backend,
        &Active {
            current: previous,
            previous: Some(active.current),
        },
    )
}

/// Remove an inactive build only. Active selection and user model files are preserved.
pub fn remove_build(id: &str) -> Result<(), String> {
    if !valid_id(id) {
        return Err("Invalid runtime build ID.".into());
    }
    let root = data_dir();
    let _lock = global_lock(&root, true)?;
    for backend in ["cpu", "cuda", "rocm", "mps", "directml"] {
        if let Ok(active) = read_active(&root, backend)
            && (active.current == id || active.previous.as_deref() == Some(id))
        {
            return Err("This runtime is active or retained for rollback. Select another build before removing it.".into());
        }
    }
    let dir = root.join(id);
    let lease = OpenOptions::new()
        .read(true)
        .write(true)
        .open(dir.join(".lease"))
        .map_err(|e| e.to_string())?;
    let lease =
        LockGuard::exclusive(lease).map_err(|_| "This runtime is still in use.".to_string())?;
    // On Windows the open lease must be closed before deleting its directory. The
    // global exclusive lock prevents a new locate() between close and removal.
    drop(lease);
    fs::remove_dir_all(dir).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_paths_are_portable_and_cannot_escape() {
        for name in [
            "../evil",
            "/evil",
            "C:/evil",
            "a\\b",
            "a/../b",
            "a//b",
            "a/CON.txt",
            "a/trailing.",
            "a/file ",
            "a/./b",
        ] {
            assert!(!safe_relative(name), "{name}");
        }
        assert!(safe_relative("worker/_internal/Unicode é.dll"));
    }

    fn fixture(root: &Path, extra: Option<&str>) -> (PathBuf, String) {
        fs::create_dir_all(root).unwrap();
        let archive_path = root.join("runtime.zip");
        let mut archive = zip::ZipWriter::new(File::create(&archive_path).unwrap());
        let mut files = BTreeMap::new();
        for name in ["worker", "assets/hubert_base.pt", "assets/rmvpe.onnx"] {
            archive
                .start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            archive.write_all(b"fixture").unwrap();
            files.insert(name.into(), digest_bytes(b"fixture"));
        }
        if let Some(name) = extra {
            archive
                .start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            archive.write_all(b"evil").unwrap();
        }
        archive.finish().unwrap();
        let manifest = Manifest {
            schema: 1,
            protocol: 2,
            id: "fixture-cpu-1".into(),
            version: "1".into(),
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            backend: "cpu".into(),
            minimum_os: "0".into(),
            executable: "worker".into(),
            assets_dir: "assets".into(),
            archive: "runtime.zip".into(),
            archive_sha256: hash_file(
                &mut File::open(&archive_path).unwrap(),
                &AtomicBool::new(false),
            )
            .unwrap(),
            archive_bytes: fs::metadata(&archive_path).unwrap().len(),
            unpacked_bytes: 21,
            files,
            packages: BTreeMap::new(),
            upstream_revision: "test".into(),
            qualification: "test only".into(),
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let path = root.join("manifest.json");
        fs::write(&path, &bytes).unwrap();
        (path, digest_bytes(&bytes))
    }

    fn scratch() -> Staging {
        staging(&std::env::temp_dir()).unwrap()
    }

    #[test]
    fn verified_install_activates_and_holds_shared_lease() {
        let temp = scratch();
        let (manifest, digest) = fixture(&temp.0.join("package"), None);
        let install = temp.0.join("installed");
        install_at(
            &install,
            &manifest,
            &digest,
            &AtomicBool::new(false),
            |_, _, _| Ok(()),
        )
        .unwrap();
        let runtime = locate_at(&install, "cpu").unwrap();
        assert!(runtime.executable.is_file());
        // Model a descriptor inherited by a concurrent fork before exec. This
        // incidental copy must neither release nor extend the owner's lease.
        let inherited = runtime._lease.0.try_clone().unwrap();
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(runtime.root.join(".lease"))
            .unwrap();
        assert!(lock.try_lock_exclusive().is_err());
        drop(runtime);
        let _exclusive = LockGuard::exclusive(lock).unwrap();
        drop(inherited);
    }

    #[test]
    fn global_lock_releases_even_when_a_fork_copy_is_still_open() {
        let temp = scratch();
        let guard = global_lock(&temp.0, true).unwrap();
        let inherited = guard.0.try_clone().unwrap();
        assert!(global_lock(&temp.0, true).is_err());
        drop(guard);
        // Closing only the owning File would leave this locked on Unix while
        // inherited is open; LockGuard::drop explicitly releases ownership.
        let replacement = global_lock(&temp.0, true).unwrap();
        drop(inherited);
        assert!(global_lock(&temp.0, true).is_err());
        drop(replacement);
        assert!(global_lock(&temp.0, false).is_ok());
    }

    #[test]
    fn failed_probe_or_verification_never_activates() {
        let temp = scratch();
        let (manifest, digest) = fixture(&temp.0.join("package"), None);
        let install = temp.0.join("installed");
        assert!(
            install_at(
                &install,
                &manifest,
                &"0".repeat(64),
                &AtomicBool::new(false),
                |_, _, _| Ok(())
            )
            .is_err()
        );
        assert!(
            install_at(
                &install,
                &manifest,
                &digest,
                &AtomicBool::new(false),
                |_, _, _| Err("probe failure".into())
            )
            .is_err()
        );
        assert!(!active_path(&install, "cpu").exists());
        assert!(!install.join("fixture-cpu-1").exists());
        assert!(
            install_at(
                &install,
                &manifest,
                &digest,
                &AtomicBool::new(true),
                |_, _, _| Ok(())
            )
            .is_err()
        );
    }

    #[test]
    fn repair_requires_unused_runtime_and_restores_verified_files() {
        let temp = scratch();
        let (manifest, digest) = fixture(&temp.0.join("package"), None);
        let install = temp.0.join("installed");
        let stop = AtomicBool::new(false);
        install_at(&install, &manifest, &digest, &stop, |_, _, _| Ok(())).unwrap();
        let runtime = locate_at(&install, "cpu").unwrap();
        assert!(install_at(&install, &manifest, &digest, &stop, |_, _, _| Ok(())).is_err());
        fs::write(&runtime.executable, b"damaged").unwrap();
        drop(runtime);
        install_at(&install, &manifest, &digest, &stop, |_, _, _| Ok(())).unwrap();
        assert_eq!(
            fs::read(locate_at(&install, "cpu").unwrap().executable).unwrap(),
            b"fixture"
        );
        assert!(read_active(&install, "cpu").unwrap().previous.is_none());
    }

    #[test]
    fn signed_manifest_does_not_authorize_archive_path_traversal() {
        let temp = scratch();
        let (manifest, digest) = fixture(&temp.0.join("package"), Some("../escaped"));
        let install = temp.0.join("installed");
        assert!(
            install_at(
                &install,
                &manifest,
                &digest,
                &AtomicBool::new(false),
                |_, _, _| Ok(())
            )
            .is_err()
        );
        assert!(!install.join("escaped").exists());
        assert!(!active_path(&install, "cpu").exists());
    }

    #[test]
    fn minimum_os_version_comparison_does_not_compare_strings() {
        assert!(version_at_least(&[14, 10], &[14, 9]));
        assert!(version_at_least(&[12, 3, 0], &[12, 3]));
        assert!(!version_at_least(&[10, 0, 17763], &[10, 0, 18362]));
        assert!(version_parts("14.x").is_err());
    }

    #[test]
    fn canceled_download_does_not_contact_the_network() {
        let temp = scratch();
        assert!(
            download(
                "https://invalid.invalid/runtime.zip",
                &temp.0.join("never"),
                1000,
                &AtomicBool::new(true)
            )
            .unwrap_err()
            .contains("canceled")
        );
        assert!(!temp.0.join("never").exists());
    }
}
