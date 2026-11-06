//! Own the complete process tree, including failed startup and delayed kernel exit.
use std::{
    io,
    process::{Child, Command, ExitStatus},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

type Retained = Vec<Box<dyn Send>>;
type ReapJob = (Child, Retained);
// Retain leases if OS thread creation fails; the next handoff retries these jobs.
static DEFERRED: Mutex<Vec<ReapJob>> = Mutex::new(Vec::new());

pub struct ManagedChild {
    child: Option<Child>,
    status: Option<ExitStatus>,
    reaper: Option<mpsc::Receiver<Result<ExitStatus, String>>>,
    retained: Retained,
    #[cfg(unix)]
    group_owned: bool,
    #[cfg(windows)]
    job: windows_sys::Win32::Foundation::HANDLE,
}

// The HANDLE is exclusively owned; retained resources require Send.
#[cfg(windows)]
unsafe impl Send for ManagedChild {}

impl ManagedChild {
    pub fn spawn(command: &mut Command) -> Result<Self, String> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
        }
        let child = command
            .spawn()
            .map_err(|e| format!("Cannot start AI worker: {e}"))?;
        let mut result = Self {
            child: Some(child),
            status: None,
            reaper: None,
            retained: Vec::new(),
            #[cfg(unix)]
            group_owned: true,
            #[cfg(windows)]
            job: std::ptr::null_mut(),
        };
        #[cfg(windows)]
        result.attach_job()?;
        let _ = &mut result;
        Ok(result)
    }

    /// Borrow to take configured pipes only. Reaping must go through try_wait(),
    /// which kills Unix descendants before releasing their leader's PID.
    pub fn child_mut(&mut self) -> &mut Child {
        self.child
            .as_mut()
            .expect("Process already transferred to its background reaper")
    }

    /// Runtime usage leases survive until confirmed exit, including deferred reap.
    pub fn retain<T: Send + 'static>(&mut self, resource: T) {
        self.retained.push(Box::new(resource));
    }

    #[cfg(windows)]
    fn attach_job(&mut self) -> Result<(), String> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            self.job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if self.job.is_null() {
                return Err("Cannot create an AI worker job.".into());
            }
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                self.job,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _,
                std::mem::size_of_val(&limits) as u32,
            ) == 0
                || AssignProcessToJobObject(self.job, self.child.as_ref().unwrap().as_raw_handle())
                    == 0
            {
                return Err(format!(
                    "Cannot supervise AI worker process tree: {}",
                    io::Error::last_os_error()
                ));
            }
        }
        // The caller sends protocol initialization after ownership is established.
        Ok(())
    }

    #[cfg(windows)]
    fn terminate_job(&mut self) {
        unsafe {
            if !self.job.is_null() {
                windows_sys::Win32::System::JobObjects::TerminateJobObject(self.job, 1);
                windows_sys::Win32::Foundation::CloseHandle(self.job);
                self.job = std::ptr::null_mut();
            }
        }
    }

    /// WNOWAIT keeps an exited leader's PID reserved until descendants are killed.
    #[cfg(unix)]
    fn unix_exited(&mut self) -> io::Result<bool> {
        if !self.group_owned {
            return Ok(true);
        }
        let child = self.child.as_ref().expect("Owned Unix child");
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                child.id() as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ECHILD) {
                // An external caller/SIGCHLD policy reaped it: never signal a
                // numeric process group whose ID may now have been reused.
                self.group_owned = false;
            }
            return Err(error);
        }
        Ok(unsafe { info.si_pid() } != 0)
    }

    #[cfg(unix)]
    fn kill_owned_group(&mut self) {
        if self.group_owned {
            let child = self.child.as_ref().expect("Owned Unix child");
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
        }
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.status {
            return Ok(Some(status));
        }
        if self.child.is_none() {
            return match self.reaper.as_ref().map(|reaper| reaper.try_recv()) {
                Some(Ok(Ok(status))) => {
                    self.status = Some(status);
                    Ok(Some(status))
                }
                Some(Ok(Err(error))) => Err(io::Error::other(error)),
                Some(Err(mpsc::TryRecvError::Disconnected)) => {
                    Err(io::Error::other("AI process reaper disconnected"))
                }
                _ => Ok(None),
            };
        }
        #[cfg(unix)]
        {
            match self.unix_exited() {
                Ok(false) => return Ok(None),
                Ok(true) => self.kill_owned_group(),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(None),
                Err(error) if error.raw_os_error() == Some(libc::ECHILD) => {
                    // Child::try_wait can return its cached status if an external
                    // caller already reaped it. Group signaling stays disabled.
                }
                Err(error) => return Err(error),
            }
        }
        let status = self.child.as_mut().unwrap().try_wait()?;
        if let Some(status) = status {
            #[cfg(unix)]
            {
                self.group_owned = false;
            }
            #[cfg(windows)]
            self.terminate_job();
            self.status = Some(status);
            self.retained.clear();
        }
        Ok(status)
    }

    /// Whole-tree termination waits at most 500ms here. A kernel/driver-delayed
    /// exit is reaped by an owning background thread, retaining runtime leases.
    pub fn terminate(&mut self) {
        if self.child.is_none() {
            return;
        }
        #[cfg(windows)]
        self.terminate_job();
        if self.status.is_some() {
            self.child.take();
            self.retained.clear();
            return;
        }
        if let Some(child) = self.child.as_mut() {
            child.stdin.take();
        }
        let deadline = Instant::now() + Duration::from_millis(500);
        loop {
            #[cfg(unix)]
            if self.unix_exited().is_ok() && self.group_owned {
                self.kill_owned_group();
                let _ = self.child.as_mut().unwrap().kill();
            }
            #[cfg(not(unix))]
            {
                let _ = self.child.as_mut().unwrap().kill();
            }
            match self.try_wait() {
                Ok(Some(_)) => {
                    self.child.take();
                    return;
                }
                Err(_) => break,
                Ok(None) => {}
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if let Some(child) = self.child.take() {
            self.reaper = Some(background_reap((child, std::mem::take(&mut self.retained))));
        }
    }

    pub fn wait_cancellable(
        &mut self,
        stop: &AtomicBool,
        timeout: Duration,
    ) -> Result<ExitStatus, String> {
        let start = Instant::now();
        loop {
            if stop.load(Ordering::Relaxed) {
                return Err("AI operation canceled.".into());
            }
            if let Some(status) = self.try_wait().map_err(|e| e.to_string())? {
                return Ok(status);
            }
            if start.elapsed() >= timeout {
                return Err("AI operation timed out.".into());
            }
            std::thread::sleep(Duration::from_millis(40));
        }
    }
}

fn background_reap(job: ReapJob) -> mpsc::Receiver<Result<ExitStatus, String>> {
    let deferred = std::mem::take(&mut *DEFERRED.lock().unwrap());
    for job in deferred {
        start_reaper(job);
    }
    start_reaper(job)
}

fn start_reaper(job: ReapJob) -> mpsc::Receiver<Result<ExitStatus, String>> {
    let (tx, rx) = mpsc::sync_channel(1);
    let job = Arc::new(Mutex::new(Some(job)));
    let owned = job.clone();
    let started = std::thread::Builder::new()
        .name("voice-warp-reaper".into())
        .spawn(move || {
            let (mut child, retained) = owned.lock().unwrap().take().unwrap();
            let result = child.wait().map_err(|e| e.to_string());
            drop(retained);
            let _ = tx.send(result);
        });
    if started.is_err()
        && let Some(job) = job.lock().unwrap().take()
    {
        // Never free a running worker's runtime lease on thread exhaustion.
        DEFERRED.lock().unwrap().push(job);
    }
    rx
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::io::{BufRead, BufReader};
    use std::{path::PathBuf, process::Stdio, sync::OnceLock};

    fn fixture(mode: &str) -> Command {
        static BINARY: OnceLock<PathBuf> = OnceLock::new();
        let binary = BINARY.get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let out = root.join("target").join(format!(
                "process-tree-fixture-{}{}",
                std::process::id(),
                std::env::consts::EXE_SUFFIX
            ));
            let status = Command::new("rustc")
                .arg("--edition=2024")
                .arg(root.join("tests/fixtures/process_tree.rs"))
                .arg("-o")
                .arg(&out)
                .status()
                .unwrap();
            assert!(status.success());
            out
        });
        let mut command = Command::new(binary);
        command
            .arg(mode)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        command
    }

    #[test]
    fn stop_is_bounded_and_releases_leases_after_reaping() {
        struct Lease(Arc<AtomicBool>);
        impl Drop for Lease {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let released = Arc::new(AtomicBool::new(false));
        let mut process = ManagedChild::spawn(&mut fixture("sleep")).unwrap();
        process.retain(Lease(released.clone()));
        let started = Instant::now();
        process.terminate();
        assert!(started.elapsed() < Duration::from_secs(2));
        let deadline = Instant::now() + Duration::from_secs(2);
        while process.try_wait().unwrap().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(process.try_wait().unwrap().is_some());
        assert!(released.load(Ordering::SeqCst));
    }

    #[test]
    fn background_reaper_retains_resources_until_the_process_exits() {
        struct Lease(Arc<AtomicBool>);
        impl Drop for Lease {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let released = Arc::new(AtomicBool::new(false));
        let mut command = fixture("wait_for_eof");
        command.stdin(Stdio::piped());
        let mut child = command.spawn().unwrap();
        let input = child.stdin.take().unwrap();
        let reaped = background_reap((child, vec![Box::new(Lease(released.clone()))]));
        // EOF is controlled by this test, so the child cannot have exited yet.
        assert!(!released.load(Ordering::SeqCst));
        assert!(matches!(reaped.try_recv(), Err(mpsc::TryRecvError::Empty)));
        drop(input);
        assert!(
            reaped
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap()
                .success()
        );
        assert!(released.load(Ordering::SeqCst));
    }

    #[cfg(unix)]
    fn alive(pid: i32) -> bool {
        #[cfg(target_os = "linux")]
        if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat"))
            && stat
                .split(") ")
                .nth(1)
                .is_some_and(|tail| tail.starts_with('Z'))
        {
            return false;
        }
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[cfg(unix)]
    #[test]
    fn exited_leader_does_not_leave_its_descendant_running() {
        let mut process = ManagedChild::spawn(&mut fixture("exit_first")).unwrap();
        let mut output = BufReader::new(process.child_mut().stdout.take().unwrap());
        let mut line = String::new();
        output.read_line(&mut line).unwrap();
        let descendant: i32 = line.trim().parse().unwrap();
        assert!(alive(descendant));
        let deadline = Instant::now() + Duration::from_secs(2);
        while !process.unix_exited().unwrap() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(process.unix_exited().unwrap());
        assert!(process.try_wait().unwrap().is_some());
        let deadline = Instant::now() + Duration::from_secs(2);
        while alive(descendant) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !alive(descendant),
            "A descendant survived its exited leader's cleanup"
        );
    }
}
