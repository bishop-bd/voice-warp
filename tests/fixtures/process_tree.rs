use std::{
    io::Write,
    process::{Command, Stdio},
    time::Duration,
};

fn main() {
    if std::env::args().nth(1).as_deref() == Some("wait_for_eof") {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).unwrap();
    } else if std::env::args().nth(1).as_deref() == Some("exit_first") {
        let descendant = Command::new(std::env::current_exe().unwrap())
            .arg("sleep")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        println!("{}", descendant.id());
        std::io::stdout().flush().unwrap();
        // Deliberately leave the descendant to the app's process-tree owner.
    } else {
        std::thread::sleep(Duration::from_secs(60));
    }
}
