// A dependency-free subprocess fixture compiled by lifecycle tests. It requires
// the Rust toolchain used to run those tests, never a user's Python installation.
use std::io::{self, BufRead, Write};
use std::time::Duration;

fn main() {
    let mode = std::env::var("VOICE_WARP_FIXTURE_MODE").unwrap_or_default();
    let mut input = io::stdin().lock();
    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    if mode == "stall_start" {
        std::thread::sleep(Duration::from_secs(60));
        return;
    }
    println!("{}", std::env::var("VOICE_WARP_FIXTURE_READY").unwrap());
    io::stdout().flush().unwrap();
    line.clear();
    input.read_line(&mut line).unwrap();
    if mode == "stall_audio" {
        std::thread::sleep(Duration::from_secs(60));
    } else {
        println!("{}", std::env::var("VOICE_WARP_FIXTURE_REPLY").unwrap_or_default());
        io::stdout().flush().unwrap();
    }
}
