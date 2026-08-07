//! GPNEC host — Rust orchestration over the Swift/Metal engine C ABI.
//!
//! Build the Swift dynamic library first (`scripts/build_bridge.sh`), then:
//!   cargo run -p gpnec-host -- --adapter lbm --steps 32

use std::env;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[link(name = "GPNECCBridge", kind = "dylib")]
unsafe extern "C" {
    fn gpnec_create(domain: *const c_char, steps_hint: i32) -> u64;
    fn gpnec_step(handle: u64, count: i32) -> i32;
    fn gpnec_steps_executed(handle: u64) -> u64;
    fn gpnec_shape(handle: u64, batch: *mut i32, nodes: *mut i32, channels: *mut i32) -> i32;
    fn gpnec_destroy(handle: u64);
    fn gpnec_version() -> *const c_char;
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let adapter = flag(&args, "--adapter").unwrap_or_else(|| "lbm".into());
    let steps: i32 = flag(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    unsafe {
        let ver = CStr::from_ptr(gpnec_version());
        println!(
            "GPNEC host (Rust) · engine {} · adapter={} steps={}",
            ver.to_string_lossy(),
            adapter,
            steps
        );

        let domain = CString::new(adapter).expect("adapter");
        let handle = gpnec_create(domain.as_ptr(), steps);
        if handle == 0 {
            eprintln!("failed to create engine (is libGPNECCBridge loaded?)");
            std::process::exit(1);
        }

        let mut batch = 0i32;
        let mut nodes = 0i32;
        let mut channels = 0i32;
        gpnec_shape(handle, &mut batch, &mut nodes, &mut channels);
        println!("shape: [{batch}, {nodes}, {channels}]");

        let rc = gpnec_step(handle, steps);
        if rc != 0 {
            eprintln!("step failed: {rc}");
            gpnec_destroy(handle);
            std::process::exit(2);
        }

        println!("steps executed: {}", gpnec_steps_executed(handle));
        gpnec_destroy(handle);
    }
}
