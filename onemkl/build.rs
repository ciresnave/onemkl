//! Build script for the `onemkl` crate.
//!
//! The only job is to thread a `--no-as-needed` linker wrapper around the
//! three layered MKL shared libraries on Linux dynamic builds. The
//! actual `-l` / `-L` plumbing is handled by `onemkl-sys`; this build
//! script just emits the extra link-args needed to keep `mkl_core` in
//! the test binaries' `DT_NEEDED` list.
//!
//! Why here and not in `onemkl-sys`?  Cargo's `rustc-link-arg`
//! directive only applies to artifacts of the crate that emits it
//! (binaries, tests, examples, cdylibs of that crate). `onemkl-sys` is
//! a plain library, so its link-args never reach `onemkl`'s test
//! binaries. Emitting them from `onemkl`'s own build script — which
//! does have a test target — gets them onto the linker command line
//! where they need to be.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        return;
    }
    if env::var_os("CARGO_FEATURE_LINK_DYNAMIC").is_none() {
        return;
    }

    // `onemkl-sys` declares `links = "mkl"` and emits `cargo:lib=<dir>`,
    // which Cargo turns into `DEP_MKL_LIB` for downstream build scripts.
    println!("cargo:rerun-if-env-changed=DEP_MKL_LIB");
    let lib_dir = match env::var_os("DEP_MKL_LIB") {
        Some(v) => PathBuf::from(v),
        None => return,
    };

    let interface = if env::var_os("CARGO_FEATURE_ILP64").is_some() {
        "mkl_intel_ilp64"
    } else {
        "mkl_intel_lp64"
    };

    let threading = if env::var_os("CARGO_FEATURE_THREADING_INTEL_OPENMP").is_some() {
        "mkl_intel_thread"
    } else if env::var_os("CARGO_FEATURE_THREADING_TBB").is_some() {
        "mkl_tbb_thread"
    } else {
        "mkl_sequential"
    };

    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    for name in [interface, threading, "mkl_core"] {
        let so = lib_dir.join(format!("lib{name}.so"));
        println!("cargo:rustc-link-arg={}", so.display());
    }
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
}
