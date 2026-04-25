//! Build script for `onemkl-sys`.
//!
//! Responsibilities:
//! 1. Locate the Intel oneAPI Math Kernel Library installation (headers + libs).
//! 2. Run `bindgen` against `mkl.h` (and any opt-in MPI-dependent headers) with
//!    the right preprocessor defines for the selected interface layer.
//! 3. Emit a link line matching the selected interface / threading / linkage
//!    feature combination.

use std::env;
use std::path::{Path, PathBuf};
use std::process;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");
    println!("cargo:rerun-if-env-changed=ONEMKL_SYS_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=ONEMKL_SYS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ONEMKL_SYS_OMP_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ONEMKL_SYS_TBB_LIB_DIR");

    let cfg = Config::from_env();
    cfg.validate_features();

    let mkl = MklPaths::locate(&cfg);
    cfg.emit_links(&mkl);
    cfg.generate_bindings(&mkl);
}

/// Resolved feature configuration.
struct Config {
    interface: Interface,
    threading: Threading,
    linkage: Linkage,
    mpi_features: MpiFeatures,
    target_os: String,
    target_env: String,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Interface {
    Lp64,
    Ilp64,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Threading {
    Sequential,
    IntelOpenMp,
    Tbb,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Linkage {
    Dynamic,
    Static,
}

#[derive(Default)]
struct MpiFeatures {
    cluster_sparse_solver: bool,
    cdft: bool,
    scalapack: bool,
    blacs: bool,
    pblas: bool,
}

impl Config {
    fn from_env() -> Self {
        let interface = match (
            cfg_feature("lp64"),
            cfg_feature("ilp64"),
        ) {
            (true, false) => Interface::Lp64,
            (false, true) => Interface::Ilp64,
            (true, true) => fatal("features `lp64` and `ilp64` are mutually exclusive"),
            (false, false) => fatal(
                "no interface layer selected; enable exactly one of `lp64` or `ilp64`",
            ),
        };

        let threading = match (
            cfg_feature("threading-sequential"),
            cfg_feature("threading-intel-openmp"),
            cfg_feature("threading-tbb"),
        ) {
            (true, false, false) => Threading::Sequential,
            (false, true, false) => Threading::IntelOpenMp,
            (false, false, true) => Threading::Tbb,
            (false, false, false) => fatal(
                "no threading layer selected; enable exactly one of \
                 `threading-sequential`, `threading-intel-openmp`, `threading-tbb`",
            ),
            _ => fatal(
                "multiple threading layers selected; enable exactly one of \
                 `threading-sequential`, `threading-intel-openmp`, `threading-tbb`",
            ),
        };

        let linkage = match (cfg_feature("link-dynamic"), cfg_feature("link-static")) {
            (true, false) => Linkage::Dynamic,
            (false, true) => Linkage::Static,
            (true, true) => fatal(
                "features `link-dynamic` and `link-static` are mutually exclusive",
            ),
            (false, false) => fatal(
                "no linkage selected; enable exactly one of `link-dynamic` or `link-static`",
            ),
        };

        let mpi_features = MpiFeatures {
            cluster_sparse_solver: cfg_feature("cluster-sparse-solver"),
            cdft: cfg_feature("cdft"),
            scalapack: cfg_feature("scalapack"),
            blacs: cfg_feature("blacs"),
            pblas: cfg_feature("pblas"),
        };

        Self {
            interface,
            threading,
            linkage,
            mpi_features,
            target_os: env::var("CARGO_CFG_TARGET_OS").unwrap_or_default(),
            target_env: env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default(),
        }
    }

    fn validate_features(&self) {
        if self.linkage == Linkage::Static && self.target_os == "windows" {
            // Static linking on Windows requires /WHOLEARCHIVE handling that
            // we do not yet emit. Surface this clearly.
            eprintln!(
                "cargo:warning=onemkl-sys: link-static on Windows is experimental; \
                 prefer link-dynamic if you hit unresolved external symbols."
            );
        }
    }

    fn is_msvc(&self) -> bool {
        self.target_env == "msvc"
    }

    /// Preprocessor defines passed to bindgen.
    fn clang_defines(&self) -> Vec<&'static str> {
        let mut defs = Vec::new();
        if self.interface == Interface::Ilp64 {
            defs.push("-DMKL_ILP64");
        }
        defs
    }

    fn emit_links(&self, mkl: &MklPaths) {
        println!("cargo:rustc-link-search=native={}", mkl.lib_dir.display());

        let suffix = match self.linkage {
            Linkage::Dynamic if self.is_msvc() => "_dll",
            _ => "",
        };
        let kind = match self.linkage {
            Linkage::Dynamic => "dylib",
            Linkage::Static => "static",
        };

        // Interface layer.
        let interface_lib = match self.interface {
            Interface::Lp64 => "mkl_intel_lp64",
            Interface::Ilp64 => "mkl_intel_ilp64",
        };
        link(kind, &format!("{interface_lib}{suffix}"));

        // Threading layer.
        let threading_lib = match self.threading {
            Threading::Sequential => "mkl_sequential",
            Threading::IntelOpenMp => "mkl_intel_thread",
            Threading::Tbb => "mkl_tbb_thread",
        };
        link(kind, &format!("{threading_lib}{suffix}"));

        // Computational core.
        link(kind, &format!("mkl_core{suffix}"));

        // MPI-dependent additions.
        if self.mpi_features.scalapack {
            let lib = match self.interface {
                Interface::Lp64 => "mkl_scalapack_lp64",
                Interface::Ilp64 => "mkl_scalapack_ilp64",
            };
            link(kind, &format!("{lib}{suffix}"));
        }
        if self.mpi_features.blacs || self.mpi_features.pblas || self.mpi_features.scalapack {
            // Default to MS-MPI on Windows, Intel MPI otherwise. Users can
            // override by linking their own BLACS lib via build-script add-on
            // crates if they need a different MPI flavor.
            let blacs_lib = if self.is_msvc() {
                match self.interface {
                    Interface::Lp64 => "mkl_blacs_msmpi_lp64",
                    Interface::Ilp64 => "mkl_blacs_msmpi_ilp64",
                }
            } else {
                match self.interface {
                    Interface::Lp64 => "mkl_blacs_intelmpi_lp64",
                    Interface::Ilp64 => "mkl_blacs_intelmpi_ilp64",
                }
            };
            // BLACS comes only as a static archive for non-MS-MPI flavors;
            // try with the requested suffix and fall through to no-suffix.
            if self.linkage == Linkage::Dynamic && self.is_msvc() {
                link("dylib", &format!("{blacs_lib}_dll"));
            } else {
                link("static", blacs_lib);
            }
        }
        if self.mpi_features.cdft {
            let cdft = if self.linkage == Linkage::Dynamic && self.is_msvc() {
                "mkl_cdft_core_dll"
            } else {
                "mkl_cdft_core"
            };
            link(
                if self.linkage == Linkage::Dynamic { "dylib" } else { "static" },
                cdft,
            );
        }

        // Companion runtime libraries for the chosen threading layer.
        match self.threading {
            Threading::Sequential => {}
            Threading::IntelOpenMp => {
                if let Some(dir) = mkl.omp_lib_dir.as_ref() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
                let omp_lib = if self.is_msvc() { "libiomp5md" } else { "iomp5" };
                println!("cargo:rustc-link-lib=dylib={omp_lib}");
            }
            Threading::Tbb => {
                if let Some(dir) = mkl.tbb_lib_dir.as_ref() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
                let tbb_lib = if self.is_msvc() { "tbb12" } else { "tbb" };
                println!("cargo:rustc-link-lib=dylib={tbb_lib}");
            }
        }

        // Platform runtime libraries.
        if self.target_os != "windows" {
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
        }

        // Re-export to dependents.
        println!("cargo:include={}", mkl.include_dir.display());
        println!("cargo:lib={}", mkl.lib_dir.display());
    }

    fn generate_bindings(&self, mkl: &MklPaths) {
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
        let wrapper = build_wrapper(&out_dir, &self.mpi_features);

        let mut builder = bindgen::Builder::default()
            .header(wrapper.to_string_lossy().into_owned())
            .clang_arg(format!("-I{}", mkl.include_dir.display()))
            .clang_args(self.clang_defines())
            .derive_default(true)
            .derive_debug(true)
            .derive_copy(true)
            .layout_tests(false)
            .generate_comments(false)
            .prepend_enum_name(false)
            .default_enum_style(bindgen::EnumVariation::ModuleConsts)
            // Limit bindgen output to declarations sourced from MKL headers
            // (and the umbrella `wrapper.h`). This excludes Windows SDK,
            // libc, MSVC intrinsics, etc. while keeping the entire MKL
            // surface area exposed.
            .allowlist_file(".*[/\\\\]wrapper\\.h")
            .allowlist_file(".*[/\\\\]mkl[_a-z0-9]*\\.h")
            .allowlist_file(".*[/\\\\]i_malloc\\.h")
            .blocklist_type("FILE")
            .blocklist_type("__.*")
            .blocklist_function("__.*");

        if let Ok(extra) = env::var("ONEMKL_SYS_BINDGEN_EXTRA_CLANG_ARGS") {
            for arg in extra.split_whitespace() {
                builder = builder.clang_arg(arg);
            }
        }

        let bindings = builder
            .generate()
            .unwrap_or_else(|e| fatal(&format!("bindgen failed: {e}")));

        let out_path = out_dir.join("bindings.rs");
        bindings
            .write_to_file(&out_path)
            .unwrap_or_else(|e| fatal(&format!("failed to write bindings: {e}")));
    }
}

/// Resolved on-disk locations.
struct MklPaths {
    include_dir: PathBuf,
    lib_dir: PathBuf,
    omp_lib_dir: Option<PathBuf>,
    tbb_lib_dir: Option<PathBuf>,
}

impl MklPaths {
    fn locate(cfg: &Config) -> Self {
        // 1. Explicit override.
        let include_dir = env::var_os("ONEMKL_SYS_INCLUDE_DIR")
            .map(PathBuf::from)
            .filter(|p| p.join("mkl.h").exists());

        let lib_dir = env::var_os("ONEMKL_SYS_LIB_DIR")
            .map(PathBuf::from)
            .filter(|p| p.is_dir());

        let (include_dir, lib_dir) = match (include_dir, lib_dir) {
            (Some(i), Some(l)) => (i, l),
            (i, l) => {
                let candidates = mkl_root_candidates();
                let root = candidates
                    .iter()
                    .find(|p| p.join("include").join("mkl.h").exists())
                    .cloned()
                    .unwrap_or_else(|| {
                        fatal(&format!(
                            "could not locate Intel oneAPI MKL.\n\
                             Tried (in order): MKLROOT, ONEMKL_SYS_INCLUDE_DIR, \
                             ONEAPI_ROOT/mkl/latest, and standard install paths.\n\
                             Set MKLROOT to your MKL install (e.g. \
                             `C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest`).\n\
                             Searched: {:?}",
                            candidates
                        ))
                    });
                let i = i.unwrap_or_else(|| root.join("include"));
                let l = l.unwrap_or_else(|| {
                    let lib = root.join("lib");
                    // Older Linux installs use `lib/intel64`.
                    let intel64 = lib.join("intel64");
                    if intel64.is_dir() { intel64 } else { lib }
                });
                (i, l)
            }
        };

        let omp_lib_dir = env::var_os("ONEMKL_SYS_OMP_LIB_DIR")
            .map(PathBuf::from)
            .or_else(|| find_compiler_lib(cfg));
        let tbb_lib_dir = env::var_os("ONEMKL_SYS_TBB_LIB_DIR")
            .map(PathBuf::from)
            .or_else(|| find_tbb_lib(cfg));

        Self { include_dir, lib_dir, omp_lib_dir, tbb_lib_dir }
    }
}

fn mkl_root_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(v) = env::var_os("MKLROOT") {
        out.push(PathBuf::from(v));
    }
    if let Some(v) = env::var_os("ONEAPI_ROOT") {
        let root = PathBuf::from(v);
        out.push(root.join("mkl").join("latest"));
    }
    // Standard Windows install.
    if cfg!(windows) {
        for base in [
            "C:/Program Files (x86)/Intel/oneAPI",
            "C:/Program Files/Intel/oneAPI",
        ] {
            out.push(PathBuf::from(base).join("mkl").join("latest"));
            // Also try the most recent versioned install if `latest` is missing.
            if let Ok(entries) = std::fs::read_dir(format!("{base}/mkl")) {
                let mut versions: Vec<_> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.is_dir() && p.file_name().is_some_and(|n| n != "latest")
                    })
                    .collect();
                // Naive lexicographic sort gives us ascending order; pick the last.
                versions.sort();
                if let Some(latest) = versions.last() {
                    out.push(latest.clone());
                }
            }
        }
    } else {
        for base in ["/opt/intel/oneapi", "/usr/local/intel/oneapi"] {
            out.push(PathBuf::from(base).join("mkl").join("latest"));
        }
    }
    out
}

fn find_compiler_lib(cfg: &Config) -> Option<PathBuf> {
    // Intel OpenMP runtime sits in the `compiler` component, not `mkl`.
    if let Some(v) = env::var_os("ONEAPI_ROOT") {
        let candidate = PathBuf::from(&v).join("compiler").join("latest").join("lib");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    if cfg.target_os == "windows" {
        for base in [
            "C:/Program Files (x86)/Intel/oneAPI",
            "C:/Program Files/Intel/oneAPI",
        ] {
            let candidate = PathBuf::from(base).join("compiler").join("latest").join("lib");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }
    None
}

fn find_tbb_lib(cfg: &Config) -> Option<PathBuf> {
    if let Some(v) = env::var_os("ONEAPI_ROOT") {
        let candidate = PathBuf::from(&v).join("tbb").join("latest").join("lib");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    if cfg.target_os == "windows" {
        for base in [
            "C:/Program Files (x86)/Intel/oneAPI",
            "C:/Program Files/Intel/oneAPI",
        ] {
            let candidate = PathBuf::from(base).join("tbb").join("latest").join("lib");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Build the umbrella header that bindgen ingests. The default is `mkl.h`,
/// optionally extended with MPI-dependent headers when those features are on.
fn build_wrapper(out_dir: &Path, mpi: &MpiFeatures) -> PathBuf {
    let mut contents = String::from("#include <mkl.h>\n");
    if mpi.cluster_sparse_solver {
        contents.push_str("#include <mkl_cluster_sparse_solver.h>\n");
    }
    if mpi.cdft {
        contents.push_str("#include <mkl_cdft.h>\n");
    }
    if mpi.scalapack {
        contents.push_str("#include <mkl_scalapack.h>\n");
    }
    if mpi.pblas {
        contents.push_str("#include <mkl_pblas.h>\n");
    }
    if mpi.blacs || mpi.pblas || mpi.scalapack {
        contents.push_str("#include <mkl_blacs.h>\n");
    }

    let path = out_dir.join("wrapper.h");
    std::fs::write(&path, contents).expect("failed to write wrapper.h");
    path
}

fn link(kind: &str, name: &str) {
    println!("cargo:rustc-link-lib={kind}={name}");
}

fn cfg_feature(name: &str) -> bool {
    let var = format!(
        "CARGO_FEATURE_{}",
        name.replace('-', "_").to_uppercase()
    );
    env::var_os(var).is_some()
}

fn fatal(msg: &str) -> ! {
    eprintln!("error: onemkl-sys: {msg}");
    process::exit(1);
}
