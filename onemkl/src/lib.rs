//! Safe, idiomatic Rust wrapper over Intel oneAPI Math Kernel Library
//! (oneMKL).
//!
//! For the raw FFI surface, see [`onemkl-sys`](onemkl_sys).
//!
//! # Quick orientation by use case
//!
//! - **Dense linear algebra** — see [`blas`] (Level 1/2/3, batched,
//!   pack/compute, JIT, mixed-precision, compact) and [`lapack`]
//!   (solvers, factorizations, eigenvalue / SVD, auxiliary routines).
//! - **Sparse linear algebra** — see [`sparse`] (CSR/COO/CSC/BSR
//!   matrices, mv/mm/trsv, format conversion, sparse arithmetic),
//!   [`pardiso`] / [`dss`] (direct solvers), [`iss`] (CG / FGMRES),
//!   [`preconditioners`] (ILU0 / ILUT).
//! - **FFT and signal processing** — see [`fft`] (1D/2D/3D/N-D
//!   complex and real-input transforms with configurable scaling),
//!   [`pde`] (DCT / DST trigonometric transforms),
//!   [`rng::convolution`] (1D conv / corr).
//! - **Statistics and random numbers** — see [`rng`] (RNG streams,
//!   distributions) and [`rng::summary_stats`]
//!   (mean / variance / min / max / sum).
//! - **Mixed-precision ML kernels** — see
//!   [`blas::mixed_precision`] for bf16 /
//!   fp16 / FP8 / int8 GEMMs and [`blas::jit`] for
//!   runtime-specialized small kernels.
//! - **Eigensolvers** — see [`lapack`] for general drivers and
//!   [`feast`] for contour-integration solvers (find all eigenvalues
//!   in an interval).
//! - **Optimization** — see [`optim`] for trust-region nonlinear
//!   least squares.
//! - **Spline / data fitting** — see [`data_fitting`] for cubic
//!   splines (natural / Bessel / Akima / Hermite).
//! - **MKL service / runtime** — see [`service`] (threading,
//!   verbose mode, version).
//!
//! Anything not yet wrapped is reachable through the raw bindings
//! re-exported as [`sys`].
//!
//! # A minimal example
//!
//! ```no_run
//! use onemkl::prelude::*;
//! use onemkl::blas::level3::gemm;
//!
//! // Compute C = A * B + 0 * C for two 2×2 matrices.
//! let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
//! let b_data = vec![5.0_f64, 6.0, 7.0, 8.0];
//! let mut c_data = vec![0.0_f64; 4];
//! let a = MatrixRef::new(&a_data, 2, 2, Layout::RowMajor).unwrap();
//! let b = MatrixRef::new(&b_data, 2, 2, Layout::RowMajor).unwrap();
//! let mut c = MatrixMut::new(&mut c_data, 2, 2, Layout::RowMajor).unwrap();
//! gemm(
//!     Transpose::NoTrans, Transpose::NoTrans,
//!     1.0, &a, &b, 0.0, &mut c,
//! ).unwrap();
//! // c_data ≈ [19, 22, 43, 50].
//! ```
//!
//! # Per-domain features
//!
//! Each major oneMKL domain is gated by a Cargo feature so users can opt
//! out of unused parts. All domain features are enabled by default.
//! Foundation modules ([`error`], [`enums`], [`matrix`], [`scalar`], and
//! [`service`]) are always built.
//!
//! | Feature             | Module                  | What's inside |
//! | ------------------- | ----------------------- | ---- |
//! | `blas`              | [`blas`]                | Level 1/2/3, batched (strided + pointer-array), pack/compute, JIT, mixed precision (bf16/f16/FP8/int8), compact |
//! | `data-fitting`      | [`data_fitting`]        | Cubic splines (natural / Bessel / Akima / Hermite) |
//! | `dss`               | [`dss`]                 | Direct sparse solver with statistics readout |
//! | `feast`             | [`feast`]               | Contour-integration eigensolver (dense, CSR, banded, generalized) |
//! | `fft`               | [`fft`]                 | DFTI 1D / 2D / 3D / N-D, complex and real-input, configurable scaling |
//! | `iss`               | [`iss`]                 | CG / FGMRES with closure or step-by-step session API |
//! | `lapack`            | [`lapack`]              | Solvers, factorizations, eigenvalue / SVD, auxiliary (lacpy / lange / gecon / laswp / larfg) |
//! | `optim`             | [`optim`]               | Trust-region nonlinear least squares |
//! | `pardiso`           | [`pardiso`]             | Stateful direct sparse solver, pardiso_64, pivot callback |
//! | `pde`               | [`pde`]                 | Trigonometric transforms (DCT / DST) for spectral PDE solvers |
//! | `preconditioners`   | [`preconditioners`]     | ILU0 / ILUT and `apply_ilu` |
//! | `rng`               | [`rng`]                 | VSL streams, distributions, 1-D conv/corr, summary statistics |
//! | `sparse`            | [`sparse`]              | Inspector-Executor sparse: CSR/COO/CSC/BSR, mv/mm/trsv, copy/convert, add/spmm/spmmd |
//! | `vm`                | [`vm`]                  | Element-wise vector math |
//!
//! For a minimal build:
//!
//! ```toml
//! onemkl = { version = "...", default-features = false, features = [
//!     "lp64", "threading-sequential", "link-dynamic",  # required
//!     "blas", "lapack",                                # whatever you need
//! ] }
//! ```
//!
//! Until a domain has a safe wrapper, the raw bindings are accessible via
//! [`sys`].
//!
//! # Building against MKL
//!
//! The build script in `onemkl-sys` searches in this order:
//!
//! 1. `MKLROOT` environment variable.
//! 2. `ONEMKL_SYS_INCLUDE_DIR` / `ONEMKL_SYS_LIB_DIR` overrides.
//! 3. `ONEAPI_ROOT/mkl/latest`.
//! 4. Platform-standard install paths (`C:\Program Files (x86)\Intel\oneAPI`
//!    on Windows, `/opt/intel/oneapi` on Linux/macOS).
//!
//! At runtime MKL's `bin/` (or `lib/`) directory must be on the loader
//! search path so that the dynamic libraries can be found.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]

pub use onemkl_sys as sys;

// Foundation modules — always built.
pub mod enums;
pub mod error;
pub mod matrix;
pub mod scalar;
pub mod service;

mod util;

// Per-domain modules — gated.
#[cfg(feature = "blas")]
pub mod blas;
#[cfg(feature = "data-fitting")]
pub mod data_fitting;
#[cfg(feature = "dss")]
pub mod dss;
#[cfg(feature = "feast")]
pub mod feast;
#[cfg(feature = "fft")]
pub mod fft;
#[cfg(feature = "iss")]
pub mod iss;
#[cfg(feature = "lapack")]
pub mod lapack;
#[cfg(feature = "optim")]
pub mod optim;
#[cfg(feature = "pardiso")]
pub mod pardiso;
#[cfg(feature = "pde")]
pub mod pde;
#[cfg(feature = "preconditioners")]
pub mod preconditioners;
#[cfg(feature = "rng")]
pub mod rng;
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "vm")]
pub mod vm;

pub use enums::{Diag, Layout, Side, Transpose, UpLo};
pub use error::{Error, Result, SparseStatus};
pub use matrix::{MatrixMut, MatrixRef};
pub use scalar::{ComplexScalar, RealScalar, Scalar};

/// Convenience prelude exporting the items most numerical Rust code
/// will reach for: foundation enums (`Layout`, `Transpose`, `UpLo`,
/// `Side`, `Diag`), the `Error` / `Result` types, matrix views, and
/// the per-domain scalar traits when their feature is enabled.
///
/// ```
/// use onemkl::prelude::*;
/// ```
pub mod prelude {
    pub use crate::enums::{Diag, Layout, Side, Transpose, UpLo};
    pub use crate::error::{Error, Result};
    pub use crate::matrix::{MatrixMut, MatrixRef};
    pub use crate::scalar::{ComplexScalar, RealScalar, Scalar};

    #[cfg(feature = "blas")]
    pub use crate::blas::{BlasScalar, ComplexBlasScalar, RealBlasScalar};

    #[cfg(feature = "lapack")]
    pub use crate::lapack::{ComplexLapackScalar, LapackScalar, RealLapackScalar};

    #[cfg(feature = "sparse")]
    pub use crate::sparse::{IndexBase, MatrixType, Operation, SparseMatrix, SparseScalar};

    #[cfg(feature = "fft")]
    pub use crate::fft::{FftPlan, FftPlanOutOfPlace, RealFftPlan};

    #[cfg(feature = "rng")]
    pub use crate::rng::{BasicRng, Stream};
}
