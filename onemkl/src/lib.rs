//! Safe, idiomatic Rust wrapper over Intel oneAPI Math Kernel Library
//! (oneMKL).
//!
//! For the raw FFI surface, see [`onemkl-sys`](onemkl_sys).
//!
//! # Per-domain features
//!
//! Each major oneMKL domain is gated by a Cargo feature so users can opt
//! out of unused parts. All domain features are enabled by default.
//! Foundation modules ([`error`], [`enums`], [`matrix`], [`scalar`], and
//! [`service`]) are always built.
//!
//! | Feature             | Module                  |
//! | ------------------- | ----------------------- |
//! | `blas`              | [`blas`]                |
//! | `data-fitting`      | [`data_fitting`]        |
//! | `dss`               | [`dss`]                 |
//! | `feast`             | [`feast`]               |
//! | `fft`               | [`fft`]                 |
//! | `iss`               | [`iss`]                 |
//! | `lapack`            | [`lapack`]              |
//! | `optim`             | [`optim`]               |
//! | `pardiso`           | [`pardiso`]             |
//! | `preconditioners`   | [`preconditioners`]     |
//! | `rng`               | [`rng`]                 |
//! | `sparse`            | [`sparse`]              |
//! | `vm`                | [`vm`]                  |
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

/// Convenience prelude for typical numerical-code use.
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
}
