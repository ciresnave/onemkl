//! Safe, idiomatic Rust wrapper over Intel oneAPI Math Kernel Library
//! (oneMKL).
//!
//! For the raw FFI surface, see [`onemkl-sys`].
//!
//! # Coverage
//!
//! This crate is being built out incrementally. Currently implemented:
//!
//! - [`blas`] — most of BLAS Levels 1, 2, and 3
//!
//! Planned: LAPACK, VM (vector math), VSL (RNG / statistics), Sparse BLAS,
//! Sparse Solvers (PARDISO, DSS, RCI ISS), FFT (DFTI), Extended Eigensolver
//! (FEAST), Data Fitting, PDE support, Nonlinear Optimization. Each domain
//! will be added in its own module mirroring the oneMKL reference's
//! organization.
//!
//! Until a domain has a safe wrapper, the raw bindings are accessible via
//! [`sys`].

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]

pub use onemkl_sys as sys;

pub mod blas;
pub mod enums;
pub mod error;
pub mod matrix;
pub mod scalar;

mod util;

pub use enums::{Diag, Layout, Side, Transpose, UpLo};
pub use error::{Error, Result, SparseStatus};
pub use matrix::{MatrixMut, MatrixRef};
pub use scalar::{ComplexScalar, RealScalar, Scalar};

/// Convenience prelude for typical BLAS-style code.
///
/// ```
/// use onemkl::prelude::*;
/// ```
pub mod prelude {
    pub use crate::blas::{BlasScalar, ComplexBlasScalar, RealBlasScalar};
    pub use crate::enums::{Diag, Layout, Side, Transpose, UpLo};
    pub use crate::error::{Error, Result};
    pub use crate::matrix::{MatrixMut, MatrixRef};
    pub use crate::scalar::{ComplexScalar, RealScalar, Scalar};
}
