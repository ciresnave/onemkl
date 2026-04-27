//! Basic Linear Algebra Subprograms (BLAS).
//!
//! oneMKL implements the standard CBLAS interface across Levels 1, 2, and 3,
//! plus a number of "BLAS-like" extensions (batched, packed, JIT, mixed
//! precision). This module provides safe, idiomatic Rust wrappers over them.
//!
//! - [`level1`] — vector-vector operations.
//! - [`level2`] — matrix-vector operations.
//! - [`level3`] — matrix-matrix operations.
//!
//! All routines are generic over the four supported scalar types via the
//! [`BlasScalar`], [`RealBlasScalar`], and [`ComplexBlasScalar`] traits.

mod scalar;

pub mod extensions;
pub mod jit;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod mixed_precision;
pub mod packed;

pub use scalar::{BlasScalar, ComplexBlasScalar, RealBlasScalar};
