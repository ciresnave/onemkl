//! Unified error types for the safe oneMKL wrapper.
//!
//! oneMKL surfaces failure information through several different mechanisms:
//!
//! - LAPACK routines return an `info` integer (`< 0` for an invalid argument
//!   index, `> 0` for a computational failure such as a singular matrix).
//! - Inspector-Executor Sparse BLAS routines return `sparse_status_t`.
//! - DFTI / FFT routines return `MKL_LONG` status codes.
//! - VSL (RNG, statistics, convolution) routines return `int` error codes.
//! - VM (vector math) reports errors through a thread-local mode/status pair.
//! - PARDISO and DSS use their own integer status codes.
//!
//! [`Error`] flattens these into a single Rust enum so that the public API
//! can return [`Result<T>`] uniformly. Lower-level callers that need the
//! raw status integer can match on the variant.

use core::fmt;
use std::num::TryFromIntError;

/// The result type used throughout `onemkl`.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Failure modes from any oneMKL routine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A user-supplied argument failed validation before reaching MKL — for
    /// example, mismatched dimensions, a stride of zero, or a buffer too
    /// short for the declared shape.
    InvalidArgument(&'static str),

    /// A user-supplied dimension or stride exceeded the range representable
    /// by [`onemkl_sys::MKL_INT`]. With the default `lp64` interface this is
    /// roughly `2^31`.
    DimensionOverflow,

    /// The `n`th argument to a LAPACK routine had an illegal value (LAPACK
    /// `info = -n`).
    LapackIllegalArgument {
        /// 1-based index of the offending argument, as reported by LAPACK.
        argument: i32,
    },

    /// A LAPACK computational routine reported a runtime failure (LAPACK
    /// `info > 0`). The exact semantics depend on the routine — see the
    /// MKL reference for details.
    LapackComputationFailure {
        /// Raw `info` value. Always positive.
        info: i32,
    },

    /// A sparse BLAS / sparse handle routine returned a non-success status.
    SparseStatus(SparseStatus),

    /// A DFTI / FFT descriptor routine returned an error.
    DftiStatus(i64),

    /// A VSL (RNG, statistics, convolution) routine returned an error.
    VslStatus(i32),

    /// A PARDISO solver routine reported a non-success error.
    PardisoStatus(i32),

    /// A VM (vector math) routine raised a domain error or other diagnostic.
    VmStatus(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::DimensionOverflow => f.write_str(
                "dimension or stride exceeds the range of MKL_INT for this build",
            ),
            Self::LapackIllegalArgument { argument } => {
                write!(f, "LAPACK reported argument #{argument} had an illegal value")
            }
            Self::LapackComputationFailure { info } => {
                write!(f, "LAPACK computational failure (info = {info})")
            }
            Self::SparseStatus(s) => write!(f, "sparse routine failure: {s}"),
            Self::DftiStatus(code) => write!(f, "DFTI routine failure (status = {code})"),
            Self::VslStatus(code) => write!(f, "VSL routine failure (status = {code})"),
            Self::PardisoStatus(code) => {
                write!(f, "PARDISO routine failure (error = {code})")
            }
            Self::VmStatus(code) => write!(f, "VM routine failure (status = {code:#x})"),
        }
    }
}

impl std::error::Error for Error {}

impl From<TryFromIntError> for Error {
    fn from(_: TryFromIntError) -> Self {
        Self::DimensionOverflow
    }
}

/// Outcomes from oneMKL Inspector-Executor Sparse BLAS routines.
///
/// Mirrors the C `sparse_status_t` enum, with an [`Unknown`](Self::Unknown)
/// catch-all for forward compatibility with future MKL releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseStatus {
    /// `SPARSE_STATUS_NOT_INITIALIZED`
    NotInitialized,
    /// `SPARSE_STATUS_ALLOC_FAILED`
    AllocFailed,
    /// `SPARSE_STATUS_INVALID_VALUE`
    InvalidValue,
    /// `SPARSE_STATUS_EXECUTION_FAILED`
    ExecutionFailed,
    /// `SPARSE_STATUS_INTERNAL_ERROR`
    InternalError,
    /// `SPARSE_STATUS_NOT_SUPPORTED`
    NotSupported,
    /// Any other unrecognized status value.
    Unknown(i32),
}

impl fmt::Display for SparseStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => f.write_str("handle not initialized"),
            Self::AllocFailed => f.write_str("allocation failed"),
            Self::InvalidValue => f.write_str("invalid value"),
            Self::ExecutionFailed => f.write_str("execution failed"),
            Self::InternalError => f.write_str("internal error"),
            Self::NotSupported => f.write_str("operation not supported"),
            Self::Unknown(code) => write!(f, "unknown status {code}"),
        }
    }
}
