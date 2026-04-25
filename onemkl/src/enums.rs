//! BLAS / LAPACK enumeration types.
//!
//! These wrap the C-level `CBLAS_*` enums in idiomatic Rust types. Each
//! variant carries a `pub const fn as_cblas` mapping back to the FFI value
//! when calling oneMKL.

use onemkl_sys as sys;

/// Storage layout of a 2-D matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major: consecutive elements of a row are contiguous.
    RowMajor,
    /// Column-major: consecutive elements of a column are contiguous (the
    /// "Fortran" / native LAPACK layout).
    ColMajor,
}

impl Layout {
    /// FFI representation suitable for passing to `cblas_*` routines.
    #[inline]
    #[must_use]
    pub const fn as_cblas(self) -> sys::CBLAS_LAYOUT::Type {
        match self {
            Self::RowMajor => sys::CBLAS_LAYOUT::CblasRowMajor,
            Self::ColMajor => sys::CBLAS_LAYOUT::CblasColMajor,
        }
    }
}

impl Default for Layout {
    /// `RowMajor`, matching Rust's natural slice ordering.
    #[inline]
    fn default() -> Self {
        Self::RowMajor
    }
}

/// Whether to operate on `A`, `Aᵀ` (transpose), or `Aᴴ` (conjugate
/// transpose).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Transpose {
    /// Use the matrix as-is.
    #[default]
    NoTrans,
    /// Use the transpose of the matrix.
    Trans,
    /// Use the conjugate transpose of the matrix.
    ///
    /// For real matrices this is equivalent to [`Transpose::Trans`].
    ConjTrans,
}

impl Transpose {
    /// FFI representation suitable for passing to `cblas_*` routines.
    #[inline]
    #[must_use]
    pub const fn as_cblas(self) -> sys::CBLAS_TRANSPOSE::Type {
        match self {
            Self::NoTrans => sys::CBLAS_TRANSPOSE::CblasNoTrans,
            Self::Trans => sys::CBLAS_TRANSPOSE::CblasTrans,
            Self::ConjTrans => sys::CBLAS_TRANSPOSE::CblasConjTrans,
        }
    }
}

/// Which triangle of a symmetric / Hermitian / triangular matrix is
/// referenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpLo {
    /// The upper triangle of the matrix is used.
    Upper,
    /// The lower triangle of the matrix is used.
    Lower,
}

impl UpLo {
    /// FFI representation suitable for passing to `cblas_*` routines.
    #[inline]
    #[must_use]
    pub const fn as_cblas(self) -> sys::CBLAS_UPLO::Type {
        match self {
            Self::Upper => sys::CBLAS_UPLO::CblasUpper,
            Self::Lower => sys::CBLAS_UPLO::CblasLower,
        }
    }
}

/// Whether a triangular matrix has unit (1) diagonal entries that need not
/// be referenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Diag {
    /// Diagonal entries are read from the matrix.
    #[default]
    NonUnit,
    /// Diagonal entries are assumed to be 1 and not read.
    Unit,
}

impl Diag {
    /// FFI representation suitable for passing to `cblas_*` routines.
    #[inline]
    #[must_use]
    pub const fn as_cblas(self) -> sys::CBLAS_DIAG::Type {
        match self {
            Self::NonUnit => sys::CBLAS_DIAG::CblasNonUnit,
            Self::Unit => sys::CBLAS_DIAG::CblasUnit,
        }
    }
}

/// Side from which a matrix is multiplied (used by `?trmm`, `?trsm`,
/// `?symm`, `?hemm`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Side {
    /// Multiply on the left: `op(A) * B`.
    #[default]
    Left,
    /// Multiply on the right: `B * op(A)`.
    Right,
}

impl Side {
    /// FFI representation suitable for passing to `cblas_*` routines.
    #[inline]
    #[must_use]
    pub const fn as_cblas(self) -> sys::CBLAS_SIDE::Type {
        match self {
            Self::Left => sys::CBLAS_SIDE::CblasLeft,
            Self::Right => sys::CBLAS_SIDE::CblasRight,
        }
    }
}
