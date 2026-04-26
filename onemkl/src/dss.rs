//! Direct Sparse Solver (DSS) — alternative interface to oneMKL PARDISO
//! with a slightly more verbose but cleaner API.
//!
//! The flow is:
//!
//! 1. [`Dss::new`] creates a handle.
//! 2. [`Dss::define_structure`] supplies the matrix sparsity pattern.
//! 3. [`Dss::reorder`] permutes for fill reduction.
//! 4. [`Dss::factor_real`] / [`factor_complex`](Dss::factor_complex)
//!    performs numerical factorization.
//! 5. [`Dss::solve_real`] / [`solve_complex`](Dss::solve_complex)
//!    solves with one or more right-hand sides.
//!
//! Drop frees the handle.

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use num_complex::{Complex32, Complex64};
use onemkl_sys as sys;

use crate::error::{Error, Result};

/// Symmetry pattern of a sparse matrix passed to DSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symmetry {
    /// `MKL_DSS_NON_SYMMETRIC` — general structure.
    NonSymmetric,
    /// `MKL_DSS_SYMMETRIC` — values and structure are symmetric.
    Symmetric,
    /// `MKL_DSS_SYMMETRIC_STRUCTURE` — values may differ but pattern matches.
    SymmetricStructure,
    /// Complex variant of `NonSymmetric`.
    NonSymmetricComplex,
    /// Complex variant of `Symmetric`.
    SymmetricComplex,
    /// Complex variant of `SymmetricStructure`.
    SymmetricStructureComplex,
}

impl Symmetry {
    #[inline]
    fn as_opt(self) -> c_int {
        let v = match self {
            Self::NonSymmetric => sys::MKL_DSS_NON_SYMMETRIC,
            Self::Symmetric => sys::MKL_DSS_SYMMETRIC,
            Self::SymmetricStructure => sys::MKL_DSS_SYMMETRIC_STRUCTURE,
            Self::NonSymmetricComplex => sys::MKL_DSS_NON_SYMMETRIC_COMPLEX,
            Self::SymmetricComplex => sys::MKL_DSS_SYMMETRIC_COMPLEX,
            Self::SymmetricStructureComplex => sys::MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX,
        };
        v as c_int
    }
}

/// Definiteness option passed to [`Dss::factor_real`] /
/// [`factor_complex`](Dss::factor_complex).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Definite {
    /// `MKL_DSS_POSITIVE_DEFINITE`.
    PositiveDefinite,
    /// `MKL_DSS_INDEFINITE` (default for symmetric/non-symmetric real).
    Indefinite,
    /// `MKL_DSS_HERMITIAN_POSITIVE_DEFINITE`.
    HermitianPositiveDefinite,
    /// `MKL_DSS_HERMITIAN_INDEFINITE`.
    HermitianIndefinite,
}

impl Definite {
    #[inline]
    fn as_opt(self) -> c_int {
        let v = match self {
            Self::PositiveDefinite => sys::MKL_DSS_POSITIVE_DEFINITE,
            Self::Indefinite => sys::MKL_DSS_INDEFINITE,
            Self::HermitianPositiveDefinite => sys::MKL_DSS_HERMITIAN_POSITIVE_DEFINITE,
            Self::HermitianIndefinite => sys::MKL_DSS_HERMITIAN_INDEFINITE,
        };
        v as c_int
    }
}

/// Index base for input CSR arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IndexBase {
    /// Zero-based (set internally via `MKL_DSS_ZERO_BASED_INDEXING`).
    Zero,
    /// One-based (DSS default).
    #[default]
    One,
}

/// Owned DSS solver handle.
pub struct Dss<T> {
    handle: sys::_MKL_DSS_HANDLE_t,
    _marker: PhantomData<T>,
}

unsafe impl<T> Send for Dss<T> {}

impl<T> Dss<T> {
    /// Create a fresh DSS handle.
    pub fn new() -> Result<Self> {
        let mut handle: sys::_MKL_DSS_HANDLE_t = ptr::null_mut();
        let opt: c_int = sys::MKL_DSS_DEFAULTS as c_int;
        let status = unsafe { sys::dss_create_(&mut handle, &opt) };
        check_dss(status)?;
        Ok(Self {
            handle,
            _marker: PhantomData,
        })
    }

    /// Provide the matrix sparsity pattern. `row_ptr` is the CSR row
    /// pointer; `col_idx` is the column indices.
    pub fn define_structure(
        &mut self,
        symmetry: Symmetry,
        rows: usize,
        cols: usize,
        row_ptr: &[i32],
        col_idx: &[i32],
        indexing: IndexBase,
    ) -> Result<()> {
        let mut opt: c_int = symmetry.as_opt();
        if matches!(indexing, IndexBase::Zero) {
            opt |= sys::MKL_DSS_ZERO_BASED_INDEXING as c_int;
        }
        if row_ptr.len() != rows + 1 {
            return Err(Error::InvalidArgument(
                "row_ptr must have length rows + 1",
            ));
        }
        let nnz: c_int = col_idx
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let nrows: c_int = rows.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ncols: c_int = cols.try_into().map_err(|_| Error::DimensionOverflow)?;
        let status = unsafe {
            sys::dss_define_structure_(
                &mut self.handle,
                &opt,
                row_ptr.as_ptr(),
                &nrows,
                &ncols,
                col_idx.as_ptr(),
                &nnz,
            )
        };
        check_dss(status)
    }

    /// Reorder the matrix for fill-in reduction.
    pub fn reorder(&mut self) -> Result<()> {
        let opt: c_int = sys::MKL_DSS_DEFAULTS as c_int;
        let perm: [c_int; 0] = [];
        let status = unsafe {
            sys::dss_reorder_(&mut self.handle, &opt, perm.as_ptr())
        };
        check_dss(status)
    }
}

impl Dss<f64> {
    /// Numerical factorization of a real matrix with the supplied
    /// definiteness hint.
    pub fn factor_real(&mut self, definite: Definite, values: &[f64]) -> Result<()> {
        let opt: c_int = definite.as_opt();
        let status = unsafe {
            sys::dss_factor_real_(
                &mut self.handle,
                &opt,
                values.as_ptr().cast(),
            )
        };
        check_dss(status)
    }

    /// Solve `A * X = B` for real `B`. `b` and `x` are interpreted as
    /// `n × nrhs` column-major matrices.
    pub fn solve_real(&mut self, b: &[f64], nrhs: i32, x: &mut [f64]) -> Result<()> {
        let opt: c_int = sys::MKL_DSS_DEFAULTS as c_int;
        let status = unsafe {
            sys::dss_solve_real_(
                &mut self.handle,
                &opt,
                b.as_ptr().cast(),
                &nrhs,
                x.as_mut_ptr().cast(),
            )
        };
        check_dss(status)
    }
}

impl Dss<Complex64> {
    /// Numerical factorization of a complex matrix.
    pub fn factor_complex(
        &mut self,
        definite: Definite,
        values: &[Complex64],
    ) -> Result<()> {
        let opt: c_int = definite.as_opt();
        let status = unsafe {
            sys::dss_factor_complex_(
                &mut self.handle,
                &opt,
                values.as_ptr().cast(),
            )
        };
        check_dss(status)
    }

    /// Solve `A * X = B` for complex `B`.
    pub fn solve_complex(
        &mut self,
        b: &[Complex64],
        nrhs: i32,
        x: &mut [Complex64],
    ) -> Result<()> {
        let opt: c_int = sys::MKL_DSS_DEFAULTS as c_int;
        let status = unsafe {
            sys::dss_solve_complex_(
                &mut self.handle,
                &opt,
                b.as_ptr().cast(),
                &nrhs,
                x.as_mut_ptr().cast(),
            )
        };
        check_dss(status)
    }
}

impl Dss<Complex32> {
    /// Numerical factorization of a complex (single-precision) matrix.
    pub fn factor_complex(
        &mut self,
        definite: Definite,
        values: &[Complex32],
    ) -> Result<()> {
        let opt: c_int = definite.as_opt();
        let status = unsafe {
            sys::dss_factor_complex_(
                &mut self.handle,
                &opt,
                values.as_ptr().cast(),
            )
        };
        check_dss(status)
    }

    /// Solve `A * X = B` for complex (single-precision) `B`.
    pub fn solve_complex(
        &mut self,
        b: &[Complex32],
        nrhs: i32,
        x: &mut [Complex32],
    ) -> Result<()> {
        let opt: c_int = sys::MKL_DSS_DEFAULTS as c_int;
        let status = unsafe {
            sys::dss_solve_complex_(
                &mut self.handle,
                &opt,
                b.as_ptr().cast(),
                &nrhs,
                x.as_mut_ptr().cast(),
            )
        };
        check_dss(status)
    }
}

impl<T> Drop for Dss<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let opt: c_int = sys::MKL_DSS_DEFAULTS as c_int;
            unsafe {
                let _ = sys::dss_delete_(&mut self.handle, &opt);
            }
        }
    }
}

#[inline]
fn check_dss(status: c_int) -> Result<()> {
    // DSS uses MKL_DSS_SUCCESS = 0 for success.
    if status == 0 {
        Ok(())
    } else {
        Err(Error::PardisoStatus(status))
    }
}
