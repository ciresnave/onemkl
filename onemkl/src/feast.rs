//! Extended Eigensolver — the FEAST contour-integration eigensolver.
//!
//! FEAST finds all eigenvalues and eigenvectors of a Hermitian /
//! symmetric matrix that lie inside a user-specified search interval
//! `[emin, emax]`. It is competitive with traditional QR-based solvers
//! when only a sub-range of the spectrum is needed.
//!
//! Currently only the dense drivers are wrapped (`?syev` for real
//! symmetric, `?heev` for complex Hermitian). CSR-input variants
//! (`?feast_scsrev` / `?feast_hcsrev`), banded variants (`?feast_sbev`
//! / `?feast_hbev`), generalized eigenvalue problems (`?feast_sygv` /
//! `?feast_hegv`), and the reverse-communication interface will be
//! added in subsequent commits.

use core::ffi::{c_char, c_int};

use num_complex::{Complex32, Complex64};
use onemkl_sys as sys;

use crate::enums::UpLo;
use crate::error::{Error, Result};

/// Result of a FEAST solve.
#[derive(Debug, Clone)]
pub struct FeastResult<T, R> {
    /// Eigenvalues in `[emin, emax]`. Length is the number of
    /// converged eigenvalues `m`.
    pub eigenvalues: Vec<R>,
    /// Eigenvectors corresponding to `eigenvalues`. Each column is one
    /// eigenvector; the matrix is column-major with leading dimension
    /// `n` (the matrix order). Total length: `n * m`.
    pub eigenvectors: Vec<T>,
    /// Number of converged eigenvalues.
    pub m: usize,
    /// Relative trace error reported by FEAST.
    pub epsout: R,
    /// Number of refinement loops used.
    pub loop_count: i32,
    /// Per-eigenpair residuals. Length matches eigenvalues.
    pub residuals: Vec<R>,
}

/// Real symmetric dense FEAST solver: returns eigenvalues of `A` in
/// `[emin, emax]`.
///
/// `m0_estimate` is an upper bound on the number of eigenvalues
/// expected in the interval; the algorithm may use less. A common
/// choice is `m0_estimate = 2 * estimate`. The buffers below are
/// sized to `m0_estimate` and trimmed to the actual count on return.
pub fn eigh_real_dense<T: FeastReal>(
    uplo: UpLo,
    a: &[T],
    n: usize,
    lda: usize,
    emin: T,
    emax: T,
    m0_estimate: usize,
) -> Result<FeastResult<T, T>> {
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    if a.len() < lda * n {
        return Err(Error::InvalidArgument(
            "A buffer is smaller than lda * n",
        ));
    }
    let m0_i: c_int = m0_estimate
        .try_into()
        .map_err(|_| Error::DimensionOverflow)?;

    let mut fpm: [c_int; 128] = [0; 128];
    unsafe { sys::feastinit(fpm.as_mut_ptr()) };

    let uplo_c: c_char = uplo.as_char() as c_char;
    let mut epsout = T::zero();
    let mut loop_count: c_int = 0;
    let mut m_actual: c_int = 0;
    let mut info: c_int = 0;
    let mut m0_inout: c_int = m0_i;
    let mut eigenvalues: Vec<T> = vec![T::zero(); m0_estimate];
    let mut eigenvectors: Vec<T> = vec![T::zero(); n * m0_estimate];
    let mut residuals: Vec<T> = vec![T::zero(); m0_estimate];

    unsafe {
        T::feast_dense_real(
            &uplo_c,
            &n_i,
            a.as_ptr(),
            &lda_i,
            fpm.as_mut_ptr(),
            &mut epsout,
            &mut loop_count,
            &emin,
            &emax,
            &mut m0_inout,
            eigenvalues.as_mut_ptr(),
            eigenvectors.as_mut_ptr(),
            &mut m_actual,
            residuals.as_mut_ptr(),
            &mut info,
        );
    }

    // info == 0 → success.
    // info == 1 → "no eigenvalue found" (warning); treat as success-with-empty.
    // info > 1 → genuine failure.
    if info > 1 || info < 0 {
        return Err(Error::LapackComputationFailure { info });
    }
    let m = m_actual.max(0) as usize;
    eigenvalues.truncate(m);
    residuals.truncate(m);
    eigenvectors.truncate(n * m);
    Ok(FeastResult {
        eigenvalues,
        eigenvectors,
        m,
        epsout,
        loop_count,
        residuals,
    })
}

/// Complex Hermitian dense FEAST solver.
pub fn eigh_complex_dense<T: FeastComplex>(
    uplo: UpLo,
    a: &[T],
    n: usize,
    lda: usize,
    emin: T::Real,
    emax: T::Real,
    m0_estimate: usize,
) -> Result<FeastResult<T, T::Real>> {
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    if a.len() < lda * n {
        return Err(Error::InvalidArgument(
            "A buffer is smaller than lda * n",
        ));
    }
    let m0_i: c_int = m0_estimate
        .try_into()
        .map_err(|_| Error::DimensionOverflow)?;

    let mut fpm: [c_int; 128] = [0; 128];
    unsafe { sys::feastinit(fpm.as_mut_ptr()) };

    let uplo_c: c_char = uplo.as_char() as c_char;
    let mut epsout = T::Real::zero();
    let mut loop_count: c_int = 0;
    let mut m_actual: c_int = 0;
    let mut info: c_int = 0;
    let mut m0_inout: c_int = m0_i;
    let mut eigenvalues: Vec<T::Real> = vec![T::Real::zero(); m0_estimate];
    let mut eigenvectors: Vec<T> = vec![T::zero_complex(); n * m0_estimate];
    let mut residuals: Vec<T::Real> = vec![T::Real::zero(); m0_estimate];

    unsafe {
        T::feast_dense_complex(
            &uplo_c,
            &n_i,
            a.as_ptr(),
            &lda_i,
            fpm.as_mut_ptr(),
            &mut epsout,
            &mut loop_count,
            &emin,
            &emax,
            &mut m0_inout,
            eigenvalues.as_mut_ptr(),
            eigenvectors.as_mut_ptr(),
            &mut m_actual,
            residuals.as_mut_ptr(),
            &mut info,
        );
    }

    // info == 0 → success.
    // info == 1 → "no eigenvalue found" (warning); treat as success-with-empty.
    // info > 1 → genuine failure.
    if info > 1 || info < 0 {
        return Err(Error::LapackComputationFailure { info });
    }
    let m = m_actual.max(0) as usize;
    eigenvalues.truncate(m);
    residuals.truncate(m);
    eigenvectors.truncate(n * m);
    Ok(FeastResult {
        eigenvalues,
        eigenvectors,
        m,
        epsout,
        loop_count,
        residuals,
    })
}

// =====================================================================
// Trait wiring
// =====================================================================

/// Real scalar types supported by FEAST dense drivers.
#[allow(missing_docs)]
pub trait FeastReal: Copy + 'static {
    fn zero() -> Self;
    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_dense_real(
        uplo: *const c_char,
        n: *const c_int,
        a: *const Self,
        lda: *const c_int,
        fpm: *mut c_int,
        epsout: *mut Self,
        loop_: *mut c_int,
        emin: *const Self,
        emax: *const Self,
        m0: *mut c_int,
        e: *mut Self,
        x: *mut Self,
        m: *mut c_int,
        res: *mut Self,
        info: *mut c_int,
    );
}

impl FeastReal for f32 {
    fn zero() -> Self {
        0.0
    }
    unsafe fn feast_dense_real(
        uplo: *const c_char,
        n: *const c_int,
        a: *const Self,
        lda: *const c_int,
        fpm: *mut c_int,
        epsout: *mut Self,
        loop_: *mut c_int,
        emin: *const Self,
        emax: *const Self,
        m0: *mut c_int,
        e: *mut Self,
        x: *mut Self,
        m: *mut c_int,
        res: *mut Self,
        info: *mut c_int,
    ) {
        unsafe {
            sys::sfeast_syev(
                uplo, n, a, lda, fpm, epsout, loop_, emin, emax, m0, e, x, m, res, info,
            )
        }
    }
}
impl FeastReal for f64 {
    fn zero() -> Self {
        0.0
    }
    unsafe fn feast_dense_real(
        uplo: *const c_char,
        n: *const c_int,
        a: *const Self,
        lda: *const c_int,
        fpm: *mut c_int,
        epsout: *mut Self,
        loop_: *mut c_int,
        emin: *const Self,
        emax: *const Self,
        m0: *mut c_int,
        e: *mut Self,
        x: *mut Self,
        m: *mut c_int,
        res: *mut Self,
        info: *mut c_int,
    ) {
        unsafe {
            sys::dfeast_syev(
                uplo, n, a, lda, fpm, epsout, loop_, emin, emax, m0, e, x, m, res, info,
            )
        }
    }
}

/// Complex scalar types supported by FEAST dense drivers.
#[allow(missing_docs)]
pub trait FeastComplex: Copy + 'static {
    /// The matching real scalar type (used for eigenvalues, residuals,
    /// and the search interval `[emin, emax]`).
    type Real: FeastReal;
    fn zero_complex() -> Self;
    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_dense_complex(
        uplo: *const c_char,
        n: *const c_int,
        a: *const Self,
        lda: *const c_int,
        fpm: *mut c_int,
        epsout: *mut Self::Real,
        loop_: *mut c_int,
        emin: *const Self::Real,
        emax: *const Self::Real,
        m0: *mut c_int,
        e: *mut Self::Real,
        x: *mut Self,
        m: *mut c_int,
        res: *mut Self::Real,
        info: *mut c_int,
    );
}

impl FeastComplex for Complex32 {
    type Real = f32;
    fn zero_complex() -> Self {
        Complex32::new(0.0, 0.0)
    }
    unsafe fn feast_dense_complex(
        uplo: *const c_char,
        n: *const c_int,
        a: *const Self,
        lda: *const c_int,
        fpm: *mut c_int,
        epsout: *mut Self::Real,
        loop_: *mut c_int,
        emin: *const Self::Real,
        emax: *const Self::Real,
        m0: *mut c_int,
        e: *mut Self::Real,
        x: *mut Self,
        m: *mut c_int,
        res: *mut Self::Real,
        info: *mut c_int,
    ) {
        unsafe {
            sys::cfeast_heev(
                uplo, n, a.cast(), lda, fpm, epsout, loop_, emin, emax, m0, e, x.cast(),
                m, res, info,
            )
        }
    }
}
impl FeastComplex for Complex64 {
    type Real = f64;
    fn zero_complex() -> Self {
        Complex64::new(0.0, 0.0)
    }
    unsafe fn feast_dense_complex(
        uplo: *const c_char,
        n: *const c_int,
        a: *const Self,
        lda: *const c_int,
        fpm: *mut c_int,
        epsout: *mut Self::Real,
        loop_: *mut c_int,
        emin: *const Self::Real,
        emax: *const Self::Real,
        m0: *mut c_int,
        e: *mut Self::Real,
        x: *mut Self,
        m: *mut c_int,
        res: *mut Self::Real,
        info: *mut c_int,
    ) {
        unsafe {
            sys::zfeast_heev(
                uplo, n, a.cast(), lda, fpm, epsout, loop_, emin, emax, m0, e, x.cast(),
                m, res, info,
            )
        }
    }
}
