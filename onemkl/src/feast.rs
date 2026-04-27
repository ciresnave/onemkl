//! Extended Eigensolver — the FEAST contour-integration eigensolver.
//!
//! FEAST finds all eigenvalues and eigenvectors of a Hermitian /
//! symmetric matrix that lie inside a user-specified search interval
//! `[emin, emax]`. It is competitive with traditional QR-based solvers
//! when only a sub-range of the spectrum is needed.
//!
//! The wrappers below cover the standard problem `A x = λ x` for
//! dense, sparse-CSR, and banded `A`, plus the generalized problem
//! `A x = λ B x` for dense `A` / `B`. The reverse-communication
//! interface (`?feast_srci` / `?feast_hrci`) is not yet wrapped.

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

/// Real symmetric CSR FEAST solver. Wraps `?feast_scsrev`.
///
/// `sa`, `isa`, `jsa` are 1-based CSR storage of `A`. The wrapper
/// stores only the upper or lower triangle as selected by `uplo`.
#[allow(clippy::too_many_arguments)]
pub fn eigh_real_csr<T: FeastReal>(
    uplo: UpLo,
    n: usize,
    sa: &[T],
    isa: &[i32],
    jsa: &[i32],
    emin: T,
    emax: T,
    m0_estimate: usize,
) -> Result<FeastResult<T, T>> {
    if isa.len() != n + 1 {
        return Err(Error::InvalidArgument("isa must have length n + 1"));
    }
    if jsa.len() != sa.len() {
        return Err(Error::InvalidArgument(
            "jsa and sa must have the same length",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
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
        T::feast_csr_real(
            &uplo_c, &n_i, sa.as_ptr(), isa.as_ptr(), jsa.as_ptr(),
            fpm.as_mut_ptr(), &mut epsout, &mut loop_count,
            &emin, &emax, &mut m0_inout,
            eigenvalues.as_mut_ptr(), eigenvectors.as_mut_ptr(),
            &mut m_actual, residuals.as_mut_ptr(), &mut info,
        );
    }
    finish_real_result(info, m_actual, n, eigenvalues, eigenvectors, residuals,
        epsout, loop_count)
}

/// Complex Hermitian CSR FEAST solver. Wraps `?feast_hcsrev`.
#[allow(clippy::too_many_arguments)]
pub fn eigh_complex_csr<T: FeastComplex>(
    uplo: UpLo,
    n: usize,
    sa: &[T],
    isa: &[i32],
    jsa: &[i32],
    emin: T::Real,
    emax: T::Real,
    m0_estimate: usize,
) -> Result<FeastResult<T, T::Real>> {
    if isa.len() != n + 1 {
        return Err(Error::InvalidArgument("isa must have length n + 1"));
    }
    if jsa.len() != sa.len() {
        return Err(Error::InvalidArgument(
            "jsa and sa must have the same length",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
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
        T::feast_csr_complex(
            &uplo_c, &n_i, sa.as_ptr(), isa.as_ptr(), jsa.as_ptr(),
            fpm.as_mut_ptr(), &mut epsout, &mut loop_count,
            &emin, &emax, &mut m0_inout,
            eigenvalues.as_mut_ptr(), eigenvectors.as_mut_ptr(),
            &mut m_actual, residuals.as_mut_ptr(), &mut info,
        );
    }
    finish_complex_result(info, m_actual, n, eigenvalues, eigenvectors, residuals,
        epsout, loop_count)
}

/// Real symmetric banded FEAST solver. Wraps `?feast_sbev`.
///
/// `kla` is the number of subdiagonals (or superdiagonals — must be
/// equal because `A` is symmetric). The matrix `a` is stored in LAPACK
/// band format with leading dimension `lda >= kla + 1`.
#[allow(clippy::too_many_arguments)]
pub fn eigh_real_banded<T: FeastReal>(
    uplo: UpLo,
    n: usize,
    kla: usize,
    a: &[T],
    lda: usize,
    emin: T,
    emax: T,
    m0_estimate: usize,
) -> Result<FeastResult<T, T>> {
    if a.len() < lda * n {
        return Err(Error::InvalidArgument("A buffer is smaller than lda * n"));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let kla_i: c_int = kla.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
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
        T::feast_banded_real(
            &uplo_c, &n_i, &kla_i, a.as_ptr(), &lda_i,
            fpm.as_mut_ptr(), &mut epsout, &mut loop_count,
            &emin, &emax, &mut m0_inout,
            eigenvalues.as_mut_ptr(), eigenvectors.as_mut_ptr(),
            &mut m_actual, residuals.as_mut_ptr(), &mut info,
        );
    }
    finish_real_result(info, m_actual, n, eigenvalues, eigenvectors, residuals,
        epsout, loop_count)
}

/// Complex Hermitian banded FEAST solver. Wraps `?feast_hbev`.
#[allow(clippy::too_many_arguments)]
pub fn eigh_complex_banded<T: FeastComplex>(
    uplo: UpLo,
    n: usize,
    kla: usize,
    a: &[T],
    lda: usize,
    emin: T::Real,
    emax: T::Real,
    m0_estimate: usize,
) -> Result<FeastResult<T, T::Real>> {
    if a.len() < lda * n {
        return Err(Error::InvalidArgument("A buffer is smaller than lda * n"));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let kla_i: c_int = kla.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
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
        T::feast_banded_complex(
            &uplo_c, &n_i, &kla_i, a.as_ptr(), &lda_i,
            fpm.as_mut_ptr(), &mut epsout, &mut loop_count,
            &emin, &emax, &mut m0_inout,
            eigenvalues.as_mut_ptr(), eigenvectors.as_mut_ptr(),
            &mut m_actual, residuals.as_mut_ptr(), &mut info,
        );
    }
    finish_complex_result(info, m_actual, n, eigenvalues, eigenvectors, residuals,
        epsout, loop_count)
}

/// Real symmetric generalized dense FEAST solver: solves
/// `A x = λ B x` with `A` symmetric and `B` symmetric positive
/// definite. Wraps `?feast_sygv`.
#[allow(clippy::too_many_arguments)]
pub fn gen_eigh_real_dense<T: FeastReal>(
    uplo: UpLo,
    n: usize,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    emin: T,
    emax: T,
    m0_estimate: usize,
) -> Result<FeastResult<T, T>> {
    if a.len() < lda * n || b.len() < ldb * n {
        return Err(Error::InvalidArgument(
            "A or B buffer is smaller than its leading dimension * n",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldb_i: c_int = ldb.try_into().map_err(|_| Error::DimensionOverflow)?;
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
        T::feast_dense_gen_real(
            &uplo_c, &n_i, a.as_ptr(), &lda_i, b.as_ptr(), &ldb_i,
            fpm.as_mut_ptr(), &mut epsout, &mut loop_count,
            &emin, &emax, &mut m0_inout,
            eigenvalues.as_mut_ptr(), eigenvectors.as_mut_ptr(),
            &mut m_actual, residuals.as_mut_ptr(), &mut info,
        );
    }
    finish_real_result(info, m_actual, n, eigenvalues, eigenvectors, residuals,
        epsout, loop_count)
}

/// Complex Hermitian generalized dense FEAST solver: solves
/// `A x = λ B x` with `A` Hermitian and `B` Hermitian positive
/// definite. Wraps `?feast_hegv`.
#[allow(clippy::too_many_arguments)]
pub fn gen_eigh_complex_dense<T: FeastComplex>(
    uplo: UpLo,
    n: usize,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    emin: T::Real,
    emax: T::Real,
    m0_estimate: usize,
) -> Result<FeastResult<T, T::Real>> {
    if a.len() < lda * n || b.len() < ldb * n {
        return Err(Error::InvalidArgument(
            "A or B buffer is smaller than its leading dimension * n",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldb_i: c_int = ldb.try_into().map_err(|_| Error::DimensionOverflow)?;
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
        T::feast_dense_gen_complex(
            &uplo_c, &n_i, a.as_ptr(), &lda_i, b.as_ptr(), &ldb_i,
            fpm.as_mut_ptr(), &mut epsout, &mut loop_count,
            &emin, &emax, &mut m0_inout,
            eigenvalues.as_mut_ptr(), eigenvectors.as_mut_ptr(),
            &mut m_actual, residuals.as_mut_ptr(), &mut info,
        );
    }
    finish_complex_result(info, m_actual, n, eigenvalues, eigenvectors, residuals,
        epsout, loop_count)
}

#[allow(clippy::too_many_arguments)]
fn finish_real_result<T: FeastReal>(
    info: c_int,
    m_actual: c_int,
    n: usize,
    mut eigenvalues: Vec<T>,
    mut eigenvectors: Vec<T>,
    mut residuals: Vec<T>,
    epsout: T,
    loop_count: c_int,
) -> Result<FeastResult<T, T>> {
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

#[allow(clippy::too_many_arguments)]
fn finish_complex_result<T: FeastComplex>(
    info: c_int,
    m_actual: c_int,
    n: usize,
    mut eigenvalues: Vec<T::Real>,
    mut eigenvectors: Vec<T>,
    mut residuals: Vec<T::Real>,
    epsout: T::Real,
    loop_count: c_int,
) -> Result<FeastResult<T, T::Real>> {
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

    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_csr_real(
        uplo: *const c_char,
        n: *const c_int,
        sa: *const Self,
        isa: *const c_int,
        jsa: *const c_int,
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

    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_banded_real(
        uplo: *const c_char,
        n: *const c_int,
        kla: *const c_int,
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

    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_dense_gen_real(
        uplo: *const c_char,
        n: *const c_int,
        a: *const Self,
        lda: *const c_int,
        b: *const Self,
        ldb: *const c_int,
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

macro_rules! impl_feast_real {
    ($ty:ty,
        syev=$syev:ident, scsrev=$scsrev:ident,
        sbev=$sbev:ident, sygv=$sygv:ident
    ) => {
        impl FeastReal for $ty {
            fn zero() -> Self {
                0.0
            }
            unsafe fn feast_dense_real(
                uplo: *const c_char, n: *const c_int,
                a: *const Self, lda: *const c_int,
                fpm: *mut c_int, epsout: *mut Self, loop_: *mut c_int,
                emin: *const Self, emax: *const Self,
                m0: *mut c_int, e: *mut Self, x: *mut Self,
                m: *mut c_int, res: *mut Self, info: *mut c_int,
            ) {
                unsafe {
                    sys::$syev(
                        uplo, n, a, lda, fpm, epsout, loop_, emin, emax, m0, e, x, m, res, info,
                    )
                }
            }
            unsafe fn feast_csr_real(
                uplo: *const c_char, n: *const c_int,
                sa: *const Self, isa: *const c_int, jsa: *const c_int,
                fpm: *mut c_int, epsout: *mut Self, loop_: *mut c_int,
                emin: *const Self, emax: *const Self,
                m0: *mut c_int, e: *mut Self, x: *mut Self,
                m: *mut c_int, res: *mut Self, info: *mut c_int,
            ) {
                unsafe {
                    sys::$scsrev(
                        uplo, n, sa, isa, jsa, fpm, epsout, loop_, emin, emax,
                        m0, e, x, m, res, info,
                    )
                }
            }
            unsafe fn feast_banded_real(
                uplo: *const c_char, n: *const c_int, kla: *const c_int,
                a: *const Self, lda: *const c_int,
                fpm: *mut c_int, epsout: *mut Self, loop_: *mut c_int,
                emin: *const Self, emax: *const Self,
                m0: *mut c_int, e: *mut Self, x: *mut Self,
                m: *mut c_int, res: *mut Self, info: *mut c_int,
            ) {
                unsafe {
                    sys::$sbev(
                        uplo, n, kla, a, lda, fpm, epsout, loop_, emin, emax,
                        m0, e, x, m, res, info,
                    )
                }
            }
            unsafe fn feast_dense_gen_real(
                uplo: *const c_char, n: *const c_int,
                a: *const Self, lda: *const c_int,
                b: *const Self, ldb: *const c_int,
                fpm: *mut c_int, epsout: *mut Self, loop_: *mut c_int,
                emin: *const Self, emax: *const Self,
                m0: *mut c_int, e: *mut Self, x: *mut Self,
                m: *mut c_int, res: *mut Self, info: *mut c_int,
            ) {
                unsafe {
                    sys::$sygv(
                        uplo, n, a, lda, b, ldb, fpm, epsout, loop_, emin, emax,
                        m0, e, x, m, res, info,
                    )
                }
            }
        }
    };
}

impl_feast_real!(f32,
    syev=sfeast_syev, scsrev=sfeast_scsrev, sbev=sfeast_sbev, sygv=sfeast_sygv);
impl_feast_real!(f64,
    syev=dfeast_syev, scsrev=dfeast_scsrev, sbev=dfeast_sbev, sygv=dfeast_sygv);

/// Complex scalar types supported by FEAST drivers.
#[allow(missing_docs)]
pub trait FeastComplex: Copy + 'static {
    /// The matching real scalar type (used for eigenvalues, residuals,
    /// and the search interval `[emin, emax]`).
    type Real: FeastReal;
    fn zero_complex() -> Self;
    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_dense_complex(
        uplo: *const c_char, n: *const c_int,
        a: *const Self, lda: *const c_int,
        fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
        emin: *const Self::Real, emax: *const Self::Real,
        m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
        m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_csr_complex(
        uplo: *const c_char, n: *const c_int,
        sa: *const Self, isa: *const c_int, jsa: *const c_int,
        fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
        emin: *const Self::Real, emax: *const Self::Real,
        m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
        m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_banded_complex(
        uplo: *const c_char, n: *const c_int, kla: *const c_int,
        a: *const Self, lda: *const c_int,
        fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
        emin: *const Self::Real, emax: *const Self::Real,
        m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
        m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn feast_dense_gen_complex(
        uplo: *const c_char, n: *const c_int,
        a: *const Self, lda: *const c_int,
        b: *const Self, ldb: *const c_int,
        fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
        emin: *const Self::Real, emax: *const Self::Real,
        m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
        m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
    );
}

macro_rules! impl_feast_complex {
    ($ty:ty, $real:ty,
        heev=$heev:ident, hcsrev=$hcsrev:ident,
        hbev=$hbev:ident, hegv=$hegv:ident
    ) => {
        impl FeastComplex for $ty {
            type Real = $real;
            fn zero_complex() -> Self {
                <$ty>::new(0.0, 0.0)
            }
            unsafe fn feast_dense_complex(
                uplo: *const c_char, n: *const c_int,
                a: *const Self, lda: *const c_int,
                fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
                emin: *const Self::Real, emax: *const Self::Real,
                m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
                m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
            ) {
                unsafe {
                    sys::$heev(
                        uplo, n, a.cast(), lda, fpm, epsout, loop_, emin, emax,
                        m0, e, x.cast(), m, res, info,
                    )
                }
            }
            unsafe fn feast_csr_complex(
                uplo: *const c_char, n: *const c_int,
                sa: *const Self, isa: *const c_int, jsa: *const c_int,
                fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
                emin: *const Self::Real, emax: *const Self::Real,
                m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
                m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
            ) {
                unsafe {
                    sys::$hcsrev(
                        uplo, n, sa.cast(), isa, jsa, fpm, epsout, loop_,
                        emin, emax, m0, e, x.cast(), m, res, info,
                    )
                }
            }
            unsafe fn feast_banded_complex(
                uplo: *const c_char, n: *const c_int, kla: *const c_int,
                a: *const Self, lda: *const c_int,
                fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
                emin: *const Self::Real, emax: *const Self::Real,
                m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
                m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
            ) {
                unsafe {
                    sys::$hbev(
                        uplo, n, kla, a.cast(), lda, fpm, epsout, loop_,
                        emin, emax, m0, e, x.cast(), m, res, info,
                    )
                }
            }
            unsafe fn feast_dense_gen_complex(
                uplo: *const c_char, n: *const c_int,
                a: *const Self, lda: *const c_int,
                b: *const Self, ldb: *const c_int,
                fpm: *mut c_int, epsout: *mut Self::Real, loop_: *mut c_int,
                emin: *const Self::Real, emax: *const Self::Real,
                m0: *mut c_int, e: *mut Self::Real, x: *mut Self,
                m: *mut c_int, res: *mut Self::Real, info: *mut c_int,
            ) {
                unsafe {
                    sys::$hegv(
                        uplo, n, a.cast(), lda, b.cast(), ldb, fpm, epsout, loop_,
                        emin, emax, m0, e, x.cast(), m, res, info,
                    )
                }
            }
        }
    };
}

impl_feast_complex!(Complex32, f32,
    heev=cfeast_heev, hcsrev=cfeast_hcsrev, hbev=cfeast_hbev, hegv=cfeast_hegv);
impl_feast_complex!(Complex64, f64,
    heev=zfeast_heev, hcsrev=zfeast_hcsrev, hbev=zfeast_hbev, hegv=zfeast_hegv);
