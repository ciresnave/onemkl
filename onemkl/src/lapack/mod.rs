//! LAPACK linear algebra routines.
//!
//! Built on the LAPACKE C interface (`LAPACKE_*`), which handles workspace
//! allocation internally and accepts both row- and column-major matrix
//! layouts. The Rust public API mirrors that interface but converts
//! info codes into [`Result`].
//!
//! Currently implemented:
//!
//! - **Linear solvers**: [`gesv`], [`getrf`], [`getrs`], [`getri`],
//!   [`posv`], [`potrf`], [`potrs`], [`sysv`] (real-only),
//!   [`hesv`] (complex-only).
//! - **Least squares & QR**: [`geqrf`], [`gelqf`], [`orgqr`] (real),
//!   `ungqr` (complex), [`gels`], [`gelsd`].
//! - **Eigenvalue & SVD**: [`syev`] (real symmetric), [`heev`]
//!   (complex Hermitian), `geev_real` / `geev_complex`, [`gesdd`],
//!   [`gesvd`].
//!
//! Planned: banded variants, packed variants, generalized eigenvalue
//! problems, generalized SVD, expert drivers, condition number, iterative
//! refinement, equilibration, auxiliary routines (norms, etc.).

mod scalar;

pub use scalar::{ComplexLapackScalar, LapackScalar, RealLapackScalar};

use crate::enums::{Layout, Transpose, UpLo};
use crate::error::{Error, Result};
use crate::matrix::{MatrixMut, MatrixRef};
use crate::util::dim_to_mkl_int;

/// Job enum used by several LAPACK routines to indicate whether to
/// compute eigenvectors / singular vectors, and how.
///
/// Each variant maps to a one-character code defined by LAPACK. The
/// admissible variants depend on the routine — see the oneMKL
/// reference for each driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Job {
    /// `'N'` — do not compute the auxiliary vectors / matrices.
    None,
    /// `'V'` — compute and return eigenvectors (used by `?syev`,
    /// `?heev`, `?geev`).
    Compute,
    /// `'A'` — compute all of `U` / `Vᵀ` (used by `?gesdd` / `?gesvd`).
    All,
    /// `'S'` — compute the first `min(m, n)` columns / rows
    /// (used by `?gesdd` / `?gesvd`).
    Some,
    /// `'O'` — overwrite the input matrix with the singular vectors
    /// (used by `?gesdd` / `?gesvd`).
    Overwrite,
}

impl Job {
    /// One-character LAPACKE job code.
    #[inline]
    #[must_use]
    pub const fn as_char(self) -> u8 {
        match self {
            Self::None => b'N',
            Self::Compute => b'V',
            Self::All => b'A',
            Self::Some => b'S',
            Self::Overwrite => b'O',
        }
    }
}

// =====================================================================
// Internal helpers
// =====================================================================

#[inline]
fn check_info(info: i32) -> Result<()> {
    if info == 0 {
        Ok(())
    } else if info < 0 {
        Err(Error::LapackIllegalArgument { argument: -info })
    } else {
        Err(Error::LapackComputationFailure { info })
    }
}

#[inline]
fn ensure_square<T>(a: &MatrixMut<'_, T>) -> Result<usize> {
    if a.rows() != a.cols() {
        return Err(Error::InvalidArgument("matrix must be square"));
    }
    Ok(a.rows())
}

#[inline]
fn ensure_square_ref<T>(a: &MatrixRef<'_, T>) -> Result<usize> {
    if a.rows() != a.cols() {
        return Err(Error::InvalidArgument("matrix must be square"));
    }
    Ok(a.rows())
}

// =====================================================================
// Linear solvers — gesv, getrf, getrs, getri
// =====================================================================

/// Solve `A * X = B` with general `A` (LU factorization with partial
/// pivoting + substitution). On success, `b` is overwritten with `X`.
///
/// `a` is overwritten with `L \ U`. `ipiv` (length `n`) receives the
/// pivot indices.
pub fn gesv<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    ipiv: &mut [i32],
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square(a)?;
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if b.rows() != n {
        return Err(Error::InvalidArgument(
            "B must have the same number of rows as A",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_gesv(
            layout.as_lapack(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(b.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        )
    };
    check_info(info)
}

/// LU factorization of a general matrix `A = P * L * U`. `a` is
/// overwritten in place. `ipiv` (length `min(m, n)`) receives the pivots.
pub fn getrf<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    ipiv: &mut [i32],
) -> Result<()> {
    let need = a.rows().min(a.cols());
    if ipiv.len() < need {
        return Err(Error::InvalidArgument(
            "ipiv must have at least min(m, n) entries",
        ));
    }
    let info = unsafe {
        T::lapacke_getrf(
            a.layout().as_lapack(),
            dim_to_mkl_int(a.rows())?,
            dim_to_mkl_int(a.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            ipiv.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Solve `op(A) * X = B` with the LU factorization produced by [`getrf`].
pub fn getrs<T: LapackScalar>(
    trans: Transpose,
    a: &MatrixRef<'_, T>,
    ipiv: &[i32],
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square_ref(a)?;
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if b.rows() != n {
        return Err(Error::InvalidArgument(
            "B must have the same number of rows as A",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_getrs(
            layout.as_lapack(),
            trans.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(b.cols())?,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        )
    };
    check_info(info)
}

/// Compute `A⁻¹` from the LU factorization produced by [`getrf`].
pub fn getri<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    ipiv: &[i32],
) -> Result<()> {
    let n = ensure_square(a)?;
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_getri(
            a.layout().as_lapack(),
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            ipiv.as_ptr(),
        )
    };
    check_info(info)
}

// =====================================================================
// Cholesky (positive definite)
// =====================================================================

/// Solve `A * X = B` with symmetric / Hermitian positive-definite `A`.
pub fn posv<T: LapackScalar>(
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square(a)?;
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if b.rows() != n {
        return Err(Error::InvalidArgument(
            "B must have the same number of rows as A",
        ));
    }
    let info = unsafe {
        T::lapacke_posv(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(b.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        )
    };
    check_info(info)
}

/// Cholesky factorization `A = U⁻ᴴ * U` (Upper) or `A = L * L⁻ᴴ` (Lower).
pub fn potrf<T: LapackScalar>(uplo: UpLo, a: &mut MatrixMut<'_, T>) -> Result<()> {
    let n = ensure_square(a)?;
    let info = unsafe {
        T::lapacke_potrf(
            a.layout().as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` given the Cholesky factor produced by [`potrf`].
pub fn potrs<T: LapackScalar>(
    uplo: UpLo,
    a: &MatrixRef<'_, T>,
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square_ref(a)?;
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if b.rows() != n {
        return Err(Error::InvalidArgument(
            "B must have the same number of rows as A",
        ));
    }
    let info = unsafe {
        T::lapacke_potrs(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(b.cols())?,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` with symmetric `A` (Bunch-Kaufman factorization).
/// Real-only.
pub fn sysv<T: RealLapackScalar>(
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    ipiv: &mut [i32],
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square(a)?;
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if b.rows() != n {
        return Err(Error::InvalidArgument(
            "B must have the same number of rows as A",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_sysv(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(b.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` with Hermitian `A`. Complex-only.
pub fn hesv<T: ComplexLapackScalar>(
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    ipiv: &mut [i32],
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square(a)?;
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if b.rows() != n {
        return Err(Error::InvalidArgument(
            "B must have the same number of rows as A",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_hesv(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(b.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        )
    };
    check_info(info)
}

// =====================================================================
// QR / LQ factorization and Q construction
// =====================================================================

/// QR factorization `A = Q * R`. `a` is overwritten with `R` above the
/// diagonal and Householder reflectors on/below. `tau` (length
/// `min(m, n)`) receives reflector scalars.
pub fn geqrf<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    tau: &mut [T],
) -> Result<()> {
    let need = a.rows().min(a.cols());
    if tau.len() < need {
        return Err(Error::InvalidArgument(
            "tau must have at least min(m, n) entries",
        ));
    }
    let info = unsafe {
        T::lapacke_geqrf(
            a.layout().as_lapack(),
            dim_to_mkl_int(a.rows())?,
            dim_to_mkl_int(a.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            tau.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// LQ factorization `A = L * Q`.
pub fn gelqf<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    tau: &mut [T],
) -> Result<()> {
    let need = a.rows().min(a.cols());
    if tau.len() < need {
        return Err(Error::InvalidArgument(
            "tau must have at least min(m, n) entries",
        ));
    }
    let info = unsafe {
        T::lapacke_gelqf(
            a.layout().as_lapack(),
            dim_to_mkl_int(a.rows())?,
            dim_to_mkl_int(a.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            tau.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Form the explicit `Q` from a QR factorization (real types: `?orgqr`,
/// complex types: `?ungqr`). `k` is the number of reflectors used.
pub fn orgqr<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    tau: &[T],
    k: usize,
) -> Result<()> {
    if tau.len() < k {
        return Err(Error::InvalidArgument(
            "tau must have at least k entries",
        ));
    }
    let info = unsafe {
        T::lapacke_orgqr(
            a.layout().as_lapack(),
            dim_to_mkl_int(a.rows())?,
            dim_to_mkl_int(a.cols())?,
            dim_to_mkl_int(k)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            tau.as_ptr(),
        )
    };
    check_info(info)
}

/// Solve linear least-squares `min ‖A * x - b‖₂` using QR (or LQ if
/// underdetermined). On exit, the solution is written to the leading
/// rows of `b`.
pub fn gels<T: LapackScalar>(
    trans: Transpose,
    a: &mut MatrixMut<'_, T>,
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    let info = unsafe {
        T::lapacke_gels(
            layout.as_lapack(),
            trans.as_char() as core::ffi::c_char,
            dim_to_mkl_int(a.rows())?,
            dim_to_mkl_int(a.cols())?,
            dim_to_mkl_int(b.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        )
    };
    check_info(info)
}

/// Solve `min ‖A * x - b‖₂` via SVD. `s` (length `min(m, n)`) receives
/// the singular values. Returns the effective rank (number of singular
/// values larger than `rcond * sigma_max`).
pub fn gelsd<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    b: &mut MatrixMut<'_, T>,
    s: &mut [T::Real],
    rcond: T::Real,
) -> Result<i32> {
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    let need = a.rows().min(a.cols());
    if s.len() < need {
        return Err(Error::InvalidArgument(
            "s must have at least min(m, n) entries",
        ));
    }
    let mut rank: i32 = 0;
    let info = unsafe {
        T::lapacke_gelsd(
            layout.as_lapack(),
            dim_to_mkl_int(a.rows())?,
            dim_to_mkl_int(a.cols())?,
            dim_to_mkl_int(b.cols())?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            s.as_mut_ptr(),
            rcond,
            &mut rank,
        )
    };
    check_info(info)?;
    Ok(rank)
}

// =====================================================================
// Eigenvalue / SVD
// =====================================================================

/// Eigen-decomposition of a real symmetric matrix `A`. With
/// [`Job::Compute`], `a` is overwritten with the orthonormal
/// eigenvectors. `w` receives the eigenvalues in ascending order.
pub fn syev<T: RealLapackScalar>(
    jobz: Job,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    w: &mut [T],
) -> Result<()> {
    let n = ensure_square(a)?;
    if w.len() < n {
        return Err(Error::InvalidArgument(
            "w must have at least n entries",
        ));
    }
    let info = unsafe {
        T::lapacke_syev(
            a.layout().as_lapack(),
            jobz.as_char() as core::ffi::c_char,
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            w.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Eigen-decomposition of a complex Hermitian matrix `A`.
pub fn heev<T: ComplexLapackScalar>(
    jobz: Job,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    w: &mut [T::Real],
) -> Result<()> {
    let n = ensure_square(a)?;
    if w.len() < n {
        return Err(Error::InvalidArgument(
            "w must have at least n entries",
        ));
    }
    let info = unsafe {
        T::lapacke_heev(
            a.layout().as_lapack(),
            jobz.as_char() as core::ffi::c_char,
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            w.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Eigen-decomposition of a real general matrix. Eigenvalues are split
/// into real (`wr`) and imaginary (`wi`) parts.
#[allow(clippy::too_many_arguments)]
pub fn geev_real<T: RealLapackScalar>(
    jobvl: Job,
    jobvr: Job,
    a: &mut MatrixMut<'_, T>,
    wr: &mut [T],
    wi: &mut [T],
    vl: Option<&mut MatrixMut<'_, T>>,
    vr: Option<&mut MatrixMut<'_, T>>,
) -> Result<()> {
    let n = ensure_square(a)?;
    if wr.len() < n || wi.len() < n {
        return Err(Error::InvalidArgument(
            "wr and wi must have at least n entries",
        ));
    }
    let lapack_layout = a.layout().as_lapack();
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let (vl_ptr, ldvl) = match vl {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let (vr_ptr, ldvr) = match vr {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let info = unsafe {
        T::lapacke_geev(
            lapack_layout,
            jobvl.as_char() as core::ffi::c_char,
            jobvr.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            lda,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl_ptr,
            ldvl,
            vr_ptr,
            ldvr,
        )
    };
    check_info(info)
}

/// Eigen-decomposition of a complex general matrix. Eigenvalues are
/// returned as a single complex vector `w`.
#[allow(clippy::too_many_arguments)]
pub fn geev_complex<T: ComplexLapackScalar>(
    jobvl: Job,
    jobvr: Job,
    a: &mut MatrixMut<'_, T>,
    w: &mut [T],
    vl: Option<&mut MatrixMut<'_, T>>,
    vr: Option<&mut MatrixMut<'_, T>>,
) -> Result<()> {
    let n = ensure_square(a)?;
    if w.len() < n {
        return Err(Error::InvalidArgument(
            "w must have at least n entries",
        ));
    }
    let lapack_layout = a.layout().as_lapack();
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let (vl_ptr, ldvl) = match vl {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let (vr_ptr, ldvr) = match vr {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let info = unsafe {
        T::lapacke_geev_complex(
            lapack_layout,
            jobvl.as_char() as core::ffi::c_char,
            jobvr.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            lda,
            w.as_mut_ptr(),
            vl_ptr,
            ldvl,
            vr_ptr,
            ldvr,
        )
    };
    check_info(info)
}

/// SVD of a general matrix using the divide-and-conquer driver.
/// Returns `A = U * Σ * V`. `s` receives singular values; `u` and `vt`
/// receive `U` and `Vᵀ` according to `jobz`.
#[allow(clippy::too_many_arguments)]
pub fn gesdd<T: LapackScalar>(
    jobz: Job,
    a: &mut MatrixMut<'_, T>,
    s: &mut [T::Real],
    u: Option<&mut MatrixMut<'_, T>>,
    vt: Option<&mut MatrixMut<'_, T>>,
) -> Result<()> {
    let m = a.rows();
    let n = a.cols();
    if s.len() < m.min(n) {
        return Err(Error::InvalidArgument(
            "s must have at least min(m, n) entries",
        ));
    }
    let lapack_layout = a.layout().as_lapack();
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let (u_ptr, ldu) = match u {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(m.max(1))?),
    };
    let (vt_ptr, ldvt) = match vt {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let info = unsafe {
        T::lapacke_gesdd(
            lapack_layout,
            jobz.as_char() as core::ffi::c_char,
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            lda,
            s.as_mut_ptr(),
            u_ptr,
            ldu,
            vt_ptr,
            ldvt,
        )
    };
    check_info(info)
}

/// SVD of a general matrix using the QR-iteration driver. Slower but
/// more memory-efficient than [`gesdd`] for some shapes.
#[allow(clippy::too_many_arguments)]
pub fn gesvd<T: LapackScalar>(
    jobu: Job,
    jobvt: Job,
    a: &mut MatrixMut<'_, T>,
    s: &mut [T::Real],
    u: Option<&mut MatrixMut<'_, T>>,
    vt: Option<&mut MatrixMut<'_, T>>,
    superb: &mut [T::Real],
) -> Result<()> {
    let m = a.rows();
    let n = a.cols();
    if s.len() < m.min(n) {
        return Err(Error::InvalidArgument(
            "s must have at least min(m, n) entries",
        ));
    }
    if superb.len() < m.min(n).saturating_sub(1) {
        return Err(Error::InvalidArgument(
            "superb must have at least min(m, n) - 1 entries",
        ));
    }
    let lapack_layout = a.layout().as_lapack();
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let (u_ptr, ldu) = match u {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(m.max(1))?),
    };
    let (vt_ptr, ldvt) = match vt {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let info = unsafe {
        T::lapacke_gesvd(
            lapack_layout,
            jobu.as_char() as core::ffi::c_char,
            jobvt.as_char() as core::ffi::c_char,
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            lda,
            s.as_mut_ptr(),
            u_ptr,
            ldu,
            vt_ptr,
            ldvt,
            superb.as_mut_ptr(),
        )
    };
    check_info(info)
}

#[inline]
fn ensure_layout(layouts: &[Layout]) -> Result<Layout> {
    let first = layouts[0];
    if layouts.iter().all(|l| *l == first) {
        Ok(first)
    } else {
        Err(Error::InvalidArgument(
            "all matrices in a LAPACK call must share the same Layout",
        ))
    }
}

// =====================================================================
// Banded variants
// =====================================================================

/// Solve `A * X = B` with a general banded `A`. `A` is `n × n` with
/// `kl` sub-diagonals and `ku` super-diagonals, stored in CBLAS-style
/// band format with leading dimension `ldab ≥ 2*kl + ku + 1`.
#[allow(clippy::too_many_arguments)]
pub fn gbsv<T: LapackScalar>(
    layout: Layout,
    n: usize,
    kl: usize,
    ku: usize,
    nrhs: usize,
    ab: &mut [T],
    ldab: usize,
    ipiv: &mut [i32],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_gbsv(
            layout.as_lapack(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(kl)?,
            dim_to_mkl_int(ku)?,
            dim_to_mkl_int(nrhs)?,
            ab.as_mut_ptr(),
            dim_to_mkl_int(ldab)?,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// LU factorization of a general banded matrix.
#[allow(clippy::too_many_arguments)]
pub fn gbtrf<T: LapackScalar>(
    layout: Layout,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: &mut [T],
    ldab: usize,
    ipiv: &mut [i32],
) -> Result<()> {
    if ipiv.len() < m.min(n) {
        return Err(Error::InvalidArgument(
            "ipiv must have at least min(m, n) entries",
        ));
    }
    let info = unsafe {
        T::lapacke_gbtrf(
            layout.as_lapack(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(kl)?,
            dim_to_mkl_int(ku)?,
            ab.as_mut_ptr(),
            dim_to_mkl_int(ldab)?,
            ipiv.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Solve with the LU factor produced by [`gbtrf`].
#[allow(clippy::too_many_arguments)]
pub fn gbtrs<T: LapackScalar>(
    layout: Layout,
    trans: Transpose,
    n: usize,
    kl: usize,
    ku: usize,
    nrhs: usize,
    ab: &[T],
    ldab: usize,
    ipiv: &[i32],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_gbtrs(
            layout.as_lapack(),
            trans.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(kl)?,
            dim_to_mkl_int(ku)?,
            dim_to_mkl_int(nrhs)?,
            ab.as_ptr(),
            dim_to_mkl_int(ldab)?,
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` with a general tridiagonal `A` given by sub-,
/// main-, and super-diagonals `dl`, `d`, `du`.
#[allow(clippy::too_many_arguments)]
pub fn gtsv<T: LapackScalar>(
    layout: Layout,
    n: usize,
    nrhs: usize,
    dl: &mut [T],
    d: &mut [T],
    du: &mut [T],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    if dl.len() < n.saturating_sub(1) || d.len() < n || du.len() < n.saturating_sub(1) {
        return Err(Error::InvalidArgument(
            "tridiagonal arrays too short: need |dl|, |du| ≥ n-1, |d| ≥ n",
        ));
    }
    let info = unsafe {
        T::lapacke_gtsv(
            layout.as_lapack(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            dl.as_mut_ptr(),
            d.as_mut_ptr(),
            du.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// LU factorization of a general tridiagonal matrix.
pub fn gttrf<T: LapackScalar>(
    n: usize,
    dl: &mut [T],
    d: &mut [T],
    du: &mut [T],
    du2: &mut [T],
    ipiv: &mut [i32],
) -> Result<()> {
    if du2.len() < n.saturating_sub(2) || ipiv.len() < n {
        return Err(Error::InvalidArgument(
            "du2 must have ≥ n-2 entries and ipiv ≥ n entries",
        ));
    }
    let info = unsafe {
        T::lapacke_gttrf(
            dim_to_mkl_int(n)?,
            dl.as_mut_ptr(),
            d.as_mut_ptr(),
            du.as_mut_ptr(),
            du2.as_mut_ptr(),
            ipiv.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Solve with the LU factor produced by [`gttrf`].
#[allow(clippy::too_many_arguments)]
pub fn gttrs<T: LapackScalar>(
    layout: Layout,
    trans: Transpose,
    n: usize,
    nrhs: usize,
    dl: &[T],
    d: &[T],
    du: &[T],
    du2: &[T],
    ipiv: &[i32],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_gttrs(
            layout.as_lapack(),
            trans.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            dl.as_ptr(),
            d.as_ptr(),
            du.as_ptr(),
            du2.as_ptr(),
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` with symmetric / Hermitian PD banded `A`.
#[allow(clippy::too_many_arguments)]
pub fn pbsv<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    kd: usize,
    nrhs: usize,
    ab: &mut [T],
    ldab: usize,
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_pbsv(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(kd)?,
            dim_to_mkl_int(nrhs)?,
            ab.as_mut_ptr(),
            dim_to_mkl_int(ldab)?,
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Cholesky factorization of a PD banded matrix.
pub fn pbtrf<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    kd: usize,
    ab: &mut [T],
    ldab: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_pbtrf(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(kd)?,
            ab.as_mut_ptr(),
            dim_to_mkl_int(ldab)?,
        )
    };
    check_info(info)
}

/// Solve with the Cholesky factor produced by [`pbtrf`].
#[allow(clippy::too_many_arguments)]
pub fn pbtrs<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    kd: usize,
    nrhs: usize,
    ab: &[T],
    ldab: usize,
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_pbtrs(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(kd)?,
            dim_to_mkl_int(nrhs)?,
            ab.as_ptr(),
            dim_to_mkl_int(ldab)?,
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` with PD tridiagonal `A`. `d` is the real diagonal
/// (length `n`); `e` is the off-diagonal (length `n - 1`).
#[allow(clippy::too_many_arguments)]
pub fn ptsv<T: LapackScalar>(
    layout: Layout,
    n: usize,
    nrhs: usize,
    d: &mut [T::Real],
    e: &mut [T],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    if d.len() < n || e.len() < n.saturating_sub(1) {
        return Err(Error::InvalidArgument(
            "PD tridiagonal arrays too short: need |d| ≥ n, |e| ≥ n-1",
        ));
    }
    let info = unsafe {
        T::lapacke_ptsv(
            layout.as_lapack(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            d.as_mut_ptr(),
            e.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Cholesky factorization of a PD tridiagonal matrix.
pub fn pttrf<T: LapackScalar>(
    n: usize,
    d: &mut [T::Real],
    e: &mut [T],
) -> Result<()> {
    if d.len() < n || e.len() < n.saturating_sub(1) {
        return Err(Error::InvalidArgument(
            "PD tridiagonal arrays too short: need |d| ≥ n, |e| ≥ n-1",
        ));
    }
    let info = unsafe {
        T::lapacke_pttrf(dim_to_mkl_int(n)?, d.as_mut_ptr(), e.as_mut_ptr())
    };
    check_info(info)
}

/// Solve with the PD-tridiagonal Cholesky factor (real types).
pub fn pttrs_real<T: RealLapackScalar>(
    layout: Layout,
    n: usize,
    nrhs: usize,
    d: &[T],
    e: &[T],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_pttrs(
            layout.as_lapack(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            d.as_ptr(),
            e.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Solve with the PD-tridiagonal Cholesky factor (complex types).
/// Takes `uplo` because the off-diagonal `e` is complex and the
/// upper / lower convention selects the conjugation.
#[allow(clippy::too_many_arguments)]
pub fn pttrs_complex<T: ComplexLapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    nrhs: usize,
    d: &[T::Real],
    e: &[T],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_pttrs_complex(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            d.as_ptr(),
            e.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

// =====================================================================
// Packed variants
// =====================================================================

/// Solve `A * X = B` with symmetric `A` stored in packed form. `ap`
/// has length `n*(n+1)/2`. Works for all four scalar types — the
/// complex variant is "complex symmetric" (rare; the more common
/// Hermitian case is [`hpsv`]).
#[allow(clippy::too_many_arguments)]
pub fn spsv<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    nrhs: usize,
    ap: &mut [T],
    ipiv: &mut [i32],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    if ap.len() < n * (n + 1) / 2 {
        return Err(Error::InvalidArgument(
            "ap must have at least n*(n+1)/2 entries",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_spsv(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            ap.as_mut_ptr(),
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Symmetric packed Bunch-Kaufman factorization.
pub fn sptrf<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    ap: &mut [T],
    ipiv: &mut [i32],
) -> Result<()> {
    if ap.len() < n * (n + 1) / 2 {
        return Err(Error::InvalidArgument(
            "ap must have at least n*(n+1)/2 entries",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_sptrf(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            ap.as_mut_ptr(),
            ipiv.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Solve with the symmetric packed factor produced by [`sptrf`].
#[allow(clippy::too_many_arguments)]
pub fn sptrs<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    nrhs: usize,
    ap: &[T],
    ipiv: &[i32],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_sptrs(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            ap.as_ptr(),
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` with symmetric / Hermitian PD `A` stored packed.
#[allow(clippy::too_many_arguments)]
pub fn ppsv<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    nrhs: usize,
    ap: &mut [T],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    if ap.len() < n * (n + 1) / 2 {
        return Err(Error::InvalidArgument(
            "ap must have at least n*(n+1)/2 entries",
        ));
    }
    let info = unsafe {
        T::lapacke_ppsv(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            ap.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// PD packed Cholesky factorization.
pub fn pptrf<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    ap: &mut [T],
) -> Result<()> {
    if ap.len() < n * (n + 1) / 2 {
        return Err(Error::InvalidArgument(
            "ap must have at least n*(n+1)/2 entries",
        ));
    }
    let info = unsafe {
        T::lapacke_pptrf(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            ap.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Solve with the PD packed Cholesky factor produced by [`pptrf`].
#[allow(clippy::too_many_arguments)]
pub fn pptrs<T: LapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    nrhs: usize,
    ap: &[T],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_pptrs(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            ap.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Solve `A * X = B` with Hermitian `A` stored in packed form.
/// Complex-only.
#[allow(clippy::too_many_arguments)]
pub fn hpsv<T: ComplexLapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    nrhs: usize,
    ap: &mut [T],
    ipiv: &mut [i32],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    if ap.len() < n * (n + 1) / 2 {
        return Err(Error::InvalidArgument(
            "ap must have at least n*(n+1)/2 entries",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_hpsv(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            ap.as_mut_ptr(),
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

/// Hermitian packed Bunch-Kaufman factorization.
pub fn hptrf<T: ComplexLapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    ap: &mut [T],
    ipiv: &mut [i32],
) -> Result<()> {
    if ap.len() < n * (n + 1) / 2 {
        return Err(Error::InvalidArgument(
            "ap must have at least n*(n+1)/2 entries",
        ));
    }
    if ipiv.len() < n {
        return Err(Error::InvalidArgument("ipiv must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_hptrf(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            ap.as_mut_ptr(),
            ipiv.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Solve with the Hermitian packed factor produced by [`hptrf`].
#[allow(clippy::too_many_arguments)]
pub fn hptrs<T: ComplexLapackScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    nrhs: usize,
    ap: &[T],
    ipiv: &[i32],
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        T::lapacke_hptrs(
            layout.as_lapack(),
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(nrhs)?,
            ap.as_ptr(),
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            dim_to_mkl_int(ldb)?,
        )
    };
    check_info(info)
}

// =====================================================================
// Generalized eigenvalue problems
// =====================================================================

/// Type of generalized eigenproblem to solve in [`sygv`] / [`hegv`]:
///
/// 1. `A x = lambda B x`
/// 2. `A B x = lambda x`
/// 3. `B A x = lambda x`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneralizedEigenType {
    /// `A x = lambda B x` (the default).
    AxLambdaBx,
    /// `A B x = lambda x`.
    AbxLambdaX,
    /// `B A x = lambda x`.
    BaxLambdaX,
}

impl GeneralizedEigenType {
    #[inline]
    fn as_int(self) -> core::ffi::c_int {
        match self {
            Self::AxLambdaBx => 1,
            Self::AbxLambdaX => 2,
            Self::BaxLambdaX => 3,
        }
    }
}

/// Solve a real symmetric-definite generalized eigenproblem.
/// `A` and `B` are real symmetric `n × n`; `B` must additionally be
/// positive-definite. Eigenvalues are written to `w`. With
/// [`Job::Compute`] the eigenvectors overwrite `A`.
#[allow(clippy::too_many_arguments)]
pub fn sygv<T: RealLapackScalar>(
    itype: GeneralizedEigenType,
    jobz: Job,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    b: &mut MatrixMut<'_, T>,
    w: &mut [T],
) -> Result<()> {
    let n = ensure_square(a)?;
    if b.rows() != n || b.cols() != n {
        return Err(Error::InvalidArgument("B must have the same shape as A"));
    }
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if w.len() < n {
        return Err(Error::InvalidArgument("w must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_sygv(
            layout.as_lapack(),
            itype.as_int(),
            jobz.as_char() as core::ffi::c_char,
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            w.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Solve a complex Hermitian-definite generalized eigenproblem.
/// `A` is Hermitian; `B` is Hermitian positive-definite. Eigenvalues
/// `w` are real (Hermitian guarantees this).
#[allow(clippy::too_many_arguments)]
pub fn hegv<T: ComplexLapackScalar>(
    itype: GeneralizedEigenType,
    jobz: Job,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    b: &mut MatrixMut<'_, T>,
    w: &mut [T::Real],
) -> Result<()> {
    let n = ensure_square(a)?;
    if b.rows() != n || b.cols() != n {
        return Err(Error::InvalidArgument("B must have the same shape as A"));
    }
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if w.len() < n {
        return Err(Error::InvalidArgument("w must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_hegv(
            layout.as_lapack(),
            itype.as_int(),
            jobz.as_char() as core::ffi::c_char,
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            w.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Real general (non-symmetric) generalized eigenproblem
/// `A x = lambda B x`. Eigenvalues are returned as
/// `(alphar + i * alphai) / beta`. Real types only.
#[allow(clippy::too_many_arguments)]
pub fn ggev_real<T: RealLapackScalar>(
    jobvl: Job,
    jobvr: Job,
    a: &mut MatrixMut<'_, T>,
    b: &mut MatrixMut<'_, T>,
    alphar: &mut [T],
    alphai: &mut [T],
    beta: &mut [T],
    vl: Option<&mut MatrixMut<'_, T>>,
    vr: Option<&mut MatrixMut<'_, T>>,
) -> Result<()> {
    let n = ensure_square(a)?;
    if b.rows() != n || b.cols() != n {
        return Err(Error::InvalidArgument("B must have the same shape as A"));
    }
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if alphar.len() < n || alphai.len() < n || beta.len() < n {
        return Err(Error::InvalidArgument(
            "alphar / alphai / beta must each have at least n entries",
        ));
    }
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let ldb = dim_to_mkl_int(b.leading_dim())?;
    let (vl_ptr, ldvl) = match vl {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let (vr_ptr, ldvr) = match vr {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let info = unsafe {
        T::lapacke_ggev(
            layout.as_lapack(),
            jobvl.as_char() as core::ffi::c_char,
            jobvr.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            lda,
            b.as_mut_ptr(),
            ldb,
            alphar.as_mut_ptr(),
            alphai.as_mut_ptr(),
            beta.as_mut_ptr(),
            vl_ptr,
            ldvl,
            vr_ptr,
            ldvr,
        )
    };
    check_info(info)
}

/// Range specification for the RRR eigensolvers
/// ([`syevr`] / [`heevr`]).
#[derive(Debug, Clone, Copy)]
pub enum EigenRange<R> {
    /// Compute all eigenvalues.
    All,
    /// Compute eigenvalues in the half-open interval `(vl, vu]`.
    Values {
        /// Lower bound (exclusive).
        vl: R,
        /// Upper bound (inclusive).
        vu: R,
    },
    /// Compute the `il`-th through `iu`-th eigenvalues, sorted in
    /// ascending order. Indices are 1-based.
    Indices {
        /// 1-based start index.
        il: i32,
        /// 1-based end index (inclusive).
        iu: i32,
    },
}

impl<R> EigenRange<R> {
    fn as_char(&self) -> u8 {
        match self {
            Self::All => b'A',
            Self::Values { .. } => b'V',
            Self::Indices { .. } => b'I',
        }
    }
}

/// Symmetric eigensolver using divide-and-conquer (`?syevd`). Real-only.
pub fn syevd<T: RealLapackScalar>(
    jobz: Job,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    w: &mut [T],
) -> Result<()> {
    let n = ensure_square(a)?;
    if w.len() < n {
        return Err(Error::InvalidArgument("w must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_syevd(
            a.layout().as_lapack(),
            jobz.as_char() as core::ffi::c_char,
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            w.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Hermitian eigensolver using divide-and-conquer (`?heevd`).
/// Complex-only.
pub fn heevd<T: ComplexLapackScalar>(
    jobz: Job,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    w: &mut [T::Real],
) -> Result<()> {
    let n = ensure_square(a)?;
    if w.len() < n {
        return Err(Error::InvalidArgument("w must have at least n entries"));
    }
    let info = unsafe {
        T::lapacke_heevd(
            a.layout().as_lapack(),
            jobz.as_char() as core::ffi::c_char,
            uplo.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            w.as_mut_ptr(),
        )
    };
    check_info(info)
}

/// Output from an RRR eigensolver call. `eigenvalues` and
/// `eigenvectors` are sized to the actual number of converged
/// eigenvalues `m`. `isuppz` reports the support indices for each
/// eigenvector when `jobz == Compute`.
#[derive(Debug, Clone)]
pub struct RrrEigResult<T, R> {
    /// Number of converged eigenvalues.
    pub m: usize,
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<R>,
    /// Eigenvectors, column-major, of shape `n × m`. Empty if
    /// `jobz == Job::None`.
    pub eigenvectors: Vec<T>,
    /// Indices of the first / last non-zero entry of each
    /// eigenvector. Empty if `jobz == Job::None`.
    pub isuppz: Vec<i32>,
}

/// Symmetric eigensolver via RRR (Relatively Robust Representations).
/// Real-only. Computes the eigenvalues / eigenvectors of `A` in the
/// requested range; `abstol` is the absolute convergence tolerance
/// (pass `0.0` to use LAPACK's default).
#[allow(clippy::too_many_arguments)]
pub fn syevr<T: RealLapackScalar + Default>(
    jobz: Job,
    range: EigenRange<T>,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    abstol: T,
) -> Result<RrrEigResult<T, T>> {
    let n = ensure_square(a)?;
    let n_i = dim_to_mkl_int(n)?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let layout = a.layout();
    let range_char = range.as_char() as core::ffi::c_char;
    let (vl, vu, il, iu) = match range {
        EigenRange::All => (T::default(), T::default(), 1_i32, n.max(1) as i32),
        EigenRange::Values { vl, vu } => (vl, vu, 1_i32, n.max(1) as i32),
        EigenRange::Indices { il, iu } => (T::default(), T::default(), il, iu),
    };
    let il = dim_to_mkl_int(il.max(1) as usize)?;
    let iu = dim_to_mkl_int(iu.max(1) as usize)?;
    let want_z = matches!(jobz, Job::Compute);
    let mut m_out: i32 = 0;
    let mut w: Vec<T> = vec![T::default(); n.max(1)];
    let mut z: Vec<T> = if want_z {
        vec![T::default(); n.max(1) * n.max(1)]
    } else {
        Vec::new()
    };
    let mut isuppz: Vec<i32> = if want_z {
        vec![0_i32; 2 * n.max(1)]
    } else {
        Vec::new()
    };
    let info = unsafe {
        T::lapacke_syevr(
            layout.as_lapack(),
            jobz.as_char() as core::ffi::c_char,
            range_char,
            uplo.as_char() as core::ffi::c_char,
            n_i,
            a.as_mut_ptr(),
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m_out,
            w.as_mut_ptr(),
            if want_z { z.as_mut_ptr() } else { core::ptr::null_mut() },
            dim_to_mkl_int(n.max(1))?,
            if want_z { isuppz.as_mut_ptr() } else { core::ptr::null_mut() },
        )
    };
    check_info(info)?;
    let m = m_out.max(0) as usize;
    w.truncate(m);
    if want_z {
        z.truncate(n.max(1) * m);
        isuppz.truncate(2 * m);
    }
    Ok(RrrEigResult {
        m,
        eigenvalues: w,
        eigenvectors: z,
        isuppz,
    })
}

/// Hermitian eigensolver via RRR. Complex-only. Eigenvalues are
/// real (stored in `Self::Real`); eigenvectors are complex.
#[allow(clippy::too_many_arguments)]
pub fn heevr<T: ComplexLapackScalar>(
    jobz: Job,
    range: EigenRange<T::Real>,
    uplo: UpLo,
    a: &mut MatrixMut<'_, T>,
    abstol: T::Real,
) -> Result<RrrEigResult<T, T::Real>>
where
    T::Real: Default,
    T: Default,
{
    let n = ensure_square(a)?;
    let n_i = dim_to_mkl_int(n)?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let layout = a.layout();
    let range_char = range.as_char() as core::ffi::c_char;
    let (vl, vu, il, iu) = match range {
        EigenRange::All => (T::Real::default(), T::Real::default(), 1_i32, n.max(1) as i32),
        EigenRange::Values { vl, vu } => (vl, vu, 1_i32, n.max(1) as i32),
        EigenRange::Indices { il, iu } => (T::Real::default(), T::Real::default(), il, iu),
    };
    let il = dim_to_mkl_int(il.max(1) as usize)?;
    let iu = dim_to_mkl_int(iu.max(1) as usize)?;
    let want_z = matches!(jobz, Job::Compute);
    let mut m_out: i32 = 0;
    let mut w: Vec<T::Real> = vec![T::Real::default(); n.max(1)];
    let mut z: Vec<T> = if want_z {
        vec![T::default(); n.max(1) * n.max(1)]
    } else {
        Vec::new()
    };
    let mut isuppz: Vec<i32> = if want_z {
        vec![0_i32; 2 * n.max(1)]
    } else {
        Vec::new()
    };
    let info = unsafe {
        T::lapacke_heevr(
            layout.as_lapack(),
            jobz.as_char() as core::ffi::c_char,
            range_char,
            uplo.as_char() as core::ffi::c_char,
            n_i,
            a.as_mut_ptr(),
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m_out,
            w.as_mut_ptr(),
            if want_z { z.as_mut_ptr() } else { core::ptr::null_mut() },
            dim_to_mkl_int(n.max(1))?,
            if want_z { isuppz.as_mut_ptr() } else { core::ptr::null_mut() },
        )
    };
    check_info(info)?;
    let m = m_out.max(0) as usize;
    w.truncate(m);
    if want_z {
        z.truncate(n.max(1) * m);
        isuppz.truncate(2 * m);
    }
    Ok(RrrEigResult {
        m,
        eigenvalues: w,
        eigenvectors: z,
        isuppz,
    })
}

/// Complex general generalized eigenproblem. Eigenvalues are
/// returned as `alpha / beta`, both complex.
#[allow(clippy::too_many_arguments)]
pub fn ggev_complex<T: ComplexLapackScalar>(
    jobvl: Job,
    jobvr: Job,
    a: &mut MatrixMut<'_, T>,
    b: &mut MatrixMut<'_, T>,
    alpha: &mut [T],
    beta: &mut [T],
    vl: Option<&mut MatrixMut<'_, T>>,
    vr: Option<&mut MatrixMut<'_, T>>,
) -> Result<()> {
    let n = ensure_square(a)?;
    if b.rows() != n || b.cols() != n {
        return Err(Error::InvalidArgument("B must have the same shape as A"));
    }
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    if alpha.len() < n || beta.len() < n {
        return Err(Error::InvalidArgument(
            "alpha / beta must each have at least n entries",
        ));
    }
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let ldb = dim_to_mkl_int(b.leading_dim())?;
    let (vl_ptr, ldvl) = match vl {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let (vr_ptr, ldvr) = match vr {
        Some(m) => (m.as_mut_ptr(), dim_to_mkl_int(m.leading_dim())?),
        None => (core::ptr::null_mut(), dim_to_mkl_int(n.max(1))?),
    };
    let info = unsafe {
        T::lapacke_ggev_complex(
            layout.as_lapack(),
            jobvl.as_char() as core::ffi::c_char,
            jobvr.as_char() as core::ffi::c_char,
            dim_to_mkl_int(n)?,
            a.as_mut_ptr(),
            lda,
            b.as_mut_ptr(),
            ldb,
            alpha.as_mut_ptr(),
            beta.as_mut_ptr(),
            vl_ptr,
            ldvl,
            vr_ptr,
            ldvr,
        )
    };
    check_info(info)
}

// =====================================================================
// Auxiliary routines
// =====================================================================

/// Which triangle of a matrix to copy in [`lacpy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LacpyPart {
    /// Upper triangle including diagonal.
    Upper,
    /// Lower triangle including diagonal.
    Lower,
    /// Full matrix.
    Full,
}

impl LacpyPart {
    #[inline]
    fn as_char(self) -> core::ffi::c_char {
        match self {
            Self::Upper => b'U' as _,
            Self::Lower => b'L' as _,
            Self::Full => b'A' as _,
        }
    }
}

/// Norm to compute in [`lange`] / use in [`gecon`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixNorm {
    /// Maximum absolute element value (`'M'` in LAPACK).
    Max,
    /// 1-norm (max column-absolute-sum, `'1'` in LAPACK).
    One,
    /// Infinity-norm (max row-absolute-sum, `'I'`).
    Infinity,
    /// Frobenius norm (Euclidean of all elements, `'F'`).
    Frobenius,
}

impl MatrixNorm {
    #[inline]
    fn as_char(self) -> core::ffi::c_char {
        match self {
            Self::Max => b'M' as _,
            Self::One => b'1' as _,
            Self::Infinity => b'I' as _,
            Self::Frobenius => b'F' as _,
        }
    }
}

/// Copy a matrix region from `a` into `b`. Wraps `LAPACKE_*lacpy`.
///
/// Useful for snapshotting an in-place factorization input, or
/// promoting a triangular factor before applying a transformation.
pub fn lacpy<T: LapackScalar>(
    part: LacpyPart,
    a: &MatrixRef<'_, T>,
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(Error::InvalidArgument(
            "source and destination must have the same shape",
        ));
    }
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    let m = dim_to_mkl_int(a.rows())?;
    let n = dim_to_mkl_int(a.cols())?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let ldb = dim_to_mkl_int(b.leading_dim())?;
    let info = unsafe {
        T::lapacke_lacpy(
            layout.as_lapack(),
            part.as_char(),
            m, n,
            a.as_ptr(), lda,
            b.as_mut_ptr(), ldb,
        )
    };
    check_info(info)
}

/// Compute a matrix norm. Wraps `LAPACKE_*lange` and returns the
/// scalar norm value.
pub fn lange<T: LapackScalar>(norm: MatrixNorm, a: &MatrixRef<'_, T>) -> Result<T::Real> {
    let m = dim_to_mkl_int(a.rows())?;
    let n = dim_to_mkl_int(a.cols())?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let value = unsafe {
        T::lapacke_lange(
            a.layout().as_lapack(),
            norm.as_char(),
            m, n,
            a.as_ptr(), lda,
        )
    };
    Ok(value)
}

/// Reciprocal condition number `1 / κ(A)` from a previously LU-
/// factored matrix and its norm. Wraps `LAPACKE_*gecon`.
///
/// `a` must contain the LU factors as produced by [`getrf`].
/// `anorm` is the matrix norm of the *original* matrix in the same
/// `norm` selection passed here.
pub fn gecon<T: LapackScalar>(
    norm: MatrixNorm,
    a: &MatrixRef<'_, T>,
    anorm: T::Real,
) -> Result<T::Real>
where
    T::Real: Default,
{
    if a.rows() != a.cols() {
        return Err(Error::InvalidArgument("gecon requires a square matrix"));
    }
    let n_i = dim_to_mkl_int(a.rows())?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let mut rcond: T::Real = T::Real::default();
    let info = unsafe {
        T::lapacke_gecon(
            a.layout().as_lapack(),
            norm.as_char(),
            n_i,
            a.as_ptr(), lda,
            anorm,
            &mut rcond,
        )
    };
    check_info(info)?;
    Ok(rcond)
}

/// Apply a sequence of row interchanges to a matrix from an LAPACK
/// pivot vector. Wraps `LAPACKE_*laswp`.
///
/// `k1`, `k2` are 1-based indices into `ipiv` selecting the range of
/// pivots to apply. `incx = 1` applies them forward; `-1` applies
/// them in reverse order.
pub fn laswp<T: LapackScalar>(
    a: &mut MatrixMut<'_, T>,
    k1: i32,
    k2: i32,
    ipiv: &[i32],
    incx: i32,
) -> Result<()> {
    let n = dim_to_mkl_int(a.cols())?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    if k1 < 1 || k2 < k1 || (k2 as usize) > ipiv.len() {
        return Err(Error::InvalidArgument(
            "k1, k2 must satisfy 1 ≤ k1 ≤ k2 ≤ ipiv.len()",
        ));
    }
    let info = unsafe {
        T::lapacke_laswp(
            a.layout().as_lapack(),
            n,
            a.as_mut_ptr(), lda,
            k1, k2,
            ipiv.as_ptr(),
            incx,
        )
    };
    check_info(info)
}

/// Result of generating a Householder reflector with [`larfg`]:
/// `H = I − τ · v · vᴴ` where `v[0] = 1` (implicitly) and the rest
/// of `v` is stored back into the input slice past `alpha`.
#[derive(Debug, Clone, Copy)]
pub struct HouseholderReflector<T> {
    /// Updated leading element of the source vector (the reflector's
    /// magnitude after reflection).
    pub alpha: T,
    /// Householder coefficient `τ`.
    pub tau: T,
}

/// Generate an elementary Householder reflector. Wraps
/// `LAPACKE_*larfg`.
///
/// On entry `alpha` is the leading element and `x` (length `n - 1`)
/// holds the rest of the source vector. On return `alpha` is replaced
/// by the reflected leading element; `x` holds the lower part of the
/// reflector vector `v` (with the implicit leading `1` not stored).
pub fn larfg<T: LapackScalar>(
    n: usize,
    alpha: &mut T,
    x: &mut [T],
    incx: i32,
) -> Result<HouseholderReflector<T>>
where
    T: Default,
{
    if n == 0 {
        return Err(Error::InvalidArgument("n must be ≥ 1"));
    }
    if x.len() < n.saturating_sub(1) {
        return Err(Error::InvalidArgument("x is too short for n - 1 elements"));
    }
    let n_i = dim_to_mkl_int(n)?;
    let mut tau: T = T::default();
    let info = unsafe {
        T::lapacke_larfg(
            n_i,
            alpha,
            x.as_mut_ptr(),
            incx,
            &mut tau,
        )
    };
    check_info(info)?;
    Ok(HouseholderReflector { alpha: *alpha, tau })
}
