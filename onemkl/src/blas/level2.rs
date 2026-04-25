//! BLAS Level 2: matrix-vector operations.
//!
//! Routines are grouped by operand structure:
//!
//! - General: [`gemv`], [`gbmv`] (banded), [`ger`], [`gerc`], [`geru`].
//! - Triangular: [`trmv`], [`trsv`], [`tbmv`], [`tbsv`], [`tpmv`] (packed),
//!   [`tpsv`] (packed).
//! - Symmetric (real-only): [`symv`], [`syr`], [`syr2`], [`sbmv`], [`spmv`]
//!   (packed), [`spr`] (packed), [`spr2`] (packed).
//! - Hermitian (complex-only): [`hemv`], [`her`], [`her2`], [`hbmv`],
//!   [`hpmv`] (packed), [`hpr`] (packed), [`hpr2`] (packed).
//!
//! All routines validate dimensions before dispatching to oneMKL.

use crate::blas::scalar::{BlasScalar, ComplexBlasScalar, RealBlasScalar};
use crate::enums::{Diag, Transpose, UpLo};
use crate::error::{Error, Result};
use crate::matrix::{MatrixMut, MatrixRef};
use crate::util::{dim_to_mkl_int, stride_to_mkl_int, vector_min_len};

// =====================================================================
// General matrix-vector
// =====================================================================

/// `y ← alpha * op(A) * x + beta * y`, where `op(A)` is `A`, `Aᵀ`, or `Aᴴ`.
///
/// Dimensions: `A` is `m × n`. With [`Transpose::NoTrans`], `x` has `n`
/// elements and `y` has `m`. With [`Transpose::Trans`] /
/// [`Transpose::ConjTrans`], `x` has `m` and `y` has `n`.
#[allow(clippy::too_many_arguments)]
pub fn gemv<T: BlasScalar>(
    trans: Transpose,
    alpha: T,
    a: &MatrixRef<'_, T>,
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    let (x_required, y_required) = match trans {
        Transpose::NoTrans => (a.cols(), a.rows()),
        Transpose::Trans | Transpose::ConjTrans => (a.rows(), a.cols()),
    };
    check_vec_len(x, incx, x_required, "x")?;
    check_vec_len(y, incy, y_required, "y")?;

    unsafe {
        T::cblas_gemv(
            a.layout().as_cblas(),
            trans.as_cblas(),
            dim_to_mkl_int(a.rows())?,
            dim_to_mkl_int(a.cols())?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// `y ← alpha * op(A) * x + beta * y` for a banded matrix `A` with `kl`
/// sub-diagonals and `ku` super-diagonals.
///
/// `A` is logically `m × n`; the band is stored in a `(kl + ku + 1) × n`
/// (column-major) or `(kl + ku + 1) × m` (row-major) buffer with leading
/// dimension `lda ≥ kl + ku + 1`. See the oneMKL reference for the exact
/// layout.
#[allow(clippy::too_many_arguments)]
pub fn gbmv<T: BlasScalar>(
    trans: Transpose,
    a: &MatrixRef<'_, T>,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    if a.leading_dim() < kl + ku + 1 {
        return Err(Error::InvalidArgument(
            "banded matrix leading dimension must be at least kl + ku + 1",
        ));
    }
    let (x_required, y_required) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    check_vec_len(x, incx, x_required, "x")?;
    check_vec_len(y, incy, y_required, "y")?;

    unsafe {
        T::cblas_gbmv(
            a.layout().as_cblas(),
            trans.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(kl)?,
            dim_to_mkl_int(ku)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// `A ← alpha * x * yᵀ + A` (real rank-1 update).
///
/// `A` is `m × n`; `x` has `m` elements, `y` has `n`.
pub fn ger<T: RealBlasScalar>(
    alpha: T,
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
    a: &mut MatrixMut<'_, T>,
) -> Result<()> {
    check_vec_len(x, incx, a.rows(), "x")?;
    check_vec_len(y, incy, a.cols(), "y")?;
    let layout = a.layout();
    let m = a.rows();
    let n = a.cols();
    let lda = a.leading_dim();
    unsafe {
        T::cblas_ger(
            layout.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(lda)?,
        );
    }
    Ok(())
}

/// `A ← alpha * x * yᴴ + A` (conjugated rank-1 update for complex types).
pub fn gerc<T: ComplexBlasScalar>(
    alpha: T,
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
    a: &mut MatrixMut<'_, T>,
) -> Result<()> {
    check_vec_len(x, incx, a.rows(), "x")?;
    check_vec_len(y, incy, a.cols(), "y")?;
    let layout = a.layout();
    let m = a.rows();
    let n = a.cols();
    let lda = a.leading_dim();
    unsafe {
        T::cblas_gerc(
            layout.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(lda)?,
        );
    }
    Ok(())
}

/// `A ← alpha * x * yᵀ + A` (unconjugated rank-1 update for complex types).
pub fn geru<T: ComplexBlasScalar>(
    alpha: T,
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
    a: &mut MatrixMut<'_, T>,
) -> Result<()> {
    check_vec_len(x, incx, a.rows(), "x")?;
    check_vec_len(y, incy, a.cols(), "y")?;
    let layout = a.layout();
    let m = a.rows();
    let n = a.cols();
    let lda = a.leading_dim();
    unsafe {
        T::cblas_geru(
            layout.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(lda)?,
        );
    }
    Ok(())
}

// =====================================================================
// Triangular matrix-vector (universal)
// =====================================================================

/// `x ← op(A) * x`, where `A` is triangular.
pub fn trmv<T: BlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    a: &MatrixRef<'_, T>,
    x: &mut [T],
    incx: i64,
) -> Result<()> {
    let n = ensure_square(a)?;
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_trmv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(n)?,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
        );
    }
    Ok(())
}

/// `x ← op(A)⁻¹ * x`, where `A` is triangular and non-singular.
pub fn trsv<T: BlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    a: &MatrixRef<'_, T>,
    x: &mut [T],
    incx: i64,
) -> Result<()> {
    let n = ensure_square(a)?;
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_trsv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(n)?,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
        );
    }
    Ok(())
}

/// Banded triangular matrix-vector multiply.
///
/// `A` is `n × n` triangular with `k` off-diagonals stored in band format
/// with leading dimension `lda ≥ k + 1`.
#[allow(clippy::too_many_arguments)]
pub fn tbmv<T: BlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &MatrixRef<'_, T>,
    x: &mut [T],
    incx: i64,
) -> Result<()> {
    if a.leading_dim() < k + 1 {
        return Err(Error::InvalidArgument(
            "banded triangular leading dimension must be at least k + 1",
        ));
    }
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_tbmv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
        );
    }
    Ok(())
}

/// Banded triangular solve.
#[allow(clippy::too_many_arguments)]
pub fn tbsv<T: BlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &MatrixRef<'_, T>,
    x: &mut [T],
    incx: i64,
) -> Result<()> {
    if a.leading_dim() < k + 1 {
        return Err(Error::InvalidArgument(
            "banded triangular leading dimension must be at least k + 1",
        ));
    }
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_tbsv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
        );
    }
    Ok(())
}

/// Packed triangular matrix-vector multiply.
///
/// `ap` is a packed triangular matrix of `n*(n+1)/2` elements following
/// the CBLAS packed-storage convention (see oneMKL reference).
pub fn tpmv<T: BlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: i64,
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_tpmv(
            layout.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(n)?,
            ap.as_ptr(),
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
        );
    }
    Ok(())
}

/// Packed triangular solve.
pub fn tpsv<T: BlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: i64,
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_tpsv(
            layout.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(n)?,
            ap.as_ptr(),
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
        );
    }
    Ok(())
}

// =====================================================================
// Symmetric (real-only)
// =====================================================================

/// `y ← alpha * A * x + beta * y` for a real symmetric `A`.
#[allow(clippy::too_many_arguments)]
pub fn symv<T: RealBlasScalar>(
    uplo: UpLo,
    alpha: T,
    a: &MatrixRef<'_, T>,
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    let n = ensure_square(a)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_symv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// `A ← alpha * x * xᵀ + A` for a real symmetric `A`.
pub fn syr<T: RealBlasScalar>(
    uplo: UpLo,
    alpha: T,
    x: &[T],
    incx: i64,
    a: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square_mut(a)?;
    check_vec_len(x, incx, n, "x")?;
    let layout = a.layout();
    let lda = a.leading_dim();
    unsafe {
        T::cblas_syr(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(lda)?,
        );
    }
    Ok(())
}

/// `A ← alpha * (x * yᵀ + y * xᵀ) + A` for a real symmetric `A`.
#[allow(clippy::too_many_arguments)]
pub fn syr2<T: RealBlasScalar>(
    uplo: UpLo,
    alpha: T,
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
    a: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square_mut(a)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    let layout = a.layout();
    let lda = a.leading_dim();
    unsafe {
        T::cblas_syr2(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(lda)?,
        );
    }
    Ok(())
}

/// Symmetric banded matrix-vector multiply. `A` is `n × n` symmetric
/// banded with `k` super-diagonals (mirrored to sub-diagonals).
#[allow(clippy::too_many_arguments)]
pub fn sbmv<T: RealBlasScalar>(
    uplo: UpLo,
    n: usize,
    k: usize,
    alpha: T,
    a: &MatrixRef<'_, T>,
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    if a.leading_dim() < k + 1 {
        return Err(Error::InvalidArgument(
            "symmetric banded leading dimension must be at least k + 1",
        ));
    }
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_sbmv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// Symmetric packed matrix-vector multiply.
#[allow(clippy::too_many_arguments)]
pub fn spmv<T: RealBlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_spmv(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            ap.as_ptr(),
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// Packed symmetric rank-1 update.
pub fn spr<T: RealBlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: i64,
    ap: &mut [T],
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_spr(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            ap.as_mut_ptr(),
        );
    }
    Ok(())
}

/// Packed symmetric rank-2 update.
#[allow(clippy::too_many_arguments)]
pub fn spr2<T: RealBlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
    ap: &mut [T],
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_spr2(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
            ap.as_mut_ptr(),
        );
    }
    Ok(())
}

// =====================================================================
// Hermitian (complex-only)
// =====================================================================

/// `y ← alpha * A * x + beta * y` for a Hermitian `A` (complex types).
#[allow(clippy::too_many_arguments)]
pub fn hemv<T: ComplexBlasScalar>(
    uplo: UpLo,
    alpha: T,
    a: &MatrixRef<'_, T>,
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    let n = ensure_square(a)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_hemv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// `A ← alpha * x * xᴴ + A` for a Hermitian `A`. `alpha` is real.
pub fn her<T: ComplexBlasScalar>(
    uplo: UpLo,
    alpha: T::Real,
    x: &[T],
    incx: i64,
    a: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square_mut(a)?;
    check_vec_len(x, incx, n, "x")?;
    let layout = a.layout();
    let lda = a.leading_dim();
    unsafe {
        T::cblas_her(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(lda)?,
        );
    }
    Ok(())
}

/// `A ← alpha * x * yᴴ + conj(alpha) * y * xᴴ + A` for Hermitian `A`.
/// `alpha` is complex.
#[allow(clippy::too_many_arguments)]
pub fn her2<T: ComplexBlasScalar>(
    uplo: UpLo,
    alpha: T,
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
    a: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let n = ensure_square_mut(a)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    let layout = a.layout();
    let lda = a.leading_dim();
    unsafe {
        T::cblas_her2(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
            a.as_mut_ptr(),
            dim_to_mkl_int(lda)?,
        );
    }
    Ok(())
}

/// Hermitian banded matrix-vector multiply.
#[allow(clippy::too_many_arguments)]
pub fn hbmv<T: ComplexBlasScalar>(
    uplo: UpLo,
    n: usize,
    k: usize,
    alpha: T,
    a: &MatrixRef<'_, T>,
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    if a.leading_dim() < k + 1 {
        return Err(Error::InvalidArgument(
            "hermitian banded leading dimension must be at least k + 1",
        ));
    }
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_hbmv(
            a.layout().as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// Hermitian packed matrix-vector multiply.
#[allow(clippy::too_many_arguments)]
pub fn hpmv<T: ComplexBlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_hpmv(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            ap.as_ptr(),
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// Packed Hermitian rank-1 update with real `alpha`.
pub fn hpr<T: ComplexBlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    n: usize,
    alpha: T::Real,
    x: &[T],
    incx: i64,
    ap: &mut [T],
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    unsafe {
        T::cblas_hpr(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            ap.as_mut_ptr(),
        );
    }
    Ok(())
}

/// Packed Hermitian rank-2 update with complex `alpha`.
#[allow(clippy::too_many_arguments)]
pub fn hpr2<T: ComplexBlasScalar>(
    layout: crate::Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
    ap: &mut [T],
) -> Result<()> {
    check_packed_len(ap, n)?;
    check_vec_len(x, incx, n, "x")?;
    check_vec_len(y, incy, n, "y")?;
    unsafe {
        T::cblas_hpr2(
            layout.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(n)?,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
            ap.as_mut_ptr(),
        );
    }
    Ok(())
}

// =====================================================================
// Helpers
// =====================================================================

fn ensure_square<T>(a: &MatrixRef<'_, T>) -> Result<usize> {
    if a.rows() != a.cols() {
        return Err(Error::InvalidArgument("matrix must be square"));
    }
    Ok(a.rows())
}

fn ensure_square_mut<T>(a: &MatrixMut<'_, T>) -> Result<usize> {
    if a.rows() != a.cols() {
        return Err(Error::InvalidArgument("matrix must be square"));
    }
    Ok(a.rows())
}

fn check_vec_len<T>(
    v: &[T],
    inc: i64,
    n: usize,
    name: &'static str,
) -> Result<()> {
    if inc == 0 {
        return Err(Error::InvalidArgument("vector stride must be non-zero"));
    }
    if v.len() < vector_min_len(n, inc) {
        return Err(match name {
            "x" => Error::InvalidArgument(
                "x is too short for the implied dimension and stride",
            ),
            "y" => Error::InvalidArgument(
                "y is too short for the implied dimension and stride",
            ),
            _ => Error::InvalidArgument(
                "vector is too short for the implied dimension and stride",
            ),
        });
    }
    Ok(())
}

fn check_packed_len<T>(ap: &[T], n: usize) -> Result<()> {
    let needed = n.checked_mul(n + 1)
        .map(|v| v / 2)
        .ok_or(Error::DimensionOverflow)?;
    if ap.len() < needed {
        return Err(Error::InvalidArgument(
            "packed buffer is too short: need n*(n+1)/2 elements",
        ));
    }
    Ok(())
}
