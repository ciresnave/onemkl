//! BLAS Level 3: matrix-matrix operations.
//!
//! Routines are grouped by operand structure:
//!
//! - General: [`gemm`], [`gemmt`] (writes only one triangle of `C`).
//! - Symmetric: [`symm`], [`syrk`], [`syr2k`].
//! - Hermitian (complex-only): [`hemm`], [`herk`], [`her2k`], [`gemm3m`]
//!   (Karatsuba-style fast complex `gemm`).
//! - Triangular: [`trmm`], [`trsm`].

use crate::blas::scalar::{BlasScalar, ComplexBlasScalar};
use crate::enums::{Diag, Layout, Side, Transpose, UpLo};
use crate::error::{Error, Result};
use crate::matrix::{MatrixMut, MatrixRef};
use crate::util::dim_to_mkl_int;

// =====================================================================
// gemm + gemmt + gemm3m
// =====================================================================

/// `C ← alpha * op(A) * op(B) + beta * C`.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: BlasScalar>(
    transa: Transpose,
    transb: Transpose,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &MatrixRef<'_, T>,
    beta: T,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout(), c.layout()])?;
    let (a_rows, a_cols) = effective_shape(transa, a.rows(), a.cols());
    let (b_rows, b_cols) = effective_shape(transb, b.rows(), b.cols());
    let m = c.rows();
    let n = c.cols();
    let k = a_cols;

    if a_rows != m {
        return Err(Error::InvalidArgument("rows of op(A) must equal rows of C"));
    }
    if b_cols != n {
        return Err(Error::InvalidArgument("cols of op(B) must equal cols of C"));
    }
    if b_rows != k {
        return Err(Error::InvalidArgument(
            "cols of op(A) must equal rows of op(B)",
        ));
    }

    unsafe {
        T::cblas_gemm(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

/// `gemm` that writes only the upper or lower triangle of `C`. Useful when
/// `C` is known to be symmetric / Hermitian. `C` must be square.
///
/// The other triangle of `C` is left unchanged.
#[allow(clippy::too_many_arguments)]
pub fn gemmt<T: BlasScalar>(
    uplo: UpLo,
    transa: Transpose,
    transb: Transpose,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &MatrixRef<'_, T>,
    beta: T,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout(), c.layout()])?;
    let n = ensure_square_mut(c)?;
    let (a_rows, a_cols) = effective_shape(transa, a.rows(), a.cols());
    let (b_rows, b_cols) = effective_shape(transb, b.rows(), b.cols());
    if a_rows != n || b_cols != n {
        return Err(Error::InvalidArgument(
            "op(A) rows and op(B) cols must equal n (= rows of square C)",
        ));
    }
    if a_cols != b_rows {
        return Err(Error::InvalidArgument(
            "cols of op(A) must equal rows of op(B)",
        ));
    }
    let k = a_cols;

    unsafe {
        T::cblas_gemmt(
            layout.as_cblas(),
            uplo.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

/// Complex `gemm` using a Karatsuba-style 3-multiplication algorithm.
///
/// Faster than [`gemm`] for large complex matrices but with slightly
/// different rounding behavior — see the oneMKL reference for accuracy
/// considerations. Available only for complex types.
#[allow(clippy::too_many_arguments)]
pub fn gemm3m<T: ComplexBlasScalar>(
    transa: Transpose,
    transb: Transpose,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &MatrixRef<'_, T>,
    beta: T,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout(), c.layout()])?;
    let (a_rows, a_cols) = effective_shape(transa, a.rows(), a.cols());
    let (b_rows, b_cols) = effective_shape(transb, b.rows(), b.cols());
    let m = c.rows();
    let n = c.cols();
    let k = a_cols;
    if a_rows != m || b_cols != n || b_rows != k {
        return Err(Error::InvalidArgument(
            "matrix shapes are not compatible for gemm3m",
        ));
    }

    unsafe {
        T::cblas_gemm3m(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

// =====================================================================
// symm / syrk / syr2k
// =====================================================================

/// `C ← alpha * A * B + beta * C` (`side == Left`) or
/// `C ← alpha * B * A + beta * C` (`side == Right`), where `A` is symmetric.
#[allow(clippy::too_many_arguments)]
pub fn symm<T: BlasScalar>(
    side: Side,
    uplo: UpLo,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &MatrixRef<'_, T>,
    beta: T,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout(), c.layout()])?;
    ensure_square(a)?;
    let m = c.rows();
    let n = c.cols();
    if b.rows() != m || b.cols() != n {
        return Err(Error::InvalidArgument("B and C must have the same shape"));
    }
    let expected_side_dim = match side {
        Side::Left => m,
        Side::Right => n,
    };
    if a.rows() != expected_side_dim {
        return Err(Error::InvalidArgument(
            "side dimension of A does not match C",
        ));
    }

    unsafe {
        T::cblas_symm(
            layout.as_cblas(),
            side.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

/// `C ← alpha * op(A) * op(A)ᵀ + beta * C` (symmetric rank-k update).
///
/// `C` is symmetric `n × n`; only the triangle indicated by `uplo` is read
/// and written. With [`Transpose::NoTrans`], `A` is `n × k`; otherwise `A`
/// is `k × n`.
#[allow(clippy::too_many_arguments)]
pub fn syrk<T: BlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    alpha: T,
    a: &MatrixRef<'_, T>,
    beta: T,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), c.layout()])?;
    let n = ensure_square_mut(c)?;
    let (a_rows, a_cols) = effective_shape(trans, a.rows(), a.cols());
    if a_rows != n {
        return Err(Error::InvalidArgument(
            "rows of op(A) must equal n (= rows of square C)",
        ));
    }
    let k = a_cols;
    unsafe {
        T::cblas_syrk(
            layout.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

/// `C ← alpha * (op(A) * op(B)ᵀ + op(B) * op(A)ᵀ) + beta * C`
/// (symmetric rank-2k update).
#[allow(clippy::too_many_arguments)]
pub fn syr2k<T: BlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &MatrixRef<'_, T>,
    beta: T,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout(), c.layout()])?;
    let n = ensure_square_mut(c)?;
    let (a_rows, a_cols) = effective_shape(trans, a.rows(), a.cols());
    let (b_rows, b_cols) = effective_shape(trans, b.rows(), b.cols());
    if a_rows != n || b_rows != n {
        return Err(Error::InvalidArgument(
            "rows of op(A) and op(B) must equal n",
        ));
    }
    if a_cols != b_cols {
        return Err(Error::InvalidArgument(
            "op(A) and op(B) must share the second dimension k",
        ));
    }
    let k = a_cols;
    unsafe {
        T::cblas_syr2k(
            layout.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

// =====================================================================
// hemm / herk / her2k (complex-only)
// =====================================================================

/// Hermitian matrix-matrix multiply (complex types).
#[allow(clippy::too_many_arguments)]
pub fn hemm<T: ComplexBlasScalar>(
    side: Side,
    uplo: UpLo,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &MatrixRef<'_, T>,
    beta: T,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout(), c.layout()])?;
    ensure_square(a)?;
    let m = c.rows();
    let n = c.cols();
    if b.rows() != m || b.cols() != n {
        return Err(Error::InvalidArgument("B and C must have the same shape"));
    }
    let expected_side_dim = match side {
        Side::Left => m,
        Side::Right => n,
    };
    if a.rows() != expected_side_dim {
        return Err(Error::InvalidArgument(
            "side dimension of A does not match C",
        ));
    }
    unsafe {
        T::cblas_hemm(
            layout.as_cblas(),
            side.as_cblas(),
            uplo.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

/// Hermitian rank-k update with **real** `alpha` and `beta`.
/// `C ← alpha * A * Aᴴ + beta * C` (`trans == NoTrans`) or
/// `C ← alpha * Aᴴ * A + beta * C` (`trans == ConjTrans`).
#[allow(clippy::too_many_arguments)]
pub fn herk<T: ComplexBlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    alpha: T::Real,
    a: &MatrixRef<'_, T>,
    beta: T::Real,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    if matches!(trans, Transpose::Trans) {
        return Err(Error::InvalidArgument(
            "herk only accepts NoTrans or ConjTrans (not Trans)",
        ));
    }
    let layout = ensure_layout(&[a.layout(), c.layout()])?;
    let n = ensure_square_mut(c)?;
    let (a_rows, a_cols) = effective_shape(trans, a.rows(), a.cols());
    if a_rows != n {
        return Err(Error::InvalidArgument(
            "rows of op(A) must equal n (= rows of square C)",
        ));
    }
    let k = a_cols;
    unsafe {
        T::cblas_herk(
            layout.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

/// Hermitian rank-2k update with complex `alpha` and **real** `beta`.
/// `C ← alpha * op(A) * op(B)ᴴ + conj(alpha) * op(B) * op(A)ᴴ + beta * C`.
#[allow(clippy::too_many_arguments)]
pub fn her2k<T: ComplexBlasScalar>(
    uplo: UpLo,
    trans: Transpose,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &MatrixRef<'_, T>,
    beta: T::Real,
    c: &mut MatrixMut<'_, T>,
) -> Result<()> {
    if matches!(trans, Transpose::Trans) {
        return Err(Error::InvalidArgument(
            "her2k only accepts NoTrans or ConjTrans (not Trans)",
        ));
    }
    let layout = ensure_layout(&[a.layout(), b.layout(), c.layout()])?;
    let n = ensure_square_mut(c)?;
    let (a_rows, a_cols) = effective_shape(trans, a.rows(), a.cols());
    let (b_rows, b_cols) = effective_shape(trans, b.rows(), b.cols());
    if a_rows != n || b_rows != n {
        return Err(Error::InvalidArgument(
            "rows of op(A) and op(B) must equal n",
        ));
    }
    if a_cols != b_cols {
        return Err(Error::InvalidArgument(
            "op(A) and op(B) must share the second dimension k",
        ));
    }
    let k = a_cols;
    unsafe {
        T::cblas_her2k(
            layout.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            dim_to_mkl_int(n)?,
            dim_to_mkl_int(k)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
            beta,
            c.as_mut_ptr(),
            dim_to_mkl_int(c.leading_dim())?,
        );
    }
    Ok(())
}

// =====================================================================
// trmm / trsm
// =====================================================================

/// Triangular matrix-matrix multiply.
/// `B ← alpha * op(A) * B` (`side == Left`) or
/// `B ← alpha * B * op(A)` (`side == Right`).
#[allow(clippy::too_many_arguments)]
pub fn trmm<T: BlasScalar>(
    side: Side,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    ensure_square(a)?;
    let m = b.rows();
    let n = b.cols();
    let expected_side_dim = match side {
        Side::Left => m,
        Side::Right => n,
    };
    if a.rows() != expected_side_dim {
        return Err(Error::InvalidArgument(
            "side dimension of A does not match B",
        ));
    }
    unsafe {
        T::cblas_trmm(
            layout.as_cblas(),
            side.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        );
    }
    Ok(())
}

/// Triangular solve with multiple right-hand sides.
/// `B ← alpha * op(A)⁻¹ * B` (`side == Left`) or
/// `B ← alpha * B * op(A)⁻¹` (`side == Right`).
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: BlasScalar>(
    side: Side,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    alpha: T,
    a: &MatrixRef<'_, T>,
    b: &mut MatrixMut<'_, T>,
) -> Result<()> {
    let layout = ensure_layout(&[a.layout(), b.layout()])?;
    ensure_square(a)?;
    let m = b.rows();
    let n = b.cols();
    let expected_side_dim = match side {
        Side::Left => m,
        Side::Right => n,
    };
    if a.rows() != expected_side_dim {
        return Err(Error::InvalidArgument(
            "side dimension of A does not match B",
        ));
    }
    unsafe {
        T::cblas_trsm(
            layout.as_cblas(),
            side.as_cblas(),
            uplo.as_cblas(),
            trans.as_cblas(),
            diag.as_cblas(),
            dim_to_mkl_int(m)?,
            dim_to_mkl_int(n)?,
            alpha,
            a.as_ptr(),
            dim_to_mkl_int(a.leading_dim())?,
            b.as_mut_ptr(),
            dim_to_mkl_int(b.leading_dim())?,
        );
    }
    Ok(())
}

// =====================================================================
// Helpers
// =====================================================================

#[inline]
fn effective_shape(t: Transpose, rows: usize, cols: usize) -> (usize, usize) {
    match t {
        Transpose::NoTrans => (rows, cols),
        Transpose::Trans | Transpose::ConjTrans => (cols, rows),
    }
}

fn ensure_layout(layouts: &[Layout]) -> Result<Layout> {
    let first = layouts[0];
    if layouts.iter().all(|l| *l == first) {
        Ok(first)
    } else {
        Err(Error::InvalidArgument(
            "all matrices in a Level 3 call must share the same Layout",
        ))
    }
}

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
