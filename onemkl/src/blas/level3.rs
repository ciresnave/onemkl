//! BLAS Level 3: matrix-matrix operations.
//!
//! Currently provides [`gemm`] (general matrix-matrix multiply); additional
//! routines (`?symm`, `?hemm`, `?syrk`, `?herk`, `?syr2k`, `?her2k`,
//! `?trmm`, `?trsm`) will be added incrementally.

use crate::blas::scalar::BlasScalar;
use crate::enums::{Layout, Transpose};
use crate::error::{Error, Result};
use crate::matrix::{MatrixMut, MatrixRef};
use crate::util::dim_to_mkl_int;

/// `C ← alpha * op(A) * op(B) + beta * C`.
///
/// All three matrices must use the same [`Layout`]. The shapes after
/// applying `transa` / `transb` must be compatible: with the post-`op`
/// shapes `op(A): m × k`, `op(B): k × n`, the output `C` is `m × n`.
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
    let layout = ensure_same_layout(a.layout(), b.layout(), c.layout())?;

    let (a_rows, a_cols) = effective_shape(transa, a.rows(), a.cols());
    let (b_rows, b_cols) = effective_shape(transb, b.rows(), b.cols());
    let m = c.rows();
    let n = c.cols();
    let k = a_cols;

    if a_rows != m {
        return Err(Error::InvalidArgument(
            "rows of op(A) must equal rows of C",
        ));
    }
    if b_cols != n {
        return Err(Error::InvalidArgument(
            "cols of op(B) must equal cols of C",
        ));
    }
    if b_rows != k {
        return Err(Error::InvalidArgument(
            "cols of op(A) must equal rows of op(B)",
        ));
    }

    let cm = dim_to_mkl_int(m)?;
    let cn = dim_to_mkl_int(n)?;
    let ck = dim_to_mkl_int(k)?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let ldb = dim_to_mkl_int(b.leading_dim())?;
    let ldc = dim_to_mkl_int(c.leading_dim())?;

    unsafe {
        T::cblas_gemm(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            cm,
            cn,
            ck,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        );
    }
    Ok(())
}

#[inline]
fn effective_shape(t: Transpose, rows: usize, cols: usize) -> (usize, usize) {
    match t {
        Transpose::NoTrans => (rows, cols),
        Transpose::Trans | Transpose::ConjTrans => (cols, rows),
    }
}

#[inline]
fn ensure_same_layout(a: Layout, b: Layout, c: Layout) -> Result<Layout> {
    if a == b && b == c {
        Ok(a)
    } else {
        Err(Error::InvalidArgument(
            "A, B, and C must share the same Layout (row-major or column-major)",
        ))
    }
}
