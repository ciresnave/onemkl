//! BLAS-like extensions provided by oneMKL.
//!
//! This module covers Intel-specific routines beyond the standard BLAS:
//!
//! - [`axpby`] — generalization of `axpy` that scales `y`.
//! - [`imatcopy`], [`omatcopy`], [`omatcopy2`], [`omatadd`] — combined
//!   transposition and scaling.
//!
//! Batched routines (`?gemm_batch_strided`, `?trsm_batch_strided`, etc.)
//! and the JIT / packed / mixed-precision GEMM APIs are wrapped in
//! follow-up modules.

use core::ffi::c_char;

use crate::blas::scalar::BlasScalar;
use crate::enums::{Layout, Transpose};
use crate::error::{Error, Result};
use crate::util::{check_two_vectors, dim_to_mkl_int, stride_to_mkl_int};

// =====================================================================
// axpby
// =====================================================================

/// `y ← alpha * x + beta * y`. Equivalent to `axpy` but also scales `y`.
pub fn axpby<T: BlasScalar>(alpha: T, x: &[T], beta: T, y: &mut [T]) -> Result<()> {
    axpby_inc(alpha, x, 1, beta, y, 1)
}

/// Strided variant of [`axpby`].
pub fn axpby_inc<T: BlasScalar>(
    alpha: T,
    x: &[T],
    incx: i64,
    beta: T,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    unsafe {
        T::cblas_axpby(
            cn,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            beta,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

// =====================================================================
// imatcopy / omatcopy / omatcopy2 / omatadd
// =====================================================================

/// In-place matrix transposition with scaling: `A ← alpha * op(A)`.
///
/// `rows` × `cols` describes the input matrix. `lda` is the leading
/// dimension before the operation; `ldb` is the leading dimension after
/// (which differs when `trans` actually transposes).
///
/// The buffer must hold both layouts:
///
/// - For [`Layout::RowMajor`], a buffer of `max(rows * lda, post_rows * ldb)`
///   elements (where `post_rows` is `cols` if transposed, else `rows`).
/// - For [`Layout::ColMajor`], `max(cols * lda, post_cols * ldb)` similarly.
///
/// See the oneMKL reference for the exact buffer-size formula.
#[allow(clippy::too_many_arguments)]
pub fn imatcopy<T: BlasScalar>(
    layout: Layout,
    trans: Transpose,
    rows: usize,
    cols: usize,
    alpha: T,
    ab: &mut [T],
    lda: usize,
    ldb: usize,
) -> Result<()> {
    check_imatcopy_buf(layout, trans, rows, cols, lda, ldb, ab.len())?;
    unsafe {
        T::mkl_imatcopy(
            layout.as_char() as c_char,
            trans.as_char() as c_char,
            rows,
            cols,
            alpha,
            ab.as_mut_ptr(),
            lda,
            ldb,
        );
    }
    Ok(())
}

/// Out-of-place matrix transposition with scaling: `B ← alpha * op(A)`.
///
/// `rows` × `cols` describes `A`. The output `B` has shape `op(rows × cols)`.
#[allow(clippy::too_many_arguments)]
pub fn omatcopy<T: BlasScalar>(
    layout: Layout,
    trans: Transpose,
    rows: usize,
    cols: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) -> Result<()> {
    check_matcopy_in(layout, rows, cols, lda, a.len(), "A")?;
    let (out_rows, out_cols) = match trans {
        Transpose::NoTrans => (rows, cols),
        Transpose::Trans | Transpose::ConjTrans => (cols, rows),
    };
    check_matcopy_in(layout, out_rows, out_cols, ldb, b.len(), "B")?;
    unsafe {
        T::mkl_omatcopy(
            layout.as_char() as c_char,
            trans.as_char() as c_char,
            rows,
            cols,
            alpha,
            a.as_ptr(),
            lda,
            b.as_mut_ptr(),
            ldb,
        );
    }
    Ok(())
}

/// Out-of-place 2-D strided transposition with scaling: like [`omatcopy`]
/// but accepts an extra inter-element stride along the contiguous
/// dimension (`stridea`/`strideb`).
#[allow(clippy::too_many_arguments)]
pub fn omatcopy2<T: BlasScalar>(
    layout: Layout,
    trans: Transpose,
    rows: usize,
    cols: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    stridea: usize,
    b: &mut [T],
    ldb: usize,
    strideb: usize,
) -> Result<()> {
    if stridea == 0 || strideb == 0 {
        return Err(Error::InvalidArgument(
            "matcopy2 strides must be non-zero",
        ));
    }
    unsafe {
        T::mkl_omatcopy2(
            layout.as_char() as c_char,
            trans.as_char() as c_char,
            rows,
            cols,
            alpha,
            a.as_ptr(),
            lda,
            stridea,
            b.as_mut_ptr(),
            ldb,
            strideb,
        );
    }
    Ok(())
}

/// Out-of-place add: `C ← alpha * op(A) + beta * op(B)`. `op(A)` and
/// `op(B)` must have the same shape (which becomes the shape of `C`).
#[allow(clippy::too_many_arguments)]
pub fn omatadd<T: BlasScalar>(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    rows: usize,
    cols: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    beta: T,
    b: &[T],
    ldb: usize,
    c: &mut [T],
    ldc: usize,
) -> Result<()> {
    let (a_rows, a_cols) = pre_op_shape(transa, rows, cols);
    let (b_rows, b_cols) = pre_op_shape(transb, rows, cols);
    check_matcopy_in(layout, a_rows, a_cols, lda, a.len(), "A")?;
    check_matcopy_in(layout, b_rows, b_cols, ldb, b.len(), "B")?;
    check_matcopy_in(layout, rows, cols, ldc, c.len(), "C")?;
    unsafe {
        T::mkl_omatadd(
            layout.as_char() as c_char,
            transa.as_char() as c_char,
            transb.as_char() as c_char,
            rows,
            cols,
            alpha,
            a.as_ptr(),
            lda,
            beta,
            b.as_ptr(),
            ldb,
            c.as_mut_ptr(),
            ldc,
        );
    }
    Ok(())
}

// =====================================================================
// Helpers
// =====================================================================

#[inline]
fn pre_op_shape(t: Transpose, post_rows: usize, post_cols: usize) -> (usize, usize) {
    match t {
        Transpose::NoTrans => (post_rows, post_cols),
        Transpose::Trans | Transpose::ConjTrans => (post_cols, post_rows),
    }
}

fn check_matcopy_in(
    layout: Layout,
    rows: usize,
    cols: usize,
    ld: usize,
    buf_len: usize,
    name: &'static str,
) -> Result<()> {
    let (contiguous, strided) = match layout {
        Layout::RowMajor => (cols, rows),
        Layout::ColMajor => (rows, cols),
    };
    if ld < contiguous {
        return Err(invalid_arg_for(name, "leading dimension is smaller than the contiguous dimension"));
    }
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let needed = (strided - 1)
        .checked_mul(ld)
        .and_then(|v| v.checked_add(contiguous))
        .ok_or(Error::DimensionOverflow)?;
    if buf_len < needed {
        return Err(invalid_arg_for(name, "buffer too small for declared dimensions"));
    }
    Ok(())
}

fn check_imatcopy_buf(
    layout: Layout,
    trans: Transpose,
    rows: usize,
    cols: usize,
    lda: usize,
    ldb: usize,
    buf_len: usize,
) -> Result<()> {
    // Pre-op shape: rows × cols with leading lda.
    check_matcopy_in(layout, rows, cols, lda, buf_len, "AB")?;
    // Post-op shape: op(rows × cols) with leading ldb.
    let (post_rows, post_cols) = match trans {
        Transpose::NoTrans => (rows, cols),
        Transpose::Trans | Transpose::ConjTrans => (cols, rows),
    };
    check_matcopy_in(layout, post_rows, post_cols, ldb, buf_len, "AB")
}

fn invalid_arg_for(name: &'static str, msg: &'static str) -> Error {
    let _ = msg;
    match name {
        "A" => Error::InvalidArgument("A buffer is invalid for declared dimensions"),
        "B" => Error::InvalidArgument("B buffer is invalid for declared dimensions"),
        "C" => Error::InvalidArgument("C buffer is invalid for declared dimensions"),
        "AB" => Error::InvalidArgument("AB buffer is invalid for declared dimensions"),
        _ => Error::InvalidArgument("matrix buffer is invalid for declared dimensions"),
    }
}
