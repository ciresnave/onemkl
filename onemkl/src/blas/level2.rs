//! BLAS Level 2: matrix-vector operations.
//!
//! Currently provides [`gemv`] (general matrix-vector multiply); additional
//! routines (`?ger`, `?gerc`, `?geru`, `?trmv`, `?trsv`, `?symv`, `?hemv`,
//! `?syr`, `?syr2`, `?her`, `?her2`, banded and packed variants) will be
//! added incrementally.

use crate::blas::scalar::BlasScalar;
use crate::enums::Transpose;
use crate::error::{Error, Result};
use crate::matrix::MatrixRef;
use crate::util::{dim_to_mkl_int, stride_to_mkl_int, vector_min_len};

/// `y ← alpha * op(A) * x + beta * y`, where `op(A)` is `A`, `Aᵀ`, or `Aᴴ`
/// according to `trans`.
///
/// Dimensions: if `trans` is [`Transpose::NoTrans`], `A` is `m × n`,
/// `x` has `n` elements (with stride `incx`), and `y` has `m` elements
/// (with stride `incy`). Otherwise `x` has `m` elements and `y` has `n`.
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
    let m = a.rows();
    let n = a.cols();
    let (x_len_required, y_len_required) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };

    if incx == 0 || incy == 0 {
        return Err(Error::InvalidArgument("vector stride must be non-zero"));
    }
    if x.len() < vector_min_len(x_len_required, incx) {
        return Err(Error::InvalidArgument(
            "x is too short for the implied dimension and stride",
        ));
    }
    if y.len() < vector_min_len(y_len_required, incy) {
        return Err(Error::InvalidArgument(
            "y is too short for the implied dimension and stride",
        ));
    }

    let cm = dim_to_mkl_int(m)?;
    let cn = dim_to_mkl_int(n)?;
    let lda = dim_to_mkl_int(a.leading_dim())?;
    let ix = stride_to_mkl_int(incx)?;
    let iy = stride_to_mkl_int(incy)?;

    unsafe {
        T::cblas_gemv(
            a.layout().as_cblas(),
            trans.as_cblas(),
            cm,
            cn,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            ix,
            beta,
            y.as_mut_ptr(),
            iy,
        );
    }
    Ok(())
}
