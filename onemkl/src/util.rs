//! Internal helpers shared across the public modules.

use onemkl_sys::MKL_INT;

use crate::error::{Error, Result};

/// Convert an unsigned dimension into [`MKL_INT`], failing on overflow.
#[inline]
pub(crate) fn dim_to_mkl_int(value: usize) -> Result<MKL_INT> {
    MKL_INT::try_from(value).map_err(|_| Error::DimensionOverflow)
}

/// Convert a signed stride into [`MKL_INT`], failing on overflow.
///
/// BLAS allows negative strides (the vector is traversed from the end), so
/// we keep the value signed.
#[inline]
pub(crate) fn stride_to_mkl_int(stride: i64) -> Result<MKL_INT> {
    MKL_INT::try_from(stride).map_err(|_| Error::DimensionOverflow)
}

/// Element count derivable from a slice of length `slice_len` accessed
/// with stride `inc`. Returns `0` if the slice is empty; errors if the
/// stride is `0`.
#[inline]
pub(crate) fn derive_n(slice_len: usize, inc: i64) -> Result<usize> {
    if inc == 0 {
        return Err(Error::InvalidArgument("vector stride must be non-zero"));
    }
    if slice_len == 0 {
        return Ok(0);
    }
    let abs_inc = inc.unsigned_abs() as usize;
    Ok((slice_len - 1) / abs_inc + 1)
}

/// Validate that two BLAS vectors with separate strides address the same
/// number of elements. Returns the agreed-upon `n`.
#[inline]
pub(crate) fn check_two_vectors<T, U>(
    x: &[T],
    incx: i64,
    y: &[U],
    incy: i64,
) -> Result<usize> {
    let nx = derive_n(x.len(), incx)?;
    let ny = derive_n(y.len(), incy)?;
    if nx != ny {
        return Err(Error::InvalidArgument(
            "x and y vectors must contain the same number of elements (length / |stride|)",
        ));
    }
    Ok(nx)
}

/// Smallest contiguous buffer length that holds `n` elements at absolute
/// stride `|inc|`.
#[inline]
pub(crate) fn vector_min_len(n: usize, inc: i64) -> usize {
    if n == 0 {
        return 0;
    }
    1 + (n - 1) * inc.unsigned_abs() as usize
}
