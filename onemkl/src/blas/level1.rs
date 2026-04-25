//! BLAS Level 1: vector-vector operations.
//!
//! All routines accept slice inputs and validate dimensions before
//! dispatching to oneMKL. Unit-stride convenience wrappers (e.g.,
//! [`asum`]) call the strided variant (e.g., [`asum_inc`]) with `inc = 1`.

use crate::blas::scalar::{BlasScalar, ComplexBlasScalar, RealBlasScalar};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::util::{check_two_vectors, derive_n, dim_to_mkl_int, stride_to_mkl_int};

// =====================================================================
// asum / nrm2 — single-vector reductions.
// =====================================================================

/// Sum of absolute values: `Σ |x_i|` (or `Σ (|re| + |im|)` for complex).
pub fn asum<T: BlasScalar>(x: &[T]) -> T::Real {
    asum_inc(x, 1).expect("unit-stride asum cannot fail")
}

/// Strided variant of [`asum`].
pub fn asum_inc<T: BlasScalar>(x: &[T], incx: i64) -> Result<T::Real> {
    let n = derive_n(x.len(), incx)?;
    let n = dim_to_mkl_int(n)?;
    let incx = stride_to_mkl_int(incx)?;
    Ok(unsafe { T::cblas_asum(n, x.as_ptr(), incx) })
}

/// Euclidean norm `‖x‖₂ = sqrt(Σ |x_i|²)`.
pub fn nrm2<T: BlasScalar>(x: &[T]) -> T::Real {
    nrm2_inc(x, 1).expect("unit-stride nrm2 cannot fail")
}

/// Strided variant of [`nrm2`].
pub fn nrm2_inc<T: BlasScalar>(x: &[T], incx: i64) -> Result<T::Real> {
    let n = derive_n(x.len(), incx)?;
    let n = dim_to_mkl_int(n)?;
    let incx = stride_to_mkl_int(incx)?;
    Ok(unsafe { T::cblas_nrm2(n, x.as_ptr(), incx) })
}

/// Index of the first element with maximum absolute value.
///
/// Returns `None` if `x` is empty. Indices are zero-based.
pub fn iamax<T: BlasScalar>(x: &[T]) -> Option<usize> {
    iamax_inc(x, 1).expect("unit-stride iamax cannot fail")
}

/// Strided variant of [`iamax`].
pub fn iamax_inc<T: BlasScalar>(x: &[T], incx: i64) -> Result<Option<usize>> {
    let n = derive_n(x.len(), incx)?;
    if n == 0 {
        return Ok(None);
    }
    let cn = dim_to_mkl_int(n)?;
    let cinc = stride_to_mkl_int(incx)?;
    Ok(Some(unsafe { T::cblas_iamax(cn, x.as_ptr(), cinc) }))
}

/// Index of the first element with minimum absolute value (oneMKL
/// extension to BLAS).
///
/// Returns `None` if `x` is empty. Indices are zero-based.
pub fn iamin<T: BlasScalar>(x: &[T]) -> Option<usize> {
    iamin_inc(x, 1).expect("unit-stride iamin cannot fail")
}

/// Strided variant of [`iamin`].
pub fn iamin_inc<T: BlasScalar>(x: &[T], incx: i64) -> Result<Option<usize>> {
    let n = derive_n(x.len(), incx)?;
    if n == 0 {
        return Ok(None);
    }
    let cn = dim_to_mkl_int(n)?;
    let cinc = stride_to_mkl_int(incx)?;
    Ok(Some(unsafe { T::cblas_iamin(cn, x.as_ptr(), cinc) }))
}

// =====================================================================
// scal — in-place scaling.
// =====================================================================

/// In-place scaling: `x ← alpha * x`.
pub fn scal<T: BlasScalar>(alpha: T, x: &mut [T]) {
    scal_inc(alpha, x, 1).expect("unit-stride scal cannot fail")
}

/// Strided variant of [`scal`].
pub fn scal_inc<T: BlasScalar>(alpha: T, x: &mut [T], incx: i64) -> Result<()> {
    let n = derive_n(x.len(), incx)?;
    let cn = dim_to_mkl_int(n)?;
    let cinc = stride_to_mkl_int(incx)?;
    unsafe {
        T::cblas_scal(cn, alpha, x.as_mut_ptr(), cinc);
    }
    Ok(())
}

/// In-place scaling of a complex vector by a real scalar:
/// `x ← alpha * x`.
///
/// This is more efficient than [`scal`] with `alpha.im == 0` because it
/// avoids the unnecessary complex-by-complex multiplication.
pub fn scal_real<T: ComplexBlasScalar>(alpha: T::Real, x: &mut [T]) {
    scal_real_inc(alpha, x, 1).expect("unit-stride scal_real cannot fail")
}

/// Strided variant of [`scal_real`].
pub fn scal_real_inc<T: ComplexBlasScalar>(
    alpha: T::Real,
    x: &mut [T],
    incx: i64,
) -> Result<()> {
    let n = derive_n(x.len(), incx)?;
    let cn = dim_to_mkl_int(n)?;
    let cinc = stride_to_mkl_int(incx)?;
    unsafe {
        T::cblas_scal_real(cn, alpha, x.as_mut_ptr(), cinc);
    }
    Ok(())
}

// =====================================================================
// axpy / copy / swap — two-vector operations.
// =====================================================================

/// `y ← alpha * x + y`. Both vectors must have the same length.
pub fn axpy<T: BlasScalar>(alpha: T, x: &[T], y: &mut [T]) -> Result<()> {
    axpy_inc(alpha, x, 1, y, 1)
}

/// Strided variant of [`axpy`].
pub fn axpy_inc<T: BlasScalar>(
    alpha: T,
    x: &[T],
    incx: i64,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    unsafe {
        T::cblas_axpy(
            cn,
            alpha,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// `y ← x`. Both vectors must have the same length.
pub fn copy<T: BlasScalar>(x: &[T], y: &mut [T]) -> Result<()> {
    copy_inc(x, 1, y, 1)
}

/// Strided variant of [`copy`].
pub fn copy_inc<T: BlasScalar>(
    x: &[T],
    incx: i64,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    unsafe {
        T::cblas_copy(
            cn,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

/// Exchange `x` and `y` element-wise.
pub fn swap<T: BlasScalar>(x: &mut [T], y: &mut [T]) -> Result<()> {
    swap_inc(x, 1, y, 1)
}

/// Strided variant of [`swap`].
pub fn swap_inc<T: BlasScalar>(
    x: &mut [T],
    incx: i64,
    y: &mut [T],
    incy: i64,
) -> Result<()> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    unsafe {
        T::cblas_swap(
            cn,
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
        );
    }
    Ok(())
}

// =====================================================================
// dot products — type-specific (real vs complex).
// =====================================================================

/// Real dot product `x · y = Σ x_i y_i`.
///
/// Available only for [`f32`] and [`f64`]. For complex types use
/// [`dotc`] or [`dotu`].
pub fn dot<T: RealBlasScalar>(x: &[T], y: &[T]) -> Result<T> {
    dot_inc(x, 1, y, 1)
}

/// Strided variant of [`dot`].
pub fn dot_inc<T: RealBlasScalar>(
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
) -> Result<T> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    Ok(unsafe {
        T::cblas_dot(
            cn,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
        )
    })
}

/// Conjugated complex dot product `Σ conj(x_i) * y_i`.
pub fn dotc<T: ComplexBlasScalar>(x: &[T], y: &[T]) -> Result<T> {
    dotc_inc(x, 1, y, 1)
}

/// Strided variant of [`dotc`].
pub fn dotc_inc<T: ComplexBlasScalar>(
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
) -> Result<T> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    Ok(unsafe {
        T::cblas_dotc(
            cn,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
        )
    })
}

/// Unconjugated complex dot product `Σ x_i * y_i`.
pub fn dotu<T: ComplexBlasScalar>(x: &[T], y: &[T]) -> Result<T> {
    dotu_inc(x, 1, y, 1)
}

/// Strided variant of [`dotu`].
pub fn dotu_inc<T: ComplexBlasScalar>(
    x: &[T],
    incx: i64,
    y: &[T],
    incy: i64,
) -> Result<T> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    Ok(unsafe {
        T::cblas_dotu(
            cn,
            x.as_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_ptr(),
            stride_to_mkl_int(incy)?,
        )
    })
}

// =====================================================================
// Givens rotations.
// =====================================================================

/// Apply a Givens plane rotation in place:
///
/// ```text
/// [x'] = [ c  s ] [x]
/// [y']   [-s  c ] [y]
/// ```
pub fn rot<T: RealBlasScalar>(x: &mut [T], y: &mut [T], c: T, s: T) -> Result<()> {
    rot_inc(x, 1, y, 1, c, s)
}

/// Strided variant of [`rot`].
pub fn rot_inc<T: RealBlasScalar>(
    x: &mut [T],
    incx: i64,
    y: &mut [T],
    incy: i64,
    c: T,
    s: T,
) -> Result<()> {
    let n = check_two_vectors(x, incx, y, incy)?;
    let cn = dim_to_mkl_int(n)?;
    unsafe {
        T::cblas_rot(
            cn,
            x.as_mut_ptr(),
            stride_to_mkl_int(incx)?,
            y.as_mut_ptr(),
            stride_to_mkl_int(incy)?,
            c,
            s,
        );
    }
    Ok(())
}

/// Generate a Givens rotation: given inputs `(a, b)`, computes `(c, s,
/// r, z)` such that the rotation `[[c, s], [-s, c]]` zeros `b` and the
/// caller's `a` is replaced with `r`. `b` is replaced with the recovery
/// parameter `z`.
///
/// Returns `(c, s)`; `a` and `b` are mutated in place.
pub fn rotg<T: RealBlasScalar>(a: &mut T, b: &mut T) -> (T, T) {
    let mut c = <T as Scalar>::zero();
    let mut s = <T as Scalar>::zero();
    unsafe {
        T::cblas_rotg(a, b, &mut c, &mut s);
    }
    (c, s)
}

