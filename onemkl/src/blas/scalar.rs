//! Scalar-typed dispatch traits for BLAS.
//!
//! These traits associate each Rust scalar with the matching `cblas_*`
//! entry points. The public BLAS API is built on top of them, generic over
//! the trait bound rather than the scalar type.
//!
//! All trait methods are `unsafe` and forward directly to the underlying
//! C ABI. They exist so generic Rust code can dispatch to the right
//! routine; safety obligations match the [oneMKL reference][ref].
//!
//! [ref]: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/

use num_complex::{Complex32, Complex64};
use onemkl_sys::{self as sys, MKL_INT};

use crate::scalar::{ComplexScalar, RealScalar, Scalar};

// =====================================================================
// BlasScalar — operations defined for all four scalar types.
// =====================================================================

/// Scalar types supported by Level 1/2/3 BLAS.
///
/// This trait is implemented for [`f32`], [`f64`], [`Complex32`], and
/// [`Complex64`]. Each method is a thin wrapper over the corresponding
/// `cblas_*` routine.
///
/// All methods are `unsafe` because they ingest raw pointers and trust
/// the lengths and strides supplied by the caller. The safe public API
/// (free functions in [`level1`](super::level1) and friends) validates
/// arguments before calling these.
pub trait BlasScalar: Scalar {
    // ---- Level 1 ----

    /// `cblas_*asum` — sum of absolute values (or `|re| + |im|` per element
    /// for complex).
    ///
    /// # Safety
    /// `x` must point to at least `(n - 1) * |incx| + 1` elements.
    unsafe fn cblas_asum(n: MKL_INT, x: *const Self, incx: MKL_INT) -> Self::Real;

    /// `cblas_*nrm2` — Euclidean norm.
    ///
    /// # Safety
    /// `x` must point to at least `(n - 1) * |incx| + 1` elements.
    unsafe fn cblas_nrm2(n: MKL_INT, x: *const Self, incx: MKL_INT) -> Self::Real;

    /// `cblas_i*amax` — index of element with maximum absolute value
    /// (`|re| + |im|` for complex).
    ///
    /// # Safety
    /// `x` must point to at least `(n - 1) * |incx| + 1` elements.
    unsafe fn cblas_iamax(n: MKL_INT, x: *const Self, incx: MKL_INT) -> usize;

    /// `cblas_i*amin` — index of element with minimum absolute value.
    /// MKL extension to BLAS.
    ///
    /// # Safety
    /// Same as [`cblas_iamax`](Self::cblas_iamax).
    unsafe fn cblas_iamin(n: MKL_INT, x: *const Self, incx: MKL_INT) -> usize;

    /// `cblas_*scal` — `x ← alpha * x`.
    ///
    /// # Safety
    /// `x` must point to at least `(n - 1) * |incx| + 1` elements and be
    /// uniquely accessible (mutable).
    unsafe fn cblas_scal(n: MKL_INT, alpha: Self, x: *mut Self, incx: MKL_INT);

    /// `cblas_*axpy` — `y ← alpha * x + y`.
    ///
    /// # Safety
    /// `x` and `y` must each point to at least `(n - 1) * |inc?| + 1`
    /// elements and not overlap.
    unsafe fn cblas_axpy(
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `cblas_*copy` — `y ← x`.
    ///
    /// # Safety
    /// As [`cblas_axpy`](Self::cblas_axpy).
    unsafe fn cblas_copy(
        n: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `cblas_*swap` — exchange `x` and `y`.
    ///
    /// # Safety
    /// Both `x` and `y` must be uniquely accessible and non-overlapping,
    /// each addressing at least `(n - 1) * |inc?| + 1` elements.
    unsafe fn cblas_swap(
        n: MKL_INT,
        x: *mut Self,
        incx: MKL_INT,
        y: *mut Self,
        incy: MKL_INT,
    );

    // ---- Level 3 ----

    /// `cblas_*gemm` — general matrix-matrix multiply
    /// `C ← alpha * op(A) * op(B) + beta * C`.
    ///
    /// # Safety
    /// `a`, `b`, `c` must each point to a matrix of the dimensions implied
    /// by `m`, `n`, `k`, `transa`, `transb`, the chosen `layout`, and the
    /// leading dimensions `lda`, `ldb`, `ldc`. `c` must be uniquely
    /// accessible.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemm(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: sys::CBLAS_TRANSPOSE::Type,
        transb: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
    );

    // ---- Level 2 ----

    /// `cblas_*gemv` — general matrix-vector multiply
    /// `y ← alpha * op(A) * x + beta * y`.
    ///
    /// # Safety
    /// `a`, `x`, `y` must point to buffers consistent with the declared
    /// shapes; `y` must be uniquely accessible.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemv(
        layout: sys::CBLAS_LAYOUT::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
    );
}

// =====================================================================
// RealBlasScalar — operations only meaningful for real scalars.
// =====================================================================

/// Real-only BLAS operations.
///
/// Implemented for [`f32`] and [`f64`].
pub trait RealBlasScalar: BlasScalar + RealScalar {
    /// `cblas_*dot` — real dot product.
    ///
    /// # Safety
    /// As [`BlasScalar::cblas_axpy`].
    unsafe fn cblas_dot(
        n: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
    ) -> Self;

    /// `cblas_*rot` — apply Givens plane rotation.
    ///
    /// # Safety
    /// `x` and `y` must be uniquely accessible, each at least
    /// `(n - 1) * |inc?| + 1` elements.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_rot(
        n: MKL_INT,
        x: *mut Self,
        incx: MKL_INT,
        y: *mut Self,
        incy: MKL_INT,
        c: Self,
        s: Self,
    );

    /// `cblas_*rotg` — generate Givens rotation `(c, s)` such that
    /// `[[c, s], [-s, c]] * [[a], [b]] = [[r], [0]]`. On return, `a` is
    /// `r` and `b` is `z` (the recovery parameter).
    ///
    /// # Safety
    /// All four pointers must be valid and uniquely accessible.
    unsafe fn cblas_rotg(a: *mut Self, b: *mut Self, c: *mut Self, s: *mut Self);
}

// =====================================================================
// ComplexBlasScalar — operations only meaningful for complex scalars.
// =====================================================================

/// Complex-only BLAS operations.
///
/// Implemented for [`Complex32`] and [`Complex64`].
pub trait ComplexBlasScalar: BlasScalar + ComplexScalar {
    /// `cblas_*dotc_sub` — conjugated dot product `Σ conj(x_i) * y_i`.
    ///
    /// # Safety
    /// As [`BlasScalar::cblas_axpy`].
    unsafe fn cblas_dotc(
        n: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
    ) -> Self;

    /// `cblas_*dotu_sub` — unconjugated dot product `Σ x_i * y_i`.
    ///
    /// # Safety
    /// As [`BlasScalar::cblas_axpy`].
    unsafe fn cblas_dotu(
        n: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
    ) -> Self;

    /// `cblas_csscal` / `cblas_zdscal` — scale a complex vector by a real
    /// scalar.
    ///
    /// # Safety
    /// As [`BlasScalar::cblas_scal`].
    unsafe fn cblas_scal_real(
        n: MKL_INT,
        alpha: Self::Real,
        x: *mut Self,
        incx: MKL_INT,
    );
}

// =====================================================================
// Implementations.
//
// Each macro expands to `BlasScalar` methods that simply forward to the
// corresponding `cblas_*` symbol. The complex specializations need a bit
// of pointer-cast bookkeeping to bridge `Complex<R>` ↔ `MKL_Complex*`.
// =====================================================================

macro_rules! impl_real_blas {
    (
        $ty:ty,
        asum = $asum:ident,
        nrm2 = $nrm2:ident,
        iamax = $iamax:ident,
        iamin = $iamin:ident,
        scal = $scal:ident,
        axpy = $axpy:ident,
        copy = $copy:ident,
        swap = $swap:ident,
        gemm = $gemm:ident,
        gemv = $gemv:ident,
        dot = $dot:ident,
        rot = $rot:ident,
        rotg = $rotg:ident,
    ) => {
        impl BlasScalar for $ty {
            #[inline]
            unsafe fn cblas_asum(n: MKL_INT, x: *const Self, incx: MKL_INT) -> Self::Real {
                unsafe { sys::$asum(n, x, incx) }
            }
            #[inline]
            unsafe fn cblas_nrm2(n: MKL_INT, x: *const Self, incx: MKL_INT) -> Self::Real {
                unsafe { sys::$nrm2(n, x, incx) }
            }
            #[inline]
            unsafe fn cblas_iamax(n: MKL_INT, x: *const Self, incx: MKL_INT) -> usize {
                unsafe { sys::$iamax(n, x, incx) }
            }
            #[inline]
            unsafe fn cblas_iamin(n: MKL_INT, x: *const Self, incx: MKL_INT) -> usize {
                unsafe { sys::$iamin(n, x, incx) }
            }
            #[inline]
            unsafe fn cblas_scal(n: MKL_INT, alpha: Self, x: *mut Self, incx: MKL_INT) {
                unsafe { sys::$scal(n, alpha, x, incx) }
            }
            #[inline]
            unsafe fn cblas_axpy(
                n: MKL_INT,
                alpha: Self,
                x: *const Self,
                incx: MKL_INT,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe { sys::$axpy(n, alpha, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_copy(
                n: MKL_INT,
                x: *const Self,
                incx: MKL_INT,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe { sys::$copy(n, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_swap(
                n: MKL_INT,
                x: *mut Self,
                incx: MKL_INT,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe { sys::$swap(n, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_gemm(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT,
                n: MKL_INT,
                k: MKL_INT,
                alpha: Self,
                a: *const Self,
                lda: MKL_INT,
                b: *const Self,
                ldb: MKL_INT,
                beta: Self,
                c: *mut Self,
                ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm(
                        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemv(
                layout: sys::CBLAS_LAYOUT::Type,
                trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT,
                n: MKL_INT,
                alpha: Self,
                a: *const Self,
                lda: MKL_INT,
                x: *const Self,
                incx: MKL_INT,
                beta: Self,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe {
                    sys::$gemv(
                        layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                    )
                }
            }
        }

        impl RealBlasScalar for $ty {
            #[inline]
            unsafe fn cblas_dot(
                n: MKL_INT,
                x: *const Self,
                incx: MKL_INT,
                y: *const Self,
                incy: MKL_INT,
            ) -> Self {
                unsafe { sys::$dot(n, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_rot(
                n: MKL_INT,
                x: *mut Self,
                incx: MKL_INT,
                y: *mut Self,
                incy: MKL_INT,
                c: Self,
                s: Self,
            ) {
                unsafe { sys::$rot(n, x, incx, y, incy, c, s) }
            }
            #[inline]
            unsafe fn cblas_rotg(a: *mut Self, b: *mut Self, c: *mut Self, s: *mut Self) {
                unsafe { sys::$rotg(a, b, c, s) }
            }
        }
    };
}

impl_real_blas!(
    f32,
    asum = cblas_sasum,
    nrm2 = cblas_snrm2,
    iamax = cblas_isamax,
    iamin = cblas_isamin,
    scal = cblas_sscal,
    axpy = cblas_saxpy,
    copy = cblas_scopy,
    swap = cblas_sswap,
    gemm = cblas_sgemm,
    gemv = cblas_sgemv,
    dot = cblas_sdot,
    rot = cblas_srot,
    rotg = cblas_srotg,
);

impl_real_blas!(
    f64,
    asum = cblas_dasum,
    nrm2 = cblas_dnrm2,
    iamax = cblas_idamax,
    iamin = cblas_idamin,
    scal = cblas_dscal,
    axpy = cblas_daxpy,
    copy = cblas_dcopy,
    swap = cblas_dswap,
    gemm = cblas_dgemm,
    gemv = cblas_dgemv,
    dot = cblas_ddot,
    rot = cblas_drot,
    rotg = cblas_drotg,
);

macro_rules! impl_complex_blas {
    (
        $ty:ty,
        asum = $asum:ident,
        nrm2 = $nrm2:ident,
        iamax = $iamax:ident,
        iamin = $iamin:ident,
        scal = $scal:ident,
        scal_real = $scal_real:ident,
        axpy = $axpy:ident,
        copy = $copy:ident,
        swap = $swap:ident,
        gemm = $gemm:ident,
        gemv = $gemv:ident,
        dotc_sub = $dotc:ident,
        dotu_sub = $dotu:ident,
    ) => {
        impl BlasScalar for $ty {
            #[inline]
            unsafe fn cblas_asum(n: MKL_INT, x: *const Self, incx: MKL_INT) -> Self::Real {
                unsafe { sys::$asum(n, x.cast(), incx) }
            }
            #[inline]
            unsafe fn cblas_nrm2(n: MKL_INT, x: *const Self, incx: MKL_INT) -> Self::Real {
                unsafe { sys::$nrm2(n, x.cast(), incx) }
            }
            #[inline]
            unsafe fn cblas_iamax(n: MKL_INT, x: *const Self, incx: MKL_INT) -> usize {
                unsafe { sys::$iamax(n, x.cast(), incx) }
            }
            #[inline]
            unsafe fn cblas_iamin(n: MKL_INT, x: *const Self, incx: MKL_INT) -> usize {
                unsafe { sys::$iamin(n, x.cast(), incx) }
            }
            #[inline]
            unsafe fn cblas_scal(n: MKL_INT, alpha: Self, x: *mut Self, incx: MKL_INT) {
                unsafe {
                    sys::$scal(n, (&alpha as *const Self).cast(), x.cast(), incx)
                }
            }
            #[inline]
            unsafe fn cblas_axpy(
                n: MKL_INT,
                alpha: Self,
                x: *const Self,
                incx: MKL_INT,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe {
                    sys::$axpy(
                        n,
                        (&alpha as *const Self).cast(),
                        x.cast(),
                        incx,
                        y.cast(),
                        incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_copy(
                n: MKL_INT,
                x: *const Self,
                incx: MKL_INT,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe { sys::$copy(n, x.cast(), incx, y.cast(), incy) }
            }
            #[inline]
            unsafe fn cblas_swap(
                n: MKL_INT,
                x: *mut Self,
                incx: MKL_INT,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe { sys::$swap(n, x.cast(), incx, y.cast(), incy) }
            }
            #[inline]
            unsafe fn cblas_gemm(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT,
                n: MKL_INT,
                k: MKL_INT,
                alpha: Self,
                a: *const Self,
                lda: MKL_INT,
                b: *const Self,
                ldb: MKL_INT,
                beta: Self,
                c: *mut Self,
                ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm(
                        layout,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        (&alpha as *const Self).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        (&beta as *const Self).cast(),
                        c.cast(),
                        ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemv(
                layout: sys::CBLAS_LAYOUT::Type,
                trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT,
                n: MKL_INT,
                alpha: Self,
                a: *const Self,
                lda: MKL_INT,
                x: *const Self,
                incx: MKL_INT,
                beta: Self,
                y: *mut Self,
                incy: MKL_INT,
            ) {
                unsafe {
                    sys::$gemv(
                        layout,
                        trans,
                        m,
                        n,
                        (&alpha as *const Self).cast(),
                        a.cast(),
                        lda,
                        x.cast(),
                        incx,
                        (&beta as *const Self).cast(),
                        y.cast(),
                        incy,
                    )
                }
            }
        }

        impl ComplexBlasScalar for $ty {
            #[inline]
            unsafe fn cblas_dotc(
                n: MKL_INT,
                x: *const Self,
                incx: MKL_INT,
                y: *const Self,
                incy: MKL_INT,
            ) -> Self {
                let mut out = <Self as Scalar>::zero();
                unsafe {
                    sys::$dotc(
                        n,
                        x.cast(),
                        incx,
                        y.cast(),
                        incy,
                        (&mut out as *mut Self).cast(),
                    );
                }
                out
            }
            #[inline]
            unsafe fn cblas_dotu(
                n: MKL_INT,
                x: *const Self,
                incx: MKL_INT,
                y: *const Self,
                incy: MKL_INT,
            ) -> Self {
                let mut out = <Self as Scalar>::zero();
                unsafe {
                    sys::$dotu(
                        n,
                        x.cast(),
                        incx,
                        y.cast(),
                        incy,
                        (&mut out as *mut Self).cast(),
                    );
                }
                out
            }
            #[inline]
            unsafe fn cblas_scal_real(
                n: MKL_INT,
                alpha: Self::Real,
                x: *mut Self,
                incx: MKL_INT,
            ) {
                unsafe { sys::$scal_real(n, alpha, x.cast(), incx) }
            }
        }
    };
}

impl_complex_blas!(
    Complex32,
    asum = cblas_scasum,
    nrm2 = cblas_scnrm2,
    iamax = cblas_icamax,
    iamin = cblas_icamin,
    scal = cblas_cscal,
    scal_real = cblas_csscal,
    axpy = cblas_caxpy,
    copy = cblas_ccopy,
    swap = cblas_cswap,
    gemm = cblas_cgemm,
    gemv = cblas_cgemv,
    dotc_sub = cblas_cdotc_sub,
    dotu_sub = cblas_cdotu_sub,
);

impl_complex_blas!(
    Complex64,
    asum = cblas_dzasum,
    nrm2 = cblas_dznrm2,
    iamax = cblas_izamax,
    iamin = cblas_izamin,
    scal = cblas_zscal,
    scal_real = cblas_zdscal,
    axpy = cblas_zaxpy,
    copy = cblas_zcopy,
    swap = cblas_zswap,
    gemm = cblas_zgemm,
    gemv = cblas_zgemv,
    dotc_sub = cblas_zdotc_sub,
    dotu_sub = cblas_zdotu_sub,
);
