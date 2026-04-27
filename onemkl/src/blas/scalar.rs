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

use core::ffi::c_char;

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
    // ----- Level 1 -----

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

    // ----- Level 2 (universal) -----

    /// `cblas_*gemv` — general matrix-vector multiply
    /// `y ← alpha * op(A) * x + beta * y`.
    ///
    /// # Safety
    /// Buffers must satisfy the dimensional requirements implied by the
    /// arguments; `y` must be uniquely accessible.
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

    /// `cblas_*gbmv` — general band matrix-vector multiply.
    ///
    /// # Safety
    /// As [`cblas_gemv`](Self::cblas_gemv); `a` is a banded matrix in the
    /// CBLAS band-storage layout with `kl` sub-diagonals and `ku`
    /// super-diagonals.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gbmv(
        layout: sys::CBLAS_LAYOUT::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        kl: MKL_INT,
        ku: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `cblas_*trmv` — triangular matrix-vector multiply `x ← op(A) * x`.
    ///
    /// # Safety
    /// `a` is `n × n` with leading dimension `lda`; `x` is uniquely
    /// accessible with at least `(n - 1) * |incx| + 1` elements.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_trmv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        n: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        x: *mut Self,
        incx: MKL_INT,
    );

    /// `cblas_*trsv` — triangular solve `x ← op(A)⁻¹ * x`.
    ///
    /// # Safety
    /// As [`cblas_trmv`](Self::cblas_trmv); `A` must be non-singular when
    /// `diag` is [`Diag::NonUnit`](crate::Diag::NonUnit).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_trsv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        n: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        x: *mut Self,
        incx: MKL_INT,
    );

    /// `cblas_*tbmv` — triangular band matrix-vector multiply.
    ///
    /// # Safety
    /// `a` is in CBLAS triangular-band storage with `k` off-diagonals.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_tbmv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        n: MKL_INT,
        k: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        x: *mut Self,
        incx: MKL_INT,
    );

    /// `cblas_*tbsv` — triangular band solve.
    ///
    /// # Safety
    /// As [`cblas_tbmv`](Self::cblas_tbmv).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_tbsv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        n: MKL_INT,
        k: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        x: *mut Self,
        incx: MKL_INT,
    );

    /// `cblas_*tpmv` — triangular packed matrix-vector multiply.
    ///
    /// # Safety
    /// `ap` is a packed triangular matrix of `n*(n+1)/2` elements.
    unsafe fn cblas_tpmv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        n: MKL_INT,
        ap: *const Self,
        x: *mut Self,
        incx: MKL_INT,
    );

    /// `cblas_*tpsv` — triangular packed solve.
    ///
    /// # Safety
    /// As [`cblas_tpmv`](Self::cblas_tpmv).
    unsafe fn cblas_tpsv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        n: MKL_INT,
        ap: *const Self,
        x: *mut Self,
        incx: MKL_INT,
    );

    // ----- Level 3 (universal) -----

    /// `cblas_*gemm` — general matrix-matrix multiply
    /// `C ← alpha * op(A) * op(B) + beta * C`.
    ///
    /// # Safety
    /// `a`, `b`, `c` must each point to a matrix of the dimensions implied
    /// by `m`, `n`, `k`, `transa`, `transb`, the chosen `layout`, and the
    /// leading dimensions; `c` must be uniquely accessible.
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

    /// `cblas_*symm` — symmetric matrix-matrix multiply.
    /// `C ← alpha * op(A) * B + beta * C` if `side == Left`, else
    /// `C ← alpha * B * op(A) + beta * C`. `A` is symmetric.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_symm(
        layout: sys::CBLAS_LAYOUT::Type,
        side: sys::CBLAS_SIDE::Type,
        uplo: sys::CBLAS_UPLO::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
    );

    /// `cblas_*syrk` — symmetric rank-k update.
    /// `C ← alpha * A * Aᵀ + beta * C` (or `Aᵀ * A` if `trans` flips).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_syrk(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
    );

    /// `cblas_*syr2k` — symmetric rank-2k update.
    /// `C ← alpha * (A * Bᵀ + B * Aᵀ) + beta * C`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_syr2k(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
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

    /// `cblas_*trmm` — triangular matrix-matrix multiply.
    /// `B ← alpha * op(A) * B` (left) or `B ← alpha * B * op(A)` (right).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_trmm(
        layout: sys::CBLAS_LAYOUT::Type,
        side: sys::CBLAS_SIDE::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        b: *mut Self,
        ldb: MKL_INT,
    );

    /// `cblas_*trsm` — triangular solve with multiple right-hand sides.
    /// `B ← alpha * op(A)⁻¹ * B` (left) or `B ← alpha * B * op(A)⁻¹` (right).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_trsm(
        layout: sys::CBLAS_LAYOUT::Type,
        side: sys::CBLAS_SIDE::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        b: *mut Self,
        ldb: MKL_INT,
    );

    /// `cblas_*gemmt` — gemm but only writes the upper or lower triangle of
    /// `C`. Useful when `C` is known to be symmetric/Hermitian.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemmt(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        transa: sys::CBLAS_TRANSPOSE::Type,
        transb: sys::CBLAS_TRANSPOSE::Type,
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

    // ----- BLAS-like extensions (universal subset) -----

    /// `cblas_*axpby` — `y ← alpha * x + beta * y`. Generalization of
    /// `axpy` that also scales `y`.
    ///
    /// # Safety
    /// As [`cblas_axpy`](Self::cblas_axpy) plus the requirement that `y`
    /// is uniquely accessible.
    unsafe fn cblas_axpby(
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `MKL_*imatcopy` — in-place matrix transposition with scaling
    /// `A ← alpha * op(A)` where the same buffer is read and written.
    ///
    /// `ordering` is `b'R'` or `b'C'`; `trans` is `b'N'`, `b'T'`, or
    /// `b'C'` (use [`Layout::as_char`](crate::Layout::as_char) and
    /// [`Transpose::as_char`](crate::Transpose::as_char)).
    ///
    /// # Safety
    /// The buffer must be large enough for both pre- and post-transpose
    /// shapes given the leading dimensions.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mkl_imatcopy(
        ordering: c_char,
        trans: c_char,
        rows: usize,
        cols: usize,
        alpha: Self,
        ab: *mut Self,
        lda: usize,
        ldb: usize,
    );

    /// `MKL_*omatcopy` — out-of-place matrix transposition with scaling
    /// `B ← alpha * op(A)`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mkl_omatcopy(
        ordering: c_char,
        trans: c_char,
        rows: usize,
        cols: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        b: *mut Self,
        ldb: usize,
    );

    /// `MKL_*omatcopy2` — out-of-place 2-D strided transposition.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mkl_omatcopy2(
        ordering: c_char,
        trans: c_char,
        rows: usize,
        cols: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        stridea: usize,
        b: *mut Self,
        ldb: usize,
        strideb: usize,
    );

    /// `MKL_*omatadd` — `C ← alpha * op(A) + beta * op(B)`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mkl_omatadd(
        ordering: c_char,
        transa: c_char,
        transb: c_char,
        rows: usize,
        cols: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        beta: Self,
        b: *const Self,
        ldb: usize,
        c: *mut Self,
        ldc: usize,
    );

    // ----- Batched (strided) extensions -----
    //
    // Each `*_batch_strided` routine performs `batch_size` independent
    // operations on contiguous slices of one buffer per operand, with a
    // user-specified element-stride between slices.

    /// `cblas_*axpy_batch_strided` — batched [`cblas_axpy`](Self::cblas_axpy).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_axpy_batch_strided(
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        stridex: MKL_INT,
        y: *mut Self,
        incy: MKL_INT,
        stridey: MKL_INT,
        batch_size: MKL_INT,
    );

    /// `cblas_*copy_batch_strided` — batched [`cblas_copy`](Self::cblas_copy).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_copy_batch_strided(
        n: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        stridex: MKL_INT,
        y: *mut Self,
        incy: MKL_INT,
        stridey: MKL_INT,
        batch_size: MKL_INT,
    );

    /// `cblas_*gemv_batch_strided` — batched [`cblas_gemv`](Self::cblas_gemv).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemv_batch_strided(
        layout: sys::CBLAS_LAYOUT::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        stridea: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        stridex: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
        stridey: MKL_INT,
        batch_size: MKL_INT,
    );

    /// `cblas_*gemm_batch_strided` — batched [`cblas_gemm`](Self::cblas_gemm).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemm_batch_strided(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: sys::CBLAS_TRANSPOSE::Type,
        transb: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        stridea: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        strideb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
        stridec: MKL_INT,
        batch_size: MKL_INT,
    );

    /// `cblas_*gemm_batch` — pointer-array batched GEMM. Each "group"
    /// shares uniform parameters (transposes, dimensions, alpha/beta);
    /// `group_size[i]` is the number of GEMMs in group `i`. The
    /// pointer arrays for `A`, `B`, `C` have total length
    /// `sum(group_size)`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemm_batch(
        layout: sys::CBLAS_LAYOUT::Type,
        transa_array: *const sys::CBLAS_TRANSPOSE::Type,
        transb_array: *const sys::CBLAS_TRANSPOSE::Type,
        m_array: *const MKL_INT,
        n_array: *const MKL_INT,
        k_array: *const MKL_INT,
        alpha_array: *const Self,
        a_array: *mut *const Self,
        lda_array: *const MKL_INT,
        b_array: *mut *const Self,
        ldb_array: *const MKL_INT,
        beta_array: *const Self,
        c_array: *mut *mut Self,
        ldc_array: *const MKL_INT,
        group_count: MKL_INT,
        group_size: *const MKL_INT,
    );

    /// `cblas_*gemv_batch` — pointer-array batched GEMV.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemv_batch(
        layout: sys::CBLAS_LAYOUT::Type,
        trans_array: *const sys::CBLAS_TRANSPOSE::Type,
        m_array: *const MKL_INT,
        n_array: *const MKL_INT,
        alpha_array: *const Self,
        a_array: *mut *const Self,
        lda_array: *const MKL_INT,
        x_array: *mut *const Self,
        incx_array: *const MKL_INT,
        beta_array: *const Self,
        y_array: *mut *mut Self,
        incy_array: *const MKL_INT,
        group_count: MKL_INT,
        group_size: *const MKL_INT,
    );

    /// `cblas_*trsm_batch` — pointer-array batched TRSM.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_trsm_batch(
        layout: sys::CBLAS_LAYOUT::Type,
        side_array: *const sys::CBLAS_SIDE::Type,
        uplo_array: *const sys::CBLAS_UPLO::Type,
        transa_array: *const sys::CBLAS_TRANSPOSE::Type,
        diag_array: *const sys::CBLAS_DIAG::Type,
        m_array: *const MKL_INT,
        n_array: *const MKL_INT,
        alpha_array: *const Self,
        a_array: *mut *const Self,
        lda_array: *const MKL_INT,
        b_array: *mut *mut Self,
        ldb_array: *const MKL_INT,
        group_count: MKL_INT,
        group_size: *const MKL_INT,
    );

    /// `cblas_*trsm_batch_strided` — batched [`cblas_trsm`](Self::cblas_trsm).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_trsm_batch_strided(
        layout: sys::CBLAS_LAYOUT::Type,
        side: sys::CBLAS_SIDE::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        diag: sys::CBLAS_DIAG::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        stridea: MKL_INT,
        b: *mut Self,
        ldb: MKL_INT,
        strideb: MKL_INT,
        batch_size: MKL_INT,
    );

    /// `cblas_*dgmm_batch_strided` — batched diagonal-matrix multiply.
    /// `C[i] ← A[i] * diag(x[i])` (right) or `C[i] ← diag(x[i]) * A[i]`
    /// (left).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_dgmm_batch_strided(
        layout: sys::CBLAS_LAYOUT::Type,
        side: sys::CBLAS_SIDE::Type,
        m: MKL_INT,
        n: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        stridea: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        stridex: MKL_INT,
        c: *mut Self,
        ldc: MKL_INT,
        stridec: MKL_INT,
        batch_size: MKL_INT,
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

    /// `cblas_*ger` — general rank-1 update `A ← α * x * yᵀ + A`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_ger(
        layout: sys::CBLAS_LAYOUT::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
        a: *mut Self,
        lda: MKL_INT,
    );

    /// `cblas_*symv` — symmetric matrix-vector multiply.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_symv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
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

    /// `cblas_*syr` — symmetric rank-1 update `A ← α * x * xᵀ + A`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_syr(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        a: *mut Self,
        lda: MKL_INT,
    );

    /// `cblas_*syr2` — symmetric rank-2 update
    /// `A ← α * (x * yᵀ + y * xᵀ) + A`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_syr2(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
        a: *mut Self,
        lda: MKL_INT,
    );

    /// `cblas_*sbmv` — symmetric band matrix-vector multiply.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_sbmv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `cblas_*spmv` — symmetric packed matrix-vector multiply.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_spmv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        ap: *const Self,
        x: *const Self,
        incx: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `cblas_*spr` — packed symmetric rank-1 update.
    unsafe fn cblas_spr(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        ap: *mut Self,
    );

    /// `cblas_*spr2` — packed symmetric rank-2 update.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_spr2(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
        ap: *mut Self,
    );
}

// =====================================================================
// ComplexBlasScalar — operations only meaningful for complex scalars.
// =====================================================================

/// Complex-only BLAS operations.
///
/// Implemented for [`Complex32`] and [`Complex64`].
pub trait ComplexBlasScalar: BlasScalar + ComplexScalar {
    /// `cblas_*dotc_sub` — conjugated dot product `Σ conj(x_i) * y_i`.
    unsafe fn cblas_dotc(
        n: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
    ) -> Self;

    /// `cblas_*dotu_sub` — unconjugated dot product `Σ x_i * y_i`.
    unsafe fn cblas_dotu(
        n: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
    ) -> Self;

    /// `cblas_csscal` / `cblas_zdscal` — scale a complex vector by a real
    /// scalar.
    unsafe fn cblas_scal_real(
        n: MKL_INT,
        alpha: Self::Real,
        x: *mut Self,
        incx: MKL_INT,
    );

    /// `cblas_*gerc` — conjugated rank-1 update
    /// `A ← α * x * yᴴ + A`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gerc(
        layout: sys::CBLAS_LAYOUT::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
        a: *mut Self,
        lda: MKL_INT,
    );

    /// `cblas_*geru` — unconjugated rank-1 update
    /// `A ← α * x * yᵀ + A`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_geru(
        layout: sys::CBLAS_LAYOUT::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
        a: *mut Self,
        lda: MKL_INT,
    );

    /// `cblas_*hemv` — Hermitian matrix-vector multiply.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_hemv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
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

    /// `cblas_*her` — Hermitian rank-1 update with real `alpha`:
    /// `A ← α * x * xᴴ + A`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_her(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self::Real,
        x: *const Self,
        incx: MKL_INT,
        a: *mut Self,
        lda: MKL_INT,
    );

    /// `cblas_*her2` — Hermitian rank-2 update with complex `alpha`:
    /// `A ← α * x * yᴴ + conj(α) * y * xᴴ + A`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_her2(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
        a: *mut Self,
        lda: MKL_INT,
    );

    /// `cblas_*hbmv` — Hermitian band matrix-vector multiply.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_hbmv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        x: *const Self,
        incx: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `cblas_*hpmv` — Hermitian packed matrix-vector multiply.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_hpmv(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        ap: *const Self,
        x: *const Self,
        incx: MKL_INT,
        beta: Self,
        y: *mut Self,
        incy: MKL_INT,
    );

    /// `cblas_*hpr` — packed Hermitian rank-1 update with real alpha.
    unsafe fn cblas_hpr(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self::Real,
        x: *const Self,
        incx: MKL_INT,
        ap: *mut Self,
    );

    /// `cblas_*hpr2` — packed Hermitian rank-2 update with complex alpha.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_hpr2(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        n: MKL_INT,
        alpha: Self,
        x: *const Self,
        incx: MKL_INT,
        y: *const Self,
        incy: MKL_INT,
        ap: *mut Self,
    );

    /// `cblas_*hemm` — Hermitian matrix-matrix multiply.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_hemm(
        layout: sys::CBLAS_LAYOUT::Type,
        side: sys::CBLAS_SIDE::Type,
        uplo: sys::CBLAS_UPLO::Type,
        m: MKL_INT,
        n: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
    );

    /// `cblas_*herk` — Hermitian rank-k update with **real** `alpha` and
    /// `beta`. `C ← alpha * A * Aᴴ + beta * C` (or `Aᴴ * A` if trans flips).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_herk(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self::Real,
        a: *const Self,
        lda: MKL_INT,
        beta: Self::Real,
        c: *mut Self,
        ldc: MKL_INT,
    );

    /// `cblas_*her2k` — Hermitian rank-2k update with complex `alpha`
    /// and **real** `beta`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_her2k(
        layout: sys::CBLAS_LAYOUT::Type,
        uplo: sys::CBLAS_UPLO::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        beta: Self::Real,
        c: *mut Self,
        ldc: MKL_INT,
    );

    /// `cblas_*gemm3m` — complex `gemm` using Karatsuba-like 3M algorithm.
    /// Numerically less stable than `gemm` but faster for large matrices.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemm3m(
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

    /// `cblas_*gemm3m_batch_strided` — batched [`cblas_gemm3m`](Self::cblas_gemm3m).
    #[allow(clippy::too_many_arguments)]
    unsafe fn cblas_gemm3m_batch_strided(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: sys::CBLAS_TRANSPOSE::Type,
        transb: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        a: *const Self,
        lda: MKL_INT,
        stridea: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        strideb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
        stridec: MKL_INT,
        batch_size: MKL_INT,
    );
}

// =====================================================================
// Implementations.
//
// Each macro expands to all trait method implementations for one scalar
// type. The complex specializations include pointer-cast bookkeeping
// to bridge `Complex<R>` ↔ `MKL_Complex*`, which are layout-compatible
// (`#[repr(C)]` two-real-fields structs).
// =====================================================================

macro_rules! impl_real_blas {
    (
        $ty:ty,
        // L1
        asum = $asum:ident, nrm2 = $nrm2:ident,
        iamax = $iamax:ident, iamin = $iamin:ident,
        scal = $scal:ident, axpy = $axpy:ident,
        copy = $copy:ident, swap = $swap:ident,
        dot = $dot:ident, rot = $rot:ident, rotg = $rotg:ident,
        // L2 universal
        gemv = $gemv:ident, gbmv = $gbmv:ident,
        trmv = $trmv:ident, trsv = $trsv:ident,
        tbmv = $tbmv:ident, tbsv = $tbsv:ident,
        tpmv = $tpmv:ident, tpsv = $tpsv:ident,
        // L2 real-only
        ger = $ger:ident, symv = $symv:ident,
        syr = $syr:ident, syr2 = $syr2:ident,
        sbmv = $sbmv:ident, spmv = $spmv:ident,
        spr = $spr:ident, spr2 = $spr2:ident,
        // L3
        gemm = $gemm:ident,
        symm = $symm:ident,
        syrk = $syrk:ident,
        syr2k = $syr2k:ident,
        trmm = $trmm:ident,
        trsm = $trsm:ident,
        gemmt = $gemmt:ident,
        // Extensions
        axpby = $axpby:ident,
        imatcopy = $imatcopy:ident,
        omatcopy = $omatcopy:ident,
        omatcopy2 = $omatcopy2:ident,
        omatadd = $omatadd:ident,
        // Batched (strided) — universal across all four scalar types
        axpy_batch_strided = $axpy_batch:ident,
        copy_batch_strided = $copy_batch:ident,
        gemv_batch_strided = $gemv_batch:ident,
        gemm_batch_strided = $gemm_batch:ident,
        gemm_batch_ptr = $gemm_batch_ptr:ident,
        gemv_batch_ptr = $gemv_batch_ptr:ident,
        trsm_batch_strided = $trsm_batch:ident,
        trsm_batch_ptr = $trsm_batch_ptr:ident,
        dgmm_batch_strided = $dgmm_batch:ident,
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
                n: MKL_INT, alpha: Self, x: *const Self, incx: MKL_INT,
                y: *mut Self, incy: MKL_INT,
            ) {
                unsafe { sys::$axpy(n, alpha, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_copy(
                n: MKL_INT, x: *const Self, incx: MKL_INT,
                y: *mut Self, incy: MKL_INT,
            ) {
                unsafe { sys::$copy(n, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_swap(
                n: MKL_INT, x: *mut Self, incx: MKL_INT,
                y: *mut Self, incy: MKL_INT,
            ) {
                unsafe { sys::$swap(n, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_gemv(
                layout: sys::CBLAS_LAYOUT::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$gemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
                }
            }
            #[inline]
            unsafe fn cblas_gbmv(
                layout: sys::CBLAS_LAYOUT::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, kl: MKL_INT, ku: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$gbmv(
                        layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe { sys::$trmv(layout, uplo, trans, diag, n, a, lda, x, incx) }
            }
            #[inline]
            unsafe fn cblas_trsv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe { sys::$trsv(layout, uplo, trans, diag, n, a, lda, x, incx) }
            }
            #[inline]
            unsafe fn cblas_tbmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, k: MKL_INT,
                a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe { sys::$tbmv(layout, uplo, trans, diag, n, k, a, lda, x, incx) }
            }
            #[inline]
            unsafe fn cblas_tbsv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, k: MKL_INT,
                a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe { sys::$tbsv(layout, uplo, trans, diag, n, k, a, lda, x, incx) }
            }
            #[inline]
            unsafe fn cblas_tpmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, ap: *const Self,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe { sys::$tpmv(layout, uplo, trans, diag, n, ap, x, incx) }
            }
            #[inline]
            unsafe fn cblas_tpsv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, ap: *const Self,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe { sys::$tpsv(layout, uplo, trans, diag, n, ap, x, incx) }
            }
            #[inline]
            unsafe fn cblas_gemm(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm(
                        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_symm(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$symm(
                        layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_syrk(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$syrk(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
                }
            }
            #[inline]
            unsafe fn cblas_syr2k(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$syr2k(
                        layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trmm(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                diag: sys::CBLAS_DIAG::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) {
                unsafe {
                    sys::$trmm(
                        layout, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trsm(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                diag: sys::CBLAS_DIAG::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) {
                unsafe {
                    sys::$trsm(
                        layout, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemmt(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                transa: sys::CBLAS_TRANSPOSE::Type, transb: sys::CBLAS_TRANSPOSE::Type,
                n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$gemmt(
                        layout, uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_axpby(
                n: MKL_INT, alpha: Self, x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe { sys::$axpby(n, alpha, x, incx, beta, y, incy) }
            }
            #[inline]
            unsafe fn mkl_imatcopy(
                ordering: c_char, trans: c_char,
                rows: usize, cols: usize, alpha: Self,
                ab: *mut Self, lda: usize, ldb: usize,
            ) {
                unsafe { sys::$imatcopy(ordering, trans, rows, cols, alpha, ab, lda, ldb) }
            }
            #[inline]
            unsafe fn mkl_omatcopy(
                ordering: c_char, trans: c_char,
                rows: usize, cols: usize, alpha: Self,
                a: *const Self, lda: usize,
                b: *mut Self, ldb: usize,
            ) {
                unsafe {
                    sys::$omatcopy(ordering, trans, rows, cols, alpha, a, lda, b, ldb)
                }
            }
            #[inline]
            unsafe fn mkl_omatcopy2(
                ordering: c_char, trans: c_char,
                rows: usize, cols: usize, alpha: Self,
                a: *const Self, lda: usize, stridea: usize,
                b: *mut Self, ldb: usize, strideb: usize,
            ) {
                unsafe {
                    sys::$omatcopy2(
                        ordering, trans, rows, cols, alpha,
                        a, lda, stridea, b, ldb, strideb,
                    )
                }
            }
            #[inline]
            unsafe fn mkl_omatadd(
                ordering: c_char, transa: c_char, transb: c_char,
                rows: usize, cols: usize, alpha: Self,
                a: *const Self, lda: usize,
                beta: Self, b: *const Self, ldb: usize,
                c: *mut Self, ldc: usize,
            ) {
                unsafe {
                    sys::$omatadd(
                        ordering, transa, transb, rows, cols, alpha,
                        a, lda, beta, b, ldb, c, ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_axpy_batch_strided(
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                y: *mut Self, incy: MKL_INT, stridey: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$axpy_batch(
                        n, alpha, x, incx, stridex, y, incy, stridey, batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_copy_batch_strided(
                n: MKL_INT,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                y: *mut Self, incy: MKL_INT, stridey: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$copy_batch(
                        n, x, incx, stridex, y, incy, stridey, batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemv_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                beta: Self,
                y: *mut Self, incy: MKL_INT, stridey: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$gemv_batch(
                        layout, trans, m, n, alpha,
                        a, lda, stridea, x, incx, stridex,
                        beta, y, incy, stridey, batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemm_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                b: *const Self, ldb: MKL_INT, strideb: MKL_INT,
                beta: Self,
                c: *mut Self, ldc: MKL_INT, stridec: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm_batch(
                        layout, transa, transb, m, n, k, alpha,
                        a, lda, stridea, b, ldb, strideb,
                        beta, c, ldc, stridec, batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemm_batch(
                layout: sys::CBLAS_LAYOUT::Type,
                transa_array: *const sys::CBLAS_TRANSPOSE::Type,
                transb_array: *const sys::CBLAS_TRANSPOSE::Type,
                m_array: *const MKL_INT,
                n_array: *const MKL_INT,
                k_array: *const MKL_INT,
                alpha_array: *const Self,
                a_array: *mut *const Self,
                lda_array: *const MKL_INT,
                b_array: *mut *const Self,
                ldb_array: *const MKL_INT,
                beta_array: *const Self,
                c_array: *mut *mut Self,
                ldc_array: *const MKL_INT,
                group_count: MKL_INT,
                group_size: *const MKL_INT,
            ) {
                unsafe {
                    sys::$gemm_batch_ptr(
                        layout, transa_array, transb_array,
                        m_array, n_array, k_array,
                        alpha_array,
                        a_array, lda_array,
                        b_array, ldb_array,
                        beta_array,
                        c_array, ldc_array,
                        group_count, group_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemv_batch(
                layout: sys::CBLAS_LAYOUT::Type,
                trans_array: *const sys::CBLAS_TRANSPOSE::Type,
                m_array: *const MKL_INT,
                n_array: *const MKL_INT,
                alpha_array: *const Self,
                a_array: *mut *const Self,
                lda_array: *const MKL_INT,
                x_array: *mut *const Self,
                incx_array: *const MKL_INT,
                beta_array: *const Self,
                y_array: *mut *mut Self,
                incy_array: *const MKL_INT,
                group_count: MKL_INT,
                group_size: *const MKL_INT,
            ) {
                unsafe {
                    sys::$gemv_batch_ptr(
                        layout, trans_array, m_array, n_array,
                        alpha_array,
                        a_array, lda_array,
                        x_array, incx_array,
                        beta_array,
                        y_array, incy_array,
                        group_count, group_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trsm_batch(
                layout: sys::CBLAS_LAYOUT::Type,
                side_array: *const sys::CBLAS_SIDE::Type,
                uplo_array: *const sys::CBLAS_UPLO::Type,
                transa_array: *const sys::CBLAS_TRANSPOSE::Type,
                diag_array: *const sys::CBLAS_DIAG::Type,
                m_array: *const MKL_INT,
                n_array: *const MKL_INT,
                alpha_array: *const Self,
                a_array: *mut *const Self,
                lda_array: *const MKL_INT,
                b_array: *mut *mut Self,
                ldb_array: *const MKL_INT,
                group_count: MKL_INT,
                group_size: *const MKL_INT,
            ) {
                unsafe {
                    sys::$trsm_batch_ptr(
                        layout, side_array, uplo_array,
                        transa_array, diag_array,
                        m_array, n_array,
                        alpha_array,
                        a_array, lda_array,
                        b_array, ldb_array,
                        group_count, group_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trsm_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                diag: sys::CBLAS_DIAG::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                b: *mut Self, ldb: MKL_INT, strideb: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$trsm_batch(
                        layout, side, uplo, trans, diag, m, n, alpha,
                        a, lda, stridea, b, ldb, strideb, batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_dgmm_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                m: MKL_INT, n: MKL_INT,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                c: *mut Self, ldc: MKL_INT, stridec: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$dgmm_batch(
                        layout, side, m, n,
                        a, lda, stridea, x, incx, stridex,
                        c, ldc, stridec, batch_size,
                    )
                }
            }
        }

        impl RealBlasScalar for $ty {
            #[inline]
            unsafe fn cblas_dot(
                n: MKL_INT, x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
            ) -> Self {
                unsafe { sys::$dot(n, x, incx, y, incy) }
            }
            #[inline]
            unsafe fn cblas_rot(
                n: MKL_INT, x: *mut Self, incx: MKL_INT,
                y: *mut Self, incy: MKL_INT, c: Self, s: Self,
            ) {
                unsafe { sys::$rot(n, x, incx, y, incy, c, s) }
            }
            #[inline]
            unsafe fn cblas_rotg(
                a: *mut Self, b: *mut Self, c: *mut Self, s: *mut Self,
            ) {
                unsafe { sys::$rotg(a, b, c, s) }
            }
            #[inline]
            unsafe fn cblas_ger(
                layout: sys::CBLAS_LAYOUT::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) {
                unsafe {
                    sys::$ger(layout, m, n, alpha, x, incx, y, incy, a, lda)
                }
            }
            #[inline]
            unsafe fn cblas_symv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$symv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
                }
            }
            #[inline]
            unsafe fn cblas_syr(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) {
                unsafe { sys::$syr(layout, uplo, n, alpha, x, incx, a, lda) }
            }
            #[inline]
            unsafe fn cblas_syr2(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) {
                unsafe {
                    sys::$syr2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
                }
            }
            #[inline]
            unsafe fn cblas_sbmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$sbmv(layout, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
                }
            }
            #[inline]
            unsafe fn cblas_spmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                ap: *const Self,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$spmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy)
                }
            }
            #[inline]
            unsafe fn cblas_spr(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                ap: *mut Self,
            ) {
                unsafe { sys::$spr(layout, uplo, n, alpha, x, incx, ap) }
            }
            #[inline]
            unsafe fn cblas_spr2(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
                ap: *mut Self,
            ) {
                unsafe {
                    sys::$spr2(layout, uplo, n, alpha, x, incx, y, incy, ap)
                }
            }
        }
    };
}

impl_real_blas!(
    f32,
    asum = cblas_sasum, nrm2 = cblas_snrm2,
    iamax = cblas_isamax, iamin = cblas_isamin,
    scal = cblas_sscal, axpy = cblas_saxpy,
    copy = cblas_scopy, swap = cblas_sswap,
    dot = cblas_sdot, rot = cblas_srot, rotg = cblas_srotg,
    gemv = cblas_sgemv, gbmv = cblas_sgbmv,
    trmv = cblas_strmv, trsv = cblas_strsv,
    tbmv = cblas_stbmv, tbsv = cblas_stbsv,
    tpmv = cblas_stpmv, tpsv = cblas_stpsv,
    ger = cblas_sger, symv = cblas_ssymv,
    syr = cblas_ssyr, syr2 = cblas_ssyr2,
    sbmv = cblas_ssbmv, spmv = cblas_sspmv,
    spr = cblas_sspr, spr2 = cblas_sspr2,
    gemm = cblas_sgemm,
    symm = cblas_ssymm, syrk = cblas_ssyrk, syr2k = cblas_ssyr2k,
    trmm = cblas_strmm, trsm = cblas_strsm, gemmt = cblas_sgemmt,
    axpby = cblas_saxpby,
    imatcopy = MKL_Simatcopy, omatcopy = MKL_Somatcopy,
    omatcopy2 = MKL_Somatcopy2, omatadd = MKL_Somatadd,
    axpy_batch_strided = cblas_saxpy_batch_strided,
    copy_batch_strided = cblas_scopy_batch_strided,
    gemv_batch_strided = cblas_sgemv_batch_strided,
    gemm_batch_strided = cblas_sgemm_batch_strided,
    gemm_batch_ptr = cblas_sgemm_batch,
    gemv_batch_ptr = cblas_sgemv_batch,
    trsm_batch_strided = cblas_strsm_batch_strided,
    trsm_batch_ptr = cblas_strsm_batch,
    dgmm_batch_strided = cblas_sdgmm_batch_strided,
);

impl_real_blas!(
    f64,
    asum = cblas_dasum, nrm2 = cblas_dnrm2,
    iamax = cblas_idamax, iamin = cblas_idamin,
    scal = cblas_dscal, axpy = cblas_daxpy,
    copy = cblas_dcopy, swap = cblas_dswap,
    dot = cblas_ddot, rot = cblas_drot, rotg = cblas_drotg,
    gemv = cblas_dgemv, gbmv = cblas_dgbmv,
    trmv = cblas_dtrmv, trsv = cblas_dtrsv,
    tbmv = cblas_dtbmv, tbsv = cblas_dtbsv,
    tpmv = cblas_dtpmv, tpsv = cblas_dtpsv,
    ger = cblas_dger, symv = cblas_dsymv,
    syr = cblas_dsyr, syr2 = cblas_dsyr2,
    sbmv = cblas_dsbmv, spmv = cblas_dspmv,
    spr = cblas_dspr, spr2 = cblas_dspr2,
    gemm = cblas_dgemm,
    symm = cblas_dsymm, syrk = cblas_dsyrk, syr2k = cblas_dsyr2k,
    trmm = cblas_dtrmm, trsm = cblas_dtrsm, gemmt = cblas_dgemmt,
    axpby = cblas_daxpby,
    imatcopy = MKL_Dimatcopy, omatcopy = MKL_Domatcopy,
    omatcopy2 = MKL_Domatcopy2, omatadd = MKL_Domatadd,
    axpy_batch_strided = cblas_daxpy_batch_strided,
    copy_batch_strided = cblas_dcopy_batch_strided,
    gemv_batch_strided = cblas_dgemv_batch_strided,
    gemm_batch_strided = cblas_dgemm_batch_strided,
    gemm_batch_ptr = cblas_dgemm_batch,
    gemv_batch_ptr = cblas_dgemv_batch,
    trsm_batch_strided = cblas_dtrsm_batch_strided,
    trsm_batch_ptr = cblas_dtrsm_batch,
    dgmm_batch_strided = cblas_ddgmm_batch_strided,
);

macro_rules! impl_complex_blas {
    (
        $ty:ty,
        // L1
        asum = $asum:ident, nrm2 = $nrm2:ident,
        iamax = $iamax:ident, iamin = $iamin:ident,
        scal = $scal:ident, scal_real = $scal_real:ident,
        axpy = $axpy:ident, copy = $copy:ident, swap = $swap:ident,
        dotc_sub = $dotc:ident, dotu_sub = $dotu:ident,
        // L2 universal
        gemv = $gemv:ident, gbmv = $gbmv:ident,
        trmv = $trmv:ident, trsv = $trsv:ident,
        tbmv = $tbmv:ident, tbsv = $tbsv:ident,
        tpmv = $tpmv:ident, tpsv = $tpsv:ident,
        // L2 complex-only
        gerc = $gerc:ident, geru = $geru:ident,
        hemv = $hemv:ident, her = $her:ident, her2 = $her2:ident,
        hbmv = $hbmv:ident, hpmv = $hpmv:ident,
        hpr = $hpr:ident, hpr2 = $hpr2:ident,
        // L3
        gemm = $gemm:ident,
        symm = $symm:ident, syrk = $syrk:ident, syr2k = $syr2k:ident,
        trmm = $trmm:ident, trsm = $trsm:ident, gemmt = $gemmt:ident,
        // L3 complex-only
        hemm = $hemm:ident, herk = $herk:ident, her2k = $her2k:ident,
        gemm3m = $gemm3m:ident,
        gemm3m_batch_strided = $gemm3m_batch:ident,
        // Extensions
        axpby = $axpby:ident,
        imatcopy = $imatcopy:ident,
        omatcopy = $omatcopy:ident,
        omatcopy2 = $omatcopy2:ident,
        omatadd = $omatadd:ident,
        // Batched (strided) — universal across all four scalar types
        axpy_batch_strided = $axpy_batch:ident,
        copy_batch_strided = $copy_batch:ident,
        gemv_batch_strided = $gemv_batch:ident,
        gemm_batch_strided = $gemm_batch:ident,
        gemm_batch_ptr = $gemm_batch_ptr:ident,
        gemv_batch_ptr = $gemv_batch_ptr:ident,
        trsm_batch_strided = $trsm_batch:ident,
        trsm_batch_ptr = $trsm_batch_ptr:ident,
        dgmm_batch_strided = $dgmm_batch:ident,
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
                n: MKL_INT, alpha: Self, x: *const Self, incx: MKL_INT,
                y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$axpy(
                        n, (&alpha as *const Self).cast(), x.cast(), incx, y.cast(), incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_copy(
                n: MKL_INT, x: *const Self, incx: MKL_INT,
                y: *mut Self, incy: MKL_INT,
            ) {
                unsafe { sys::$copy(n, x.cast(), incx, y.cast(), incy) }
            }
            #[inline]
            unsafe fn cblas_swap(
                n: MKL_INT, x: *mut Self, incx: MKL_INT,
                y: *mut Self, incy: MKL_INT,
            ) {
                unsafe { sys::$swap(n, x.cast(), incx, y.cast(), incy) }
            }
            #[inline]
            unsafe fn cblas_gemv(
                layout: sys::CBLAS_LAYOUT::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$gemv(
                        layout, trans, m, n, (&alpha as *const Self).cast(),
                        a.cast(), lda, x.cast(), incx, (&beta as *const Self).cast(),
                        y.cast(), incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gbmv(
                layout: sys::CBLAS_LAYOUT::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, kl: MKL_INT, ku: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$gbmv(
                        layout, trans, m, n, kl, ku, (&alpha as *const Self).cast(),
                        a.cast(), lda, x.cast(), incx, (&beta as *const Self).cast(),
                        y.cast(), incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe {
                    sys::$trmv(layout, uplo, trans, diag, n, a.cast(), lda, x.cast(), incx)
                }
            }
            #[inline]
            unsafe fn cblas_trsv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe {
                    sys::$trsv(layout, uplo, trans, diag, n, a.cast(), lda, x.cast(), incx)
                }
            }
            #[inline]
            unsafe fn cblas_tbmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, k: MKL_INT,
                a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe {
                    sys::$tbmv(
                        layout, uplo, trans, diag, n, k, a.cast(), lda, x.cast(), incx,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_tbsv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, k: MKL_INT,
                a: *const Self, lda: MKL_INT,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe {
                    sys::$tbsv(
                        layout, uplo, trans, diag, n, k, a.cast(), lda, x.cast(), incx,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_tpmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, ap: *const Self,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe {
                    sys::$tpmv(layout, uplo, trans, diag, n, ap.cast(), x.cast(), incx)
                }
            }
            #[inline]
            unsafe fn cblas_tpsv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, diag: sys::CBLAS_DIAG::Type,
                n: MKL_INT, ap: *const Self,
                x: *mut Self, incx: MKL_INT,
            ) {
                unsafe {
                    sys::$tpsv(layout, uplo, trans, diag, n, ap.cast(), x.cast(), incx)
                }
            }
            #[inline]
            unsafe fn cblas_gemm(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm(
                        layout, transa, transb, m, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_symm(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$symm(
                        layout, side, uplo, m, n,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_syrk(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$syrk(
                        layout, uplo, trans, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_syr2k(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$syr2k(
                        layout, uplo, trans, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trmm(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                diag: sys::CBLAS_DIAG::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) {
                unsafe {
                    sys::$trmm(
                        layout, side, uplo, trans, diag, m, n,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trsm(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                diag: sys::CBLAS_DIAG::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) {
                unsafe {
                    sys::$trsm(
                        layout, side, uplo, trans, diag, m, n,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemmt(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                transa: sys::CBLAS_TRANSPOSE::Type, transb: sys::CBLAS_TRANSPOSE::Type,
                n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$gemmt(
                        layout, uplo, transa, transb, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_axpby(
                n: MKL_INT, alpha: Self, x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$axpby(
                        n,
                        (&alpha as *const Self).cast(),
                        x.cast(), incx,
                        (&beta as *const Self).cast(),
                        y.cast(), incy,
                    )
                }
            }
            #[inline]
            unsafe fn mkl_imatcopy(
                ordering: c_char, trans: c_char,
                rows: usize, cols: usize, alpha: Self,
                ab: *mut Self, lda: usize, ldb: usize,
            ) {
                unsafe {
                    sys::$imatcopy(
                        ordering, trans, rows, cols,
                        core::mem::transmute_copy(&alpha),
                        ab.cast(), lda, ldb,
                    )
                }
            }
            #[inline]
            unsafe fn mkl_omatcopy(
                ordering: c_char, trans: c_char,
                rows: usize, cols: usize, alpha: Self,
                a: *const Self, lda: usize,
                b: *mut Self, ldb: usize,
            ) {
                unsafe {
                    sys::$omatcopy(
                        ordering, trans, rows, cols,
                        core::mem::transmute_copy(&alpha),
                        a.cast(), lda, b.cast(), ldb,
                    )
                }
            }
            #[inline]
            unsafe fn mkl_omatcopy2(
                ordering: c_char, trans: c_char,
                rows: usize, cols: usize, alpha: Self,
                a: *const Self, lda: usize, stridea: usize,
                b: *mut Self, ldb: usize, strideb: usize,
            ) {
                unsafe {
                    sys::$omatcopy2(
                        ordering, trans, rows, cols,
                        core::mem::transmute_copy(&alpha),
                        a.cast(), lda, stridea, b.cast(), ldb, strideb,
                    )
                }
            }
            #[inline]
            unsafe fn mkl_omatadd(
                ordering: c_char, transa: c_char, transb: c_char,
                rows: usize, cols: usize, alpha: Self,
                a: *const Self, lda: usize,
                beta: Self, b: *const Self, ldb: usize,
                c: *mut Self, ldc: usize,
            ) {
                unsafe {
                    sys::$omatadd(
                        ordering, transa, transb, rows, cols,
                        core::mem::transmute_copy(&alpha),
                        a.cast(), lda,
                        core::mem::transmute_copy(&beta),
                        b.cast(), ldb,
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_axpy_batch_strided(
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                y: *mut Self, incy: MKL_INT, stridey: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$axpy_batch(
                        n,
                        (&alpha as *const Self).cast(),
                        x.cast(), incx, stridex,
                        y.cast(), incy, stridey,
                        batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_copy_batch_strided(
                n: MKL_INT,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                y: *mut Self, incy: MKL_INT, stridey: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$copy_batch(
                        n,
                        x.cast(), incx, stridex,
                        y.cast(), incy, stridey,
                        batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemv_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                beta: Self,
                y: *mut Self, incy: MKL_INT, stridey: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$gemv_batch(
                        layout, trans, m, n,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, stridea,
                        x.cast(), incx, stridex,
                        (&beta as *const Self).cast(),
                        y.cast(), incy, stridey,
                        batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemm_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                b: *const Self, ldb: MKL_INT, strideb: MKL_INT,
                beta: Self,
                c: *mut Self, ldc: MKL_INT, stridec: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm_batch(
                        layout, transa, transb, m, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, stridea,
                        b.cast(), ldb, strideb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc, stridec,
                        batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemm_batch(
                layout: sys::CBLAS_LAYOUT::Type,
                transa_array: *const sys::CBLAS_TRANSPOSE::Type,
                transb_array: *const sys::CBLAS_TRANSPOSE::Type,
                m_array: *const MKL_INT,
                n_array: *const MKL_INT,
                k_array: *const MKL_INT,
                alpha_array: *const Self,
                a_array: *mut *const Self,
                lda_array: *const MKL_INT,
                b_array: *mut *const Self,
                ldb_array: *const MKL_INT,
                beta_array: *const Self,
                c_array: *mut *mut Self,
                ldc_array: *const MKL_INT,
                group_count: MKL_INT,
                group_size: *const MKL_INT,
            ) {
                unsafe {
                    sys::$gemm_batch_ptr(
                        layout, transa_array, transb_array,
                        m_array, n_array, k_array,
                        alpha_array.cast(),
                        a_array.cast(), lda_array,
                        b_array.cast(), ldb_array,
                        beta_array.cast(),
                        c_array.cast(), ldc_array,
                        group_count, group_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemv_batch(
                layout: sys::CBLAS_LAYOUT::Type,
                trans_array: *const sys::CBLAS_TRANSPOSE::Type,
                m_array: *const MKL_INT,
                n_array: *const MKL_INT,
                alpha_array: *const Self,
                a_array: *mut *const Self,
                lda_array: *const MKL_INT,
                x_array: *mut *const Self,
                incx_array: *const MKL_INT,
                beta_array: *const Self,
                y_array: *mut *mut Self,
                incy_array: *const MKL_INT,
                group_count: MKL_INT,
                group_size: *const MKL_INT,
            ) {
                unsafe {
                    sys::$gemv_batch_ptr(
                        layout, trans_array, m_array, n_array,
                        alpha_array.cast(),
                        a_array.cast(), lda_array,
                        x_array.cast(), incx_array,
                        beta_array.cast(),
                        y_array.cast(), incy_array,
                        group_count, group_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trsm_batch(
                layout: sys::CBLAS_LAYOUT::Type,
                side_array: *const sys::CBLAS_SIDE::Type,
                uplo_array: *const sys::CBLAS_UPLO::Type,
                transa_array: *const sys::CBLAS_TRANSPOSE::Type,
                diag_array: *const sys::CBLAS_DIAG::Type,
                m_array: *const MKL_INT,
                n_array: *const MKL_INT,
                alpha_array: *const Self,
                a_array: *mut *const Self,
                lda_array: *const MKL_INT,
                b_array: *mut *mut Self,
                ldb_array: *const MKL_INT,
                group_count: MKL_INT,
                group_size: *const MKL_INT,
            ) {
                unsafe {
                    sys::$trsm_batch_ptr(
                        layout, side_array, uplo_array,
                        transa_array, diag_array,
                        m_array, n_array,
                        alpha_array.cast(),
                        a_array.cast(), lda_array,
                        b_array.cast(), ldb_array,
                        group_count, group_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_trsm_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, trans: sys::CBLAS_TRANSPOSE::Type,
                diag: sys::CBLAS_DIAG::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                b: *mut Self, ldb: MKL_INT, strideb: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$trsm_batch(
                        layout, side, uplo, trans, diag, m, n,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, stridea,
                        b.cast(), ldb, strideb,
                        batch_size,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_dgmm_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                m: MKL_INT, n: MKL_INT,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                x: *const Self, incx: MKL_INT, stridex: MKL_INT,
                c: *mut Self, ldc: MKL_INT, stridec: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$dgmm_batch(
                        layout, side, m, n,
                        a.cast(), lda, stridea,
                        x.cast(), incx, stridex,
                        c.cast(), ldc, stridec,
                        batch_size,
                    )
                }
            }
        }

        impl ComplexBlasScalar for $ty {
            #[inline]
            unsafe fn cblas_dotc(
                n: MKL_INT, x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
            ) -> Self {
                let mut out = <Self as Scalar>::zero();
                unsafe {
                    sys::$dotc(
                        n, x.cast(), incx, y.cast(), incy,
                        (&mut out as *mut Self).cast(),
                    );
                }
                out
            }
            #[inline]
            unsafe fn cblas_dotu(
                n: MKL_INT, x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
            ) -> Self {
                let mut out = <Self as Scalar>::zero();
                unsafe {
                    sys::$dotu(
                        n, x.cast(), incx, y.cast(), incy,
                        (&mut out as *mut Self).cast(),
                    );
                }
                out
            }
            #[inline]
            unsafe fn cblas_scal_real(
                n: MKL_INT, alpha: Self::Real, x: *mut Self, incx: MKL_INT,
            ) {
                unsafe { sys::$scal_real(n, alpha, x.cast(), incx) }
            }
            #[inline]
            unsafe fn cblas_gerc(
                layout: sys::CBLAS_LAYOUT::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) {
                unsafe {
                    sys::$gerc(
                        layout, m, n, (&alpha as *const Self).cast(),
                        x.cast(), incx, y.cast(), incy, a.cast(), lda,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_geru(
                layout: sys::CBLAS_LAYOUT::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) {
                unsafe {
                    sys::$geru(
                        layout, m, n, (&alpha as *const Self).cast(),
                        x.cast(), incx, y.cast(), incy, a.cast(), lda,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_hemv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$hemv(
                        layout, uplo, n, (&alpha as *const Self).cast(),
                        a.cast(), lda, x.cast(), incx,
                        (&beta as *const Self).cast(), y.cast(), incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_her(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self::Real,
                x: *const Self, incx: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) {
                unsafe {
                    sys::$her(layout, uplo, n, alpha, x.cast(), incx, a.cast(), lda)
                }
            }
            #[inline]
            unsafe fn cblas_her2(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) {
                unsafe {
                    sys::$her2(
                        layout, uplo, n, (&alpha as *const Self).cast(),
                        x.cast(), incx, y.cast(), incy, a.cast(), lda,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_hbmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$hbmv(
                        layout, uplo, n, k, (&alpha as *const Self).cast(),
                        a.cast(), lda, x.cast(), incx,
                        (&beta as *const Self).cast(), y.cast(), incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_hpmv(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                ap: *const Self,
                x: *const Self, incx: MKL_INT,
                beta: Self, y: *mut Self, incy: MKL_INT,
            ) {
                unsafe {
                    sys::$hpmv(
                        layout, uplo, n, (&alpha as *const Self).cast(),
                        ap.cast(), x.cast(), incx,
                        (&beta as *const Self).cast(), y.cast(), incy,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_hpr(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self::Real,
                x: *const Self, incx: MKL_INT,
                ap: *mut Self,
            ) {
                unsafe { sys::$hpr(layout, uplo, n, alpha, x.cast(), incx, ap.cast()) }
            }
            #[inline]
            unsafe fn cblas_hpr2(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                n: MKL_INT, alpha: Self,
                x: *const Self, incx: MKL_INT,
                y: *const Self, incy: MKL_INT,
                ap: *mut Self,
            ) {
                unsafe {
                    sys::$hpr2(
                        layout, uplo, n, (&alpha as *const Self).cast(),
                        x.cast(), incx, y.cast(), incy, ap.cast(),
                    )
                }
            }
            #[inline]
            unsafe fn cblas_hemm(
                layout: sys::CBLAS_LAYOUT::Type, side: sys::CBLAS_SIDE::Type,
                uplo: sys::CBLAS_UPLO::Type, m: MKL_INT, n: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$hemm(
                        layout, side, uplo, m, n,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_herk(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, n: MKL_INT, k: MKL_INT,
                alpha: Self::Real,
                a: *const Self, lda: MKL_INT,
                beta: Self::Real,
                c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$herk(
                        layout, uplo, trans, n, k, alpha, a.cast(), lda, beta,
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_her2k(
                layout: sys::CBLAS_LAYOUT::Type, uplo: sys::CBLAS_UPLO::Type,
                trans: sys::CBLAS_TRANSPOSE::Type, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self::Real,
                c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$her2k(
                        layout, uplo, trans, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                        beta, c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemm3m(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT,
                b: *const Self, ldb: MKL_INT,
                beta: Self, c: *mut Self, ldc: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm3m(
                        layout, transa, transb, m, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, b.cast(), ldb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc,
                    )
                }
            }
            #[inline]
            unsafe fn cblas_gemm3m_batch_strided(
                layout: sys::CBLAS_LAYOUT::Type,
                transa: sys::CBLAS_TRANSPOSE::Type,
                transb: sys::CBLAS_TRANSPOSE::Type,
                m: MKL_INT, n: MKL_INT, k: MKL_INT, alpha: Self,
                a: *const Self, lda: MKL_INT, stridea: MKL_INT,
                b: *const Self, ldb: MKL_INT, strideb: MKL_INT,
                beta: Self,
                c: *mut Self, ldc: MKL_INT, stridec: MKL_INT,
                batch_size: MKL_INT,
            ) {
                unsafe {
                    sys::$gemm3m_batch(
                        layout, transa, transb, m, n, k,
                        (&alpha as *const Self).cast(),
                        a.cast(), lda, stridea,
                        b.cast(), ldb, strideb,
                        (&beta as *const Self).cast(),
                        c.cast(), ldc, stridec,
                        batch_size,
                    )
                }
            }
        }
    };
}

impl_complex_blas!(
    Complex32,
    asum = cblas_scasum, nrm2 = cblas_scnrm2,
    iamax = cblas_icamax, iamin = cblas_icamin,
    scal = cblas_cscal, scal_real = cblas_csscal,
    axpy = cblas_caxpy, copy = cblas_ccopy, swap = cblas_cswap,
    dotc_sub = cblas_cdotc_sub, dotu_sub = cblas_cdotu_sub,
    gemv = cblas_cgemv, gbmv = cblas_cgbmv,
    trmv = cblas_ctrmv, trsv = cblas_ctrsv,
    tbmv = cblas_ctbmv, tbsv = cblas_ctbsv,
    tpmv = cblas_ctpmv, tpsv = cblas_ctpsv,
    gerc = cblas_cgerc, geru = cblas_cgeru,
    hemv = cblas_chemv, her = cblas_cher, her2 = cblas_cher2,
    hbmv = cblas_chbmv, hpmv = cblas_chpmv,
    hpr = cblas_chpr, hpr2 = cblas_chpr2,
    gemm = cblas_cgemm,
    symm = cblas_csymm, syrk = cblas_csyrk, syr2k = cblas_csyr2k,
    trmm = cblas_ctrmm, trsm = cblas_ctrsm, gemmt = cblas_cgemmt,
    hemm = cblas_chemm, herk = cblas_cherk, her2k = cblas_cher2k,
    gemm3m = cblas_cgemm3m,
    gemm3m_batch_strided = cblas_cgemm3m_batch_strided,
    axpby = cblas_caxpby,
    imatcopy = MKL_Cimatcopy, omatcopy = MKL_Comatcopy,
    omatcopy2 = MKL_Comatcopy2, omatadd = MKL_Comatadd,
    axpy_batch_strided = cblas_caxpy_batch_strided,
    copy_batch_strided = cblas_ccopy_batch_strided,
    gemv_batch_strided = cblas_cgemv_batch_strided,
    gemm_batch_strided = cblas_cgemm_batch_strided,
    gemm_batch_ptr = cblas_cgemm_batch,
    gemv_batch_ptr = cblas_cgemv_batch,
    trsm_batch_strided = cblas_ctrsm_batch_strided,
    trsm_batch_ptr = cblas_ctrsm_batch,
    dgmm_batch_strided = cblas_cdgmm_batch_strided,
);

impl_complex_blas!(
    Complex64,
    asum = cblas_dzasum, nrm2 = cblas_dznrm2,
    iamax = cblas_izamax, iamin = cblas_izamin,
    scal = cblas_zscal, scal_real = cblas_zdscal,
    axpy = cblas_zaxpy, copy = cblas_zcopy, swap = cblas_zswap,
    dotc_sub = cblas_zdotc_sub, dotu_sub = cblas_zdotu_sub,
    gemv = cblas_zgemv, gbmv = cblas_zgbmv,
    trmv = cblas_ztrmv, trsv = cblas_ztrsv,
    tbmv = cblas_ztbmv, tbsv = cblas_ztbsv,
    tpmv = cblas_ztpmv, tpsv = cblas_ztpsv,
    gerc = cblas_zgerc, geru = cblas_zgeru,
    hemv = cblas_zhemv, her = cblas_zher, her2 = cblas_zher2,
    hbmv = cblas_zhbmv, hpmv = cblas_zhpmv,
    hpr = cblas_zhpr, hpr2 = cblas_zhpr2,
    gemm = cblas_zgemm,
    symm = cblas_zsymm, syrk = cblas_zsyrk, syr2k = cblas_zsyr2k,
    trmm = cblas_ztrmm, trsm = cblas_ztrsm, gemmt = cblas_zgemmt,
    hemm = cblas_zhemm, herk = cblas_zherk, her2k = cblas_zher2k,
    gemm3m = cblas_zgemm3m,
    gemm3m_batch_strided = cblas_zgemm3m_batch_strided,
    axpby = cblas_zaxpby,
    imatcopy = MKL_Zimatcopy, omatcopy = MKL_Zomatcopy,
    omatcopy2 = MKL_Zomatcopy2, omatadd = MKL_Zomatadd,
    axpy_batch_strided = cblas_zaxpy_batch_strided,
    copy_batch_strided = cblas_zcopy_batch_strided,
    gemv_batch_strided = cblas_zgemv_batch_strided,
    gemm_batch_strided = cblas_zgemm_batch_strided,
    gemm_batch_ptr = cblas_zgemm_batch,
    gemv_batch_ptr = cblas_zgemv_batch,
    trsm_batch_strided = cblas_ztrsm_batch_strided,
    trsm_batch_ptr = cblas_ztrsm_batch,
    dgmm_batch_strided = cblas_zdgmm_batch_strided,
);
