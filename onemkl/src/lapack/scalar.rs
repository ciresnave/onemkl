//! Scalar-typed dispatch traits for LAPACK (via the LAPACKE C interface).

use core::ffi::{c_char, c_int};

use num_complex::{Complex32, Complex64};
use onemkl_sys::{self as sys, MKL_INT};

use crate::scalar::{ComplexScalar, RealScalar, Scalar};

/// LAPACK operations supported across all four scalar types.
#[allow(missing_docs)]
pub trait LapackScalar: Scalar {
    // ---- Linear solvers ----

    /// `LAPACKE_*gesv` — solve `A * X = B` (LU + substitution).
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gesv(
        layout: c_int, n: MKL_INT, nrhs: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        ipiv: *mut MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*getrf` — LU factorization.
    unsafe fn lapacke_getrf(
        layout: c_int, m: MKL_INT, n: MKL_INT,
        a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
    ) -> i32;

    /// `LAPACKE_*getrs` — solve with LU factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_getrs(
        layout: c_int, trans: c_char, n: MKL_INT, nrhs: MKL_INT,
        a: *const Self, lda: MKL_INT,
        ipiv: *const MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*getri` — invert an LU-factored matrix.
    unsafe fn lapacke_getri(
        layout: c_int, n: MKL_INT,
        a: *mut Self, lda: MKL_INT, ipiv: *const MKL_INT,
    ) -> i32;

    /// `LAPACKE_*posv` — Cholesky solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_posv(
        layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*potrf` — Cholesky factorization.
    unsafe fn lapacke_potrf(
        layout: c_int, uplo: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*potrs` — solve with Cholesky factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_potrs(
        layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
        a: *const Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    // ---- QR & LQ ----

    unsafe fn lapacke_geqrf(
        layout: c_int, m: MKL_INT, n: MKL_INT,
        a: *mut Self, lda: MKL_INT, tau: *mut Self,
    ) -> i32;

    unsafe fn lapacke_gelqf(
        layout: c_int, m: MKL_INT, n: MKL_INT,
        a: *mut Self, lda: MKL_INT, tau: *mut Self,
    ) -> i32;

    /// Real types call `?orgqr`; complex types call `?ungqr`.
    unsafe fn lapacke_orgqr(
        layout: c_int, m: MKL_INT, n: MKL_INT, k: MKL_INT,
        a: *mut Self, lda: MKL_INT, tau: *const Self,
    ) -> i32;

    // ---- Least squares ----

    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gels(
        layout: c_int, trans: c_char,
        m: MKL_INT, n: MKL_INT, nrhs: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gelsd(
        layout: c_int,
        m: MKL_INT, n: MKL_INT, nrhs: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
        s: *mut Self::Real, rcond: Self::Real,
        rank: *mut MKL_INT,
    ) -> i32;

    // ---- SVD ----

    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gesdd(
        layout: c_int, jobz: c_char,
        m: MKL_INT, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        s: *mut Self::Real,
        u: *mut Self, ldu: MKL_INT,
        vt: *mut Self, ldvt: MKL_INT,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gesvd(
        layout: c_int, jobu: c_char, jobvt: c_char,
        m: MKL_INT, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        s: *mut Self::Real,
        u: *mut Self, ldu: MKL_INT,
        vt: *mut Self, ldvt: MKL_INT,
        superb: *mut Self::Real,
    ) -> i32;

    // ---- Banded ----

    /// `LAPACKE_*gbsv` — general band solve (LU + substitution).
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gbsv(
        layout: c_int, n: MKL_INT, kl: MKL_INT, ku: MKL_INT, nrhs: MKL_INT,
        ab: *mut Self, ldab: MKL_INT, ipiv: *mut MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*gbtrf` — general band LU factorization.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gbtrf(
        layout: c_int, m: MKL_INT, n: MKL_INT, kl: MKL_INT, ku: MKL_INT,
        ab: *mut Self, ldab: MKL_INT, ipiv: *mut MKL_INT,
    ) -> i32;

    /// `LAPACKE_*gbtrs` — solve with general band LU factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gbtrs(
        layout: c_int, trans: c_char,
        n: MKL_INT, kl: MKL_INT, ku: MKL_INT, nrhs: MKL_INT,
        ab: *const Self, ldab: MKL_INT, ipiv: *const MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*gtsv` — general tridiagonal solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gtsv(
        layout: c_int, n: MKL_INT, nrhs: MKL_INT,
        dl: *mut Self, d: *mut Self, du: *mut Self,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*gttrf` — general tridiagonal LU factorization. Note
    /// no `matrix_layout` argument.
    unsafe fn lapacke_gttrf(
        n: MKL_INT,
        dl: *mut Self, d: *mut Self, du: *mut Self, du2: *mut Self,
        ipiv: *mut MKL_INT,
    ) -> i32;

    /// `LAPACKE_*gttrs` — solve with general tridiagonal LU factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_gttrs(
        layout: c_int, trans: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        dl: *const Self, d: *const Self, du: *const Self, du2: *const Self,
        ipiv: *const MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*pbsv` — symmetric / Hermitian positive-definite band
    /// solve (Cholesky).
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_pbsv(
        layout: c_int, uplo: c_char,
        n: MKL_INT, kd: MKL_INT, nrhs: MKL_INT,
        ab: *mut Self, ldab: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*pbtrf` — PD band Cholesky factorization.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_pbtrf(
        layout: c_int, uplo: c_char,
        n: MKL_INT, kd: MKL_INT,
        ab: *mut Self, ldab: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*pbtrs` — solve with PD band Cholesky factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_pbtrs(
        layout: c_int, uplo: c_char,
        n: MKL_INT, kd: MKL_INT, nrhs: MKL_INT,
        ab: *const Self, ldab: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*ptsv` — PD tridiagonal solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_ptsv(
        layout: c_int, n: MKL_INT, nrhs: MKL_INT,
        d: *mut Self::Real, e: *mut Self,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*pttrf` — PD tridiagonal Cholesky factorization. Note
    /// no `matrix_layout` argument.
    unsafe fn lapacke_pttrf(
        n: MKL_INT,
        d: *mut Self::Real, e: *mut Self,
    ) -> i32;

    // ---- Packed ----

    /// `LAPACKE_*spsv` — symmetric packed solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_spsv(
        layout: c_int, uplo: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        ap: *mut Self, ipiv: *mut MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*sptrf` — symmetric packed factorization.
    unsafe fn lapacke_sptrf(
        layout: c_int, uplo: c_char,
        n: MKL_INT, ap: *mut Self, ipiv: *mut MKL_INT,
    ) -> i32;

    /// `LAPACKE_*sptrs` — solve with symmetric packed factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_sptrs(
        layout: c_int, uplo: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        ap: *const Self, ipiv: *const MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*ppsv` — PD packed Cholesky solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_ppsv(
        layout: c_int, uplo: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        ap: *mut Self, b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*pptrf` — PD packed Cholesky factorization.
    unsafe fn lapacke_pptrf(
        layout: c_int, uplo: c_char,
        n: MKL_INT, ap: *mut Self,
    ) -> i32;

    /// `LAPACKE_*pptrs` — solve with PD packed Cholesky factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_pptrs(
        layout: c_int, uplo: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        ap: *const Self, b: *mut Self, ldb: MKL_INT,
    ) -> i32;
}

/// Real-only LAPACK operations.
#[allow(missing_docs)]
pub trait RealLapackScalar: LapackScalar + RealScalar {
    /// `LAPACKE_*sysv` — symmetric solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_sysv(
        layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
        a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*syev` — symmetric eigenproblem.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_syev(
        layout: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT, w: *mut Self,
    ) -> i32;

    /// `LAPACKE_*geev` for real types — eigenvalues split into wr/wi.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_geev(
        layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        wr: *mut Self, wi: *mut Self,
        vl: *mut Self, ldvl: MKL_INT,
        vr: *mut Self, ldvr: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*pttrs` (real) — solve with PD tridiagonal Cholesky
    /// factor. No uplo because the matrix is symmetric.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_pttrs(
        layout: c_int, n: MKL_INT, nrhs: MKL_INT,
        d: *const Self, e: *const Self,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*sygv` — generalized symmetric-definite eigenproblem
    /// `A x = lambda B x` (`itype = 1`), `A B x = lambda x` (`= 2`),
    /// or `B A x = lambda x` (`= 3`).
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_sygv(
        layout: c_int, itype: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
        w: *mut Self,
    ) -> i32;

    /// `LAPACKE_*ggev` (real) — generalized non-symmetric eigenproblem.
    /// Eigenvalues are returned as (alphar + i*alphai) / beta.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_ggev(
        layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
        alphar: *mut Self, alphai: *mut Self, beta: *mut Self,
        vl: *mut Self, ldvl: MKL_INT,
        vr: *mut Self, ldvr: MKL_INT,
    ) -> i32;
}

/// Complex-only LAPACK operations.
#[allow(missing_docs)]
pub trait ComplexLapackScalar: LapackScalar + ComplexScalar {
    /// `LAPACKE_*hesv` — Hermitian solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_hesv(
        layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
        a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*heev` — Hermitian eigenproblem.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_heev(
        layout: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT, w: *mut Self::Real,
    ) -> i32;

    /// `LAPACKE_*geev` for complex types — eigenvalues as one complex vector.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_geev_complex(
        layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        w: *mut Self,
        vl: *mut Self, ldvl: MKL_INT,
        vr: *mut Self, ldvr: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*pttrs` (complex) — solve with PD tridiagonal Cholesky
    /// factor. Takes uplo because the off-diagonal `e` is complex.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_pttrs_complex(
        layout: c_int, uplo: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        d: *const Self::Real, e: *const Self,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*hpsv` — Hermitian packed solve.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_hpsv(
        layout: c_int, uplo: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        ap: *mut Self, ipiv: *mut MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*hptrf` — Hermitian packed factorization.
    unsafe fn lapacke_hptrf(
        layout: c_int, uplo: c_char,
        n: MKL_INT, ap: *mut Self, ipiv: *mut MKL_INT,
    ) -> i32;

    /// `LAPACKE_*hptrs` — solve with Hermitian packed factor.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_hptrs(
        layout: c_int, uplo: c_char,
        n: MKL_INT, nrhs: MKL_INT,
        ap: *const Self, ipiv: *const MKL_INT,
        b: *mut Self, ldb: MKL_INT,
    ) -> i32;

    /// `LAPACKE_*hegv` — generalized Hermitian-definite eigenproblem
    /// (`itype` semantics same as `sygv`). Eigenvalues `w` are real.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_hegv(
        layout: c_int, itype: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
        w: *mut Self::Real,
    ) -> i32;

    /// `LAPACKE_*ggev` (complex) — generalized non-symmetric
    /// eigenproblem. Eigenvalues are returned as `alpha / beta`,
    /// each complex.
    #[allow(clippy::too_many_arguments)]
    unsafe fn lapacke_ggev_complex(
        layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
        a: *mut Self, lda: MKL_INT,
        b: *mut Self, ldb: MKL_INT,
        alpha: *mut Self, beta: *mut Self,
        vl: *mut Self, ldvl: MKL_INT,
        vr: *mut Self, ldvr: MKL_INT,
    ) -> i32;
}

// =====================================================================
// Implementations
// =====================================================================

macro_rules! impl_lapack_real {
    ($ty:ty,
        gesv=$gesv:ident, getrf=$getrf:ident, getrs=$getrs:ident, getri=$getri:ident,
        posv=$posv:ident, potrf=$potrf:ident, potrs=$potrs:ident,
        geqrf=$geqrf:ident, gelqf=$gelqf:ident, orgqr=$orgqr:ident,
        gels=$gels:ident, gelsd=$gelsd:ident,
        gesdd=$gesdd:ident, gesvd=$gesvd:ident,
        sysv=$sysv:ident, syev=$syev:ident, geev=$geev:ident,
        gbsv=$gbsv:ident, gbtrf=$gbtrf:ident, gbtrs=$gbtrs:ident,
        gtsv=$gtsv:ident, gttrf=$gttrf:ident, gttrs=$gttrs:ident,
        pbsv=$pbsv:ident, pbtrf=$pbtrf:ident, pbtrs=$pbtrs:ident,
        ptsv=$ptsv:ident, pttrf=$pttrf:ident, pttrs=$pttrs:ident,
        spsv=$spsv:ident, sptrf=$sptrf:ident, sptrs=$sptrs:ident,
        ppsv=$ppsv:ident, pptrf=$pptrf:ident, pptrs=$pptrs:ident,
        sygv=$sygv:ident, ggev_real=$ggev_real:ident,
    ) => {
        impl LapackScalar for $ty {
            unsafe fn lapacke_gesv(
                layout: c_int, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$gesv(layout, n, nrhs, a, lda, ipiv, b, ldb) }
            }
            unsafe fn lapacke_getrf(
                layout: c_int, m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$getrf(layout, m, n, a, lda, ipiv) }
            }
            unsafe fn lapacke_getrs(
                layout: c_int, trans: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *const Self, lda: MKL_INT, ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$getrs(layout, trans, n, nrhs, a, lda, ipiv, b, ldb) }
            }
            unsafe fn lapacke_getri(
                layout: c_int, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *const MKL_INT,
            ) -> i32 {
                unsafe { sys::$getri(layout, n, a, lda, ipiv) }
            }
            unsafe fn lapacke_posv(
                layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT, b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$posv(layout, uplo, n, nrhs, a, lda, b, ldb) }
            }
            unsafe fn lapacke_potrf(
                layout: c_int, uplo: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) -> i32 {
                unsafe { sys::$potrf(layout, uplo, n, a, lda) }
            }
            unsafe fn lapacke_potrs(
                layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *const Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$potrs(layout, uplo, n, nrhs, a, lda, b, ldb) }
            }
            unsafe fn lapacke_geqrf(
                layout: c_int, m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, tau: *mut Self,
            ) -> i32 {
                unsafe { sys::$geqrf(layout, m, n, a, lda, tau) }
            }
            unsafe fn lapacke_gelqf(
                layout: c_int, m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, tau: *mut Self,
            ) -> i32 {
                unsafe { sys::$gelqf(layout, m, n, a, lda, tau) }
            }
            unsafe fn lapacke_orgqr(
                layout: c_int, m: MKL_INT, n: MKL_INT, k: MKL_INT,
                a: *mut Self, lda: MKL_INT, tau: *const Self,
            ) -> i32 {
                unsafe { sys::$orgqr(layout, m, n, k, a, lda, tau) }
            }
            unsafe fn lapacke_gels(
                layout: c_int, trans: c_char,
                m: MKL_INT, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$gels(layout, trans, m, n, nrhs, a, lda, b, ldb) }
            }
            unsafe fn lapacke_gelsd(
                layout: c_int,
                m: MKL_INT, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
                s: *mut Self::Real, rcond: Self::Real,
                rank: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$gelsd(layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank) }
            }
            unsafe fn lapacke_gesdd(
                layout: c_int, jobz: c_char,
                m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                s: *mut Self::Real,
                u: *mut Self, ldu: MKL_INT,
                vt: *mut Self, ldvt: MKL_INT,
            ) -> i32 {
                unsafe { sys::$gesdd(layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt) }
            }
            unsafe fn lapacke_gesvd(
                layout: c_int, jobu: c_char, jobvt: c_char,
                m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                s: *mut Self::Real,
                u: *mut Self, ldu: MKL_INT,
                vt: *mut Self, ldvt: MKL_INT,
                superb: *mut Self::Real,
            ) -> i32 {
                unsafe {
                    sys::$gesvd(
                        layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb,
                    )
                }
            }
            unsafe fn lapacke_gbsv(
                layout: c_int, n: MKL_INT, kl: MKL_INT, ku: MKL_INT, nrhs: MKL_INT,
                ab: *mut Self, ldab: MKL_INT, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$gbsv(layout, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb) }
            }
            unsafe fn lapacke_gbtrf(
                layout: c_int, m: MKL_INT, n: MKL_INT, kl: MKL_INT, ku: MKL_INT,
                ab: *mut Self, ldab: MKL_INT, ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$gbtrf(layout, m, n, kl, ku, ab, ldab, ipiv) }
            }
            unsafe fn lapacke_gbtrs(
                layout: c_int, trans: c_char,
                n: MKL_INT, kl: MKL_INT, ku: MKL_INT, nrhs: MKL_INT,
                ab: *const Self, ldab: MKL_INT, ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gbtrs(layout, trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb)
                }
            }
            unsafe fn lapacke_gtsv(
                layout: c_int, n: MKL_INT, nrhs: MKL_INT,
                dl: *mut Self, d: *mut Self, du: *mut Self,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$gtsv(layout, n, nrhs, dl, d, du, b, ldb) }
            }
            unsafe fn lapacke_gttrf(
                n: MKL_INT,
                dl: *mut Self, d: *mut Self, du: *mut Self, du2: *mut Self,
                ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$gttrf(n, dl, d, du, du2, ipiv) }
            }
            unsafe fn lapacke_gttrs(
                layout: c_int, trans: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                dl: *const Self, d: *const Self, du: *const Self, du2: *const Self,
                ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gttrs(layout, trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb)
                }
            }
            unsafe fn lapacke_pbsv(
                layout: c_int, uplo: c_char,
                n: MKL_INT, kd: MKL_INT, nrhs: MKL_INT,
                ab: *mut Self, ldab: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$pbsv(layout, uplo, n, kd, nrhs, ab, ldab, b, ldb) }
            }
            unsafe fn lapacke_pbtrf(
                layout: c_int, uplo: c_char,
                n: MKL_INT, kd: MKL_INT,
                ab: *mut Self, ldab: MKL_INT,
            ) -> i32 {
                unsafe { sys::$pbtrf(layout, uplo, n, kd, ab, ldab) }
            }
            unsafe fn lapacke_pbtrs(
                layout: c_int, uplo: c_char,
                n: MKL_INT, kd: MKL_INT, nrhs: MKL_INT,
                ab: *const Self, ldab: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$pbtrs(layout, uplo, n, kd, nrhs, ab, ldab, b, ldb) }
            }
            unsafe fn lapacke_ptsv(
                layout: c_int, n: MKL_INT, nrhs: MKL_INT,
                d: *mut Self::Real, e: *mut Self,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$ptsv(layout, n, nrhs, d, e, b, ldb) }
            }
            unsafe fn lapacke_pttrf(
                n: MKL_INT,
                d: *mut Self::Real, e: *mut Self,
            ) -> i32 {
                unsafe { sys::$pttrf(n, d, e) }
            }
            unsafe fn lapacke_spsv(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *mut Self, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$spsv(layout, uplo, n, nrhs, ap, ipiv, b, ldb) }
            }
            unsafe fn lapacke_sptrf(
                layout: c_int, uplo: c_char,
                n: MKL_INT, ap: *mut Self, ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$sptrf(layout, uplo, n, ap, ipiv) }
            }
            unsafe fn lapacke_sptrs(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *const Self, ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$sptrs(layout, uplo, n, nrhs, ap, ipiv, b, ldb) }
            }
            unsafe fn lapacke_ppsv(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *mut Self, b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$ppsv(layout, uplo, n, nrhs, ap, b, ldb) }
            }
            unsafe fn lapacke_pptrf(
                layout: c_int, uplo: c_char,
                n: MKL_INT, ap: *mut Self,
            ) -> i32 {
                unsafe { sys::$pptrf(layout, uplo, n, ap) }
            }
            unsafe fn lapacke_pptrs(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *const Self, b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$pptrs(layout, uplo, n, nrhs, ap, b, ldb) }
            }
        }

        impl RealLapackScalar for $ty {
            unsafe fn lapacke_pttrs(
                layout: c_int, n: MKL_INT, nrhs: MKL_INT,
                d: *const Self, e: *const Self,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$pttrs(layout, n, nrhs, d, e, b, ldb) }
            }
            unsafe fn lapacke_sysv(
                layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$sysv(layout, uplo, n, nrhs, a, lda, ipiv, b, ldb) }
            }
            unsafe fn lapacke_syev(
                layout: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, w: *mut Self,
            ) -> i32 {
                unsafe { sys::$syev(layout, jobz, uplo, n, a, lda, w) }
            }
            unsafe fn lapacke_geev(
                layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                wr: *mut Self, wi: *mut Self,
                vl: *mut Self, ldvl: MKL_INT,
                vr: *mut Self, ldvr: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$geev(
                        layout, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr,
                    )
                }
            }
            unsafe fn lapacke_sygv(
                layout: c_int, itype: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
                w: *mut Self,
            ) -> i32 {
                unsafe { sys::$sygv(layout, itype, jobz, uplo, n, a, lda, b, ldb, w) }
            }
            unsafe fn lapacke_ggev(
                layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
                alphar: *mut Self, alphai: *mut Self, beta: *mut Self,
                vl: *mut Self, ldvl: MKL_INT,
                vr: *mut Self, ldvr: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$ggev_real(
                        layout, jobvl, jobvr, n, a, lda, b, ldb,
                        alphar, alphai, beta, vl, ldvl, vr, ldvr,
                    )
                }
            }
        }
    };
}

impl_lapack_real! {
    f32,
    gesv=LAPACKE_sgesv, getrf=LAPACKE_sgetrf, getrs=LAPACKE_sgetrs, getri=LAPACKE_sgetri,
    posv=LAPACKE_sposv, potrf=LAPACKE_spotrf, potrs=LAPACKE_spotrs,
    geqrf=LAPACKE_sgeqrf, gelqf=LAPACKE_sgelqf, orgqr=LAPACKE_sorgqr,
    gels=LAPACKE_sgels, gelsd=LAPACKE_sgelsd,
    gesdd=LAPACKE_sgesdd, gesvd=LAPACKE_sgesvd,
    sysv=LAPACKE_ssysv, syev=LAPACKE_ssyev, geev=LAPACKE_sgeev,
    gbsv=LAPACKE_sgbsv, gbtrf=LAPACKE_sgbtrf, gbtrs=LAPACKE_sgbtrs,
    gtsv=LAPACKE_sgtsv, gttrf=LAPACKE_sgttrf, gttrs=LAPACKE_sgttrs,
    pbsv=LAPACKE_spbsv, pbtrf=LAPACKE_spbtrf, pbtrs=LAPACKE_spbtrs,
    ptsv=LAPACKE_sptsv, pttrf=LAPACKE_spttrf, pttrs=LAPACKE_spttrs,
    spsv=LAPACKE_sspsv, sptrf=LAPACKE_ssptrf, sptrs=LAPACKE_ssptrs,
    ppsv=LAPACKE_sppsv, pptrf=LAPACKE_spptrf, pptrs=LAPACKE_spptrs,
    sygv=LAPACKE_ssygv, ggev_real=LAPACKE_sggev,
}

impl_lapack_real! {
    f64,
    gesv=LAPACKE_dgesv, getrf=LAPACKE_dgetrf, getrs=LAPACKE_dgetrs, getri=LAPACKE_dgetri,
    posv=LAPACKE_dposv, potrf=LAPACKE_dpotrf, potrs=LAPACKE_dpotrs,
    geqrf=LAPACKE_dgeqrf, gelqf=LAPACKE_dgelqf, orgqr=LAPACKE_dorgqr,
    gels=LAPACKE_dgels, gelsd=LAPACKE_dgelsd,
    gesdd=LAPACKE_dgesdd, gesvd=LAPACKE_dgesvd,
    sysv=LAPACKE_dsysv, syev=LAPACKE_dsyev, geev=LAPACKE_dgeev,
    gbsv=LAPACKE_dgbsv, gbtrf=LAPACKE_dgbtrf, gbtrs=LAPACKE_dgbtrs,
    gtsv=LAPACKE_dgtsv, gttrf=LAPACKE_dgttrf, gttrs=LAPACKE_dgttrs,
    pbsv=LAPACKE_dpbsv, pbtrf=LAPACKE_dpbtrf, pbtrs=LAPACKE_dpbtrs,
    ptsv=LAPACKE_dptsv, pttrf=LAPACKE_dpttrf, pttrs=LAPACKE_dpttrs,
    spsv=LAPACKE_dspsv, sptrf=LAPACKE_dsptrf, sptrs=LAPACKE_dsptrs,
    ppsv=LAPACKE_dppsv, pptrf=LAPACKE_dpptrf, pptrs=LAPACKE_dpptrs,
    sygv=LAPACKE_dsygv, ggev_real=LAPACKE_dggev,
}

macro_rules! impl_lapack_complex {
    ($ty:ty, $real:ty,
        gesv=$gesv:ident, getrf=$getrf:ident, getrs=$getrs:ident, getri=$getri:ident,
        posv=$posv:ident, potrf=$potrf:ident, potrs=$potrs:ident,
        geqrf=$geqrf:ident, gelqf=$gelqf:ident, ungqr=$ungqr:ident,
        gels=$gels:ident, gelsd=$gelsd:ident,
        gesdd=$gesdd:ident, gesvd=$gesvd:ident,
        hesv=$hesv:ident, heev=$heev:ident, geev=$geev:ident,
        gbsv=$gbsv:ident, gbtrf=$gbtrf:ident, gbtrs=$gbtrs:ident,
        gtsv=$gtsv:ident, gttrf=$gttrf:ident, gttrs=$gttrs:ident,
        pbsv=$pbsv:ident, pbtrf=$pbtrf:ident, pbtrs=$pbtrs:ident,
        ptsv=$ptsv:ident, pttrf=$pttrf:ident, pttrs=$pttrs:ident,
        spsv=$spsv:ident, sptrf=$sptrf:ident, sptrs=$sptrs:ident,
        ppsv=$ppsv:ident, pptrf=$pptrf:ident, pptrs=$pptrs:ident,
        hpsv=$hpsv:ident, hptrf=$hptrf:ident, hptrs=$hptrs:ident,
        hegv=$hegv:ident, ggev_complex=$ggev_complex:ident,
    ) => {
        impl LapackScalar for $ty {
            unsafe fn lapacke_gesv(
                layout: c_int, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$gesv(layout, n, nrhs, a.cast(), lda, ipiv, b.cast(), ldb) }
            }
            unsafe fn lapacke_getrf(
                layout: c_int, m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$getrf(layout, m, n, a.cast(), lda, ipiv) }
            }
            unsafe fn lapacke_getrs(
                layout: c_int, trans: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *const Self, lda: MKL_INT, ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$getrs(layout, trans, n, nrhs, a.cast(), lda, ipiv, b.cast(), ldb) }
            }
            unsafe fn lapacke_getri(
                layout: c_int, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *const MKL_INT,
            ) -> i32 {
                unsafe { sys::$getri(layout, n, a.cast(), lda, ipiv) }
            }
            unsafe fn lapacke_posv(
                layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT, b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$posv(layout, uplo, n, nrhs, a.cast(), lda, b.cast(), ldb) }
            }
            unsafe fn lapacke_potrf(
                layout: c_int, uplo: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
            ) -> i32 {
                unsafe { sys::$potrf(layout, uplo, n, a.cast(), lda) }
            }
            unsafe fn lapacke_potrs(
                layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *const Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$potrs(layout, uplo, n, nrhs, a.cast(), lda, b.cast(), ldb) }
            }
            unsafe fn lapacke_geqrf(
                layout: c_int, m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, tau: *mut Self,
            ) -> i32 {
                unsafe { sys::$geqrf(layout, m, n, a.cast(), lda, tau.cast()) }
            }
            unsafe fn lapacke_gelqf(
                layout: c_int, m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, tau: *mut Self,
            ) -> i32 {
                unsafe { sys::$gelqf(layout, m, n, a.cast(), lda, tau.cast()) }
            }
            unsafe fn lapacke_orgqr(
                layout: c_int, m: MKL_INT, n: MKL_INT, k: MKL_INT,
                a: *mut Self, lda: MKL_INT, tau: *const Self,
            ) -> i32 {
                unsafe { sys::$ungqr(layout, m, n, k, a.cast(), lda, tau.cast()) }
            }
            unsafe fn lapacke_gels(
                layout: c_int, trans: c_char,
                m: MKL_INT, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$gels(layout, trans, m, n, nrhs, a.cast(), lda, b.cast(), ldb) }
            }
            unsafe fn lapacke_gelsd(
                layout: c_int,
                m: MKL_INT, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
                s: *mut Self::Real, rcond: Self::Real,
                rank: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$gelsd(layout, m, n, nrhs, a.cast(), lda, b.cast(), ldb, s, rcond, rank) }
            }
            unsafe fn lapacke_gesdd(
                layout: c_int, jobz: c_char,
                m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                s: *mut Self::Real,
                u: *mut Self, ldu: MKL_INT,
                vt: *mut Self, ldvt: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gesdd(
                        layout, jobz, m, n, a.cast(), lda, s, u.cast(), ldu, vt.cast(), ldvt,
                    )
                }
            }
            unsafe fn lapacke_gesvd(
                layout: c_int, jobu: c_char, jobvt: c_char,
                m: MKL_INT, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                s: *mut Self::Real,
                u: *mut Self, ldu: MKL_INT,
                vt: *mut Self, ldvt: MKL_INT,
                superb: *mut Self::Real,
            ) -> i32 {
                unsafe {
                    sys::$gesvd(
                        layout, jobu, jobvt, m, n, a.cast(), lda, s,
                        u.cast(), ldu, vt.cast(), ldvt, superb,
                    )
                }
            }
            unsafe fn lapacke_gbsv(
                layout: c_int, n: MKL_INT, kl: MKL_INT, ku: MKL_INT, nrhs: MKL_INT,
                ab: *mut Self, ldab: MKL_INT, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gbsv(
                        layout, n, kl, ku, nrhs, ab.cast(), ldab, ipiv, b.cast(), ldb,
                    )
                }
            }
            unsafe fn lapacke_gbtrf(
                layout: c_int, m: MKL_INT, n: MKL_INT, kl: MKL_INT, ku: MKL_INT,
                ab: *mut Self, ldab: MKL_INT, ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$gbtrf(layout, m, n, kl, ku, ab.cast(), ldab, ipiv) }
            }
            unsafe fn lapacke_gbtrs(
                layout: c_int, trans: c_char,
                n: MKL_INT, kl: MKL_INT, ku: MKL_INT, nrhs: MKL_INT,
                ab: *const Self, ldab: MKL_INT, ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gbtrs(
                        layout, trans, n, kl, ku, nrhs, ab.cast(), ldab, ipiv, b.cast(), ldb,
                    )
                }
            }
            unsafe fn lapacke_gtsv(
                layout: c_int, n: MKL_INT, nrhs: MKL_INT,
                dl: *mut Self, d: *mut Self, du: *mut Self,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gtsv(layout, n, nrhs, dl.cast(), d.cast(), du.cast(), b.cast(), ldb)
                }
            }
            unsafe fn lapacke_gttrf(
                n: MKL_INT,
                dl: *mut Self, d: *mut Self, du: *mut Self, du2: *mut Self,
                ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gttrf(n, dl.cast(), d.cast(), du.cast(), du2.cast(), ipiv)
                }
            }
            unsafe fn lapacke_gttrs(
                layout: c_int, trans: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                dl: *const Self, d: *const Self, du: *const Self, du2: *const Self,
                ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$gttrs(
                        layout, trans, n, nrhs,
                        dl.cast(), d.cast(), du.cast(), du2.cast(), ipiv,
                        b.cast(), ldb,
                    )
                }
            }
            unsafe fn lapacke_pbsv(
                layout: c_int, uplo: c_char,
                n: MKL_INT, kd: MKL_INT, nrhs: MKL_INT,
                ab: *mut Self, ldab: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$pbsv(layout, uplo, n, kd, nrhs, ab.cast(), ldab, b.cast(), ldb)
                }
            }
            unsafe fn lapacke_pbtrf(
                layout: c_int, uplo: c_char,
                n: MKL_INT, kd: MKL_INT,
                ab: *mut Self, ldab: MKL_INT,
            ) -> i32 {
                unsafe { sys::$pbtrf(layout, uplo, n, kd, ab.cast(), ldab) }
            }
            unsafe fn lapacke_pbtrs(
                layout: c_int, uplo: c_char,
                n: MKL_INT, kd: MKL_INT, nrhs: MKL_INT,
                ab: *const Self, ldab: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$pbtrs(layout, uplo, n, kd, nrhs, ab.cast(), ldab, b.cast(), ldb)
                }
            }
            unsafe fn lapacke_ptsv(
                layout: c_int, n: MKL_INT, nrhs: MKL_INT,
                d: *mut Self::Real, e: *mut Self,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$ptsv(layout, n, nrhs, d, e.cast(), b.cast(), ldb) }
            }
            unsafe fn lapacke_pttrf(
                n: MKL_INT,
                d: *mut Self::Real, e: *mut Self,
            ) -> i32 {
                unsafe { sys::$pttrf(n, d, e.cast()) }
            }
            unsafe fn lapacke_spsv(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *mut Self, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$spsv(layout, uplo, n, nrhs, ap.cast(), ipiv, b.cast(), ldb)
                }
            }
            unsafe fn lapacke_sptrf(
                layout: c_int, uplo: c_char,
                n: MKL_INT, ap: *mut Self, ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$sptrf(layout, uplo, n, ap.cast(), ipiv) }
            }
            unsafe fn lapacke_sptrs(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *const Self, ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$sptrs(layout, uplo, n, nrhs, ap.cast(), ipiv, b.cast(), ldb)
                }
            }
            unsafe fn lapacke_ppsv(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *mut Self, b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$ppsv(layout, uplo, n, nrhs, ap.cast(), b.cast(), ldb)
                }
            }
            unsafe fn lapacke_pptrf(
                layout: c_int, uplo: c_char,
                n: MKL_INT, ap: *mut Self,
            ) -> i32 {
                unsafe { sys::$pptrf(layout, uplo, n, ap.cast()) }
            }
            unsafe fn lapacke_pptrs(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *const Self, b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$pptrs(layout, uplo, n, nrhs, ap.cast(), b.cast(), ldb)
                }
            }
        }

        impl ComplexLapackScalar for $ty {
            unsafe fn lapacke_pttrs_complex(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                d: *const Self::Real, e: *const Self,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$pttrs(layout, uplo, n, nrhs, d, e.cast(), b.cast(), ldb)
                }
            }
            unsafe fn lapacke_hesv(
                layout: c_int, uplo: c_char, n: MKL_INT, nrhs: MKL_INT,
                a: *mut Self, lda: MKL_INT, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe { sys::$hesv(layout, uplo, n, nrhs, a.cast(), lda, ipiv, b.cast(), ldb) }
            }
            unsafe fn lapacke_heev(
                layout: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT, w: *mut $real,
            ) -> i32 {
                unsafe { sys::$heev(layout, jobz, uplo, n, a.cast(), lda, w) }
            }
            unsafe fn lapacke_geev_complex(
                layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                w: *mut Self,
                vl: *mut Self, ldvl: MKL_INT,
                vr: *mut Self, ldvr: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$geev(
                        layout, jobvl, jobvr, n, a.cast(), lda, w.cast(),
                        vl.cast(), ldvl, vr.cast(), ldvr,
                    )
                }
            }
            unsafe fn lapacke_hpsv(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *mut Self, ipiv: *mut MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$hpsv(layout, uplo, n, nrhs, ap.cast(), ipiv, b.cast(), ldb)
                }
            }
            unsafe fn lapacke_hptrf(
                layout: c_int, uplo: c_char,
                n: MKL_INT, ap: *mut Self, ipiv: *mut MKL_INT,
            ) -> i32 {
                unsafe { sys::$hptrf(layout, uplo, n, ap.cast(), ipiv) }
            }
            unsafe fn lapacke_hptrs(
                layout: c_int, uplo: c_char,
                n: MKL_INT, nrhs: MKL_INT,
                ap: *const Self, ipiv: *const MKL_INT,
                b: *mut Self, ldb: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$hptrs(layout, uplo, n, nrhs, ap.cast(), ipiv, b.cast(), ldb)
                }
            }
            unsafe fn lapacke_hegv(
                layout: c_int, itype: c_int, jobz: c_char, uplo: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
                w: *mut Self::Real,
            ) -> i32 {
                unsafe {
                    sys::$hegv(
                        layout, itype, jobz, uplo, n, a.cast(), lda, b.cast(), ldb, w,
                    )
                }
            }
            unsafe fn lapacke_ggev_complex(
                layout: c_int, jobvl: c_char, jobvr: c_char, n: MKL_INT,
                a: *mut Self, lda: MKL_INT,
                b: *mut Self, ldb: MKL_INT,
                alpha: *mut Self, beta: *mut Self,
                vl: *mut Self, ldvl: MKL_INT,
                vr: *mut Self, ldvr: MKL_INT,
            ) -> i32 {
                unsafe {
                    sys::$ggev_complex(
                        layout, jobvl, jobvr, n, a.cast(), lda, b.cast(), ldb,
                        alpha.cast(), beta.cast(),
                        vl.cast(), ldvl, vr.cast(), ldvr,
                    )
                }
            }
        }
    };
}

impl_lapack_complex! {
    Complex32, f32,
    gesv=LAPACKE_cgesv, getrf=LAPACKE_cgetrf, getrs=LAPACKE_cgetrs, getri=LAPACKE_cgetri,
    posv=LAPACKE_cposv, potrf=LAPACKE_cpotrf, potrs=LAPACKE_cpotrs,
    geqrf=LAPACKE_cgeqrf, gelqf=LAPACKE_cgelqf, ungqr=LAPACKE_cungqr,
    gels=LAPACKE_cgels, gelsd=LAPACKE_cgelsd,
    gesdd=LAPACKE_cgesdd, gesvd=LAPACKE_cgesvd,
    hesv=LAPACKE_chesv, heev=LAPACKE_cheev, geev=LAPACKE_cgeev,
    gbsv=LAPACKE_cgbsv, gbtrf=LAPACKE_cgbtrf, gbtrs=LAPACKE_cgbtrs,
    gtsv=LAPACKE_cgtsv, gttrf=LAPACKE_cgttrf, gttrs=LAPACKE_cgttrs,
    pbsv=LAPACKE_cpbsv, pbtrf=LAPACKE_cpbtrf, pbtrs=LAPACKE_cpbtrs,
    ptsv=LAPACKE_cptsv, pttrf=LAPACKE_cpttrf, pttrs=LAPACKE_cpttrs,
    spsv=LAPACKE_cspsv, sptrf=LAPACKE_csptrf, sptrs=LAPACKE_csptrs,
    ppsv=LAPACKE_cppsv, pptrf=LAPACKE_cpptrf, pptrs=LAPACKE_cpptrs,
    hpsv=LAPACKE_chpsv, hptrf=LAPACKE_chptrf, hptrs=LAPACKE_chptrs,
    hegv=LAPACKE_chegv, ggev_complex=LAPACKE_cggev,
}

impl_lapack_complex! {
    Complex64, f64,
    gesv=LAPACKE_zgesv, getrf=LAPACKE_zgetrf, getrs=LAPACKE_zgetrs, getri=LAPACKE_zgetri,
    posv=LAPACKE_zposv, potrf=LAPACKE_zpotrf, potrs=LAPACKE_zpotrs,
    geqrf=LAPACKE_zgeqrf, gelqf=LAPACKE_zgelqf, ungqr=LAPACKE_zungqr,
    gels=LAPACKE_zgels, gelsd=LAPACKE_zgelsd,
    gesdd=LAPACKE_zgesdd, gesvd=LAPACKE_zgesvd,
    hesv=LAPACKE_zhesv, heev=LAPACKE_zheev, geev=LAPACKE_zgeev,
    gbsv=LAPACKE_zgbsv, gbtrf=LAPACKE_zgbtrf, gbtrs=LAPACKE_zgbtrs,
    gtsv=LAPACKE_zgtsv, gttrf=LAPACKE_zgttrf, gttrs=LAPACKE_zgttrs,
    pbsv=LAPACKE_zpbsv, pbtrf=LAPACKE_zpbtrf, pbtrs=LAPACKE_zpbtrs,
    ptsv=LAPACKE_zptsv, pttrf=LAPACKE_zpttrf, pttrs=LAPACKE_zpttrs,
    spsv=LAPACKE_zspsv, sptrf=LAPACKE_zsptrf, sptrs=LAPACKE_zsptrs,
    ppsv=LAPACKE_zppsv, pptrf=LAPACKE_zpptrf, pptrs=LAPACKE_zpptrs,
    hpsv=LAPACKE_zhpsv, hptrf=LAPACKE_zhptrf, hptrs=LAPACKE_zhptrs,
    hegv=LAPACKE_zhegv, ggev_complex=LAPACKE_zggev,
}
