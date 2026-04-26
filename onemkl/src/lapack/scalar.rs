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
        }

        impl RealLapackScalar for $ty {
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
}

impl_lapack_real! {
    f64,
    gesv=LAPACKE_dgesv, getrf=LAPACKE_dgetrf, getrs=LAPACKE_dgetrs, getri=LAPACKE_dgetri,
    posv=LAPACKE_dposv, potrf=LAPACKE_dpotrf, potrs=LAPACKE_dpotrs,
    geqrf=LAPACKE_dgeqrf, gelqf=LAPACKE_dgelqf, orgqr=LAPACKE_dorgqr,
    gels=LAPACKE_dgels, gelsd=LAPACKE_dgelsd,
    gesdd=LAPACKE_dgesdd, gesvd=LAPACKE_dgesvd,
    sysv=LAPACKE_dsysv, syev=LAPACKE_dsyev, geev=LAPACKE_dgeev,
}

macro_rules! impl_lapack_complex {
    ($ty:ty, $real:ty,
        gesv=$gesv:ident, getrf=$getrf:ident, getrs=$getrs:ident, getri=$getri:ident,
        posv=$posv:ident, potrf=$potrf:ident, potrs=$potrs:ident,
        geqrf=$geqrf:ident, gelqf=$gelqf:ident, ungqr=$ungqr:ident,
        gels=$gels:ident, gelsd=$gelsd:ident,
        gesdd=$gesdd:ident, gesvd=$gesvd:ident,
        hesv=$hesv:ident, heev=$heev:ident, geev=$geev:ident,
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
        }

        impl ComplexLapackScalar for $ty {
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
}

impl_lapack_complex! {
    Complex64, f64,
    gesv=LAPACKE_zgesv, getrf=LAPACKE_zgetrf, getrs=LAPACKE_zgetrs, getri=LAPACKE_zgetri,
    posv=LAPACKE_zposv, potrf=LAPACKE_zpotrf, potrs=LAPACKE_zpotrs,
    geqrf=LAPACKE_zgeqrf, gelqf=LAPACKE_zgelqf, ungqr=LAPACKE_zungqr,
    gels=LAPACKE_zgels, gelsd=LAPACKE_zgelsd,
    gesdd=LAPACKE_zgesdd, gesvd=LAPACKE_zgesvd,
    hesv=LAPACKE_zhesv, heev=LAPACKE_zheev, geev=LAPACKE_zgeev,
}
