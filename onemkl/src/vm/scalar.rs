//! Scalar dispatch traits for the Vector Math (VM) module.
//!
//! Each trait method is `unsafe` and forwards directly to the matching
//! `v[smcdz]*` C function. Argument validation is done at the public
//! free-function level in [`super`].

use num_complex::{Complex32, Complex64};
use onemkl_sys::{self as sys, MKL_INT};

use crate::scalar::{ComplexScalar, RealScalar, Scalar};

// =====================================================================
// VmScalar — universal element-wise operations.
// =====================================================================

/// Element-wise math operations supported across all four scalar types.
pub trait VmScalar: Scalar {
    // ---- Arithmetic ----
    /// `r[i] = a[i] + b[i]`
    unsafe fn vm_add(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    /// `r[i] = a[i] - b[i]`
    unsafe fn vm_sub(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    /// `r[i] = a[i] * b[i]`
    unsafe fn vm_mul(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    /// `r[i] = a[i] / b[i]`
    unsafe fn vm_div(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);

    // ---- Power & root ----
    /// `r[i] = sqrt(a[i])`
    unsafe fn vm_sqrt(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = a[i] ^ b[i]`
    unsafe fn vm_pow(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    /// `r[i] = a[i] ^ b` (`b` a scalar)
    unsafe fn vm_powx(n: MKL_INT, a: *const Self, b: Self, r: *mut Self);

    // ---- Exp / log ----
    /// `r[i] = exp(a[i])`
    unsafe fn vm_exp(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = ln(a[i])`
    unsafe fn vm_ln(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = log10(a[i])`
    unsafe fn vm_log10(n: MKL_INT, a: *const Self, r: *mut Self);

    // ---- Trigonometric ----
    /// `r[i] = cos(a[i])`
    unsafe fn vm_cos(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = sin(a[i])`
    unsafe fn vm_sin(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = tan(a[i])`
    unsafe fn vm_tan(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = acos(a[i])`
    unsafe fn vm_acos(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = asin(a[i])`
    unsafe fn vm_asin(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = atan(a[i])`
    unsafe fn vm_atan(n: MKL_INT, a: *const Self, r: *mut Self);

    // ---- Hyperbolic ----
    /// `r[i] = cosh(a[i])`
    unsafe fn vm_cosh(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = sinh(a[i])`
    unsafe fn vm_sinh(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = tanh(a[i])`
    unsafe fn vm_tanh(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = acosh(a[i])`
    unsafe fn vm_acosh(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = asinh(a[i])`
    unsafe fn vm_asinh(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = atanh(a[i])`
    unsafe fn vm_atanh(n: MKL_INT, a: *const Self, r: *mut Self);
}

// =====================================================================
// RealVmScalar — real-only operations.
// =====================================================================

/// Real-only element-wise operations.
#[allow(missing_docs)] // Each method is documented in the corresponding free function
pub trait RealVmScalar: VmScalar + RealScalar {
    // Arithmetic / abs / sqr / inv
    unsafe fn vm_abs_real(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_sqr(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_inv(n: MKL_INT, a: *const Self, r: *mut Self);
    #[allow(clippy::too_many_arguments)]
    unsafe fn vm_linear_frac(
        n: MKL_INT, a: *const Self, b: *const Self,
        scale_a: Self, shift_a: Self, scale_b: Self, shift_b: Self,
        r: *mut Self,
    );

    // Power & root (real-only because complex has no Sqr/Inv/Cbrt/InvCbrt/InvSqrt)
    unsafe fn vm_inv_sqrt(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_cbrt(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_inv_cbrt(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_powr(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_pow2o3(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_pow3o2(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_hypot(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);

    // Exp / log
    unsafe fn vm_exp2(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_exp10(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_expm1(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_log2(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_log1p(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_logb(n: MKL_INT, a: *const Self, r: *mut Self);

    // Trig (degree, π-scaled, multi-output)
    unsafe fn vm_cosd(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_sind(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_tand(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_cospi(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_sinpi(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_tanpi(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_sincos(n: MKL_INT, a: *const Self, s: *mut Self, c: *mut Self);
    unsafe fn vm_sincospi(n: MKL_INT, a: *const Self, s: *mut Self, c: *mut Self);

    // Inverse trig
    unsafe fn vm_atan2(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_acospi(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_asinpi(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_atanpi(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_atan2pi(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);

    // Special
    unsafe fn vm_erf(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_erfc(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_erf_inv(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_erfc_inv(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_erfcx(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_cdf_norm(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_cdf_norm_inv(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_lgamma(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_tgamma(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_exp_int1(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_i0(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_i1(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_j0(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_j1(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_y0(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_y1(n: MKL_INT, a: *const Self, r: *mut Self);

    // Rounding
    unsafe fn vm_floor(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_ceil(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_trunc(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_round(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_nearby_int(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_rint(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_frac(n: MKL_INT, a: *const Self, r: *mut Self);
    unsafe fn vm_modf(n: MKL_INT, a: *const Self, ip: *mut Self, fp: *mut Self);

    // Misc binary
    unsafe fn vm_fmod(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_remainder(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_copysign(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_next_after(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_fdim(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_fmax(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_fmin(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_max_mag(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
    unsafe fn vm_min_mag(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self);
}

// =====================================================================
// ComplexVmScalar — complex-only operations.
// =====================================================================

/// Complex-only element-wise operations, including type-changing ones
/// (real → complex and complex → real).
#[allow(missing_docs)]
pub trait ComplexVmScalar: VmScalar + ComplexScalar {
    /// `r[i] = |a[i]|`. Output is real.
    unsafe fn vm_abs_complex(n: MKL_INT, a: *const Self, r: *mut Self::Real);
    /// `r[i] = arg(a[i])`. Output is real (in radians).
    unsafe fn vm_arg(n: MKL_INT, a: *const Self, r: *mut Self::Real);
    /// `r[i] = conj(a[i])`. Output is complex.
    unsafe fn vm_conj(n: MKL_INT, a: *const Self, r: *mut Self);
    /// `r[i] = a[i] * conj(b[i])`.
    unsafe fn vm_mul_by_conj(
        n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self,
    );
    /// `r[i] = exp(i * a[i])`. Input is real, output complex.
    unsafe fn vm_cis(n: MKL_INT, a: *const Self::Real, r: *mut Self);
}

// =====================================================================
// Implementations
// =====================================================================

macro_rules! impl_vm_scalar_real {
    ($ty:ty,
        add=$add:ident, sub=$sub:ident, mul=$mul:ident, div=$div:ident,
        sqrt=$sqrt:ident,
        pow=$pow:ident, powx=$powx:ident,
        exp=$exp:ident, ln=$ln:ident, log10=$log10:ident,
        cos=$cos:ident, sin=$sin:ident, tan=$tan:ident,
        acos=$acos:ident, asin=$asin:ident, atan=$atan:ident,
        cosh=$cosh:ident, sinh=$sinh:ident, tanh=$tanh:ident,
        acosh=$acosh:ident, asinh=$asinh:ident, atanh=$atanh:ident,
        // Real-only
        abs=$abs:ident, sqr=$sqr:ident, inv=$inv:ident,
        linear_frac=$linear_frac:ident,
        inv_sqrt=$inv_sqrt:ident, cbrt=$cbrt:ident, inv_cbrt=$inv_cbrt:ident,
        powr=$powr:ident, pow2o3=$pow2o3:ident, pow3o2=$pow3o2:ident,
        hypot=$hypot:ident,
        exp2=$exp2:ident, exp10=$exp10:ident, expm1=$expm1:ident,
        log2=$log2:ident, log1p=$log1p:ident, logb=$logb:ident,
        cosd=$cosd:ident, sind=$sind:ident, tand=$tand:ident,
        cospi=$cospi:ident, sinpi=$sinpi:ident, tanpi=$tanpi:ident,
        sincos=$sincos:ident, sincospi=$sincospi:ident,
        atan2=$atan2:ident, acospi=$acospi:ident, asinpi=$asinpi:ident,
        atanpi=$atanpi:ident, atan2pi=$atan2pi:ident,
        erf=$erf:ident, erfc=$erfc:ident, erf_inv=$erf_inv:ident,
        erfc_inv=$erfc_inv:ident, erfcx=$erfcx:ident,
        cdf_norm=$cdf_norm:ident, cdf_norm_inv=$cdf_norm_inv:ident,
        lgamma=$lgamma:ident, tgamma=$tgamma:ident,
        exp_int1=$exp_int1:ident,
        i0=$i0:ident, i1=$i1:ident, j0=$j0:ident, j1=$j1:ident,
        y0=$y0:ident, y1=$y1:ident,
        floor=$floor:ident, ceil=$ceil:ident, trunc=$trunc:ident,
        round=$round:ident, nearby_int=$nearby_int:ident,
        rint=$rint:ident, frac=$frac:ident, modf=$modf:ident,
        fmod=$fmod:ident, remainder=$remainder:ident,
        copysign=$copysign:ident, next_after=$next_after:ident,
        fdim=$fdim:ident, fmax=$fmax:ident, fmin=$fmin:ident,
        max_mag=$max_mag:ident, min_mag=$min_mag:ident,
    ) => {
        impl VmScalar for $ty {
            unsafe fn vm_add(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$add(n, a, b, r) } }
            unsafe fn vm_sub(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$sub(n, a, b, r) } }
            unsafe fn vm_mul(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$mul(n, a, b, r) } }
            unsafe fn vm_div(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$div(n, a, b, r) } }
            unsafe fn vm_sqrt(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sqrt(n, a, r) } }
            unsafe fn vm_pow(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$pow(n, a, b, r) } }
            unsafe fn vm_powx(n: MKL_INT, a: *const Self, b: Self, r: *mut Self) { unsafe { sys::$powx(n, a, b, r) } }
            unsafe fn vm_exp(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$exp(n, a, r) } }
            unsafe fn vm_ln(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$ln(n, a, r) } }
            unsafe fn vm_log10(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$log10(n, a, r) } }
            unsafe fn vm_cos(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cos(n, a, r) } }
            unsafe fn vm_sin(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sin(n, a, r) } }
            unsafe fn vm_tan(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$tan(n, a, r) } }
            unsafe fn vm_acos(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$acos(n, a, r) } }
            unsafe fn vm_asin(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$asin(n, a, r) } }
            unsafe fn vm_atan(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$atan(n, a, r) } }
            unsafe fn vm_cosh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cosh(n, a, r) } }
            unsafe fn vm_sinh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sinh(n, a, r) } }
            unsafe fn vm_tanh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$tanh(n, a, r) } }
            unsafe fn vm_acosh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$acosh(n, a, r) } }
            unsafe fn vm_asinh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$asinh(n, a, r) } }
            unsafe fn vm_atanh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$atanh(n, a, r) } }
        }

        impl RealVmScalar for $ty {
            unsafe fn vm_abs_real(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$abs(n, a, r) } }
            unsafe fn vm_sqr(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sqr(n, a, r) } }
            unsafe fn vm_inv(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$inv(n, a, r) } }
            unsafe fn vm_linear_frac(
                n: MKL_INT, a: *const Self, b: *const Self,
                scale_a: Self, shift_a: Self, scale_b: Self, shift_b: Self,
                r: *mut Self,
            ) {
                unsafe { sys::$linear_frac(n, a, b, scale_a, shift_a, scale_b, shift_b, r) }
            }
            unsafe fn vm_inv_sqrt(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$inv_sqrt(n, a, r) } }
            unsafe fn vm_cbrt(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cbrt(n, a, r) } }
            unsafe fn vm_inv_cbrt(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$inv_cbrt(n, a, r) } }
            unsafe fn vm_powr(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$powr(n, a, b, r) } }
            unsafe fn vm_pow2o3(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$pow2o3(n, a, r) } }
            unsafe fn vm_pow3o2(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$pow3o2(n, a, r) } }
            unsafe fn vm_hypot(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$hypot(n, a, b, r) } }
            unsafe fn vm_exp2(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$exp2(n, a, r) } }
            unsafe fn vm_exp10(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$exp10(n, a, r) } }
            unsafe fn vm_expm1(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$expm1(n, a, r) } }
            unsafe fn vm_log2(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$log2(n, a, r) } }
            unsafe fn vm_log1p(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$log1p(n, a, r) } }
            unsafe fn vm_logb(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$logb(n, a, r) } }
            unsafe fn vm_cosd(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cosd(n, a, r) } }
            unsafe fn vm_sind(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sind(n, a, r) } }
            unsafe fn vm_tand(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$tand(n, a, r) } }
            unsafe fn vm_cospi(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cospi(n, a, r) } }
            unsafe fn vm_sinpi(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sinpi(n, a, r) } }
            unsafe fn vm_tanpi(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$tanpi(n, a, r) } }
            unsafe fn vm_sincos(n: MKL_INT, a: *const Self, s: *mut Self, c: *mut Self) { unsafe { sys::$sincos(n, a, s, c) } }
            unsafe fn vm_sincospi(n: MKL_INT, a: *const Self, s: *mut Self, c: *mut Self) { unsafe { sys::$sincospi(n, a, s, c) } }
            unsafe fn vm_atan2(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$atan2(n, a, b, r) } }
            unsafe fn vm_acospi(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$acospi(n, a, r) } }
            unsafe fn vm_asinpi(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$asinpi(n, a, r) } }
            unsafe fn vm_atanpi(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$atanpi(n, a, r) } }
            unsafe fn vm_atan2pi(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$atan2pi(n, a, b, r) } }
            unsafe fn vm_erf(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$erf(n, a, r) } }
            unsafe fn vm_erfc(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$erfc(n, a, r) } }
            unsafe fn vm_erf_inv(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$erf_inv(n, a, r) } }
            unsafe fn vm_erfc_inv(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$erfc_inv(n, a, r) } }
            unsafe fn vm_erfcx(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$erfcx(n, a, r) } }
            unsafe fn vm_cdf_norm(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cdf_norm(n, a, r) } }
            unsafe fn vm_cdf_norm_inv(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cdf_norm_inv(n, a, r) } }
            unsafe fn vm_lgamma(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$lgamma(n, a, r) } }
            unsafe fn vm_tgamma(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$tgamma(n, a, r) } }
            unsafe fn vm_exp_int1(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$exp_int1(n, a, r) } }
            unsafe fn vm_i0(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$i0(n, a, r) } }
            unsafe fn vm_i1(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$i1(n, a, r) } }
            unsafe fn vm_j0(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$j0(n, a, r) } }
            unsafe fn vm_j1(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$j1(n, a, r) } }
            unsafe fn vm_y0(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$y0(n, a, r) } }
            unsafe fn vm_y1(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$y1(n, a, r) } }
            unsafe fn vm_floor(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$floor(n, a, r) } }
            unsafe fn vm_ceil(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$ceil(n, a, r) } }
            unsafe fn vm_trunc(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$trunc(n, a, r) } }
            unsafe fn vm_round(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$round(n, a, r) } }
            unsafe fn vm_nearby_int(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$nearby_int(n, a, r) } }
            unsafe fn vm_rint(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$rint(n, a, r) } }
            unsafe fn vm_frac(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$frac(n, a, r) } }
            unsafe fn vm_modf(n: MKL_INT, a: *const Self, ip: *mut Self, fp: *mut Self) { unsafe { sys::$modf(n, a, ip, fp) } }
            unsafe fn vm_fmod(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$fmod(n, a, b, r) } }
            unsafe fn vm_remainder(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$remainder(n, a, b, r) } }
            unsafe fn vm_copysign(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$copysign(n, a, b, r) } }
            unsafe fn vm_next_after(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$next_after(n, a, b, r) } }
            unsafe fn vm_fdim(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$fdim(n, a, b, r) } }
            unsafe fn vm_fmax(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$fmax(n, a, b, r) } }
            unsafe fn vm_fmin(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$fmin(n, a, b, r) } }
            unsafe fn vm_max_mag(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$max_mag(n, a, b, r) } }
            unsafe fn vm_min_mag(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$min_mag(n, a, b, r) } }
        }
    };
}

impl_vm_scalar_real! {
    f32,
    add=vsAdd, sub=vsSub, mul=vsMul, div=vsDiv,
    sqrt=vsSqrt,
    pow=vsPow, powx=vsPowx,
    exp=vsExp, ln=vsLn, log10=vsLog10,
    cos=vsCos, sin=vsSin, tan=vsTan, acos=vsAcos, asin=vsAsin, atan=vsAtan,
    cosh=vsCosh, sinh=vsSinh, tanh=vsTanh,
    acosh=vsAcosh, asinh=vsAsinh, atanh=vsAtanh,
    abs=vsAbs, sqr=vsSqr, inv=vsInv, linear_frac=vsLinearFrac,
    inv_sqrt=vsInvSqrt, cbrt=vsCbrt, inv_cbrt=vsInvCbrt,
    powr=vsPowr, pow2o3=vsPow2o3, pow3o2=vsPow3o2, hypot=vsHypot,
    exp2=vsExp2, exp10=vsExp10, expm1=vsExpm1,
    log2=vsLog2, log1p=vsLog1p, logb=vsLogb,
    cosd=vsCosd, sind=vsSind, tand=vsTand,
    cospi=vsCospi, sinpi=vsSinpi, tanpi=vsTanpi,
    sincos=vsSinCos, sincospi=vsSinCospi,
    atan2=vsAtan2, acospi=vsAcospi, asinpi=vsAsinpi,
    atanpi=vsAtanpi, atan2pi=vsAtan2pi,
    erf=vsErf, erfc=vsErfc, erf_inv=vsErfInv, erfc_inv=vsErfcInv, erfcx=vsErfcx,
    cdf_norm=vsCdfNorm, cdf_norm_inv=vsCdfNormInv,
    lgamma=vsLGamma, tgamma=vsTGamma,
    exp_int1=vsExpInt1,
    i0=vsI0, i1=vsI1, j0=vsJ0, j1=vsJ1, y0=vsY0, y1=vsY1,
    floor=vsFloor, ceil=vsCeil, trunc=vsTrunc, round=vsRound,
    nearby_int=vsNearbyInt, rint=vsRint, frac=vsFrac, modf=vsModf,
    fmod=vsFmod, remainder=vsRemainder,
    copysign=vsCopySign, next_after=vsNextAfter,
    fdim=vsFdim, fmax=vsFmax, fmin=vsFmin,
    max_mag=vsMaxMag, min_mag=vsMinMag,
}

impl_vm_scalar_real! {
    f64,
    add=vdAdd, sub=vdSub, mul=vdMul, div=vdDiv,
    sqrt=vdSqrt,
    pow=vdPow, powx=vdPowx,
    exp=vdExp, ln=vdLn, log10=vdLog10,
    cos=vdCos, sin=vdSin, tan=vdTan, acos=vdAcos, asin=vdAsin, atan=vdAtan,
    cosh=vdCosh, sinh=vdSinh, tanh=vdTanh,
    acosh=vdAcosh, asinh=vdAsinh, atanh=vdAtanh,
    abs=vdAbs, sqr=vdSqr, inv=vdInv, linear_frac=vdLinearFrac,
    inv_sqrt=vdInvSqrt, cbrt=vdCbrt, inv_cbrt=vdInvCbrt,
    powr=vdPowr, pow2o3=vdPow2o3, pow3o2=vdPow3o2, hypot=vdHypot,
    exp2=vdExp2, exp10=vdExp10, expm1=vdExpm1,
    log2=vdLog2, log1p=vdLog1p, logb=vdLogb,
    cosd=vdCosd, sind=vdSind, tand=vdTand,
    cospi=vdCospi, sinpi=vdSinpi, tanpi=vdTanpi,
    sincos=vdSinCos, sincospi=vdSinCospi,
    atan2=vdAtan2, acospi=vdAcospi, asinpi=vdAsinpi,
    atanpi=vdAtanpi, atan2pi=vdAtan2pi,
    erf=vdErf, erfc=vdErfc, erf_inv=vdErfInv, erfc_inv=vdErfcInv, erfcx=vdErfcx,
    cdf_norm=vdCdfNorm, cdf_norm_inv=vdCdfNormInv,
    lgamma=vdLGamma, tgamma=vdTGamma,
    exp_int1=vdExpInt1,
    i0=vdI0, i1=vdI1, j0=vdJ0, j1=vdJ1, y0=vdY0, y1=vdY1,
    floor=vdFloor, ceil=vdCeil, trunc=vdTrunc, round=vdRound,
    nearby_int=vdNearbyInt, rint=vdRint, frac=vdFrac, modf=vdModf,
    fmod=vdFmod, remainder=vdRemainder,
    copysign=vdCopySign, next_after=vdNextAfter,
    fdim=vdFdim, fmax=vdFmax, fmin=vdFmin,
    max_mag=vdMaxMag, min_mag=vdMinMag,
}

macro_rules! impl_vm_scalar_complex {
    ($ty:ty, $real:ty,
        add=$add:ident, sub=$sub:ident, mul=$mul:ident, div=$div:ident,
        sqrt=$sqrt:ident,
        pow=$pow:ident, powx=$powx:ident,
        exp=$exp:ident, ln=$ln:ident, log10=$log10:ident,
        cos=$cos:ident, sin=$sin:ident, tan=$tan:ident,
        acos=$acos:ident, asin=$asin:ident, atan=$atan:ident,
        cosh=$cosh:ident, sinh=$sinh:ident, tanh=$tanh:ident,
        acosh=$acosh:ident, asinh=$asinh:ident, atanh=$atanh:ident,
        abs_complex=$abs:ident, arg=$arg:ident, conj=$conj:ident,
        mul_by_conj=$mul_by_conj:ident, cis=$cis:ident,
    ) => {
        impl VmScalar for $ty {
            unsafe fn vm_add(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$add(n, a.cast(), b.cast(), r.cast()) } }
            unsafe fn vm_sub(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$sub(n, a.cast(), b.cast(), r.cast()) } }
            unsafe fn vm_mul(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$mul(n, a.cast(), b.cast(), r.cast()) } }
            unsafe fn vm_div(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$div(n, a.cast(), b.cast(), r.cast()) } }
            unsafe fn vm_sqrt(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sqrt(n, a.cast(), r.cast()) } }
            unsafe fn vm_pow(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) { unsafe { sys::$pow(n, a.cast(), b.cast(), r.cast()) } }
            unsafe fn vm_powx(n: MKL_INT, a: *const Self, b: Self, r: *mut Self) {
                unsafe { sys::$powx(n, a.cast(), core::mem::transmute_copy(&b), r.cast()) }
            }
            unsafe fn vm_exp(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$exp(n, a.cast(), r.cast()) } }
            unsafe fn vm_ln(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$ln(n, a.cast(), r.cast()) } }
            unsafe fn vm_log10(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$log10(n, a.cast(), r.cast()) } }
            unsafe fn vm_cos(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cos(n, a.cast(), r.cast()) } }
            unsafe fn vm_sin(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sin(n, a.cast(), r.cast()) } }
            unsafe fn vm_tan(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$tan(n, a.cast(), r.cast()) } }
            unsafe fn vm_acos(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$acos(n, a.cast(), r.cast()) } }
            unsafe fn vm_asin(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$asin(n, a.cast(), r.cast()) } }
            unsafe fn vm_atan(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$atan(n, a.cast(), r.cast()) } }
            unsafe fn vm_cosh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$cosh(n, a.cast(), r.cast()) } }
            unsafe fn vm_sinh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$sinh(n, a.cast(), r.cast()) } }
            unsafe fn vm_tanh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$tanh(n, a.cast(), r.cast()) } }
            unsafe fn vm_acosh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$acosh(n, a.cast(), r.cast()) } }
            unsafe fn vm_asinh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$asinh(n, a.cast(), r.cast()) } }
            unsafe fn vm_atanh(n: MKL_INT, a: *const Self, r: *mut Self) { unsafe { sys::$atanh(n, a.cast(), r.cast()) } }
        }

        impl ComplexVmScalar for $ty {
            unsafe fn vm_abs_complex(n: MKL_INT, a: *const Self, r: *mut $real) {
                unsafe { sys::$abs(n, a.cast(), r) }
            }
            unsafe fn vm_arg(n: MKL_INT, a: *const Self, r: *mut $real) {
                unsafe { sys::$arg(n, a.cast(), r) }
            }
            unsafe fn vm_conj(n: MKL_INT, a: *const Self, r: *mut Self) {
                unsafe { sys::$conj(n, a.cast(), r.cast()) }
            }
            unsafe fn vm_mul_by_conj(n: MKL_INT, a: *const Self, b: *const Self, r: *mut Self) {
                unsafe { sys::$mul_by_conj(n, a.cast(), b.cast(), r.cast()) }
            }
            unsafe fn vm_cis(n: MKL_INT, a: *const $real, r: *mut Self) {
                unsafe { sys::$cis(n, a, r.cast()) }
            }
        }
    };
}

impl_vm_scalar_complex! {
    Complex32, f32,
    add=vcAdd, sub=vcSub, mul=vcMul, div=vcDiv,
    sqrt=vcSqrt,
    pow=vcPow, powx=vcPowx,
    exp=vcExp, ln=vcLn, log10=vcLog10,
    cos=vcCos, sin=vcSin, tan=vcTan, acos=vcAcos, asin=vcAsin, atan=vcAtan,
    cosh=vcCosh, sinh=vcSinh, tanh=vcTanh,
    acosh=vcAcosh, asinh=vcAsinh, atanh=vcAtanh,
    abs_complex=vcAbs, arg=vcArg, conj=vcConj,
    mul_by_conj=vcMulByConj, cis=vcCIS,
}

impl_vm_scalar_complex! {
    Complex64, f64,
    add=vzAdd, sub=vzSub, mul=vzMul, div=vzDiv,
    sqrt=vzSqrt,
    pow=vzPow, powx=vzPowx,
    exp=vzExp, ln=vzLn, log10=vzLog10,
    cos=vzCos, sin=vzSin, tan=vzTan, acos=vzAcos, asin=vzAsin, atan=vzAtan,
    cosh=vzCosh, sinh=vzSinh, tanh=vzTanh,
    acosh=vzAcosh, asinh=vzAsinh, atanh=vzAtanh,
    abs_complex=vzAbs, arg=vzArg, conj=vzConj,
    mul_by_conj=vzMulByConj, cis=vzCIS,
}
