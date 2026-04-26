//! Vector Mathematical Functions (VM).
//!
//! Element-wise mathematical functions on slices of real or complex values.
//! Operations are organized by category:
//!
//! - **Arithmetic**: [`add`], [`sub`], [`mul`], [`div`], [`sqr`], [`abs`],
//!   [`linear_frac`], [`mul_by_conj`] (complex), [`conj`] (complex).
//! - **Power & root**: [`pow`], [`powx`], [`powr`], [`pow2o3`], [`pow3o2`],
//!   [`sqrt`], [`inv_sqrt`], [`cbrt`], [`inv_cbrt`], [`hypot`], [`inv`].
//! - **Exponential / logarithm**: [`exp`], [`exp2`], [`exp10`], [`expm1`],
//!   [`ln`], [`log2`], [`log10`], [`log1p`], [`logb`].
//! - **Trigonometric**: [`cos`], [`sin`], [`tan`], [`sincos`], [`cosd`],
//!   [`sind`], [`tand`], [`cospi`], [`sinpi`], [`tanpi`], [`sincospi`],
//!   [`cis`].
//! - **Inverse trigonometric**: [`acos`], [`asin`], [`atan`], [`atan2`],
//!   [`acospi`], [`asinpi`], [`atanpi`], [`atan2pi`].
//! - **Hyperbolic**: [`cosh`], [`sinh`], [`tanh`], [`acosh`], [`asinh`],
//!   [`atanh`].
//! - **Special**: [`erf`], [`erfc`], [`erf_inv`], [`erfc_inv`], [`erfcx`],
//!   [`cdf_norm`], [`cdf_norm_inv`], [`lgamma`], [`tgamma`], [`exp_int1`],
//!   Bessel ([`i0`], [`i1`], [`j0`], [`j1`], [`jn`], [`y0`], [`y1`],
//!   [`yn`]).
//! - **Rounding**: [`floor`], [`ceil`], [`trunc`], [`round`],
//!   [`nearby_int`], [`rint`], [`modf`], [`frac`].
//! - **Misc**: [`copysign`], [`next_after`], [`fdim`], [`fmax`], [`fmin`],
//!   [`max_mag`], [`min_mag`], [`fmod`], [`remainder`], [`arg`] (complex).
//!
//! Each function takes input and output slices of equal length. The
//! "scalar" trait family ([`VmScalar`], [`RealVmScalar`],
//! [`ComplexVmScalar`]) dispatches to the correct `v[smcdz]*` routine.

mod scalar;

pub use scalar::{ComplexVmScalar, RealVmScalar, VmScalar};

use crate::error::{Error, Result};
use crate::util::dim_to_mkl_int;

// =====================================================================
// Macros to define public free functions over the trait family.
//
// `unary_universal` — y[i] = f(x[i]) for all four scalar types.
// `unary_real`      — y[i] = f(x[i]) for f32 / f64 only.
// `binary_universal`/`binary_real` — z[i] = f(x[i], y[i]).
// =====================================================================

macro_rules! unary_universal {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            #[doc = concat!("Element-wise `", stringify!($name), "`. `y[i] = ", stringify!($name), "(x[i])`.")]
            pub fn $name<T: VmScalar>(x: &[T], y: &mut [T]) -> Result<()> {
                check_same_len(x.len(), y.len())?;
                let n = dim_to_mkl_int(x.len())?;
                unsafe { T::$method(n, x.as_ptr(), y.as_mut_ptr()); }
                Ok(())
            }
        )*
    };
}

macro_rules! unary_real {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            #[doc = concat!("Element-wise `", stringify!($name), "` (real-only).")]
            pub fn $name<T: RealVmScalar>(x: &[T], y: &mut [T]) -> Result<()> {
                check_same_len(x.len(), y.len())?;
                let n = dim_to_mkl_int(x.len())?;
                unsafe { T::$method(n, x.as_ptr(), y.as_mut_ptr()); }
                Ok(())
            }
        )*
    };
}

macro_rules! unary_complex_to_real {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            #[doc = concat!("Element-wise `", stringify!($name), "`. Maps a complex vector to a real vector.")]
            pub fn $name<T: ComplexVmScalar>(x: &[T], y: &mut [T::Real]) -> Result<()> {
                check_same_len(x.len(), y.len())?;
                let n = dim_to_mkl_int(x.len())?;
                unsafe { T::$method(n, x.as_ptr(), y.as_mut_ptr()); }
                Ok(())
            }
        )*
    };
}

macro_rules! unary_complex {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            #[doc = concat!("Element-wise `", stringify!($name), "` (complex-only).")]
            pub fn $name<T: ComplexVmScalar>(x: &[T], y: &mut [T]) -> Result<()> {
                check_same_len(x.len(), y.len())?;
                let n = dim_to_mkl_int(x.len())?;
                unsafe { T::$method(n, x.as_ptr(), y.as_mut_ptr()); }
                Ok(())
            }
        )*
    };
}

macro_rules! binary_universal {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            #[doc = concat!("Element-wise `", stringify!($name), "`. `z[i] = ", stringify!($name), "(x[i], y[i])`.")]
            pub fn $name<T: VmScalar>(x: &[T], y: &[T], z: &mut [T]) -> Result<()> {
                check_three_same_len(x.len(), y.len(), z.len())?;
                let n = dim_to_mkl_int(x.len())?;
                unsafe { T::$method(n, x.as_ptr(), y.as_ptr(), z.as_mut_ptr()); }
                Ok(())
            }
        )*
    };
}

macro_rules! binary_real {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            #[doc = concat!("Element-wise `", stringify!($name), "` (real-only).")]
            pub fn $name<T: RealVmScalar>(x: &[T], y: &[T], z: &mut [T]) -> Result<()> {
                check_three_same_len(x.len(), y.len(), z.len())?;
                let n = dim_to_mkl_int(x.len())?;
                unsafe { T::$method(n, x.as_ptr(), y.as_ptr(), z.as_mut_ptr()); }
                Ok(())
            }
        )*
    };
}

// =====================================================================
// Public API
// =====================================================================

unary_universal! {
    sqrt => vm_sqrt,
    exp => vm_exp,
    ln => vm_ln,
    log10 => vm_log10,
    cos => vm_cos,
    sin => vm_sin,
    tan => vm_tan,
    acos => vm_acos,
    asin => vm_asin,
    atan => vm_atan,
    cosh => vm_cosh,
    sinh => vm_sinh,
    tanh => vm_tanh,
    acosh => vm_acosh,
    asinh => vm_asinh,
    atanh => vm_atanh,
}

binary_universal! {
    add => vm_add,
    sub => vm_sub,
    mul => vm_mul,
    div => vm_div,
    pow => vm_pow,
}

unary_real! {
    abs => vm_abs_real,
    sqr => vm_sqr,
    inv => vm_inv,
    inv_sqrt => vm_inv_sqrt,
    cbrt => vm_cbrt,
    inv_cbrt => vm_inv_cbrt,
    pow2o3 => vm_pow2o3,
    pow3o2 => vm_pow3o2,
    exp2 => vm_exp2,
    exp10 => vm_exp10,
    expm1 => vm_expm1,
    log2 => vm_log2,
    log1p => vm_log1p,
    logb => vm_logb,
    cosd => vm_cosd,
    sind => vm_sind,
    tand => vm_tand,
    cospi => vm_cospi,
    sinpi => vm_sinpi,
    tanpi => vm_tanpi,
    acospi => vm_acospi,
    asinpi => vm_asinpi,
    atanpi => vm_atanpi,
    erf => vm_erf,
    erfc => vm_erfc,
    erf_inv => vm_erf_inv,
    erfc_inv => vm_erfc_inv,
    erfcx => vm_erfcx,
    cdf_norm => vm_cdf_norm,
    cdf_norm_inv => vm_cdf_norm_inv,
    lgamma => vm_lgamma,
    tgamma => vm_tgamma,
    exp_int1 => vm_exp_int1,
    i0 => vm_i0,
    i1 => vm_i1,
    j0 => vm_j0,
    j1 => vm_j1,
    y0 => vm_y0,
    y1 => vm_y1,
    floor => vm_floor,
    ceil => vm_ceil,
    trunc => vm_trunc,
    round => vm_round,
    nearby_int => vm_nearby_int,
    rint => vm_rint,
    frac => vm_frac,
}

binary_real! {
    atan2 => vm_atan2,
    atan2pi => vm_atan2pi,
    hypot => vm_hypot,
    powr => vm_powr,
    fmod => vm_fmod,
    remainder => vm_remainder,
    copysign => vm_copysign,
    next_after => vm_next_after,
    fdim => vm_fdim,
    fmax => vm_fmax,
    fmin => vm_fmin,
    max_mag => vm_max_mag,
    min_mag => vm_min_mag,
}

unary_complex_to_real! {
    abs_complex => vm_abs_complex,
    arg => vm_arg,
}

unary_complex! {
    conj => vm_conj,
}

// =====================================================================
// Functions that don't fit the macro patterns
// =====================================================================

/// Element-wise `y[i] = x[i] ^ b` (`b` is a scalar). Universal.
pub fn powx<T: VmScalar>(x: &[T], b: T, y: &mut [T]) -> Result<()> {
    check_same_len(x.len(), y.len())?;
    let n = dim_to_mkl_int(x.len())?;
    unsafe {
        T::vm_powx(n, x.as_ptr(), b, y.as_mut_ptr());
    }
    Ok(())
}

/// Element-wise sine and cosine: writes `sin(x[i])` to `s` and
/// `cos(x[i])` to `c`. Real-only.
pub fn sincos<T: RealVmScalar>(x: &[T], s: &mut [T], c: &mut [T]) -> Result<()> {
    if x.len() != s.len() || x.len() != c.len() {
        return Err(Error::InvalidArgument(
            "sincos: x, s, c must have the same length",
        ));
    }
    let n = dim_to_mkl_int(x.len())?;
    unsafe {
        T::vm_sincos(n, x.as_ptr(), s.as_mut_ptr(), c.as_mut_ptr());
    }
    Ok(())
}

/// Element-wise sine and cosine of `π * x`. Real-only.
pub fn sincospi<T: RealVmScalar>(x: &[T], s: &mut [T], c: &mut [T]) -> Result<()> {
    if x.len() != s.len() || x.len() != c.len() {
        return Err(Error::InvalidArgument(
            "sincospi: x, s, c must have the same length",
        ));
    }
    let n = dim_to_mkl_int(x.len())?;
    unsafe {
        T::vm_sincospi(n, x.as_ptr(), s.as_mut_ptr(), c.as_mut_ptr());
    }
    Ok(())
}

/// Element-wise integer / fractional split: writes `trunc(x[i])` to
/// `int_part` and the fractional remainder to `frac_part`. Real-only.
pub fn modf<T: RealVmScalar>(
    x: &[T],
    int_part: &mut [T],
    frac_part: &mut [T],
) -> Result<()> {
    if x.len() != int_part.len() || x.len() != frac_part.len() {
        return Err(Error::InvalidArgument(
            "modf: x, int_part, frac_part must have the same length",
        ));
    }
    let n = dim_to_mkl_int(x.len())?;
    unsafe {
        T::vm_modf(n, x.as_ptr(), int_part.as_mut_ptr(), frac_part.as_mut_ptr());
    }
    Ok(())
}

/// Element-wise `y[i] = exp(i * x[i])` (cosine + i*sine). Maps a real
/// input to a complex output.
pub fn cis<T: ComplexVmScalar>(x: &[T::Real], y: &mut [T]) -> Result<()> {
    check_same_len(x.len(), y.len())?;
    let n = dim_to_mkl_int(x.len())?;
    unsafe {
        T::vm_cis(n, x.as_ptr(), y.as_mut_ptr());
    }
    Ok(())
}

/// Element-wise `z[i] = x[i] * conj(y[i])` for complex types.
pub fn mul_by_conj<T: ComplexVmScalar>(x: &[T], y: &[T], z: &mut [T]) -> Result<()> {
    check_three_same_len(x.len(), y.len(), z.len())?;
    let n = dim_to_mkl_int(x.len())?;
    unsafe {
        T::vm_mul_by_conj(n, x.as_ptr(), y.as_ptr(), z.as_mut_ptr());
    }
    Ok(())
}

/// Element-wise `r[i] = (scale_a * x[i] + shift_a) / (scale_b * y[i] + shift_b)`.
/// Real-only.
#[allow(clippy::too_many_arguments)]
pub fn linear_frac<T: RealVmScalar>(
    x: &[T],
    y: &[T],
    scale_a: T,
    shift_a: T,
    scale_b: T,
    shift_b: T,
    r: &mut [T],
) -> Result<()> {
    check_three_same_len(x.len(), y.len(), r.len())?;
    let n = dim_to_mkl_int(x.len())?;
    unsafe {
        T::vm_linear_frac(
            n,
            x.as_ptr(),
            y.as_ptr(),
            scale_a,
            shift_a,
            scale_b,
            shift_b,
            r.as_mut_ptr(),
        );
    }
    Ok(())
}

// =====================================================================
// Helpers
// =====================================================================

#[inline]
fn check_same_len(a: usize, b: usize) -> Result<()> {
    if a != b {
        Err(Error::InvalidArgument(
            "input and output slices must have the same length",
        ))
    } else {
        Ok(())
    }
}

#[inline]
fn check_three_same_len(a: usize, b: usize, c: usize) -> Result<()> {
    if a != b || a != c {
        Err(Error::InvalidArgument(
            "all input and output slices must have the same length",
        ))
    } else {
        Ok(())
    }
}
