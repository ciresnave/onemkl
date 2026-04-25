//! Scalar type abstractions used throughout oneMKL.
//!
//! oneMKL operates on four basic scalar types:
//!
//! | Type      | Rust type                          | MKL C type      |
//! | --------- | ---------------------------------- | --------------- |
//! | `s` real  | [`f32`]                            | `float`         |
//! | `d` real  | [`f64`]                            | `double`        |
//! | `c` cplx  | [`num_complex::Complex32`]         | `MKL_Complex8`  |
//! | `z` cplx  | [`num_complex::Complex64`]         | `MKL_Complex16` |
//!
//! The trait hierarchy lets generic code be written once and dispatch to the
//! right underlying routine based on the scalar type.

use core::ffi::c_void;

use num_complex::{Complex32, Complex64};
use onemkl_sys::{MKL_Complex8, MKL_Complex16};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

/// Common operations and metadata for any scalar type supported by oneMKL.
///
/// This trait is sealed — only [`f32`], [`f64`], [`Complex32`], and
/// [`Complex64`] can implement it.
pub trait Scalar: sealed::Sealed + Copy + Send + Sync + 'static {
    /// The corresponding *real* scalar type.
    ///
    /// `f32` for `f32` and `Complex32`; `f64` for `f64` and `Complex64`.
    type Real: RealScalar;

    /// Additive identity (`0` or `0 + 0i`).
    fn zero() -> Self;
    /// Multiplicative identity (`1` or `1 + 0i`).
    fn one() -> Self;

    /// Cast to a `*const c_void` for FFI use. Equivalent to
    /// `&self as *const Self as *const c_void` but kept on the trait to
    /// document intent.
    #[inline]
    fn as_void_ptr(&self) -> *const c_void {
        self as *const Self as *const c_void
    }

    /// Cast to a `*mut c_void` for FFI use.
    #[inline]
    fn as_mut_void_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }
}

/// Marker trait for the real scalar types ([`f32`] and [`f64`]).
pub trait RealScalar: Scalar<Real = Self> + num_traits::Float {}

/// Marker trait for the complex scalar types ([`Complex32`] and
/// [`Complex64`]).
///
/// Provides bridging between [`num_complex::Complex<R>`] and the
/// corresponding `MKL_Complex*` C structs. The two are layout-compatible —
/// both are `#[repr(C)]` with two real fields — so casts between pointers
/// of the two types are valid.
pub trait ComplexScalar: Scalar {
    /// The matching `MKL_Complex*` FFI struct.
    type MklType: Copy;

    /// Compose a complex value from real and imaginary parts.
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self;

    /// Real part.
    fn re(self) -> Self::Real;

    /// Imaginary part.
    fn im(self) -> Self::Real;

    /// Re-interpret a const pointer as the matching MKL FFI struct.
    ///
    /// # Safety
    ///
    /// The returned pointer aliases `p` for the duration of the call. The
    /// memory layout of `Self` and [`Self::MklType`] is identical (both are
    /// `#[repr(C)]` with `re, im` of the same primitive type) so reads
    /// through either pointer observe the same value.
    #[inline]
    unsafe fn as_mkl_ptr(p: *const Self) -> *const Self::MklType {
        p.cast::<Self::MklType>()
    }

    /// Re-interpret a mutable pointer as the matching MKL FFI struct.
    ///
    /// # Safety
    ///
    /// Same layout argument as [`as_mkl_ptr`](Self::as_mkl_ptr). The caller
    /// must additionally uphold standard `&mut` aliasing rules for `p`.
    #[inline]
    unsafe fn as_mkl_mut_ptr(p: *mut Self) -> *mut Self::MklType {
        p.cast::<Self::MklType>()
    }
}

impl Scalar for f32 {
    type Real = f32;
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
}
impl RealScalar for f32 {}

impl Scalar for f64 {
    type Real = f64;
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
}
impl RealScalar for f64 {}

impl Scalar for Complex32 {
    type Real = f32;
    #[inline]
    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }
    #[inline]
    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }
}
impl ComplexScalar for Complex32 {
    type MklType = MKL_Complex8;
    #[inline]
    fn from_re_im(re: f32, im: f32) -> Self {
        Complex32::new(re, im)
    }
    #[inline]
    fn re(self) -> f32 {
        self.re
    }
    #[inline]
    fn im(self) -> f32 {
        self.im
    }
}

impl Scalar for Complex64 {
    type Real = f64;
    #[inline]
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    #[inline]
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
}
impl ComplexScalar for Complex64 {
    type MklType = MKL_Complex16;
    #[inline]
    fn from_re_im(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    #[inline]
    fn re(self) -> f64 {
        self.re
    }
    #[inline]
    fn im(self) -> f64 {
        self.im
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Complex32`/`Complex64` must be layout-compatible with the MKL
    /// complex C structs, otherwise our pointer casts in `BlasScalar`
    /// implementations would invoke undefined behavior.
    #[test]
    fn complex_layouts_match_mkl() {
        use core::mem::{align_of, size_of};

        assert_eq!(size_of::<Complex32>(), size_of::<MKL_Complex8>());
        assert_eq!(align_of::<Complex32>(), align_of::<MKL_Complex8>());
        assert_eq!(size_of::<Complex64>(), size_of::<MKL_Complex16>());
        assert_eq!(align_of::<Complex64>(), align_of::<MKL_Complex16>());
    }
}
