//! Discrete Fourier Transform (DFT) — fast Fourier transform via the
//! oneMKL DFTI interface.
//!
//! The [`FftPlan`] type owns a committed DFTI descriptor and provides
//! [`forward`](FftPlan::forward) and [`backward`](FftPlan::backward)
//! methods for executing transforms. A typestate distinguishes the
//! "uninitialized" [`FftPlanBuilder`] from the committed [`FftPlan`]:
//! configuration is only allowed before commit, execution only after.
//!
//! ```no_run
//! use num_complex::Complex64;
//! use onemkl::fft::{FftPlan, FftPrecision};
//!
//! let mut plan = FftPlan::<f64>::complex_1d(8).unwrap();
//! let mut buf: Vec<Complex64> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();
//! plan.forward_in_place(&mut buf).unwrap();
//! plan.backward_in_place(&mut buf).unwrap();
//! ```

use core::ffi::c_long;
use core::marker::PhantomData;
use core::ptr;

use num_complex::Complex;
use onemkl_sys::{self as sys, DFTI_DESCRIPTOR_HANDLE};

use crate::error::{Error, Result};

/// Precision of an FFT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftPrecision {
    /// Single precision (32-bit).
    Single,
    /// Double precision (64-bit).
    Double,
}

impl FftPrecision {
    #[inline]
    fn as_dfti(self) -> sys::DFTI_CONFIG_VALUE::Type {
        match self {
            Self::Single => sys::DFTI_CONFIG_VALUE::DFTI_SINGLE,
            Self::Double => sys::DFTI_CONFIG_VALUE::DFTI_DOUBLE,
        }
    }
}

/// Trait associating a Rust real scalar type with its DFTI precision
/// code.
pub trait FftScalar: Copy + 'static {
    /// DFTI precision code corresponding to this real type.
    const PRECISION: FftPrecision;
}
impl FftScalar for f32 {
    const PRECISION: FftPrecision = FftPrecision::Single;
}
impl FftScalar for f64 {
    const PRECISION: FftPrecision = FftPrecision::Double;
}

/// Marker for a complex-to-complex FFT plan.
pub struct ComplexDomain<T>(PhantomData<T>);
/// Marker for a real-input forward FFT plan.
pub struct RealDomain<T>(PhantomData<T>);

// =====================================================================
// FftPlan — a committed DFTI descriptor.
// =====================================================================

/// A committed DFTI descriptor ready to execute transforms.
///
/// `D` is a domain marker (currently [`ComplexDomain`]). Drops the
/// descriptor on `Drop`.
pub struct FftPlan<T: FftScalar, D = ComplexDomain<T>> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    _marker: PhantomData<(T, D)>,
}

unsafe impl<T: FftScalar, D> Send for FftPlan<T, D> {}

impl<T: FftScalar> FftPlan<T, ComplexDomain<T>> {
    /// Create a 1-D complex-to-complex plan of the given length.
    pub fn complex_1d(length: usize) -> Result<Self> {
        let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        let precision = T::PRECISION.as_dfti();
        let domain = sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX;
        let dim: c_long = 1;
        let n: c_long = length.try_into().map_err(|_| Error::DimensionOverflow)?;
        let status = unsafe {
            sys::DftiCreateDescriptor(&mut handle, precision, domain, dim, n)
        };
        check_dfti(status)?;
        // Use not-in-place by default; users can reconfigure if they want.
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_PLACEMENT,
                sys::DFTI_CONFIG_VALUE::DFTI_INPLACE,
            )
        };
        check_dfti(status).map_err(|e| {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            e
        })?;
        let status = unsafe { sys::DftiCommitDescriptor(handle) };
        check_dfti(status).map_err(|e| {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            e
        })?;
        Ok(Self {
            handle,
            _marker: PhantomData,
        })
    }

    /// Forward transform in place: `x ← FFT(x)`.
    pub fn forward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        let status = unsafe {
            sys::DftiComputeForward(self.handle, x.as_mut_ptr().cast())
        };
        check_dfti(status)
    }

    /// Backward transform in place: `x ← IFFT(x)` (un-normalized; you
    /// may want to divide by `length` afterwards).
    pub fn backward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        let status = unsafe {
            sys::DftiComputeBackward(self.handle, x.as_mut_ptr().cast())
        };
        check_dfti(status)
    }
}

impl<T: FftScalar, D> Drop for FftPlan<T, D> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// Out-of-place 1-D plan
// =====================================================================

/// A complex-to-complex plan configured for out-of-place execution.
pub struct FftPlanOutOfPlace<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for FftPlanOutOfPlace<T> {}

impl<T: FftScalar> FftPlanOutOfPlace<T> {
    /// Build a 1-D out-of-place complex plan.
    pub fn complex_1d(length: usize) -> Result<Self> {
        let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        let n: c_long = length.try_into().map_err(|_| Error::DimensionOverflow)?;
        let status = unsafe {
            sys::DftiCreateDescriptor(
                &mut handle,
                T::PRECISION.as_dfti(),
                sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX,
                1,
                n,
            )
        };
        check_dfti(status)?;
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_PLACEMENT,
                sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE,
            )
        };
        if let Err(e) = check_dfti(status) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
        let status = unsafe { sys::DftiCommitDescriptor(handle) };
        if let Err(e) = check_dfti(status) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
        Ok(Self {
            handle,
            _marker: PhantomData,
        })
    }

    /// Forward transform: `out ← FFT(in)`.
    pub fn forward(&mut self, input: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() != out.len() {
            return Err(Error::InvalidArgument(
                "FFT input and output must have the same length",
            ));
        }
        let status = unsafe {
            sys::DftiComputeForward(
                self.handle,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        };
        check_dfti(status)
    }

    /// Backward transform: `out ← IFFT(in)` (un-normalized).
    pub fn backward(&mut self, input: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() != out.len() {
            return Err(Error::InvalidArgument(
                "FFT input and output must have the same length",
            ));
        }
        let status = unsafe {
            sys::DftiComputeBackward(
                self.handle,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        };
        check_dfti(status)
    }
}

impl<T: FftScalar> Drop for FftPlanOutOfPlace<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// Helpers
// =====================================================================

#[inline]
fn check_dfti(status: c_long) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::DftiStatus(status as i64))
    }
}
