//! Discrete Fourier Transform (DFT) — fast Fourier transform via the
//! oneMKL DFTI interface.
//!
//! This module exposes three plan types:
//!
//! - [`FftPlan`] — in-place complex-to-complex FFT, 1-D, 2-D, 3-D, or
//!   arbitrary dimensionality.
//! - [`FftPlanOutOfPlace`] — out-of-place complex-to-complex FFT.
//! - [`RealFftPlan`] — real-to-complex (forward) and complex-to-real
//!   (backward) using the CCE (conjugate-even) storage format.
//!
//! All plans own a committed DFTI descriptor and free it on drop.
//!
//! ```no_run
//! use num_complex::Complex64;
//! use onemkl::fft::FftPlan;
//!
//! // 1-D in-place FFT.
//! let mut plan = FftPlan::<f64>::complex_1d(8).unwrap();
//! let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); 8];
//! buf[0] = Complex64::new(1.0, 0.0);
//! plan.forward_in_place(&mut buf).unwrap();
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
    /// Convert to `f64` for variadic FFI calls. DFTI's variadic
    /// `DftiSetValue` reads the value with `va_arg(ap, double)`
    /// regardless of descriptor precision, then converts internally
    /// — so callers must pass `double` even for `DFTI_SINGLE`.
    fn to_f64(self) -> f64;
}
impl FftScalar for f32 {
    const PRECISION: FftPrecision = FftPrecision::Single;
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl FftScalar for f64 {
    const PRECISION: FftPrecision = FftPrecision::Double;
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

// =====================================================================
// Internal helpers
// =====================================================================

#[inline]
fn check_dfti(status: c_long) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::DftiStatus(status as i64))
    }
}

#[inline]
fn apply_scales<T: FftScalar>(
    handle: DFTI_DESCRIPTOR_HANDLE,
    scales: Option<(T, T)>,
) -> Result<()> {
    if let Some((forward, backward)) = scales {
        // DFTI's variadic interface reads scales as `double`
        // regardless of descriptor precision, so cast f32 → f64.
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_FORWARD_SCALE,
                forward.to_f64(),
            )
        };
        check_dfti(status)?;
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_BACKWARD_SCALE,
                backward.to_f64(),
            )
        };
        check_dfti(status)?;
    }
    Ok(())
}

#[inline]
fn create_complex_descriptor<T: FftScalar>(
    dims: &[usize],
    placement: sys::DFTI_CONFIG_VALUE::Type,
    scales: Option<(T, T)>,
) -> Result<DFTI_DESCRIPTOR_HANDLE> {
    if dims.is_empty() {
        return Err(Error::InvalidArgument(
            "FFT plan needs at least one dimension",
        ));
    }
    let dims_long: Vec<c_long> = dims
        .iter()
        .map(|&d| {
            c_long::try_from(d).map_err(|_| Error::DimensionOverflow)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
    let ndim_long: c_long = c_long::try_from(dims.len())
        .map_err(|_| Error::DimensionOverflow)?;
    let precision = T::PRECISION.as_dfti();
    let domain = sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX;

    let status = unsafe {
        if dims.len() == 1 {
            sys::DftiCreateDescriptor(&mut handle, precision, domain, ndim_long, dims_long[0])
        } else {
            sys::DftiCreateDescriptor(
                &mut handle,
                precision,
                domain,
                ndim_long,
                dims_long.as_ptr(),
            )
        }
    };
    check_dfti(status)?;

    let status = unsafe {
        sys::DftiSetValue(
            handle,
            sys::DFTI_CONFIG_PARAM::DFTI_PLACEMENT,
            placement,
        )
    };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    if let Err(e) = apply_scales(handle, scales) {
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
    Ok(handle)
}

#[inline]
fn create_real_descriptor<T: FftScalar>(
    dims: &[usize],
    placement: sys::DFTI_CONFIG_VALUE::Type,
    scales: Option<(T, T)>,
) -> Result<DFTI_DESCRIPTOR_HANDLE> {
    if dims.is_empty() {
        return Err(Error::InvalidArgument(
            "FFT plan needs at least one dimension",
        ));
    }
    let dims_long: Vec<c_long> = dims
        .iter()
        .map(|&d| c_long::try_from(d).map_err(|_| Error::DimensionOverflow))
        .collect::<Result<Vec<_>>>()?;

    let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
    let ndim_long: c_long = c_long::try_from(dims.len())
        .map_err(|_| Error::DimensionOverflow)?;
    let precision = T::PRECISION.as_dfti();
    let domain = sys::DFTI_CONFIG_VALUE::DFTI_REAL;

    let status = unsafe {
        if dims.len() == 1 {
            sys::DftiCreateDescriptor(&mut handle, precision, domain, ndim_long, dims_long[0])
        } else {
            sys::DftiCreateDescriptor(
                &mut handle,
                precision,
                domain,
                ndim_long,
                dims_long.as_ptr(),
            )
        }
    };
    check_dfti(status)?;

    // Pack CCE output as one contiguous Complex<T> array of size
    // floor(n0/2)+1 along the first dimension (MKL's fastest-varying
    // dimension, since DFTI defaults to Fortran-style strides) and
    // full size along the rest.
    let status = unsafe {
        sys::DftiSetValue(
            handle,
            sys::DFTI_CONFIG_PARAM::DFTI_CONJUGATE_EVEN_STORAGE,
            sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX_COMPLEX,
        )
    };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    // MKL's default INPUT_STRIDES are computed for in-place storage,
    // which assumes the real buffer has been padded to hold the
    // complex output. For not-in-place real ↔ complex transforms we
    // must override both stride arrays so MKL walks each side with
    // its own contiguous layout.
    if placement == sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE && dims.len() > 1 {
        // MKL DFTI uses Fortran-style strides where stride[i+1] is the
        // step along dimension i. CCE shrinks the LAST physical
        // dimension of the spectrum (the slowest-varying one in
        // column-major), not the first.
        let last = dims.len() - 1;
        let mut input_strides: Vec<c_long> = Vec::with_capacity(dims.len() + 1);
        input_strides.push(0);
        let mut acc: c_long = 1;
        for &d in &dims_long {
            input_strides.push(acc);
            acc = acc.checked_mul(d).ok_or(Error::DimensionOverflow)?;
        }

        let mut output_strides: Vec<c_long> = Vec::with_capacity(dims.len() + 1);
        output_strides.push(0);
        let cce_last: c_long = c_long::try_from(cce_complex_len(dims[last]))
            .map_err(|_| Error::DimensionOverflow)?;
        let mut acc: c_long = 1;
        for (i, &d) in dims_long.iter().enumerate() {
            output_strides.push(acc);
            let step = if i == last { cce_last } else { d };
            acc = acc.checked_mul(step).ok_or(Error::DimensionOverflow)?;
        }

        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_INPUT_STRIDES,
                input_strides.as_ptr(),
            )
        };
        if let Err(e) = check_dfti(status) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_OUTPUT_STRIDES,
                output_strides.as_ptr(),
            )
        };
        if let Err(e) = check_dfti(status) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
    }

    let status = unsafe {
        sys::DftiSetValue(
            handle,
            sys::DFTI_CONFIG_PARAM::DFTI_PLACEMENT,
            placement,
        )
    };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    if let Err(e) = apply_scales(handle, scales) {
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
    Ok(handle)
}

/// Number of complex elements in a CCE forward output for an `n`-point
/// real input along the FFT dimension: `n/2 + 1`.
#[inline]
#[must_use]
pub fn cce_complex_len(real_n: usize) -> usize {
    real_n / 2 + 1
}

// =====================================================================
// FftPlan — in-place complex-to-complex
// =====================================================================

/// In-place complex-to-complex FFT plan.
pub struct FftPlan<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    total_complex_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for FftPlan<T> {}

impl<T: FftScalar> FftPlan<T> {
    /// 1-D in-place plan of length `n`.
    pub fn complex_1d(n: usize) -> Result<Self> {
        Self::complex_nd(&[n])
    }

    /// 2-D in-place plan of size `n0 × n1`.
    pub fn complex_2d(n0: usize, n1: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1])
    }

    /// 3-D in-place plan of size `n0 × n1 × n2`.
    pub fn complex_3d(n0: usize, n1: usize, n2: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1, n2])
    }

    /// N-D in-place plan with the supplied dimensions.
    pub fn complex_nd(dims: &[usize]) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_INPLACE,
            None,
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// N-D in-place plan with explicit forward / backward scales. Set
    /// `backward_scale = 1.0 / N` (where `N = dims.iter().product()`)
    /// to make `IFFT(FFT(x)) = x`. Set both to `1.0 / sqrt(N)` for a
    /// unitary transform pair.
    pub fn complex_nd_with_scales(
        dims: &[usize],
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_INPLACE,
            Some((forward_scale, backward_scale)),
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// Forward transform in place.
    pub fn forward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        if x.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
            ));
        }
        let status = unsafe {
            sys::DftiComputeForward(self.handle, x.as_mut_ptr().cast())
        };
        check_dfti(status)
    }

    /// Backward transform in place (un-normalized).
    pub fn backward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        if x.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
            ));
        }
        let status = unsafe {
            sys::DftiComputeBackward(self.handle, x.as_mut_ptr().cast())
        };
        check_dfti(status)
    }
}

impl<T: FftScalar> Drop for FftPlan<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// FftPlanOutOfPlace — out-of-place complex-to-complex
// =====================================================================

/// Out-of-place complex-to-complex FFT plan.
pub struct FftPlanOutOfPlace<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    total_complex_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for FftPlanOutOfPlace<T> {}

impl<T: FftScalar> FftPlanOutOfPlace<T> {
    /// 1-D out-of-place plan.
    pub fn complex_1d(n: usize) -> Result<Self> {
        Self::complex_nd(&[n])
    }

    /// 2-D out-of-place plan.
    pub fn complex_2d(n0: usize, n1: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1])
    }

    /// 3-D out-of-place plan.
    pub fn complex_3d(n0: usize, n1: usize, n2: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1, n2])
    }

    /// N-D out-of-place plan.
    pub fn complex_nd(dims: &[usize]) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE,
            None,
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// N-D out-of-place plan with explicit forward / backward scales.
    /// See [`FftPlan::complex_nd_with_scales`] for normalization
    /// guidance.
    pub fn complex_nd_with_scales(
        dims: &[usize],
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE,
            Some((forward_scale, backward_scale)),
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// Forward transform `out ← FFT(in)`. The two buffers must have the
    /// same length (≥ the plan's total size).
    pub fn forward(&mut self, input: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() < self.total_complex_len || out.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
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

    /// Backward transform `out ← IFFT(in)` (un-normalized).
    pub fn backward(&mut self, input: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() < self.total_complex_len || out.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
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
// RealFftPlan — real-to-complex / complex-to-real (CCE format)
// =====================================================================

/// Real-input FFT plan. Forward maps a real `n0 × n1 × ...` array to a
/// `(n0/2 + 1) × n1 × ...` complex array (CCE format); backward does
/// the reverse.
///
/// The plan is always out-of-place because the real and complex
/// buffers have different sizes.
pub struct RealFftPlan<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    real_total_len: usize,
    complex_total_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for RealFftPlan<T> {}

impl<T: FftScalar> RealFftPlan<T> {
    /// 1-D real FFT plan of length `n`. Real buffer has `n` elements;
    /// CCE complex buffer has `n / 2 + 1` elements.
    pub fn real_1d(n: usize) -> Result<Self> {
        Self::real_nd(&[n])
    }

    /// 2-D real FFT plan of size `n0 × n1`.
    pub fn real_2d(n0: usize, n1: usize) -> Result<Self> {
        Self::real_nd(&[n0, n1])
    }

    /// 3-D real FFT plan of size `n0 × n1 × n2`.
    pub fn real_3d(n0: usize, n1: usize, n2: usize) -> Result<Self> {
        Self::real_nd(&[n0, n1, n2])
    }

    /// N-D real FFT plan with the supplied dimensions.
    ///
    /// CCE storage shrinks the last dimension to
    /// `dims[dims.len() - 1] / 2 + 1` in the complex output.
    pub fn real_nd(dims: &[usize]) -> Result<Self> {
        Self::real_nd_inner(dims, None)
    }

    /// N-D real FFT plan with explicit forward / backward scales. See
    /// [`FftPlan::complex_nd_with_scales`] for normalization
    /// guidance.
    pub fn real_nd_with_scales(
        dims: &[usize],
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        Self::real_nd_inner(dims, Some((forward_scale, backward_scale)))
    }

    fn real_nd_inner(dims: &[usize], scales: Option<(T, T)>) -> Result<Self> {
        let handle = create_real_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE,
            scales,
        )?;
        let real_total: usize = dims.iter().product();
        let mut complex_dims = dims.to_vec();
        let last = complex_dims.len() - 1;
        complex_dims[last] = cce_complex_len(complex_dims[last]);
        let complex_total: usize = complex_dims.iter().product();
        Ok(Self {
            handle,
            real_total_len: real_total,
            complex_total_len: complex_total,
            _marker: PhantomData,
        })
    }

    /// Number of real elements per transform.
    #[inline]
    #[must_use]
    pub fn real_len(&self) -> usize {
        self.real_total_len
    }

    /// Number of complex elements per transform (CCE format).
    #[inline]
    #[must_use]
    pub fn complex_len(&self) -> usize {
        self.complex_total_len
    }

    /// Forward transform `out ← FFT(in)`.
    pub fn forward(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() < self.real_total_len {
            return Err(Error::InvalidArgument(
                "real input buffer too small for FFT plan size",
            ));
        }
        if out.len() < self.complex_total_len {
            return Err(Error::InvalidArgument(
                "complex output buffer too small (need n/2 + 1 along first dim)",
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

    /// Backward transform `out ← IFFT(in)` (un-normalized).
    ///
    /// The complex input must satisfy CCE Hermitian symmetry —
    /// otherwise the output is not real and the result is undefined.
    pub fn backward(&mut self, input: &[Complex<T>], out: &mut [T]) -> Result<()> {
        if input.len() < self.complex_total_len {
            return Err(Error::InvalidArgument(
                "complex input buffer too small (need n/2 + 1 along first dim)",
            ));
        }
        if out.len() < self.real_total_len {
            return Err(Error::InvalidArgument(
                "real output buffer too small for FFT plan size",
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

impl<T: FftScalar> Drop for RealFftPlan<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}
