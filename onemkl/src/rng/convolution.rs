//! 1-D convolution and correlation tasks (`vsl?Conv*` / `vsl?Corr*`).
//!
//! The simplest entry points are [`convolve_1d`] and [`correlate_1d`],
//! which build, execute, and tear down a task in one call. For
//! repeated execution against changing inputs, build a [`Conv1d`] /
//! [`Corr1d`] task once and call [`Conv1d::execute`] / [`Corr1d::execute`]
//! repeatedly.
//!
//! ```no_run
//! use onemkl::rng::convolution::{convolve_1d, ConvMode};
//!
//! let x = [1.0_f64, 2.0, 3.0];
//! let y = [1.0_f64, 1.0];
//! let z = convolve_1d::<f64>(ConvMode::Auto, &x, &y).unwrap();
//! // Convolution length is x.len() + y.len() - 1 = 4.
//! ```

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use onemkl_sys::{self as sys, VSLConvTaskPtr, VSLCorrTaskPtr};

use crate::error::{Error, Result};

/// Algorithm used by a convolution task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ConvMode {
    /// Let MKL pick between direct and FFT.
    #[default]
    Auto,
    /// Force direct evaluation (good for short inputs).
    Direct,
    /// Force FFT-based evaluation (good for long inputs).
    Fft,
}

impl ConvMode {
    #[inline]
    fn as_int(self) -> c_int {
        let v = match self {
            Self::Auto => sys::VSL_CONV_MODE_AUTO,
            Self::Direct => sys::VSL_CONV_MODE_DIRECT,
            Self::Fft => sys::VSL_CONV_MODE_FFT,
        };
        v as c_int
    }
}

/// Algorithm used by a correlation task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CorrMode {
    /// Let MKL pick between direct and FFT.
    #[default]
    Auto,
    /// Force direct evaluation.
    Direct,
    /// Force FFT-based evaluation.
    Fft,
}

impl CorrMode {
    #[inline]
    fn as_int(self) -> c_int {
        let v = match self {
            Self::Auto => sys::VSL_CORR_MODE_AUTO,
            Self::Direct => sys::VSL_CORR_MODE_DIRECT,
            Self::Fft => sys::VSL_CORR_MODE_FFT,
        };
        v as c_int
    }
}

/// Real scalar types supported by 1-D convolution / correlation tasks.
#[allow(missing_docs)]
pub trait ConvCorrScalar: Copy + 'static {
    unsafe fn conv_new_task_1d(
        task: *mut VSLConvTaskPtr,
        mode: c_int,
        xshape: c_int,
        yshape: c_int,
        zshape: c_int,
    ) -> c_int;

    unsafe fn corr_new_task_1d(
        task: *mut VSLCorrTaskPtr,
        mode: c_int,
        xshape: c_int,
        yshape: c_int,
        zshape: c_int,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    unsafe fn conv_exec_1d(
        task: VSLConvTaskPtr,
        x: *const Self,
        xstride: c_int,
        y: *const Self,
        ystride: c_int,
        z: *mut Self,
        zstride: c_int,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    unsafe fn corr_exec_1d(
        task: VSLCorrTaskPtr,
        x: *const Self,
        xstride: c_int,
        y: *const Self,
        ystride: c_int,
        z: *mut Self,
        zstride: c_int,
    ) -> c_int;
}

macro_rules! impl_conv_corr {
    ($ty:ty,
        conv_new=$conv_new:ident, corr_new=$corr_new:ident,
        conv_exec=$conv_exec:ident, corr_exec=$corr_exec:ident
    ) => {
        impl ConvCorrScalar for $ty {
            unsafe fn conv_new_task_1d(
                task: *mut VSLConvTaskPtr,
                mode: c_int,
                xshape: c_int,
                yshape: c_int,
                zshape: c_int,
            ) -> c_int {
                unsafe { sys::$conv_new(task, mode, xshape, yshape, zshape) }
            }
            unsafe fn corr_new_task_1d(
                task: *mut VSLCorrTaskPtr,
                mode: c_int,
                xshape: c_int,
                yshape: c_int,
                zshape: c_int,
            ) -> c_int {
                unsafe { sys::$corr_new(task, mode, xshape, yshape, zshape) }
            }
            unsafe fn conv_exec_1d(
                task: VSLConvTaskPtr,
                x: *const Self, xstride: c_int,
                y: *const Self, ystride: c_int,
                z: *mut Self, zstride: c_int,
            ) -> c_int {
                unsafe { sys::$conv_exec(task, x, xstride, y, ystride, z, zstride) }
            }
            unsafe fn corr_exec_1d(
                task: VSLCorrTaskPtr,
                x: *const Self, xstride: c_int,
                y: *const Self, ystride: c_int,
                z: *mut Self, zstride: c_int,
            ) -> c_int {
                unsafe { sys::$corr_exec(task, x, xstride, y, ystride, z, zstride) }
            }
        }
    };
}

impl_conv_corr!(f32,
    conv_new=vslsConvNewTask1D, corr_new=vslsCorrNewTask1D,
    conv_exec=vslsConvExec1D, corr_exec=vslsCorrExec1D);
impl_conv_corr!(f64,
    conv_new=vsldConvNewTask1D, corr_new=vsldCorrNewTask1D,
    conv_exec=vsldConvExec1D, corr_exec=vsldCorrExec1D);

/// Owned 1-D convolution task. Can be reused across multiple
/// [`execute`](Self::execute) calls.
pub struct Conv1d<T: ConvCorrScalar> {
    task: VSLConvTaskPtr,
    xshape: usize,
    yshape: usize,
    zshape: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: ConvCorrScalar + Send> Send for Conv1d<T> {}

impl<T: ConvCorrScalar> Conv1d<T> {
    /// Build a new convolution task for inputs of length `xshape` /
    /// `yshape` and an output buffer of length `zshape`. The natural
    /// full-convolution length is `xshape + yshape - 1`.
    pub fn new(mode: ConvMode, xshape: usize, yshape: usize, zshape: usize) -> Result<Self> {
        let xs: c_int = xshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = yshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut task: VSLConvTaskPtr = ptr::null_mut();
        let status = unsafe { T::conv_new_task_1d(&mut task, mode.as_int(), xs, ys, zs) };
        check_vsl(status)?;
        Ok(Self {
            task,
            xshape,
            yshape,
            zshape,
            _marker: PhantomData,
        })
    }

    /// Execute the task: `z = x * y` (convolution).
    /// All three buffers may be strided; pass `xstride = ystride =
    /// zstride = 1` for the dense case.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        x: &[T],
        xstride: usize,
        y: &[T],
        ystride: usize,
        z: &mut [T],
        zstride: usize,
    ) -> Result<()> {
        if x.len() < (self.xshape - 1) * xstride + 1 {
            return Err(Error::InvalidArgument("x is too short for xstride * xshape"));
        }
        if y.len() < (self.yshape - 1) * ystride + 1 {
            return Err(Error::InvalidArgument("y is too short for ystride * yshape"));
        }
        if z.len() < (self.zshape - 1) * zstride + 1 {
            return Err(Error::InvalidArgument("z is too short for zstride * zshape"));
        }
        let xs: c_int = xstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = ystride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let status = unsafe {
            T::conv_exec_1d(self.task, x.as_ptr(), xs, y.as_ptr(), ys, z.as_mut_ptr(), zs)
        };
        check_vsl(status)
    }
}

impl<T: ConvCorrScalar> Drop for Conv1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::vslConvDeleteTask(&mut self.task);
            }
        }
    }
}

/// Owned 1-D correlation task. Reusable across [`execute`](Self::execute)
/// calls.
pub struct Corr1d<T: ConvCorrScalar> {
    task: VSLCorrTaskPtr,
    xshape: usize,
    yshape: usize,
    zshape: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: ConvCorrScalar + Send> Send for Corr1d<T> {}

impl<T: ConvCorrScalar> Corr1d<T> {
    /// Build a new correlation task. See [`Conv1d::new`] for shape
    /// conventions — they apply equally here.
    pub fn new(mode: CorrMode, xshape: usize, yshape: usize, zshape: usize) -> Result<Self> {
        let xs: c_int = xshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = yshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut task: VSLCorrTaskPtr = ptr::null_mut();
        let status = unsafe { T::corr_new_task_1d(&mut task, mode.as_int(), xs, ys, zs) };
        check_vsl(status)?;
        Ok(Self {
            task,
            xshape,
            yshape,
            zshape,
            _marker: PhantomData,
        })
    }

    /// Execute the task: `z = x ⋆ y` (cross-correlation).
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        x: &[T],
        xstride: usize,
        y: &[T],
        ystride: usize,
        z: &mut [T],
        zstride: usize,
    ) -> Result<()> {
        if x.len() < (self.xshape - 1) * xstride + 1 {
            return Err(Error::InvalidArgument("x is too short for xstride * xshape"));
        }
        if y.len() < (self.yshape - 1) * ystride + 1 {
            return Err(Error::InvalidArgument("y is too short for ystride * yshape"));
        }
        if z.len() < (self.zshape - 1) * zstride + 1 {
            return Err(Error::InvalidArgument("z is too short for zstride * zshape"));
        }
        let xs: c_int = xstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = ystride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let status = unsafe {
            T::corr_exec_1d(self.task, x.as_ptr(), xs, y.as_ptr(), ys, z.as_mut_ptr(), zs)
        };
        check_vsl(status)
    }
}

impl<T: ConvCorrScalar> Drop for Corr1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::vslCorrDeleteTask(&mut self.task);
            }
        }
    }
}

/// One-shot 1-D convolution: `z = x * y`. Output length is
/// `x.len() + y.len() - 1`.
pub fn convolve_1d<T: ConvCorrScalar + Default>(
    mode: ConvMode,
    x: &[T],
    y: &[T],
) -> Result<Vec<T>> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::InvalidArgument("x and y must each be non-empty"));
    }
    let zshape = x.len() + y.len() - 1;
    let mut task = Conv1d::<T>::new(mode, x.len(), y.len(), zshape)?;
    let mut z: Vec<T> = (0..zshape).map(|_| T::default()).collect();
    task.execute(x, 1, y, 1, &mut z, 1)?;
    Ok(z)
}

/// One-shot 1-D cross-correlation: `z = x ⋆ y`. Output length is
/// `x.len() + y.len() - 1`.
pub fn correlate_1d<T: ConvCorrScalar + Default>(
    mode: CorrMode,
    x: &[T],
    y: &[T],
) -> Result<Vec<T>> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::InvalidArgument("x and y must each be non-empty"));
    }
    let zshape = x.len() + y.len() - 1;
    let mut task = Corr1d::<T>::new(mode, x.len(), y.len(), zshape)?;
    let mut z: Vec<T> = (0..zshape).map(|_| T::default()).collect();
    task.execute(x, 1, y, 1, &mut z, 1)?;
    Ok(z)
}

#[inline]
fn check_vsl(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::PardisoStatus(status))
    }
}
