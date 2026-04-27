//! PDE support — trigonometric transforms (DCT / DST) used in
//! spectral PDE solvers and signal processing.
//!
//! oneMKL exposes six trig-transform variants (sine, cosine, plus
//! four staggered combinations). The flow is:
//!
//! 1. [`TrigTransform::new`] — initialize and commit a plan for a
//!    given `n`-point transform.
//! 2. [`TrigTransform::forward`] / [`backward`](TrigTransform::backward)
//!    — run the transform on an in-place buffer.
//! 3. `Drop` releases the descriptor.
//!
//! Both single (`f32`) and double (`f64`) precision are supported via
//! the [`TrigTransformScalar`] trait. The transform is applied
//! in-place to the user-supplied buffer.
//!
//! ```ignore
//! use onemkl::pde::{TrigTransform, TrigTransformType};
//!
//! // 8-point DCT.
//! let mut f = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
//! //              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//! //              n+1 = 9 elements (sine / cosine transforms work on
//! //              n+1-point buffers — the boundary samples).
//! let mut plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, 8).unwrap();
//! plan.forward(&mut f).unwrap();
//! ```

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use onemkl_sys::{self as sys, DFTI_DESCRIPTOR_HANDLE};

use crate::error::{Error, Result};

/// Variant of the trigonometric transform. Each maps to one of
/// MKL's `MKL_*_TRANSFORM` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrigTransformType {
    /// Sine transform (DST-I).
    Sine,
    /// Cosine transform (DCT-I).
    Cosine,
    /// Staggered cosine transform (DCT-II / DCT-III pair).
    StaggeredCosine,
    /// Staggered sine transform (DST-II / DST-III pair).
    StaggeredSine,
    /// Twice-staggered cosine transform.
    Staggered2Cosine,
    /// Twice-staggered sine transform.
    Staggered2Sine,
}

impl TrigTransformType {
    #[inline]
    fn as_int(self) -> c_int {
        let v = match self {
            Self::Sine => sys::MKL_SINE_TRANSFORM,
            Self::Cosine => sys::MKL_COSINE_TRANSFORM,
            Self::StaggeredCosine => sys::MKL_STAGGERED_COSINE_TRANSFORM,
            Self::StaggeredSine => sys::MKL_STAGGERED_SINE_TRANSFORM,
            Self::Staggered2Cosine => sys::MKL_STAGGERED2_COSINE_TRANSFORM,
            Self::Staggered2Sine => sys::MKL_STAGGERED2_SINE_TRANSFORM,
        };
        v as c_int
    }

    /// Number of input samples expected for an `n`-point transform of
    /// this type. Sine and cosine work on `n + 1` boundary samples;
    /// the staggered variants work on `n` interior samples.
    #[inline]
    #[must_use]
    pub fn buffer_len(self, n: usize) -> usize {
        match self {
            Self::Sine | Self::Cosine => n + 1,
            _ => n,
        }
    }
}

/// Scalar types supported by the trig transform routines.
#[allow(missing_docs)]
pub trait TrigTransformScalar: Copy + Default + 'static {
    unsafe fn init_trig_transform(
        n: *mut c_int,
        tt_type: *mut c_int,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
    unsafe fn commit_trig_transform(
        f: *mut Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
    unsafe fn forward_trig_transform(
        f: *mut Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
    unsafe fn backward_trig_transform(
        f: *mut Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
}

impl TrigTransformScalar for f32 {
    unsafe fn init_trig_transform(
        n: *mut c_int, tt_type: *mut c_int,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_init_trig_transform(n, tt_type, ipar, dpar, stat) }
    }
    unsafe fn commit_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_commit_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn forward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_forward_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn backward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_backward_trig_transform(f, handle, ipar, dpar, stat) }
    }
}

impl TrigTransformScalar for f64 {
    unsafe fn init_trig_transform(
        n: *mut c_int, tt_type: *mut c_int,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_init_trig_transform(n, tt_type, ipar, dpar, stat) }
    }
    unsafe fn commit_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_commit_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn forward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_forward_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn backward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_backward_trig_transform(f, handle, ipar, dpar, stat) }
    }
}

#[inline]
fn check_tr(stat: c_int) -> Result<()> {
    // The trig transform routines return 0 on success (standard
    // convention), not TR_SUCCESS = 1501 (which is the optimization
    // routines' convention).
    if stat == 0 {
        Ok(())
    } else {
        Err(Error::LapackComputationFailure { info: stat })
    }
}

/// An owned trigonometric transform plan.
pub struct TrigTransform<T: TrigTransformScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    ipar: Vec<c_int>,
    dpar: Vec<T>,
    n: c_int,
    transform_type: TrigTransformType,
    _marker: PhantomData<T>,
}

unsafe impl<T: TrigTransformScalar + Send> Send for TrigTransform<T> {}

impl<T: TrigTransformScalar> TrigTransform<T> {
    /// Build a plan for an `n`-point transform of the given type.
    /// Sine and cosine transforms operate on `n + 1`-element buffers
    /// (boundary samples included); the staggered variants operate
    /// on `n`-element buffers (interior samples). See
    /// [`TrigTransformType::buffer_len`].
    pub fn new(transform_type: TrigTransformType, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidArgument("n must be ≥ 1"));
        }
        let mut n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut tt_int: c_int = transform_type.as_int();
        // Allocate ipar of size 128 and dpar generously sized to fit
        // the largest variant's workspace.
        let mut ipar: Vec<c_int> = vec![0; 128];
        let mut dpar: Vec<T> = vec![T::default(); 5 * n + 2];
        let mut stat: c_int = 0;

        unsafe {
            T::init_trig_transform(
                &mut n_i,
                &mut tt_int,
                ipar.as_mut_ptr(),
                dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)?;

        let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        // commit_trig_transform takes f as input; it inspects the
        // first element to determine alignment / size. We can pass a
        // throwaway buffer here.
        let mut probe: Vec<T> = vec![T::default(); transform_type.buffer_len(n)];
        unsafe {
            T::commit_trig_transform(
                probe.as_mut_ptr(),
                &mut handle,
                ipar.as_mut_ptr(),
                dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)?;

        Ok(Self {
            handle,
            ipar,
            dpar,
            n: n_i,
            transform_type,
            _marker: PhantomData,
        })
    }

    /// Length of the buffer this plan expects on each transform call.
    #[inline]
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.transform_type.buffer_len(self.n as usize)
    }

    /// Apply the forward transform in place. `f` must have at least
    /// [`buffer_len`](Self::buffer_len) elements.
    pub fn forward(&mut self, f: &mut [T]) -> Result<()> {
        let need = self.buffer_len();
        if f.len() < need {
            return Err(Error::InvalidArgument(
                "buffer too small for the trig transform",
            ));
        }
        let mut stat: c_int = 0;
        unsafe {
            T::forward_trig_transform(
                f.as_mut_ptr(),
                &mut self.handle,
                self.ipar.as_mut_ptr(),
                self.dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)
    }

    /// Apply the backward / inverse transform in place.
    pub fn backward(&mut self, f: &mut [T]) -> Result<()> {
        let need = self.buffer_len();
        if f.len() < need {
            return Err(Error::InvalidArgument(
                "buffer too small for the trig transform",
            ));
        }
        let mut stat: c_int = 0;
        unsafe {
            T::backward_trig_transform(
                f.as_mut_ptr(),
                &mut self.handle,
                self.ipar.as_mut_ptr(),
                self.dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)
    }
}

impl<T: TrigTransformScalar> Drop for TrigTransform<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut stat: c_int = 0;
            unsafe {
                sys::free_trig_transform(
                    &mut self.handle,
                    self.ipar.as_mut_ptr(),
                    &mut stat,
                );
            }
        }
    }
}
