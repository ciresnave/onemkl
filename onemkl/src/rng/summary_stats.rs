//! Summary statistics — `vsl?SS*` task family.
//!
//! Computes per-variable statistics (mean, variance, min, max, sum)
//! over a `(p × n)` data matrix where `p` is the number of variables
//! and `n` is the number of observations.
//!
//! Data is row-major in the natural sense: variable `i`'s
//! observations live at `data[i * n + 0 .. i * n + n]`.
//!
//! ```no_run
//! use onemkl::rng::summary_stats::SummaryStats;
//!
//! // 1 variable, 5 observations.
//! let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
//! let mut ss = SummaryStats::<f64>::new(&data, 1, 5).unwrap();
//! let mean = ss.mean().unwrap();
//! assert!((mean[0] - 3.0).abs() < 1e-12);
//! ```

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use onemkl_sys::{self as sys, VSLSSTaskPtr};

use crate::error::{Error, Result};

/// Real scalar types supported by the summary-statistics task family.
#[allow(missing_docs)]
pub trait SsScalar: Copy + Default + 'static {
    #[allow(clippy::too_many_arguments)]
    unsafe fn ss_new_task(
        task: *mut VSLSSTaskPtr,
        p: *const c_int,
        n: *const c_int,
        xstorage: *const c_int,
        x: *const Self,
        w: *const Self,
        indices: *const c_int,
    ) -> c_int;

    unsafe fn ss_edit_task(
        task: VSLSSTaskPtr,
        parameter: c_int,
        address: *const Self,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    unsafe fn ss_edit_moments(
        task: VSLSSTaskPtr,
        mean: *mut Self,
        r2m: *mut Self,
        r3m: *mut Self,
        r4m: *mut Self,
        c2m: *mut Self,
        c3m: *mut Self,
        c4m: *mut Self,
    ) -> c_int;

    unsafe fn ss_compute(
        task: VSLSSTaskPtr,
        estimates: u64,
        method: c_int,
    ) -> c_int;
}

macro_rules! impl_ss_scalar {
    ($ty:ty,
        new=$new:ident, edit=$edit:ident,
        moments=$moments:ident, compute=$compute:ident
    ) => {
        impl SsScalar for $ty {
            unsafe fn ss_new_task(
                task: *mut VSLSSTaskPtr,
                p: *const c_int,
                n: *const c_int,
                xstorage: *const c_int,
                x: *const Self,
                w: *const Self,
                indices: *const c_int,
            ) -> c_int {
                unsafe { sys::$new(task, p, n, xstorage, x, w, indices) }
            }
            unsafe fn ss_edit_task(
                task: VSLSSTaskPtr,
                parameter: c_int,
                address: *const Self,
            ) -> c_int {
                unsafe { sys::$edit(task, parameter, address) }
            }
            unsafe fn ss_edit_moments(
                task: VSLSSTaskPtr,
                mean: *mut Self,
                r2m: *mut Self,
                r3m: *mut Self,
                r4m: *mut Self,
                c2m: *mut Self,
                c3m: *mut Self,
                c4m: *mut Self,
            ) -> c_int {
                unsafe { sys::$moments(task, mean, r2m, r3m, r4m, c2m, c3m, c4m) }
            }
            unsafe fn ss_compute(
                task: VSLSSTaskPtr,
                estimates: u64,
                method: c_int,
            ) -> c_int {
                unsafe { sys::$compute(task, estimates, method) }
            }
        }
    };
}

impl_ss_scalar!(f32,
    new=vslsSSNewTask, edit=vslsSSEditTask,
    moments=vslsSSEditMoments, compute=vslsSSCompute);
impl_ss_scalar!(f64,
    new=vsldSSNewTask, edit=vsldSSEditTask,
    moments=vsldSSEditMoments, compute=vsldSSCompute);

/// Owned summary-statistics task. Each compute method registers output
/// buffers internally and runs `vsl?SSCompute`.
pub struct SummaryStats<'data, T: SsScalar> {
    task: VSLSSTaskPtr,
    p: usize,
    _n: usize,
    // MKL stores raw pointers to these inside the task, so they must
    // outlive every Edit/Compute call. Boxed so the addresses are
    // stable across moves of `Self`.
    _p_holder: Box<c_int>,
    _n_holder: Box<c_int>,
    _xstorage_holder: Box<c_int>,
    _data: &'data [T],
    _marker: PhantomData<T>,
}

unsafe impl<T: SsScalar + Send> Send for SummaryStats<'_, T> {}

impl<'data, T: SsScalar> SummaryStats<'data, T> {
    /// Build a task over `num_variables × num_observations` data laid
    /// out so variable `i` occupies `data[i * num_observations ..]`.
    pub fn new(
        data: &'data [T],
        num_variables: usize,
        num_observations: usize,
    ) -> Result<Self> {
        if data.len() < num_variables * num_observations {
            return Err(Error::InvalidArgument(
                "data buffer is smaller than num_variables * num_observations",
            ));
        }
        let p_box: Box<c_int> = Box::new(
            num_variables.try_into().map_err(|_| Error::DimensionOverflow)?,
        );
        let n_box: Box<c_int> = Box::new(
            num_observations.try_into().map_err(|_| Error::DimensionOverflow)?,
        );
        let xstorage_box: Box<c_int> = Box::new(sys::VSL_SS_MATRIX_STORAGE_ROWS as c_int);
        let mut task: VSLSSTaskPtr = ptr::null_mut();
        let status = unsafe {
            T::ss_new_task(
                &mut task,
                &*p_box,
                &*n_box,
                &*xstorage_box,
                data.as_ptr(),
                ptr::null(),
                ptr::null(),
            )
        };
        check_vsl(status)?;
        Ok(Self {
            task,
            p: num_variables,
            _n: num_observations,
            _p_holder: p_box,
            _n_holder: n_box,
            _xstorage_holder: xstorage_box,
            _data: data,
            _marker: PhantomData,
        })
    }

    /// Per-variable mean. Length is `num_variables`.
    pub fn mean(&mut self) -> Result<Vec<T>> {
        let mut mean = vec![T::default(); self.p];
        let status = unsafe {
            T::ss_edit_task(self.task, sys::VSL_SS_ED_MEAN as c_int, mean.as_mut_ptr())
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_MEAN as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(mean)
    }

    /// Per-variable variance (2nd central moment). Length is
    /// `num_variables`. Computing variance also requires mean and the
    /// raw 2nd moment to be registered; the wrapper allocates
    /// temporary buffers for both and discards them.
    pub fn variance(&mut self) -> Result<Vec<T>> {
        let mut mean = vec![T::default(); self.p];
        let mut r2m = vec![T::default(); self.p];
        let mut c2m = vec![T::default(); self.p];
        let status = unsafe {
            T::ss_edit_task(self.task, sys::VSL_SS_ED_MEAN as c_int, mean.as_mut_ptr())
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_edit_task(self.task, sys::VSL_SS_ED_2R_MOM as c_int, r2m.as_mut_ptr())
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_edit_task(self.task, sys::VSL_SS_ED_2C_MOM as c_int, c2m.as_mut_ptr())
        };
        check_vsl(status)?;
        let estimates = (sys::VSL_SS_MEAN | sys::VSL_SS_2R_MOM | sys::VSL_SS_2C_MOM) as u64;
        let status = unsafe {
            T::ss_compute(self.task, estimates, sys::VSL_SS_METHOD_FAST as c_int)
        };
        check_vsl(status)?;
        Ok(c2m)
    }

    /// Per-variable minimum. Length is `num_variables`. The MKL min
    /// estimator must be seeded from an observation, so the wrapper
    /// initializes the result with the first observation of each
    /// variable before computing.
    pub fn min(&mut self) -> Result<Vec<T>> {
        let mut min = self.seed_from_first_observation();
        let status = unsafe {
            T::ss_edit_task(self.task, sys::VSL_SS_ED_MIN as c_int, min.as_mut_ptr())
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_MIN as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(min)
    }

    /// Per-variable maximum. See [`min`](Self::min) for the seeding
    /// note.
    pub fn max(&mut self) -> Result<Vec<T>> {
        let mut max = self.seed_from_first_observation();
        let status = unsafe {
            T::ss_edit_task(self.task, sys::VSL_SS_ED_MAX as c_int, max.as_mut_ptr())
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_MAX as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(max)
    }

    /// Build a length-`p` vector seeded from the first observation of
    /// each variable. Used to initialize min/max estimators.
    fn seed_from_first_observation(&self) -> Vec<T> {
        let n = self._n;
        (0..self.p).map(|i| self._data[i * n]).collect()
    }

    /// Per-variable sum. Length is `num_variables`.
    pub fn sum(&mut self) -> Result<Vec<T>> {
        let mut sum = vec![T::default(); self.p];
        let status = unsafe {
            T::ss_edit_task(self.task, sys::VSL_SS_ED_SUM as c_int, sum.as_mut_ptr())
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_SUM as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(sum)
    }
}

impl<T: SsScalar> Drop for SummaryStats<'_, T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::vslSSDeleteTask(&mut self.task);
            }
        }
    }
}

#[inline]
fn check_vsl(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::PardisoStatus(status))
    }
}
