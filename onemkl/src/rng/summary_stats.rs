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

    #[allow(clippy::too_many_arguments)]
    unsafe fn ss_edit_quantiles(
        task: VSLSSTaskPtr,
        quant_order_n: *const c_int,
        quant_orders: *const Self,
        quants_out: *mut Self,
        order_stats_out: *mut Self,
        order_stats_storage: *const c_int,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    unsafe fn ss_edit_stream_quantiles(
        task: VSLSSTaskPtr,
        quant_order_n: *const c_int,
        quant_orders: *const Self,
        quants_out: *mut Self,
        params_n: *const c_int,
        params: *const Self,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    unsafe fn ss_edit_robust_covariance(
        task: VSLSSTaskPtr,
        cov_storage: *const c_int,
        n_params: *const c_int,
        params: *const Self,
        mean_out: *mut Self,
        cov_out: *mut Self,
    ) -> c_int;

    unsafe fn ss_edit_outliers_detection(
        task: VSLSSTaskPtr,
        n_params: *const c_int,
        params: *const Self,
        weights_out: *mut Self,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    unsafe fn ss_edit_missing_values(
        task: VSLSSTaskPtr,
        n_params: *const c_int,
        params: *const Self,
        init_estimates_n: *const c_int,
        init_estimates: *const Self,
        prior_n: *const c_int,
        prior: *const Self,
        simul_vals_n: *const c_int,
        simul_vals_out: *mut Self,
        estimates_n: *const c_int,
        estimates_out: *mut Self,
    ) -> c_int;
}

macro_rules! impl_ss_scalar {
    ($ty:ty,
        new=$new:ident, edit=$edit:ident,
        moments=$moments:ident, compute=$compute:ident,
        quantiles=$quantiles:ident, stream_quantiles=$stream_quantiles:ident,
        robust_cov=$robust_cov:ident, outliers=$outliers:ident,
        missing=$missing:ident,
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
            unsafe fn ss_edit_quantiles(
                task: VSLSSTaskPtr,
                quant_order_n: *const c_int,
                quant_orders: *const Self,
                quants_out: *mut Self,
                order_stats_out: *mut Self,
                order_stats_storage: *const c_int,
            ) -> c_int {
                unsafe {
                    sys::$quantiles(
                        task,
                        quant_order_n,
                        quant_orders,
                        quants_out,
                        order_stats_out,
                        order_stats_storage,
                    )
                }
            }
            unsafe fn ss_edit_stream_quantiles(
                task: VSLSSTaskPtr,
                quant_order_n: *const c_int,
                quant_orders: *const Self,
                quants_out: *mut Self,
                params_n: *const c_int,
                params: *const Self,
            ) -> c_int {
                unsafe {
                    sys::$stream_quantiles(
                        task,
                        quant_order_n,
                        quant_orders,
                        quants_out,
                        params_n,
                        params,
                    )
                }
            }
            unsafe fn ss_edit_robust_covariance(
                task: VSLSSTaskPtr,
                cov_storage: *const c_int,
                n_params: *const c_int,
                params: *const Self,
                mean_out: *mut Self,
                cov_out: *mut Self,
            ) -> c_int {
                unsafe {
                    sys::$robust_cov(
                        task, cov_storage, n_params, params, mean_out, cov_out,
                    )
                }
            }
            unsafe fn ss_edit_outliers_detection(
                task: VSLSSTaskPtr,
                n_params: *const c_int,
                params: *const Self,
                weights_out: *mut Self,
            ) -> c_int {
                unsafe { sys::$outliers(task, n_params, params, weights_out) }
            }
            unsafe fn ss_edit_missing_values(
                task: VSLSSTaskPtr,
                n_params: *const c_int,
                params: *const Self,
                init_estimates_n: *const c_int,
                init_estimates: *const Self,
                prior_n: *const c_int,
                prior: *const Self,
                simul_vals_n: *const c_int,
                simul_vals_out: *mut Self,
                estimates_n: *const c_int,
                estimates_out: *mut Self,
            ) -> c_int {
                unsafe {
                    sys::$missing(
                        task,
                        n_params,
                        params,
                        init_estimates_n,
                        init_estimates,
                        prior_n,
                        prior,
                        simul_vals_n,
                        simul_vals_out,
                        estimates_n,
                        estimates_out,
                    )
                }
            }
        }
    };
}

impl_ss_scalar!(f32,
    new=vslsSSNewTask, edit=vslsSSEditTask,
    moments=vslsSSEditMoments, compute=vslsSSCompute,
    quantiles=vslsSSEditQuantiles, stream_quantiles=vslsSSEditStreamQuantiles,
    robust_cov=vslsSSEditRobustCovariance, outliers=vslsSSEditOutliersDetection,
    missing=vslsSSEditMissingValues,
);
impl_ss_scalar!(f64,
    new=vsldSSNewTask, edit=vsldSSEditTask,
    moments=vsldSSEditMoments, compute=vsldSSCompute,
    quantiles=vsldSSEditQuantiles, stream_quantiles=vsldSSEditStreamQuantiles,
    robust_cov=vsldSSEditRobustCovariance, outliers=vsldSSEditOutliersDetection,
    missing=vsldSSEditMissingValues,
);

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

    /// Per-variable quantiles at the requested `orders` (e.g.
    /// `[0.25, 0.5, 0.75]` for Q1 / median / Q3). Returns a flat
    /// vector laid out as `num_variables × orders.len()`, where
    /// `result[i * orders.len() + j]` is variable `i`'s `orders[j]`
    /// quantile.
    ///
    /// All order values must lie in `(0, 1)`.
    pub fn quantiles(&mut self, orders: &[T]) -> Result<Vec<T>>
    where
        T: PartialOrd,
    {
        if orders.is_empty() {
            return Err(Error::InvalidArgument("orders must be non-empty"));
        }
        let order_n: c_int = orders
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out = vec![T::default(); self.p * orders.len()];
        let status = unsafe {
            T::ss_edit_quantiles(
                self.task,
                &order_n,
                orders.as_ptr(),
                out.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_QUANTS as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(out)
    }

    /// Streaming-quantile estimator (Zhang-Wang). Suitable for
    /// online data where you can't hold the full sample but want an
    /// approximate quantile. `params` controls the estimator
    /// (typically `[buffer_size_per_quantile]`). Returns the same
    /// `num_variables × orders.len()` layout as [`Self::quantiles`].
    pub fn stream_quantiles(
        &mut self,
        orders: &[T],
        params: &[T],
    ) -> Result<Vec<T>> {
        if orders.is_empty() {
            return Err(Error::InvalidArgument("orders must be non-empty"));
        }
        let order_n: c_int = orders
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let params_n: c_int = params
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out = vec![T::default(); self.p * orders.len()];
        let status = unsafe {
            T::ss_edit_stream_quantiles(
                self.task,
                &order_n,
                orders.as_ptr(),
                out.as_mut_ptr(),
                &params_n,
                params.as_ptr(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_STREAM_QUANTS as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(out)
    }

    /// Robust covariance + mean via Tukey's bi-weight S-estimator
    /// (`VSL_SS_METHOD_TBS` in MKL).
    ///
    /// Returns `(mean, cov)` where `cov` is the `p × p` covariance
    /// stored in column-major (full) layout. `params` is the TBS
    /// configuration — pass an empty slice to accept MKL's defaults,
    /// or supply `[breakdown_point]` (typical value `0.5`) to control
    /// the maximum fraction of outliers the estimator tolerates.
    ///
    /// Note: MKL exposes BACON only for *outlier weights*
    /// (see [`Self::outliers_bacon`]); for a full robust covariance
    /// estimate the supported algorithm is TBS.
    pub fn robust_covariance_tbs(
        &mut self,
        params: &[T],
    ) -> Result<(Vec<T>, Vec<T>)> {
        let cov_storage: c_int = sys::VSL_SS_MATRIX_STORAGE_FULL as c_int;
        let n_params: c_int = params
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut mean = vec![T::default(); self.p];
        let mut cov = vec![T::default(); self.p * self.p];
        let status = unsafe {
            T::ss_edit_robust_covariance(
                self.task,
                &cov_storage,
                &n_params,
                if params.is_empty() {
                    ptr::null()
                } else {
                    params.as_ptr()
                },
                mean.as_mut_ptr(),
                cov.as_mut_ptr(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_ROBUST_COV as u64,
                sys::VSL_SS_METHOD_TBS as c_int,
            )
        };
        check_vsl(status)?;
        Ok((mean, cov))
    }

    /// Per-observation outlier weights via BACON (Blocked Adaptive
    /// Computationally-efficient Outlier Nominators). Returns a
    /// vector of length `num_observations`: `1.0` for inliers,
    /// `0.0` for flagged outliers.
    ///
    /// `params` is BACON's 3-element configuration:
    ///
    /// - `params[0]` = initialization method
    ///   (`VSL_SS_METHOD_BACON_MEDIAN_INIT` or
    ///   `VSL_SS_METHOD_BACON_MAHALANOBIS_INIT` from `onemkl-sys`),
    /// - `params[1]` = significance level (e.g. `0.05`),
    /// - `params[2]` = initial subset size as fraction of `n`.
    ///
    /// Use [`SummaryStats::<f64>::bacon_default_params`] for sensible
    /// defaults.
    pub fn outliers_bacon(&mut self, params: &[T; 3]) -> Result<Vec<T>> {
        let n_params: c_int = 3;
        let mut weights = vec![T::default(); self._n];
        let status = unsafe {
            T::ss_edit_outliers_detection(
                self.task,
                &n_params,
                params.as_ptr(),
                weights.as_mut_ptr(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_OUTLIERS as u64,
                sys::VSL_SS_METHOD_BACON as c_int,
            )
        };
        check_vsl(status)?;
        Ok(weights)
    }

    /// Imputed mean + covariance for data containing missing values
    /// (marked as `NaN`). Runs MKL's multiple-imputation EM
    /// estimator with the supplied `params` (length-5 vector;
    /// `VSL_SS_MI_PARAMS_SIZE` in `onemkl-sys`).
    ///
    /// Returns `(mean, cov)`. The simulated imputations themselves
    /// are written into the *original* data buffer in-place where
    /// supported by MKL — pass a mutable copy if you need to retain
    /// the original.
    ///
    /// Most callers should use
    /// [`SummaryStats::<f64>::impute_default_params`] which supplies
    /// a sensible default `params` array.
    pub fn impute_missing(&mut self, params: &[T; 5]) -> Result<(Vec<T>, Vec<T>)> {
        let n_params: c_int = 5;
        let init_n: c_int = 0;
        let prior_n: c_int = 0;
        let simul_n: c_int = 0;
        let estimates_n: c_int = (self.p + self.p * self.p) as c_int;
        let mut estimates = vec![T::default(); self.p + self.p * self.p];
        let status = unsafe {
            T::ss_edit_missing_values(
                self.task,
                &n_params,
                params.as_ptr(),
                &init_n,
                ptr::null(),
                &prior_n,
                ptr::null(),
                &simul_n,
                ptr::null_mut(),
                &estimates_n,
                estimates.as_mut_ptr(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_MISSING_VALS as u64,
                sys::VSL_SS_METHOD_MI as c_int,
            )
        };
        check_vsl(status)?;
        let mean = estimates[..self.p].to_vec();
        let cov = estimates[self.p..].to_vec();
        Ok((mean, cov))
    }
}

impl SummaryStats<'_, f64> {
    /// Default BACON parameters: median initialization, α = 0.05,
    /// β = 0.05.
    #[inline]
    #[must_use]
    pub fn bacon_default_params() -> [f64; 3] {
        [
            sys::VSL_SS_METHOD_BACON_MEDIAN_INIT as f64,
            0.05,
            0.05,
        ]
    }

    /// Default multiple-imputation parameters: copy input, 25 EM
    /// iterations, 1e-4 stopping tolerance.
    #[inline]
    #[must_use]
    pub fn impute_default_params() -> [f64; 5] {
        [
            sys::VSL_SS_METHOD_MI as f64, // method id
            25.0,                          // max iterations
            1.0e-4,                        // tolerance
            1.0,                           // copy data flag
            0.0,                           // reserved
        ]
    }
}

impl SummaryStats<'_, f32> {
    /// `f32` counterpart of [`SummaryStats::<f64>::bacon_default_params`].
    #[inline]
    #[must_use]
    pub fn bacon_default_params() -> [f32; 3] {
        [
            sys::VSL_SS_METHOD_BACON_MEDIAN_INIT as f32,
            0.05,
            0.05,
        ]
    }

    /// `f32` counterpart of [`SummaryStats::<f64>::impute_default_params`].
    #[inline]
    #[must_use]
    pub fn impute_default_params() -> [f32; 5] {
        [
            sys::VSL_SS_METHOD_MI as f32,
            25.0,
            1.0e-4,
            1.0,
            0.0,
        ]
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
