//! Data fitting — 1-D spline interpolation and integration.
//!
//! Wraps oneMKL's Data Fitting `df*` task interface in an owned
//! [`CubicSpline1d`] type that bundles the underlying `DFTaskPtr`
//! handle and the spline coefficient storage. The task is freed on
//! drop.
//!
//! Cubic spline subtypes covered: [`natural`](CubicSpline1d::natural),
//! [`bessel`](CubicSpline1d::bessel), [`akima`](CubicSpline1d::akima),
//! and [`hermite`](CubicSpline1d::hermite) (which takes user-supplied
//! first derivatives at each knot).

use core::ffi::c_int;
use core::ptr;

use onemkl_sys::{self as sys, DFTaskPtr};

use crate::error::{Error, Result};

// =====================================================================
// Trait wiring
// =====================================================================

/// Real scalar types supported by the data-fitting interface.
#[allow(missing_docs)]
pub trait DataFittingScalar: Copy + 'static {
    #[allow(clippy::too_many_arguments)]
    unsafe fn df_new_task_1d(
        task: *mut DFTaskPtr, nx: c_int, x: *const Self, xhint: c_int,
        ny: c_int, y: *const Self, yhint: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    unsafe fn df_edit_pp_spline_1d(
        task: DFTaskPtr, s_order: c_int, s_type: c_int, bc_type: c_int,
        bc: *const Self, ic_type: c_int, ic: *const Self, scoeff: *const Self,
        scoeffhint: c_int,
    ) -> c_int;
    unsafe fn df_construct_1d(
        task: DFTaskPtr, s_format: c_int, method: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    unsafe fn df_interpolate_1d(
        task: DFTaskPtr,
        estimation_type: c_int,
        method: c_int,
        nsite: c_int,
        site: *const Self,
        sitehint: c_int,
        ndorder: c_int,
        dorder: *const c_int,
        datahint: *const Self,
        r: *mut Self,
        rhint: c_int,
        cell: *mut c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    unsafe fn df_integrate_1d(
        task: DFTaskPtr,
        method: c_int,
        nlim: c_int,
        llim: *const Self,
        llimhint: c_int,
        rlim: *const Self,
        rlimhint: c_int,
        ldatahint: *const Self,
        rdatahint: *const Self,
        r: *mut Self,
        rhint: c_int,
    ) -> c_int;
}

macro_rules! impl_data_fitting_real {
    ($ty:ty,
        new_task=$new:ident, edit_spline=$edit:ident,
        construct=$ctor:ident, interpolate=$interp:ident,
        integrate=$integ:ident,
    ) => {
        impl DataFittingScalar for $ty {
            unsafe fn df_new_task_1d(
                task: *mut DFTaskPtr, nx: c_int, x: *const Self, xhint: c_int,
                ny: c_int, y: *const Self, yhint: c_int,
            ) -> c_int {
                unsafe { sys::$new(task, nx, x, xhint, ny, y, yhint) }
            }
            unsafe fn df_edit_pp_spline_1d(
                task: DFTaskPtr, s_order: c_int, s_type: c_int, bc_type: c_int,
                bc: *const Self, ic_type: c_int, ic: *const Self,
                scoeff: *const Self, scoeffhint: c_int,
            ) -> c_int {
                unsafe {
                    sys::$edit(
                        task, s_order, s_type, bc_type, bc, ic_type, ic, scoeff,
                        scoeffhint,
                    )
                }
            }
            unsafe fn df_construct_1d(
                task: DFTaskPtr, s_format: c_int, method: c_int,
            ) -> c_int {
                unsafe { sys::$ctor(task, s_format, method) }
            }
            unsafe fn df_interpolate_1d(
                task: DFTaskPtr,
                estimation_type: c_int,
                method: c_int,
                nsite: c_int,
                site: *const Self,
                sitehint: c_int,
                ndorder: c_int,
                dorder: *const c_int,
                datahint: *const Self,
                r: *mut Self,
                rhint: c_int,
                cell: *mut c_int,
            ) -> c_int {
                unsafe {
                    sys::$interp(
                        task, estimation_type, method, nsite, site, sitehint, ndorder,
                        dorder, datahint, r, rhint, cell,
                    )
                }
            }
            unsafe fn df_integrate_1d(
                task: DFTaskPtr,
                method: c_int,
                nlim: c_int,
                llim: *const Self,
                llimhint: c_int,
                rlim: *const Self,
                rlimhint: c_int,
                ldatahint: *const Self,
                rdatahint: *const Self,
                r: *mut Self,
                rhint: c_int,
            ) -> c_int {
                unsafe {
                    sys::$integ(
                        task, method, nlim, llim, llimhint, rlim, rlimhint, ldatahint,
                        rdatahint, r, rhint,
                    )
                }
            }
        }
    };
}

impl_data_fitting_real!(
    f32,
    new_task=dfsNewTask1D, edit_spline=dfsEditPPSpline1D,
    construct=dfsConstruct1D, interpolate=dfsInterpolate1D,
    integrate=dfsIntegrate1D,
);
impl_data_fitting_real!(
    f64,
    new_task=dfdNewTask1D, edit_spline=dfdEditPPSpline1D,
    construct=dfdConstruct1D, interpolate=dfdInterpolate1D,
    integrate=dfdIntegrate1D,
);

// Note: oneMKL's data-fitting routines expose only real (`?s` / `?d`)
// variants — there are no complex spline interpolators — so
// [`DataFittingScalar`] is not implemented for `Complex32` / `Complex64`.

// =====================================================================
// CubicSpline1d
// =====================================================================

/// A natural cubic spline interpolant of a real-valued 1-D function.
///
/// The struct owns the spline coefficient storage and the underlying
/// task handle; both are released on drop.
pub struct CubicSpline1d<T: DataFittingScalar> {
    task: DFTaskPtr,
    // Inputs kept alive — MKL holds raw pointers into them while the
    // task is open.
    _x: Box<[T]>,
    _y: Box<[T]>,
    // Hermite splines need user-supplied derivatives at each knot; MKL
    // also keeps a raw pointer into this for the task's lifetime.
    _ic: Option<Box<[T]>>,
    // Coefficient storage. For cubic spline of n knots, MKL writes
    // 4 * (n - 1) coefficients here.
    _coeffs: Box<[T]>,
}

unsafe impl<T: DataFittingScalar + Send> Send for CubicSpline1d<T> {}

impl<T: DataFittingScalar + num_traits::Zero + Default> CubicSpline1d<T> {
    /// Build a natural cubic spline interpolating `(x[i], y[i])`.
    /// `x` must be strictly increasing.
    pub fn natural(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        Self::build(
            x,
            y,
            sys::DF_PP_NATURAL as c_int,
            sys::DF_BC_FREE_END as c_int,
            None,
        )
    }

    /// Build a Bessel cubic spline interpolating `(x[i], y[i])`. The
    /// derivative at each knot is set to the slope of the parabola
    /// through that knot and its two neighbors.
    pub fn bessel(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        Self::build(
            x,
            y,
            sys::DF_PP_BESSEL as c_int,
            sys::DF_BC_NOT_A_KNOT as c_int,
            None,
        )
    }

    /// Build an Akima cubic spline interpolating `(x[i], y[i])`. Akima
    /// splines are more robust to outliers than natural cubics
    /// because the derivative at each knot is a weighted average of
    /// nearby slopes.
    pub fn akima(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        Self::build(
            x,
            y,
            sys::DF_PP_AKIMA as c_int,
            sys::DF_BC_NOT_A_KNOT as c_int,
            None,
        )
    }

    /// Build a Hermite cubic spline interpolating `(x[i], y[i])` with
    /// the supplied first derivatives at each knot. `derivatives` must
    /// have the same length as `x` and `y`.
    pub fn hermite(x: Vec<T>, y: Vec<T>, derivatives: Vec<T>) -> Result<Self> {
        if derivatives.len() != x.len() {
            return Err(Error::InvalidArgument(
                "derivatives must have the same length as x and y",
            ));
        }
        Self::build(
            x,
            y,
            sys::DF_PP_HERMITE as c_int,
            sys::DF_BC_NOT_A_KNOT as c_int,
            Some(derivatives),
        )
    }

    fn build(
        x: Vec<T>,
        y: Vec<T>,
        s_type: c_int,
        bc_type: c_int,
        ic: Option<Vec<T>>,
    ) -> Result<Self> {
        if x.len() != y.len() {
            return Err(Error::InvalidArgument(
                "x and y must have the same length",
            ));
        }
        let n = x.len();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "cubic spline needs at least 2 knots",
            ));
        }

        let x_box = x.into_boxed_slice();
        let y_box = y.into_boxed_slice();
        let ic_box: Option<Box<[T]>> = ic.map(|v| v.into_boxed_slice());
        let coeffs_box: Box<[T]> = (0..(n - 1) * sys::DF_PP_CUBIC as usize)
            .map(|_| T::default())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let nx: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ny: c_int = 1;

        let mut task: DFTaskPtr = ptr::null_mut();
        let status = unsafe {
            T::df_new_task_1d(
                &mut task,
                nx,
                x_box.as_ptr(),
                sys::DF_NON_UNIFORM_PARTITION as c_int,
                ny,
                y_box.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        };
        check_df(status)?;

        let this = Self {
            task,
            _x: x_box,
            _y: y_box,
            _ic: ic_box,
            _coeffs: coeffs_box,
        };

        let (ic_type, ic_ptr) = match this._ic.as_ref() {
            Some(buf) => (sys::DF_IC_1ST_DER as c_int, buf.as_ptr()),
            None => (sys::DF_NO_IC as c_int, ptr::null()),
        };
        let status = unsafe {
            T::df_edit_pp_spline_1d(
                this.task,
                sys::DF_PP_CUBIC as c_int,
                s_type,
                bc_type,
                ptr::null(),
                ic_type,
                ic_ptr,
                this._coeffs.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        };
        check_df(status)?;

        let status = unsafe {
            T::df_construct_1d(
                this.task,
                sys::DF_PP_SPLINE as c_int,
                sys::DF_METHOD_STD as c_int,
            )
        };
        check_df(status)?;
        Ok(this)
    }

    /// Evaluate the spline at the given sites, returning their function
    /// values.
    pub fn interpolate(&self, sites: &[T]) -> Result<Vec<T>>
    where
        T: Default,
    {
        let nsite: c_int = sites.len().try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..sites.len()).map(|_| T::default()).collect();
        let dorder: [c_int; 1] = [1]; // request the function value (not derivatives)
        let status = unsafe {
            T::df_interpolate_1d(
                self.task,
                sys::DF_INTERP as c_int,
                sys::DF_METHOD_PP as c_int,
                nsite,
                sites.as_ptr(),
                sys::DF_NO_HINT as c_int,
                1,
                dorder.as_ptr(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null_mut(),
            )
        };
        check_df(status)?;
        Ok(out)
    }

    /// Integrate the spline over `[a, b]` for each pair `(a, b)` in
    /// `(left, right)`. Returns one result per pair.
    pub fn integrate(&self, left: &[T], right: &[T]) -> Result<Vec<T>>
    where
        T: Default,
    {
        if left.len() != right.len() {
            return Err(Error::InvalidArgument(
                "left and right limit slices must have the same length",
            ));
        }
        let nlim: c_int = left.len().try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..left.len()).map(|_| T::default()).collect();
        let status = unsafe {
            T::df_integrate_1d(
                self.task,
                sys::DF_METHOD_PP as c_int,
                nlim,
                left.as_ptr(),
                sys::DF_NO_HINT as c_int,
                right.as_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        };
        check_df(status)?;
        Ok(out)
    }
}

impl<T: DataFittingScalar> Drop for CubicSpline1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::dfDeleteTask(&mut self.task);
            }
        }
    }
}

#[inline]
fn check_df(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::VslStatus(status))
    }
}
