//! Iterative sparse solvers — Conjugate Gradient (CG) and Flexible
//! GMRES (FGMRES) via oneMKL's RCI (reverse-communication interface).
//!
//! The high-level wrappers in this module take a closure for the
//! matrix-vector product (and optionally one for preconditioning) and
//! run the RCI loop internally. The closure is called repeatedly with
//! the residual / search direction and must write the corresponding
//! `A * x` (or `M⁻¹ * x`) into the output slice.
//!
//! oneMKL only provides double-precision (`d`) variants for these
//! solvers; the public functions therefore operate on `&mut [f64]`.
//!
//! ```no_run
//! use onemkl::iss::{solve_cg, IssOptions};
//!
//! let n = 3;
//! let b = [3.0_f64, 2.0, 3.0];
//! let mut x = [0.0_f64; 3];
//!
//! // A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]] — implemented as a closure.
//! let a_mul = |v: &[f64], out: &mut [f64]| {
//!     out[0] = 4.0 * v[0] - v[1];
//!     out[1] = -v[0] + 4.0 * v[1] - v[2];
//!     out[2] = -v[1] + 4.0 * v[2];
//! };
//!
//! let opts = IssOptions::default();
//! let _iters = solve_cg(&b, &mut x, opts, a_mul).unwrap();
//! ```

use core::ffi::c_int;

use onemkl_sys as sys;

use crate::error::{Error, Result};

/// Options shared by the CG and FGMRES wrappers.
#[derive(Debug, Clone, Copy)]
pub struct IssOptions {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Relative residual norm at which to stop (`|r| <= tol * |r₀|`).
    pub relative_tolerance: f64,
    /// Absolute residual norm at which to stop.
    pub absolute_tolerance: f64,
    /// FGMRES restart length (ignored by `solve_cg`). Default is 20.
    pub restart_length: usize,
}

impl Default for IssOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            relative_tolerance: 1e-10,
            absolute_tolerance: 0.0,
            restart_length: 20,
        }
    }
}

/// Diagnostic information returned alongside the converged solution.
#[derive(Debug, Clone, Copy)]
pub struct IssResult {
    /// Iteration count reported by `*_get`.
    pub iterations: usize,
    /// `‖r₀‖` — norm of the initial residual.
    pub initial_residual_norm: f64,
    /// `‖rₖ‖` — norm of the final residual.
    pub final_residual_norm: f64,
    /// Stopping condition that ended the loop.
    pub stop_reason: IssStopReason,
}

/// Why the iterative solver stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssStopReason {
    /// `‖rₖ‖` is below the configured relative or absolute tolerance.
    Converged,
    /// The loop hit `max_iterations` without converging.
    MaxIterations,
    /// The solver returned a non-standard `rci_request` code.
    Other(i32),
}

const RCI_OK: c_int = 0;
const RCI_NEED_MV: c_int = 1;
const RCI_NEED_PRECOND: c_int = 3;

// =====================================================================
// CG
// =====================================================================

/// Solve `A * x = b` with the Conjugate Gradient method. `A` must be
/// symmetric positive-definite; `mat_vec` computes `A * v`.
///
/// `b` is the right-hand side; `x` is the initial guess on entry and
/// the converged solution on return. Returns the iteration count
/// reported by `dcg_get`.
pub fn solve_cg<F>(
    b: &[f64],
    x: &mut [f64],
    opts: IssOptions,
    mat_vec: F,
) -> Result<IssResult>
where
    F: FnMut(&[f64], &mut [f64]),
{
    solve_cg_preconditioned::<F, fn(&[f64], &mut [f64])>(b, x, opts, mat_vec, None)
}

/// Solve `A * x = b` with preconditioned CG. `precondition` computes
/// `M⁻¹ * v` for some preconditioner matrix `M`.
pub fn solve_cg_preconditioned<F, P>(
    b: &[f64],
    x: &mut [f64],
    opts: IssOptions,
    mut mat_vec: F,
    mut precondition: Option<P>,
) -> Result<IssResult>
where
    F: FnMut(&[f64], &mut [f64]),
    P: FnMut(&[f64], &mut [f64]),
{
    if b.len() != x.len() {
        return Err(Error::InvalidArgument(
            "b and x must have the same length",
        ));
    }
    let n = b.len();
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;

    let mut ipar: [c_int; 128] = [0; 128];
    let mut dpar: [f64; 128] = [0.0; 128];
    let mut tmp: Vec<f64> = vec![0.0; 4 * n];
    let mut rci: c_int = 0;

    unsafe {
        sys::dcg_init(
            &n_i,
            x.as_ptr(),
            b.as_ptr(),
            &mut rci,
            ipar.as_mut_ptr(),
            dpar.as_mut_ptr(),
            tmp.as_mut_ptr(),
        );
    }
    if rci != 0 {
        return Err(Error::LapackComputationFailure { info: rci });
    }

    // Configure tolerance and iteration limit.
    ipar[4] = opts.max_iterations.try_into().unwrap_or(c_int::MAX);
    ipar[7] = 1; // automatic stopping by max iterations
    ipar[8] = 1; // automatic stopping by residual norm
    ipar[9] = 0; // no user-defined stopping
    ipar[10] = if precondition.is_some() { 1 } else { 0 };
    dpar[0] = opts.relative_tolerance;
    dpar[1] = opts.absolute_tolerance;

    unsafe {
        sys::dcg_check(
            &n_i,
            x.as_ptr(),
            b.as_ptr(),
            &mut rci,
            ipar.as_mut_ptr(),
            dpar.as_mut_ptr(),
            tmp.as_mut_ptr(),
        );
    }
    if rci < 0 {
        return Err(Error::LapackComputationFailure { info: rci });
    }

    // RCI loop.
    let mut last_rci;
    loop {
        unsafe {
            sys::dcg(
                &n_i,
                x.as_mut_ptr(),
                b.as_ptr(),
                &mut rci,
                ipar.as_mut_ptr(),
                dpar.as_mut_ptr(),
                tmp.as_mut_ptr(),
            );
        }
        last_rci = rci;
        match rci {
            RCI_OK => break,
            RCI_NEED_MV => {
                // tmp[0..n] holds the source; result goes to tmp[n..2n].
                let (src, rest) = tmp.split_at_mut(n);
                let dst = &mut rest[..n];
                mat_vec(src, dst);
            }
            RCI_NEED_PRECOND => {
                if let Some(p) = precondition.as_mut() {
                    // Source at tmp[2n..3n], destination at tmp[3n..4n].
                    let (lo, hi) = tmp.split_at_mut(3 * n);
                    let src = &lo[2 * n..3 * n];
                    let dst = &mut hi[..n];
                    p(src, dst);
                } else {
                    return Err(Error::InvalidArgument(
                        "CG asked for preconditioning but none was provided",
                    ));
                }
            }
            other if other < 0 => {
                return Err(Error::LapackComputationFailure { info: other });
            }
            _ => {
                // Unknown but non-negative: ignore and continue.
            }
        }
    }

    let mut itercount: c_int = 0;
    unsafe {
        sys::dcg_get(
            &n_i,
            x.as_ptr(),
            b.as_ptr(),
            &mut rci,
            ipar.as_mut_ptr(),
            dpar.as_mut_ptr(),
            tmp.as_mut_ptr(),
            &mut itercount,
        );
    }
    let iterations = itercount.max(0) as usize;
    Ok(IssResult {
        iterations,
        initial_residual_norm: dpar[2],
        final_residual_norm: dpar[4],
        stop_reason: classify_stop(last_rci, iterations, opts.max_iterations, dpar[4], &dpar),
    })
}

// =====================================================================
// FGMRES
// =====================================================================

/// Solve `A * x = b` with Flexible GMRES. `mat_vec` computes `A * v`.
///
/// `b` is the right-hand side; `x` is the initial guess on entry and
/// the converged solution on return. Returns the iteration count.
pub fn solve_fgmres<F>(
    b: &mut [f64],
    x: &mut [f64],
    opts: IssOptions,
    mut mat_vec: F,
) -> Result<IssResult>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if b.len() != x.len() {
        return Err(Error::InvalidArgument(
            "b and x must have the same length",
        ));
    }
    let n = b.len();
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let restart = opts.restart_length.min(n);
    let restart_i: c_int = restart.try_into().map_err(|_| Error::DimensionOverflow)?;

    let mut ipar: [c_int; 128] = [0; 128];
    let mut dpar: [f64; 128] = [0.0; 128];
    // Required tmp size for FGMRES — see oneMKL FGMRES Interface
    // Description.
    let tmp_len = ((2 * restart + 1) * n) + restart * (restart + 9) / 2 + 1;
    let mut tmp: Vec<f64> = vec![0.0; tmp_len];
    let mut rci: c_int = 0;

    unsafe {
        sys::dfgmres_init(
            &n_i,
            x.as_ptr(),
            b.as_ptr(),
            &mut rci,
            ipar.as_mut_ptr(),
            dpar.as_mut_ptr(),
            tmp.as_mut_ptr(),
        );
    }
    if rci != 0 {
        return Err(Error::LapackComputationFailure { info: rci });
    }

    ipar[4] = opts.max_iterations.try_into().unwrap_or(c_int::MAX);
    ipar[7] = 1;
    ipar[8] = 1;
    ipar[9] = 0;
    ipar[10] = 0; // no preconditioning in this simplified entry point
    ipar[11] = 1; // automatic test for convergence
    ipar[14] = restart_i;
    dpar[0] = opts.relative_tolerance;
    dpar[1] = opts.absolute_tolerance;

    unsafe {
        sys::dfgmres_check(
            &n_i,
            x.as_ptr(),
            b.as_ptr(),
            &mut rci,
            ipar.as_mut_ptr(),
            dpar.as_mut_ptr(),
            tmp.as_mut_ptr(),
        );
    }
    if rci < 0 {
        return Err(Error::LapackComputationFailure { info: rci });
    }

    let mut last_rci;
    loop {
        unsafe {
            sys::dfgmres(
                &n_i,
                x.as_mut_ptr(),
                b.as_mut_ptr(),
                &mut rci,
                ipar.as_mut_ptr(),
                dpar.as_mut_ptr(),
                tmp.as_mut_ptr(),
            );
        }
        last_rci = rci;
        match rci {
            RCI_OK => break,
            RCI_NEED_MV => {
                // Source index is `ipar[21] - 1`, destination is
                // `ipar[22] - 1` (oneMKL uses 1-based offsets here).
                let src_off = (ipar[21] - 1).max(0) as usize;
                let dst_off = (ipar[22] - 1).max(0) as usize;
                debug_assert!(src_off + n <= tmp.len());
                debug_assert!(dst_off + n <= tmp.len());
                // Borrow disjoint slices via split_at_mut around the
                // smaller of the two offsets.
                let (low, high, low_off, high_off) = if src_off < dst_off {
                    let (a, b) = tmp.split_at_mut(dst_off);
                    (a, b, src_off, 0)
                } else if src_off > dst_off {
                    let (a, b) = tmp.split_at_mut(src_off);
                    (b, a, 0, dst_off)
                } else {
                    return Err(Error::InvalidArgument(
                        "FGMRES requested mat-vec with overlapping src and dst slices",
                    ));
                };
                if src_off < dst_off {
                    let src = &low[low_off..low_off + n];
                    let dst = &mut high[..n];
                    mat_vec(src, dst);
                } else {
                    let src = &low[..n];
                    let dst = &mut high[high_off..high_off + n];
                    mat_vec(src, dst);
                }
            }
            other if other < 0 => {
                return Err(Error::LapackComputationFailure { info: other });
            }
            _ => {}
        }
    }

    let mut itercount: c_int = 0;
    unsafe {
        sys::dfgmres_get(
            &n_i,
            x.as_mut_ptr(),
            b.as_mut_ptr(),
            &mut rci,
            ipar.as_mut_ptr(),
            dpar.as_mut_ptr(),
            tmp.as_mut_ptr(),
            &mut itercount,
        );
    }
    let iterations = itercount.max(0) as usize;
    Ok(IssResult {
        iterations,
        initial_residual_norm: dpar[2],
        final_residual_norm: dpar[4],
        stop_reason: classify_stop(last_rci, iterations, opts.max_iterations, dpar[4], &dpar),
    })
}

fn classify_stop(
    last_rci: c_int,
    iterations: usize,
    max_iterations: usize,
    final_residual: f64,
    dpar: &[f64; 128],
) -> IssStopReason {
    if last_rci == 0 {
        // dpar[3] is the residual threshold computed by *_check.
        if final_residual <= dpar[3] {
            IssStopReason::Converged
        } else if iterations >= max_iterations {
            IssStopReason::MaxIterations
        } else {
            IssStopReason::Converged
        }
    } else {
        IssStopReason::Other(last_rci as i32)
    }
}
