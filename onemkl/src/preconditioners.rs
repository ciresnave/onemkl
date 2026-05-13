//! Preconditioners for iterative sparse solvers — Incomplete LU
//! factorization (`ILU0` and `ILUT`).
//!
//! These produce a sparse LU-style factor `M` of a CSR matrix `A`
//! suitable for left preconditioning: pass the resulting factor to a
//! triangular solve to apply `M⁻¹`.
//!
//! ```no_run
//! use onemkl::preconditioners::ilu0;
//!
//! // 3x3 SPD CSR (1-based): same pattern as A.
//! let ia = vec![1_i32, 4, 7, 10];
//! let ja = vec![1_i32, 2, 3,
//!               1, 2, 3,
//!               1, 2, 3];
//! let a = vec![4.0_f64, -1.0, 0.0,
//!              -1.0, 4.0, -1.0,
//!              0.0, -1.0, 4.0];
//! let alu = ilu0(3, &a, &ia, &ja).unwrap();
//! ```
//!
//! Both routines are double-precision only (oneMKL doesn't expose
//! `?csrilu0` / `?csrilut` for other types).

use core::ffi::c_int;
use core::ptr;

use onemkl_sys as sys;

use crate::error::{Error, Result, SparseStatus};

/// Compute the ILU(0) factorization of a square CSR matrix.
///
/// Returns the factored values `alu` in the same sparsity pattern as
/// the input. `ia` / `ja` follow oneMKL's 1-based CSR convention.
pub fn ilu0(n: usize, a: &[f64], ia: &[i32], ja: &[i32]) -> Result<Vec<f64>> {
    if ia.len() != n + 1 {
        return Err(Error::InvalidArgument(
            "ia must have length n + 1 for CSR storage",
        ));
    }
    if a.len() != ja.len() {
        return Err(Error::InvalidArgument(
            "a and ja must have the same length",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let mut alu = vec![0.0_f64; a.len()];
    // ipar / dpar are configuration arrays expected by RCI conventions.
    // For ILU0, only a few entries are read; the rest can stay zero.
    let ipar: [c_int; 128] = [0; 128];
    let dpar: [f64; 128] = [0.0; 128];
    let mut ierr: c_int = 0;

    unsafe {
        sys::dcsrilu0(
            &n_i,
            a.as_ptr(),
            ia.as_ptr(),
            ja.as_ptr(),
            alu.as_mut_ptr(),
            ipar.as_ptr(),
            dpar.as_ptr(),
            &mut ierr,
        );
    }
    if ierr != 0 {
        return Err(Error::LapackComputationFailure { info: ierr });
    }
    Ok(alu)
}

/// Result of an [`ilut`] factorization — values, row pointer, and
/// column index for the new (denser) sparsity pattern.
#[derive(Debug, Clone)]
pub struct IlutResult {
    /// Factored values.
    pub alut: Vec<f64>,
    /// Row pointer of the factor (length `n + 1`, 1-based).
    pub ialut: Vec<i32>,
    /// Column indices of the factor (1-based).
    pub jalut: Vec<i32>,
}

/// Compute the threshold-based ILUT factorization of a CSR matrix.
///
/// `tol` is the drop tolerance (absolute value below which an entry is
/// dropped). `max_fill` is the maximum number of off-diagonal entries
/// kept per row of `L` and `U` separately (the lower-triangle and
/// upper-triangle fill).
///
/// Output buffers are sized to the worst case:
/// `(2 * max_fill + 1) * n - max_fill * (max_fill + 1) + 1` entries
/// for `alut` / `jalut`.
pub fn ilut(
    n: usize,
    a: &[f64],
    ia: &[i32],
    ja: &[i32],
    tol: f64,
    max_fill: i32,
) -> Result<IlutResult> {
    if ia.len() != n + 1 {
        return Err(Error::InvalidArgument(
            "ia must have length n + 1 for CSR storage",
        ));
    }
    if a.len() != ja.len() {
        return Err(Error::InvalidArgument(
            "a and ja must have the same length",
        ));
    }
    if max_fill < 0 {
        return Err(Error::InvalidArgument(
            "max_fill must be non-negative",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let mf = max_fill as usize;
    let max_nnz = (2 * mf + 1) * n
        - mf.checked_mul(mf + 1).ok_or(Error::DimensionOverflow)?
        + 1;
    let mut alut = vec![0.0_f64; max_nnz];
    let mut jalut = vec![0_i32; max_nnz];
    let mut ialut = vec![0_i32; n + 1];
    // ipar / dpar must be configured for ILUT — defaults shown below
    // come from oneMKL's recommended settings.
    let mut ipar: [c_int; 128] = [0; 128];
    let mut dpar: [f64; 128] = [0.0; 128];
    ipar[1] = 6;     // stdout for diagnostics
    ipar[5] = 1;     // produce error messages
    ipar[30] = 1;    // replace zero pivots
    dpar[30] = 1.0e-16; // pivot replacement threshold
    dpar[31] = 1.0e-10; // pivot replacement value
    let mut ierr: c_int = 0;

    unsafe {
        sys::dcsrilut(
            &n_i,
            a.as_ptr(),
            ia.as_ptr(),
            ja.as_ptr(),
            alut.as_mut_ptr(),
            ialut.as_mut_ptr(),
            jalut.as_mut_ptr(),
            &tol,
            &max_fill,
            ipar.as_ptr(),
            dpar.as_ptr(),
            &mut ierr,
        );
    }
    // Negative ierr = error; positive ierr = warning (maxfil >= n
    // clamped, tol negated, etc.). Both shapes return a usable factor.
    if ierr < 0 {
        return Err(Error::LapackComputationFailure { info: ierr });
    }

    // Trim alut / jalut to the actual nnz reported by ialut.
    let nnz = (*ialut.last().unwrap() as i64 - 1).max(0) as usize;
    alut.truncate(nnz);
    jalut.truncate(nnz);
    Ok(IlutResult { alut, ialut, jalut })
}

/// Apply an ILU(0) or ILUT preconditioner: compute `M⁻¹ * v` for the
/// factor stored in `alu` / `ia` / `ja`. The factor combines
/// unit-lower `L` (implicit unit diagonal) with upper-triangular `U`
/// in standard ILU storage; the function performs two triangular
/// solves via the Inspector-Executor sparse API to produce the
/// result.
///
/// `v` is the input vector. Returns a freshly allocated solution
/// vector of length `n`.
pub fn apply_ilu(
    n: usize,
    alu: &[f64],
    ia: &[i32],
    ja: &[i32],
    v: &[f64],
) -> Result<Vec<f64>> {
    if ia.len() != n + 1 {
        return Err(Error::InvalidArgument(
            "ia must have length n + 1 for CSR storage",
        ));
    }
    if v.len() != n {
        return Err(Error::InvalidArgument("v must have length n"));
    }
    if alu.len() != ja.len() {
        return Err(Error::InvalidArgument(
            "alu and ja must have the same length",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let mut tmp = vec![0.0_f64; n];
    let mut out = vec![0.0_f64; n];

    // Build a CSR handle around the caller's borrowed buffers. The
    // FFI takes the index / value pointers as *mut even though MKL
    // does not mutate them during a triangular solve; casting away
    // const is safe here.
    let mut handle: sys::sparse_matrix_t = ptr::null_mut();
    let rows_start_ptr = ia.as_ptr() as *mut c_int;
    let rows_end_ptr = unsafe { rows_start_ptr.add(1) };
    let status = unsafe {
        sys::mkl_sparse_d_create_csr(
            &mut handle,
            sys::sparse_index_base_t::SPARSE_INDEX_BASE_ONE,
            n_i,
            n_i,
            rows_start_ptr,
            rows_end_ptr,
            ja.as_ptr() as *mut c_int,
            alu.as_ptr() as *mut f64,
        )
    };
    check_sparse(status)?;

    // L * tmp = v, lower triangle with implicit unit diagonal.
    let descr_l = sys::matrix_descr {
        type_: sys::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_TRIANGULAR,
        mode: sys::sparse_fill_mode_t::SPARSE_FILL_MODE_LOWER,
        diag: sys::sparse_diag_type_t::SPARSE_DIAG_UNIT,
    };
    let status = unsafe {
        sys::mkl_sparse_d_trsv(
            sys::sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE,
            1.0,
            handle,
            descr_l,
            v.as_ptr(),
            tmp.as_mut_ptr(),
        )
    };
    if let Err(e) = check_sparse(status) {
        unsafe {
            let _ = sys::mkl_sparse_destroy(handle);
        }
        return Err(e);
    }

    // U * out = tmp, upper triangle with explicit non-unit diagonal.
    let descr_u = sys::matrix_descr {
        type_: sys::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_TRIANGULAR,
        mode: sys::sparse_fill_mode_t::SPARSE_FILL_MODE_UPPER,
        diag: sys::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT,
    };
    let status = unsafe {
        sys::mkl_sparse_d_trsv(
            sys::sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE,
            1.0,
            handle,
            descr_u,
            tmp.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    let solve_result = check_sparse(status);

    unsafe {
        let _ = sys::mkl_sparse_destroy(handle);
    }
    solve_result?;
    Ok(out)
}

#[inline]
fn check_sparse(status: sys::sparse_status_t::Type) -> Result<()> {
    if status == sys::sparse_status_t::SPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        let s = match status {
            sys::sparse_status_t::SPARSE_STATUS_NOT_INITIALIZED => {
                SparseStatus::NotInitialized
            }
            sys::sparse_status_t::SPARSE_STATUS_ALLOC_FAILED => SparseStatus::AllocFailed,
            sys::sparse_status_t::SPARSE_STATUS_INVALID_VALUE => SparseStatus::InvalidValue,
            sys::sparse_status_t::SPARSE_STATUS_EXECUTION_FAILED => {
                SparseStatus::ExecutionFailed
            }
            sys::sparse_status_t::SPARSE_STATUS_INTERNAL_ERROR => SparseStatus::InternalError,
            sys::sparse_status_t::SPARSE_STATUS_NOT_SUPPORTED => SparseStatus::NotSupported,
            other => SparseStatus::Unknown(other as i32),
        };
        Err(Error::SparseStatus(s))
    }
}
