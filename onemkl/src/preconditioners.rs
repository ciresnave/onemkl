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

use onemkl_sys as sys;

use crate::error::{Error, Result};

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
/// factor stored in `alu` / `ia` / `ja`. The factor combines unit-lower
/// `L` (implicit unit diagonal) with upper-triangular `U` in standard
/// ILU storage; the function performs two triangular solves
/// (`mkl_dcsrtrsv` for `L` then `U`) to produce the result.
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
        return Err(Error::InvalidArgument(
            "v must have length n",
        ));
    }
    if alu.len() != ja.len() {
        return Err(Error::InvalidArgument(
            "alu and ja must have the same length",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let mut tmp = vec![0.0_f64; n];
    let mut out = vec![0.0_f64; n];

    // L * tmp = v, with unit diagonal.
    let uplo_l = b'L' as core::ffi::c_char;
    let trans = b'N' as core::ffi::c_char;
    let diag_unit = b'U' as core::ffi::c_char;
    unsafe {
        sys::mkl_dcsrtrsv(
            &uplo_l,
            &trans,
            &diag_unit,
            &n_i,
            alu.as_ptr(),
            ia.as_ptr(),
            ja.as_ptr(),
            v.as_ptr(),
            tmp.as_mut_ptr(),
        );
    }

    // U * out = tmp, non-unit diagonal.
    let uplo_u = b'U' as core::ffi::c_char;
    let diag_nonunit = b'N' as core::ffi::c_char;
    unsafe {
        sys::mkl_dcsrtrsv(
            &uplo_u,
            &trans,
            &diag_nonunit,
            &n_i,
            alu.as_ptr(),
            ia.as_ptr(),
            ja.as_ptr(),
            tmp.as_ptr(),
            out.as_mut_ptr(),
        );
    }
    Ok(out)
}
