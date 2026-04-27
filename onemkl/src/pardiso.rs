//! Intel oneMKL PARDISO direct sparse solver.
//!
//! PARDISO solves `A * X = B` for symmetric, structurally symmetric, or
//! general sparse `A`, with real or complex coefficients. The solver
//! works in three phases — analysis, numerical factorization, and
//! solve — which can be invoked independently and reused across many
//! right-hand sides.
//!
//! The Rust API exposes a [`Pardiso`] handle that owns the internal
//! state array and tears it down on `Drop`. CSR input is taken in
//! 1-based form by default (oneMKL's convention; set
//! [`PardisoBuilder::indexing`] to switch).
//!
//! ```no_run
//! use onemkl::pardiso::{Pardiso, PardisoMatrixType, IndexBase};
//!
//! // Real symmetric positive-definite 3x3 stored in upper-triangular
//! // CSR (one-indexed):
//! let ia = vec![1_i32, 4, 6, 7];
//! let ja = vec![1_i32, 2, 3, 2, 3, 3];
//! let a = vec![4.0_f64, -1.0, 0.0, 4.0, -1.0, 4.0];
//! let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
//!     .with_indexing(IndexBase::One);
//! let b = vec![1.0_f64, 2.0, 3.0];
//! let mut x = vec![0.0_f64; 3];
//! solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
//! ```

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;
use std::ffi::CString;
use std::path::Path;

use num_complex::{Complex32, Complex64};
use onemkl_sys as sys;

use crate::error::{Error, Result};

/// PARDISO matrix type (`mtype` parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PardisoMatrixType {
    /// Real, structurally symmetric (mtype = 1).
    RealStructSym,
    /// Real, symmetric, positive-definite (mtype = 2).
    RealSpd,
    /// Real, symmetric, indefinite (mtype = -2).
    RealSymIndefinite,
    /// Complex, structurally symmetric (mtype = 3).
    ComplexStructSym,
    /// Complex, Hermitian, positive-definite (mtype = 4).
    ComplexHpd,
    /// Complex, Hermitian, indefinite (mtype = -4).
    ComplexHermIndefinite,
    /// Complex, symmetric (mtype = 6).
    ComplexSym,
    /// Real, unsymmetric (mtype = 11).
    RealUnsym,
    /// Complex, unsymmetric (mtype = 13).
    ComplexUnsym,
}

impl PardisoMatrixType {
    #[inline]
    fn as_int(self) -> i32 {
        match self {
            Self::RealStructSym => 1,
            Self::RealSpd => 2,
            Self::RealSymIndefinite => -2,
            Self::ComplexStructSym => 3,
            Self::ComplexHpd => 4,
            Self::ComplexHermIndefinite => -4,
            Self::ComplexSym => 6,
            Self::RealUnsym => 11,
            Self::ComplexUnsym => 13,
        }
    }
}

/// Index base for the input CSR row pointer / column index arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IndexBase {
    /// 0-based (set `iparm[34] = 1` internally).
    Zero,
    /// 1-based (PARDISO default).
    #[default]
    One,
}

/// Trait constraining `T` to be a scalar PARDISO can handle. PARDISO
/// dispatches via the `mtype` parameter rather than per-type entry
/// points, but we still gate construction on the scalar type to catch
/// trivial mismatches at compile time.
pub trait PardisoScalar: Copy + 'static {
    /// `true` for complex scalar types — used to forbid pairing a real
    /// `mtype` with a complex `T` and vice versa.
    const IS_COMPLEX: bool;
}
impl PardisoScalar for f32 {
    const IS_COMPLEX: bool = false;
}
impl PardisoScalar for f64 {
    const IS_COMPLEX: bool = false;
}
impl PardisoScalar for Complex32 {
    const IS_COMPLEX: bool = true;
}
impl PardisoScalar for Complex64 {
    const IS_COMPLEX: bool = true;
}

/// Convenience builder that delays construction of the internal state
/// until the first solve so the user can tweak `indexing` and message
/// level.
#[derive(Debug)]
pub struct Pardiso<T> {
    pt: [*mut core::ffi::c_void; 64],
    iparm: [c_int; 64],
    mtype: c_int,
    msglvl: c_int,
    indexing: IndexBase,
    // Optional permutation array. If `None`, MKL receives NULL and
    // chooses a fill-reducing permutation internally. Stored on the
    // struct so its address stays valid across phase calls (MKL may
    // read it mid-pipeline). Used in particular for Schur complement
    // extraction (`iparm[35] = 1`) where a non-NULL perm marks which
    // rows / columns are in the Schur block.
    perm: Option<Vec<c_int>>,
    initialized: bool,
    factorized: bool,
    factorized_n: c_int,
    _marker: PhantomData<T>,
}

unsafe impl<T> Send for Pardiso<T> {}

impl<T: PardisoScalar> Pardiso<T> {
    /// Build a new solver for the given matrix type. Initializes
    /// `iparm` with PARDISO's defaults via `pardisoinit`.
    #[must_use]
    pub fn new(mtype: PardisoMatrixType) -> Self {
        let mtype_is_complex = matches!(
            mtype,
            PardisoMatrixType::ComplexStructSym
                | PardisoMatrixType::ComplexHpd
                | PardisoMatrixType::ComplexHermIndefinite
                | PardisoMatrixType::ComplexSym
                | PardisoMatrixType::ComplexUnsym
        );
        assert_eq!(
            mtype_is_complex,
            T::IS_COMPLEX,
            "scalar type and matrix type disagree on real-vs-complex",
        );

        let mut pt: [*mut core::ffi::c_void; 64] = [ptr::null_mut(); 64];
        let mut iparm: [c_int; 64] = [0; 64];
        let mtype_int = mtype.as_int();

        unsafe {
            sys::pardisoinit(
                pt.as_mut_ptr() as sys::_MKL_DSS_HANDLE_t,
                &mtype_int,
                iparm.as_mut_ptr(),
            );
        }

        Self {
            pt,
            iparm,
            mtype: mtype_int,
            msglvl: 0,
            indexing: IndexBase::One,
            perm: None,
            initialized: true,
            factorized: false,
            factorized_n: 0,
            _marker: PhantomData,
        }
    }

    /// Set the index base for input arrays. `iparm[34]` is updated
    /// accordingly when execution starts.
    #[must_use]
    pub fn with_indexing(mut self, indexing: IndexBase) -> Self {
        self.indexing = indexing;
        self
    }

    /// Set the verbosity level (`msglvl` argument). `0` is silent
    /// (default), `1` prints statistics to stdout.
    #[must_use]
    pub fn with_message_level(mut self, level: i32) -> Self {
        self.msglvl = level;
        self
    }

    /// Borrow the `iparm` array — see the oneMKL PARDISO reference for
    /// the meaning of each entry.
    #[inline]
    pub fn iparm(&mut self) -> &mut [c_int; 64] {
        &mut self.iparm
    }

    /// Install a permutation / partition array. The vector must have
    /// length `n` (the order of the matrix passed to subsequent
    /// `analyze_and_factorize` / `solve` calls). Pass `None` to clear
    /// it back to NULL.
    ///
    /// Common uses:
    /// - With `iparm[4] = 1`, MKL uses `perm` as a user-supplied
    ///   fill-reducing permutation rather than computing its own.
    /// - With `iparm[4] = 2`, MKL writes the fill-reducing permutation
    ///   it chose into `perm`.
    /// - With `iparm[35] = 1`, `perm[i] = 1` marks row `i` as part of
    ///   the Schur complement block; `perm[i] = 0` marks it as
    ///   interior. The Schur complement can then be read out via
    ///   [`export`](Self::export).
    pub fn set_perm(&mut self, perm: Option<Vec<c_int>>) {
        self.perm = perm;
    }

    /// Borrow the installed permutation array, if any. Useful for
    /// reading back the fill-reducing permutation MKL wrote when
    /// `iparm[4] = 2`.
    #[inline]
    pub fn perm(&self) -> Option<&[c_int]> {
        self.perm.as_deref()
    }

    /// Run the analysis (phase 11) followed by numerical factorization
    /// (phase 22). After this call, [`solve`](Self::solve) can be
    /// called repeatedly with different right-hand sides.
    pub fn analyze_and_factorize(
        &mut self,
        n: usize,
        a: &[T],
        ia: &[i32],
        ja: &[i32],
    ) -> Result<()> {
        self.run_phase(12, n, a, ia, ja, 0, &[], &mut [])
    }

    /// Solve `A * X = B`, returning `X` in `x`. If the matrix has not
    /// been factorized yet, runs phases 11+22+33; otherwise just phase
    /// 33 with the cached factorization. To force re-factorization of a
    /// new matrix, call [`reset`](Self::reset) first.
    pub fn solve(
        &mut self,
        n: usize,
        a: &[T],
        ia: &[i32],
        ja: &[i32],
        b: &[T],
        x: &mut [T],
    ) -> Result<()> {
        let nrhs = (b.len() / n).max(1);
        let phase = if self.factorized && self.factorized_n as usize == n {
            33
        } else {
            13
        };
        self.run_phase(phase, n, a, ia, ja, nrhs as i32, b, x)
    }

    /// Solve with multiple right-hand sides simultaneously. `b` and
    /// `x` are interpreted as `n × nrhs` column-major matrices.
    pub fn solve_multi(
        &mut self,
        n: usize,
        nrhs: i32,
        a: &[T],
        ia: &[i32],
        ja: &[i32],
        b: &[T],
        x: &mut [T],
    ) -> Result<()> {
        let phase = if self.factorized && self.factorized_n as usize == n {
            33
        } else {
            13
        };
        self.run_phase(phase, n, a, ia, ja, nrhs, b, x)
    }

    /// Free the internal numerical factorization (phase 0) but keep
    /// the analysis. Useful when reusing the symbolic structure with
    /// new numerical values.
    pub fn release_factorization(&mut self) -> Result<()> {
        // Empty arrays — phase 0 only inspects pt and iparm.
        let dummy_a: [T; 0] = [];
        let dummy_i: [i32; 0] = [];
        self.run_phase(0, 0, &dummy_a, &dummy_i, &dummy_i, 0, &[], &mut [])?;
        self.factorized = false;
        Ok(())
    }

    /// Enable diagonal extraction. Sets `iparm[55] = 1` so that
    /// [`get_diagonal`](Self::get_diagonal) is valid after the
    /// subsequent factorization. Must be called before
    /// [`analyze_and_factorize`](Self::analyze_and_factorize) or the
    /// first [`solve`](Self::solve).
    #[must_use]
    pub fn with_diagonal_enabled(mut self) -> Self {
        self.iparm[55] = 1;
        self
    }

    /// Diagonal of the factorized matrix.
    ///
    /// Returns `(df, da)` where `df` is the diagonal of the `D` factor
    /// and `da` is the diagonal of the input matrix `A`. Wraps
    /// `pardiso_getdiag`. Requires both:
    ///   - [`with_diagonal_enabled`](Self::with_diagonal_enabled)
    ///     before factorization, and
    ///   - the matrix has been factorized.
    pub fn get_diagonal(&mut self, n: usize) -> Result<(Vec<T>, Vec<T>)>
    where
        T: Default,
    {
        if !self.factorized {
            return Err(Error::InvalidArgument(
                "matrix must be factorized before get_diagonal",
            ));
        }
        if self.iparm[55] != 1 {
            return Err(Error::InvalidArgument(
                "with_diagonal_enabled must be called before factorization",
            ));
        }
        let mut df: Vec<T> = (0..n).map(|_| T::default()).collect();
        let mut da: Vec<T> = (0..n).map(|_| T::default()).collect();
        let mnum: c_int = 1;
        let mut error: c_int = 0;
        unsafe {
            sys::pardiso_getdiag(
                self.pt.as_mut_ptr() as sys::_MKL_DSS_HANDLE_t,
                df.as_mut_ptr().cast(),
                da.as_mut_ptr().cast(),
                &mnum,
                &mut error,
            );
        }
        if error != 0 {
            return Err(Error::PardisoStatus(error));
        }
        Ok((df, da))
    }

    /// Persist the in-memory factorization to disk under `dir`. The
    /// directory must exist; PARDISO writes files named after the
    /// matrix number (`mnum`) into it. Restore with
    /// [`load_handle`](Self::load_handle). Wraps `pardiso_handle_store`.
    pub fn save_handle<P: AsRef<Path>>(&mut self, dir: P) -> Result<()> {
        let dir_c = path_to_cstring(dir.as_ref())?;
        let mut error: c_int = 0;
        unsafe {
            sys::pardiso_handle_store(
                self.pt.as_mut_ptr() as sys::_MKL_DSS_HANDLE_t,
                dir_c.as_ptr(),
                &mut error,
            );
        }
        if error != 0 {
            return Err(Error::PardisoStatus(error));
        }
        Ok(())
    }

    /// Restore a previously saved factorization into the current
    /// handle. The current handle's settings (matrix type, indexing)
    /// must match those used when [`save_handle`](Self::save_handle)
    /// was called. Wraps `pardiso_handle_restore`.
    pub fn load_handle<P: AsRef<Path>>(&mut self, dir: P) -> Result<()> {
        let dir_c = path_to_cstring(dir.as_ref())?;
        let mut error: c_int = 0;
        unsafe {
            sys::pardiso_handle_restore(
                self.pt.as_mut_ptr() as sys::_MKL_DSS_HANDLE_t,
                dir_c.as_ptr(),
                &mut error,
            );
        }
        if error != 0 {
            return Err(Error::PardisoStatus(error));
        }
        // We don't know n until the user solves, but the handle is now
        // valid for solve calls. Mark it factorized — the user is
        // expected to pass the same n they saved with.
        self.factorized = true;
        Ok(())
    }

    /// Export factor data (typically the Schur complement) from a
    /// factorized handle. Wraps `pardiso_export`.
    ///
    /// `pardiso_export` uses a step-based protocol: pass increasing
    /// `step` values together with `iparm` to extract progressively
    /// larger pieces of the factor data. The exact step / iparm
    /// combination required for each output (Schur complement,
    /// L / U factors, etc.) is documented in the oneMKL PARDISO Export
    /// reference and varies by MKL version.
    ///
    /// Pass `None` for `values` and / or `ja` when only `ia` is being
    /// filled.
    ///
    /// **Note:** Schur complement extraction (`iparm[35] = 1` plus a
    /// partition `perm`) has additional MKL-version-specific
    /// requirements that we have not fully validated; calling this
    /// method in that mode may crash inside MKL. The infrastructure —
    /// [`set_perm`](Self::set_perm) and `pardiso_export` itself — is
    /// in place once the correct call sequence is determined.
    pub fn export(
        &mut self,
        step: i32,
        values: Option<&mut [T]>,
        ia: &mut [i32],
        ja: Option<&mut [i32]>,
    ) -> Result<()> {
        let step_c: c_int = step;
        let mut error: c_int = 0;
        let values_ptr = values
            .map(|v| v.as_mut_ptr().cast::<core::ffi::c_void>())
            .unwrap_or(ptr::null_mut());
        let ja_ptr = ja
            .map(|j| j.as_mut_ptr())
            .unwrap_or(ptr::null_mut());
        unsafe {
            sys::pardiso_export(
                self.pt.as_mut_ptr() as *mut core::ffi::c_void,
                values_ptr,
                ia.as_mut_ptr(),
                ja_ptr,
                &step_c,
                self.iparm.as_ptr(),
                &mut error,
            );
        }
        if error != 0 {
            return Err(Error::PardisoStatus(error));
        }
        Ok(())
    }

    /// Delete the on-disk files written by
    /// [`save_handle`](Self::save_handle) under `dir`. Free function;
    /// does not need a live `Pardiso` instance. Wraps
    /// `pardiso_handle_delete`.
    pub fn delete_handle_files<P: AsRef<Path>>(dir: P) -> Result<()> {
        let dir_c = path_to_cstring(dir.as_ref())?;
        let mut error: c_int = 0;
        unsafe {
            sys::pardiso_handle_delete(dir_c.as_ptr(), &mut error);
        }
        if error != 0 {
            return Err(Error::PardisoStatus(error));
        }
        Ok(())
    }

    /// Free everything (phase -1). After this, the solver must be
    /// rebuilt with [`Pardiso::new`].
    pub fn reset(&mut self) -> Result<()> {
        let dummy_a: [T; 0] = [];
        let dummy_i: [i32; 0] = [];
        self.run_phase(-1, 0, &dummy_a, &dummy_i, &dummy_i, 0, &[], &mut [])?;
        self.initialized = false;
        self.factorized = false;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_phase(
        &mut self,
        phase: i32,
        n: usize,
        a: &[T],
        ia: &[i32],
        ja: &[i32],
        nrhs: i32,
        b: &[T],
        x: &mut [T],
    ) -> Result<()> {
        let n_i32: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let phase_c = phase;
        let maxfct: c_int = 1;
        let mnum: c_int = 1;
        let mut error: c_int = 0;
        // 0 → Fortran/1-based; 1 → C/0-based.
        self.iparm[34] = match self.indexing {
            IndexBase::One => 0,
            IndexBase::Zero => 1,
        };
        let mtype = self.mtype;
        let perm_ptr: *mut c_int = match self.perm.as_mut() {
            Some(p) => p.as_mut_ptr(),
            None => ptr::null_mut(),
        };

        unsafe {
            sys::pardiso(
                self.pt.as_mut_ptr() as sys::_MKL_DSS_HANDLE_t,
                &maxfct,
                &mnum,
                &mtype,
                &phase_c,
                &n_i32,
                a.as_ptr().cast(),
                ia.as_ptr(),
                ja.as_ptr(),
                perm_ptr,
                &nrhs,
                self.iparm.as_mut_ptr(),
                &self.msglvl,
                b.as_ptr() as *mut core::ffi::c_void,
                x.as_mut_ptr().cast(),
                &mut error,
            );
        }

        if error != 0 {
            return Err(Error::PardisoStatus(error));
        }
        if phase == 22 || phase == 12 || phase == 13 {
            self.factorized = true;
            self.factorized_n = n_i32;
        }
        Ok(())
    }
}

fn path_to_cstring(p: &Path) -> Result<CString> {
    let s = p
        .to_str()
        .ok_or(Error::InvalidArgument("path is not valid UTF-8"))?;
    CString::new(s).map_err(|_| Error::InvalidArgument("path contains a null byte"))
}

impl<T> Drop for Pardiso<T> {
    fn drop(&mut self) {
        if !self.initialized {
            return;
        }
        // Best-effort cleanup; ignore any error here.
        let phase: c_int = -1;
        let maxfct: c_int = 1;
        let mnum: c_int = 1;
        let n: c_int = 0;
        let nrhs: c_int = 0;
        let msglvl: c_int = 0;
        let mut error: c_int = 0;
        unsafe {
            sys::pardiso(
                self.pt.as_mut_ptr() as sys::_MKL_DSS_HANDLE_t,
                &maxfct, &mnum, &self.mtype, &phase, &n,
                ptr::null(), ptr::null(), ptr::null(),
                ptr::null_mut(), &nrhs,
                self.iparm.as_mut_ptr(), &msglvl,
                ptr::null_mut(), ptr::null_mut(),
                &mut error,
            );
        }
    }
}
