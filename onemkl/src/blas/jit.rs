//! JIT GEMM — runtime-specialized small-matrix kernels.
//!
//! For small fixed-shape GEMMs called many times (transformer
//! attention heads, MLP blocks, batched element-wise compute),
//! oneMKL can JIT-compile a kernel specialized to the exact M, N, K,
//! transposes, leading dimensions, and `alpha` / `beta`. The
//! resulting kernel skips the dispatch overhead and the parameter
//! checks of the regular GEMM call, often yielding 1.5–3× speedup
//! for shapes where M·N·K ≲ 64³.
//!
//! Usage:
//!
//! ```ignore
//! use onemkl::blas::jit::JitGemm;
//! use onemkl::{Layout, Transpose};
//!
//! let plan = JitGemm::<f64>::new(
//!     Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
//!     32, 32, 32, 1.0, 32, 32, 0.0, 32,
//! ).unwrap();
//! // ... call plan.execute(a, b, c) many times ...
//! ```
//!
//! `Drop` frees the JIT'd kernel.

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use num_complex::{Complex32, Complex64};
use onemkl_sys as sys;

use crate::enums::{Layout, Transpose};
use crate::error::{Error, Result};

/// Status returned by `mkl_jit_create_*gemm`. `Success` means a JIT
/// kernel was generated; `NoJit` means the host doesn't support JIT
/// for the requested shape (e.g. unsupported architecture or the
/// shape is too large) and MKL has fallen back to the standard GEMM
/// path internally — the wrapper still works either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JitStatus {
    /// A specialized kernel was compiled successfully.
    Compiled,
    /// JIT not available; the kernel falls back to standard GEMM.
    Fallback,
}

impl JitStatus {
    #[inline]
    fn from_raw(status: sys::mkl_jit_status_t::Type) -> Result<Self> {
        match status {
            s if s == sys::mkl_jit_status_t::MKL_JIT_SUCCESS as c_int => Ok(Self::Compiled),
            s if s == sys::mkl_jit_status_t::MKL_NO_JIT as c_int => Ok(Self::Fallback),
            other => Err(Error::LapackComputationFailure { info: other }),
        }
    }
}

/// Trait wiring up the four `mkl_cblas_jit_create_*gemm` /
/// `mkl_jit_get_*gemm_ptr` variants for f32 / f64 / Complex32 /
/// Complex64.
#[allow(missing_docs)]
pub trait JitGemmScalar: Copy + 'static {
    /// Function-pointer type the kernel returns to.
    type Kernel: Copy;

    #[allow(clippy::too_many_arguments)]
    unsafe fn jit_create(
        jitter: *mut *mut core::ffi::c_void,
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type,
        transb: sys::MKL_TRANSPOSE::Type,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: Self,
        lda: c_int,
        ldb: c_int,
        beta: Self,
        ldc: c_int,
    ) -> sys::mkl_jit_status_t::Type;

    unsafe fn jit_get_kernel(jitter: *const core::ffi::c_void) -> Option<Self::Kernel>;

    /// Call the kernel: `c ← alpha · op(a) · op(b) + beta · c` with
    /// the parameters baked in at JIT time.
    unsafe fn invoke_kernel(
        kernel: Self::Kernel,
        jitter: *mut core::ffi::c_void,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self,
    );
}

impl JitGemmScalar for f32 {
    type Kernel = unsafe extern "C" fn(
        *mut core::ffi::c_void,
        *mut f32,
        *mut f32,
        *mut f32,
    );

    unsafe fn jit_create(
        jitter: *mut *mut core::ffi::c_void,
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type,
        transb: sys::MKL_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self, lda: c_int, ldb: c_int,
        beta: Self, ldc: c_int,
    ) -> sys::mkl_jit_status_t::Type {
        unsafe {
            sys::mkl_cblas_jit_create_sgemm(
                jitter, layout, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc,
            )
        }
    }
    unsafe fn jit_get_kernel(jitter: *const core::ffi::c_void) -> Option<Self::Kernel> {
        unsafe { sys::mkl_jit_get_sgemm_ptr(jitter) }
    }
    unsafe fn invoke_kernel(
        kernel: Self::Kernel,
        jitter: *mut core::ffi::c_void,
        a: *mut Self, b: *mut Self, c: *mut Self,
    ) {
        unsafe { kernel(jitter, a, b, c) }
    }
}

impl JitGemmScalar for f64 {
    type Kernel = unsafe extern "C" fn(
        *mut core::ffi::c_void,
        *mut f64,
        *mut f64,
        *mut f64,
    );

    unsafe fn jit_create(
        jitter: *mut *mut core::ffi::c_void,
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type,
        transb: sys::MKL_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self, lda: c_int, ldb: c_int,
        beta: Self, ldc: c_int,
    ) -> sys::mkl_jit_status_t::Type {
        unsafe {
            sys::mkl_cblas_jit_create_dgemm(
                jitter, layout, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc,
            )
        }
    }
    unsafe fn jit_get_kernel(jitter: *const core::ffi::c_void) -> Option<Self::Kernel> {
        unsafe { sys::mkl_jit_get_dgemm_ptr(jitter) }
    }
    unsafe fn invoke_kernel(
        kernel: Self::Kernel,
        jitter: *mut core::ffi::c_void,
        a: *mut Self, b: *mut Self, c: *mut Self,
    ) {
        unsafe { kernel(jitter, a, b, c) }
    }
}

impl JitGemmScalar for Complex32 {
    type Kernel = unsafe extern "C" fn(
        *mut core::ffi::c_void,
        *mut sys::MKL_Complex8,
        *mut sys::MKL_Complex8,
        *mut sys::MKL_Complex8,
    );

    unsafe fn jit_create(
        jitter: *mut *mut core::ffi::c_void,
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type,
        transb: sys::MKL_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self, lda: c_int, ldb: c_int,
        beta: Self, ldc: c_int,
    ) -> sys::mkl_jit_status_t::Type {
        unsafe {
            sys::mkl_cblas_jit_create_cgemm(
                jitter, layout, transa, transb, m, n, k,
                (&alpha as *const Self).cast(),
                lda, ldb,
                (&beta as *const Self).cast(),
                ldc,
            )
        }
    }
    unsafe fn jit_get_kernel(jitter: *const core::ffi::c_void) -> Option<Self::Kernel> {
        unsafe { sys::mkl_jit_get_cgemm_ptr(jitter) }
    }
    unsafe fn invoke_kernel(
        kernel: Self::Kernel,
        jitter: *mut core::ffi::c_void,
        a: *mut Self, b: *mut Self, c: *mut Self,
    ) {
        unsafe { kernel(jitter, a.cast(), b.cast(), c.cast()) }
    }
}

impl JitGemmScalar for Complex64 {
    type Kernel = unsafe extern "C" fn(
        *mut core::ffi::c_void,
        *mut sys::MKL_Complex16,
        *mut sys::MKL_Complex16,
        *mut sys::MKL_Complex16,
    );

    unsafe fn jit_create(
        jitter: *mut *mut core::ffi::c_void,
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type,
        transb: sys::MKL_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self, lda: c_int, ldb: c_int,
        beta: Self, ldc: c_int,
    ) -> sys::mkl_jit_status_t::Type {
        unsafe {
            sys::mkl_cblas_jit_create_zgemm(
                jitter, layout, transa, transb, m, n, k,
                (&alpha as *const Self).cast(),
                lda, ldb,
                (&beta as *const Self).cast(),
                ldc,
            )
        }
    }
    unsafe fn jit_get_kernel(jitter: *const core::ffi::c_void) -> Option<Self::Kernel> {
        unsafe { sys::mkl_jit_get_zgemm_ptr(jitter) }
    }
    unsafe fn invoke_kernel(
        kernel: Self::Kernel,
        jitter: *mut core::ffi::c_void,
        a: *mut Self, b: *mut Self, c: *mut Self,
    ) {
        unsafe { kernel(jitter, a.cast(), b.cast(), c.cast()) }
    }
}

#[inline]
fn map_layout(layout: Layout) -> sys::MKL_LAYOUT::Type {
    match layout {
        Layout::RowMajor => sys::MKL_LAYOUT::MKL_ROW_MAJOR,
        Layout::ColMajor => sys::MKL_LAYOUT::MKL_COL_MAJOR,
    }
}

#[inline]
fn map_transpose(t: Transpose) -> sys::MKL_TRANSPOSE::Type {
    match t {
        Transpose::NoTrans => sys::MKL_TRANSPOSE::MKL_NOTRANS,
        Transpose::Trans => sys::MKL_TRANSPOSE::MKL_TRANS,
        Transpose::ConjTrans => sys::MKL_TRANSPOSE::MKL_CONJTRANS,
    }
}

/// JIT-compiled GEMM kernel for a fixed shape.
pub struct JitGemm<T: JitGemmScalar> {
    jitter: *mut core::ffi::c_void,
    kernel: T::Kernel,
    /// Shape information retained for input validation.
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    transa: Transpose,
    transb: Transpose,
    layout: Layout,
    /// Whether the underlying kernel was actually JIT-compiled or
    /// fell back to a regular GEMM.
    status: JitStatus,
    _marker: PhantomData<T>,
}

unsafe impl<T: JitGemmScalar + Send> Send for JitGemm<T> {}

impl<T: JitGemmScalar> JitGemm<T> {
    /// Compile a JIT'd GEMM kernel for the supplied shape and
    /// scaling factors. `alpha` and `beta` are baked into the kernel
    /// at compile time — to use different values, build a different
    /// `JitGemm`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layout: Layout,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        lda: usize,
        ldb: usize,
        beta: T,
        ldc: usize,
    ) -> Result<Self> {
        let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;
        let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let k_i: c_int = k.try_into().map_err(|_| Error::DimensionOverflow)?;
        let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ldb_i: c_int = ldb.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ldc_i: c_int = ldc.try_into().map_err(|_| Error::DimensionOverflow)?;

        let mut jitter: *mut core::ffi::c_void = ptr::null_mut();
        let raw_status = unsafe {
            T::jit_create(
                &mut jitter,
                map_layout(layout),
                map_transpose(transa),
                map_transpose(transb),
                m_i, n_i, k_i, alpha, lda_i, ldb_i, beta, ldc_i,
            )
        };
        let status = JitStatus::from_raw(raw_status)?;
        let kernel = unsafe { T::jit_get_kernel(jitter) }
            .ok_or(Error::InvalidArgument(
                "MKL returned a NULL kernel pointer for the JIT'd GEMM",
            ))?;
        Ok(Self {
            jitter,
            kernel,
            m, n, k,
            lda, ldb, ldc,
            transa, transb, layout,
            status,
            _marker: PhantomData,
        })
    }

    /// Whether the kernel was JIT-compiled or fell back to standard
    /// GEMM. Functional behavior is the same either way; this is
    /// informational for performance tuning.
    #[inline]
    #[must_use]
    pub fn status(&self) -> JitStatus {
        self.status
    }

    /// Execute the JIT'd kernel: `c ← alpha · op(a) · op(b) + beta · c`
    /// with the parameters baked in at construction time.
    pub fn execute(&self, a: &[T], b: &[T], c: &mut [T]) -> Result<()> {
        let (a_rows, a_cols) = match self.transa {
            Transpose::NoTrans => (self.m, self.k),
            _ => (self.k, self.m),
        };
        let (b_rows, b_cols) = match self.transb {
            Transpose::NoTrans => (self.k, self.n),
            _ => (self.n, self.k),
        };
        let needed_a = match self.layout {
            Layout::RowMajor => a_rows.saturating_mul(self.lda),
            Layout::ColMajor => a_cols.saturating_mul(self.lda),
        };
        let needed_b = match self.layout {
            Layout::RowMajor => b_rows.saturating_mul(self.ldb),
            Layout::ColMajor => b_cols.saturating_mul(self.ldb),
        };
        let needed_c = match self.layout {
            Layout::RowMajor => self.m.saturating_mul(self.ldc),
            Layout::ColMajor => self.n.saturating_mul(self.ldc),
        };
        if a.len() < needed_a {
            return Err(Error::InvalidArgument("A buffer too small for op(A) shape"));
        }
        if b.len() < needed_b {
            return Err(Error::InvalidArgument("B buffer too small for op(B) shape"));
        }
        if c.len() < needed_c {
            return Err(Error::InvalidArgument("C buffer too small for shape"));
        }
        // The kernel takes A and B as *mut even though they are inputs
        // — this matches MKL's C signature; the kernel does not
        // actually mutate them.
        unsafe {
            T::invoke_kernel(
                self.kernel,
                self.jitter,
                a.as_ptr() as *mut T,
                b.as_ptr() as *mut T,
                c.as_mut_ptr(),
            );
        }
        Ok(())
    }
}

impl<T: JitGemmScalar> Drop for JitGemm<T> {
    fn drop(&mut self) {
        if !self.jitter.is_null() {
            unsafe {
                let _ = sys::mkl_jit_destroy(self.jitter);
            }
        }
    }
}
