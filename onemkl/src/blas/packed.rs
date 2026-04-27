//! Pack / Compute GEMM API.
//!
//! When the same operand is used in many GEMMs (e.g. a fixed weight
//! matrix multiplied by many input batches), MKL can pack it once
//! into an internal layout and reuse the packed form in repeated
//! `gemm_compute` calls. This avoids re-doing the same data
//! reordering on every call and yields a meaningful speedup for
//! medium-sized matrices.
//!
//! Two-step usage:
//!
//! ```ignore
//! use onemkl::blas::packed::{gemm_compute_packed_a, PackedMatrix};
//! use onemkl::{Layout, Transpose};
//!
//! // Pack A once.
//! let packed_a = PackedMatrix::<f64>::pack_a(
//!     Layout::RowMajor, Transpose::NoTrans,
//!     m, n, k, alpha, &a_storage, lda,
//! ).unwrap();
//!
//! // Reuse in many GEMMs against different B / C.
//! for batch in batches {
//!     gemm_compute_packed_a(
//!         Layout::RowMajor, Transpose::NoTrans,
//!         m, n, k, &packed_a, &batch.b, ldb, 0.0, &mut batch.c, ldc,
//!     ).unwrap();
//! }
//! ```
//!
//! Pack/compute is available for f32 and f64 only — MKL doesn't ship
//! complex variants.

use core::ffi::c_int;
use core::marker::PhantomData;

use onemkl_sys as sys;

use crate::enums::{Layout, Transpose};
use crate::error::{Error, Result};

/// Which GEMM operand is being packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackIdentifier {
    /// Pack the `A` operand (left-hand side).
    A,
    /// Pack the `B` operand (right-hand side).
    B,
}

impl PackIdentifier {
    #[inline]
    fn as_cblas(self) -> sys::CBLAS_IDENTIFIER::Type {
        match self {
            Self::A => sys::CBLAS_IDENTIFIER::CblasAMatrix,
            Self::B => sys::CBLAS_IDENTIFIER::CblasBMatrix,
        }
    }
}

/// Real scalar types supported by the pack / compute GEMM API.
#[allow(missing_docs)]
pub trait PackedGemmScalar: Copy + 'static {
    unsafe fn gemm_pack_get_size(
        identifier: sys::CBLAS_IDENTIFIER::Type,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_pack(
        layout: sys::CBLAS_LAYOUT::Type,
        identifier: sys::CBLAS_IDENTIFIER::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: Self,
        src: *const Self,
        ld: c_int,
        dest: *mut Self,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_compute(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        a: *const Self,
        lda: c_int,
        b: *const Self,
        ldb: c_int,
        beta: Self,
        c: *mut Self,
        ldc: c_int,
    );
}

impl PackedGemmScalar for f32 {
    unsafe fn gemm_pack_get_size(
        identifier: sys::CBLAS_IDENTIFIER::Type,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> usize {
        unsafe { sys::cblas_sgemm_pack_get_size(identifier, m, n, k) }
    }
    unsafe fn gemm_pack(
        layout: sys::CBLAS_LAYOUT::Type,
        identifier: sys::CBLAS_IDENTIFIER::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self, src: *const Self, ld: c_int, dest: *mut Self,
    ) {
        unsafe {
            sys::cblas_sgemm_pack(
                layout, identifier, trans, m, n, k, alpha, src, ld, dest,
            )
        }
    }
    unsafe fn gemm_compute(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        a: *const Self, lda: c_int,
        b: *const Self, ldb: c_int,
        beta: Self, c: *mut Self, ldc: c_int,
    ) {
        unsafe {
            sys::cblas_sgemm_compute(
                layout, transa, transb, m, n, k, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }
}

impl PackedGemmScalar for f64 {
    unsafe fn gemm_pack_get_size(
        identifier: sys::CBLAS_IDENTIFIER::Type,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> usize {
        unsafe { sys::cblas_dgemm_pack_get_size(identifier, m, n, k) }
    }
    unsafe fn gemm_pack(
        layout: sys::CBLAS_LAYOUT::Type,
        identifier: sys::CBLAS_IDENTIFIER::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self, src: *const Self, ld: c_int, dest: *mut Self,
    ) {
        unsafe {
            sys::cblas_dgemm_pack(
                layout, identifier, trans, m, n, k, alpha, src, ld, dest,
            )
        }
    }
    unsafe fn gemm_compute(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        a: *const Self, lda: c_int,
        b: *const Self, ldb: c_int,
        beta: Self, c: *mut Self, ldc: c_int,
    ) {
        unsafe {
            sys::cblas_dgemm_compute(
                layout, transa, transb, m, n, k, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }
}

/// A GEMM operand pre-packed into MKL's internal layout. The packed
/// form encodes the `alpha` scaling that was applied at pack time;
/// pass `1.0` if you want to apply scaling separately at compute
/// time.
pub struct PackedMatrix<T: PackedGemmScalar> {
    /// Backing storage. The packed format is opaque; we expose only
    /// the typed pointer to MKL's compute kernel.
    buffer: Vec<T>,
    identifier: PackIdentifier,
    m: usize,
    n: usize,
    k: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: PackedGemmScalar + Send> Send for PackedMatrix<T> {}

impl<T: PackedGemmScalar> PackedMatrix<T> {
    /// Pack the `A` operand of a future GEMM. `src` / `ld` describe
    /// the unpacked source matrix in the usual CBLAS convention; the
    /// resulting [`PackedMatrix`] can then be used in repeated
    /// [`gemm_compute_packed_a`] calls.
    #[allow(clippy::too_many_arguments)]
    pub fn pack_a(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        src: &[T],
        ld: usize,
    ) -> Result<Self> {
        Self::pack(layout, PackIdentifier::A, trans, m, n, k, alpha, src, ld)
    }

    /// Pack the `B` operand of a future GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn pack_b(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        src: &[T],
        ld: usize,
    ) -> Result<Self> {
        Self::pack(layout, PackIdentifier::B, trans, m, n, k, alpha, src, ld)
    }

    #[allow(clippy::too_many_arguments)]
    fn pack(
        layout: Layout,
        identifier: PackIdentifier,
        trans: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        src: &[T],
        ld: usize,
    ) -> Result<Self> {
        let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;
        let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let k_i: c_int = k.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ld_i: c_int = ld.try_into().map_err(|_| Error::DimensionOverflow)?;

        let bytes_needed =
            unsafe { T::gemm_pack_get_size(identifier.as_cblas(), m_i, n_i, k_i) };
        let elem_size = core::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(Error::InvalidArgument("zero-sized scalar"));
        }
        let elements_needed = bytes_needed.div_ceil(elem_size);
        // Validate src is large enough for the unpacked source. The
        // source has shape (m × k) for A or (k × n) for B, possibly
        // transposed.
        let (src_rows, src_cols) = match identifier {
            PackIdentifier::A => match trans {
                Transpose::NoTrans => (m, k),
                _ => (k, m),
            },
            PackIdentifier::B => match trans {
                Transpose::NoTrans => (k, n),
                _ => (n, k),
            },
        };
        let needed_src = match layout {
            Layout::RowMajor => src_rows.saturating_mul(ld),
            Layout::ColMajor => src_cols.saturating_mul(ld),
        };
        if src.len() < needed_src {
            return Err(Error::InvalidArgument(
                "src buffer too small for pack source shape",
            ));
        }

        let mut buffer: Vec<T> = Vec::with_capacity(elements_needed);
        // Safety: MKL writes `bytes_needed` bytes into the buffer;
        // we set len after the pack call. The buffer's allocation
        // is sized to fit those bytes (rounded up to an element
        // count).
        unsafe {
            T::gemm_pack(
                layout.as_cblas(),
                identifier.as_cblas(),
                trans.as_cblas(),
                m_i, n_i, k_i,
                alpha,
                src.as_ptr(),
                ld_i,
                buffer.as_mut_ptr(),
            );
            buffer.set_len(elements_needed);
        }
        Ok(Self {
            buffer,
            identifier,
            m, n, k,
            _marker: PhantomData,
        })
    }

    /// Which operand this packed matrix represents.
    #[inline]
    #[must_use]
    pub fn identifier(&self) -> PackIdentifier {
        self.identifier
    }
}

const CBLAS_PACKED: c_int = 151; // sys::CBLAS_STORAGE::CblasPacked

#[inline]
fn trans_as_compute_int(t: Transpose) -> c_int {
    t.as_cblas() as c_int
}

/// Run `C ← op(A_packed) · op(B) + beta · C` where `A_packed` was
/// produced by [`PackedMatrix::pack_a`].
///
/// `transb` applies to the unpacked `B` operand. `alpha` is baked
/// into `A_packed` (it was applied at pack time), so this function
/// has no `alpha` parameter.
#[allow(clippy::too_many_arguments)]
pub fn gemm_compute_packed_a<T: PackedGemmScalar>(
    layout: Layout,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_packed: &PackedMatrix<T>,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) -> Result<()> {
    if a_packed.identifier != PackIdentifier::A {
        return Err(Error::InvalidArgument(
            "a_packed must be packed as the A operand",
        ));
    }
    if a_packed.m != m || a_packed.n != n || a_packed.k != k {
        return Err(Error::InvalidArgument(
            "a_packed shape does not match the requested compute shape",
        ));
    }
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let k_i: c_int = k.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldb_i: c_int = ldb.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldc_i: c_int = ldc.try_into().map_err(|_| Error::DimensionOverflow)?;
    // For packed compute, MKL ignores lda but still reads it as a
    // parameter; pass m to keep it well-defined.
    let lda_i: c_int = m_i;
    unsafe {
        T::gemm_compute(
            layout.as_cblas(),
            CBLAS_PACKED,
            trans_as_compute_int(transb),
            m_i, n_i, k_i,
            a_packed.buffer.as_ptr(),
            lda_i,
            b.as_ptr(),
            ldb_i,
            beta,
            c.as_mut_ptr(),
            ldc_i,
        );
    }
    Ok(())
}

/// Run `C ← op(A) · op(B_packed) + beta · C` where `B_packed` was
/// produced by [`PackedMatrix::pack_b`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_compute_packed_b<T: PackedGemmScalar>(
    layout: Layout,
    transa: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    b_packed: &PackedMatrix<T>,
    beta: T,
    c: &mut [T],
    ldc: usize,
) -> Result<()> {
    if b_packed.identifier != PackIdentifier::B {
        return Err(Error::InvalidArgument(
            "b_packed must be packed as the B operand",
        ));
    }
    if b_packed.m != m || b_packed.n != n || b_packed.k != k {
        return Err(Error::InvalidArgument(
            "b_packed shape does not match the requested compute shape",
        ));
    }
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let k_i: c_int = k.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldc_i: c_int = ldc.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldb_i: c_int = n_i;
    unsafe {
        T::gemm_compute(
            layout.as_cblas(),
            trans_as_compute_int(transa),
            CBLAS_PACKED,
            m_i, n_i, k_i,
            a.as_ptr(),
            lda_i,
            b_packed.buffer.as_ptr(),
            ldb_i,
            beta,
            c.as_mut_ptr(),
            ldc_i,
        );
    }
    Ok(())
}
