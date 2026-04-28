//! Mixed-precision GEMM kernels.
//!
//! Modern ML workloads run GEMMs at reduced precision for memory
//! bandwidth and throughput, while accumulating in higher precision
//! to limit error growth. oneMKL exposes:
//!
//! | Function | A / B type | C type | Use case |
//! | --- | --- | --- | --- |
//! | [`gemm_bf16_f32`] | bf16 | f32 | mixed-precision training, fp32 master weights |
//! | [`gemm_f16_f32`] | fp16 | f32 | similar to bf16 but narrower exponent |
//! | [`gemm_e5m2_f32`] | FP8 (E5M2) | f32 | activations / gradients (wide range) |
//! | [`gemm_e4m3_f32`] | FP8 (E4M3) | f32 | weights (high precision per range) |
//! | [`gemm_s8u8_s32`] | int8 / uint8 | int32 | quantized inference |
//! | [`gemm_s16_s32`] | int16 | int32 | quantized inference (wider) |
//! | [`hgemm`] | fp16 | fp16 | pure half-precision (legacy / offload) |
//!
//! # Bit-level inputs
//!
//! oneMKL takes the reduced-precision operands as raw bit patterns:
//! `&[u16]` for bf16 / fp16 / hgemm inputs, `&[u8]` for FP8, `&[i8]`
//! and `&[u8]` for int8, `&[i16]` for int16. This matches MKL's
//! C ABI and lets callers integrate with whatever bf16 / f16 / fp8
//! type they're already using (e.g. `half::bf16` from the `half`
//! crate, custom newtypes, or raw bytes from `bytemuck`).
//!
//! Convert via `bytemuck::cast_slice` or
//! [`std::slice::from_raw_parts`]:
//!
//! ```ignore
//! let bf16_values: Vec<half::bf16> = ...;
//! // half::bf16 is #[repr(transparent)] over u16, so casting is sound.
//! let bits: &[u16] = bytemuck::cast_slice(&bf16_values);
//! gemm_bf16_f32(...args..., bits, ...);
//! ```

use core::ffi::{c_char, c_int, c_short, c_ushort, c_void};

use onemkl_sys as sys;

use crate::enums::{Layout, Transpose};
use crate::error::{Error, Result};

/// Offset semantics for the integer GEMM kernels (`gemm_s8u8_s32`,
/// `gemm_s16_s32`). Selects how the per-row, per-column, or constant
/// `cb` offset is broadcast into `C`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CblasOffset {
    /// Single row of length `n` broadcast across all rows.
    #[default]
    Row,
    /// Single column of length `m` broadcast across all columns.
    Col,
    /// Single scalar broadcast everywhere.
    Fix,
}

impl CblasOffset {
    #[inline]
    fn as_cblas(self) -> sys::CBLAS_OFFSET::Type {
        match self {
            Self::Row => sys::CBLAS_OFFSET::CblasRowOffset,
            Self::Col => sys::CBLAS_OFFSET::CblasColOffset,
            Self::Fix => sys::CBLAS_OFFSET::CblasFixOffset,
        }
    }
}

#[inline]
fn validate_gemm_dims(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
    layout: Layout,
) -> Result<()> {
    let (a_rows, a_cols) = match transa {
        Transpose::NoTrans => (m, k),
        _ => (k, m),
    };
    let (b_rows, b_cols) = match transb {
        Transpose::NoTrans => (k, n),
        _ => (n, k),
    };
    let needed_a = match layout {
        Layout::RowMajor => a_rows.saturating_mul(lda),
        Layout::ColMajor => a_cols.saturating_mul(lda),
    };
    let needed_b = match layout {
        Layout::RowMajor => b_rows.saturating_mul(ldb),
        Layout::ColMajor => b_cols.saturating_mul(ldb),
    };
    let needed_c = match layout {
        Layout::RowMajor => m.saturating_mul(ldc),
        Layout::ColMajor => n.saturating_mul(ldc),
    };
    if a_len < needed_a {
        return Err(Error::InvalidArgument("A buffer too small for op(A) shape"));
    }
    if b_len < needed_b {
        return Err(Error::InvalidArgument("B buffer too small for op(B) shape"));
    }
    if c_len < needed_c {
        return Err(Error::InvalidArgument("C buffer too small for shape"));
    }
    Ok(())
}

#[inline]
fn dim(x: usize) -> Result<c_int> {
    x.try_into().map_err(|_| Error::DimensionOverflow)
}

/// `C ← alpha · op(A) · op(B) + beta · C` where `A` and `B` are bf16
/// and `C` is f32. Wraps `cblas_gemm_bf16bf16f32`.
///
/// `a` and `b` are slices of bf16 bit patterns (one `u16` per
/// element). See the module-level docs for converting from your
/// preferred bf16 representation.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_f32(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u16],
    lda: usize,
    b: &[u16],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> Result<()> {
    validate_gemm_dims(
        transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc, layout,
    )?;
    unsafe {
        sys::cblas_gemm_bf16bf16f32(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            alpha,
            a.as_ptr(),
            dim(lda)?,
            b.as_ptr(),
            dim(ldb)?,
            beta,
            c.as_mut_ptr(),
            dim(ldc)?,
        );
    }
    Ok(())
}

/// `C ← alpha · op(A) · op(B) + beta · C` where `A` and `B` are fp16
/// and `C` is f32. Wraps `cblas_gemm_f16f16f32`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f16_f32(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u16],
    lda: usize,
    b: &[u16],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> Result<()> {
    validate_gemm_dims(
        transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc, layout,
    )?;
    unsafe {
        sys::cblas_gemm_f16f16f32(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            alpha,
            a.as_ptr(),
            dim(lda)?,
            b.as_ptr(),
            dim(ldb)?,
            beta,
            c.as_mut_ptr(),
            dim(ldc)?,
        );
    }
    Ok(())
}

/// `C ← alpha · op(A) · op(B) + beta · C` where `A` and `B` are FP8
/// in E5M2 format (5-bit exponent, 2-bit mantissa) and `C` is f32.
/// Wraps `cblas_gemm_e5m2e5m2f32`.
///
/// E5M2 has wider dynamic range than E4M3 — typically used for
/// activations and gradients during FP8 mixed-precision training.
#[allow(clippy::too_many_arguments)]
pub fn gemm_e5m2_f32(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u8],
    lda: usize,
    b: &[u8],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> Result<()> {
    validate_gemm_dims(
        transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc, layout,
    )?;
    unsafe {
        sys::cblas_gemm_e5m2e5m2f32(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            alpha,
            a.as_ptr(),
            dim(lda)?,
            b.as_ptr(),
            dim(ldb)?,
            beta,
            c.as_mut_ptr(),
            dim(ldc)?,
        );
    }
    Ok(())
}

/// `C ← alpha · op(A) · op(B) + beta · C` where `A` and `B` are FP8
/// in E4M3 format (4-bit exponent, 3-bit mantissa) and `C` is f32.
/// Wraps `cblas_gemm_e4m3e4m3f32`.
///
/// E4M3 has higher per-range precision than E5M2 — typically used
/// for weights during FP8 mixed-precision inference / training.
#[allow(clippy::too_many_arguments)]
pub fn gemm_e4m3_f32(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u8],
    lda: usize,
    b: &[u8],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> Result<()> {
    validate_gemm_dims(
        transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc, layout,
    )?;
    unsafe {
        sys::cblas_gemm_e4m3e4m3f32(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            alpha,
            a.as_ptr(),
            dim(lda)?,
            b.as_ptr(),
            dim(ldb)?,
            beta,
            c.as_mut_ptr(),
            dim(ldc)?,
        );
    }
    Ok(())
}

/// `C ← alpha · (op(A) + ao) · (op(B) + bo) + beta · C + cb` where
/// `A` is signed int8, `B` is unsigned int8, and `C` is int32 with a
/// per-row / per-col / scalar bias `cb` selected by `offset_c`.
/// Wraps `cblas_gemm_s8u8s32`.
///
/// This is the canonical quantized-inference primitive: the `ao` /
/// `bo` zero points and `cb` bias capture the asymmetric
/// quantization scheme used by most quantized models.
#[allow(clippy::too_many_arguments)]
pub fn gemm_s8u8_s32(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    offset_c: CblasOffset,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[i8],
    lda: usize,
    ao: i8,
    b: &[u8],
    ldb: usize,
    bo: i8,
    beta: f32,
    c: &mut [i32],
    ldc: usize,
    cb: &[i32],
) -> Result<()> {
    validate_gemm_dims(
        transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc, layout,
    )?;
    let needed_cb = match offset_c {
        CblasOffset::Row => n,
        CblasOffset::Col => m,
        CblasOffset::Fix => 1,
    };
    if cb.len() < needed_cb {
        return Err(Error::InvalidArgument(
            "cb buffer too small for the selected offset mode",
        ));
    }
    unsafe {
        sys::cblas_gemm_s8u8s32(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            offset_c.as_cblas(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            alpha,
            a.as_ptr() as *const c_void,
            dim(lda)?,
            ao as c_char,
            b.as_ptr() as *const c_void,
            dim(ldb)?,
            bo as c_char,
            beta,
            c.as_mut_ptr(),
            dim(ldc)?,
            cb.as_ptr(),
        );
    }
    Ok(())
}

/// `C ← alpha · (op(A) + ao) · (op(B) + bo) + beta · C + cb` for
/// int16 inputs and int32 output. Wraps `cblas_gemm_s16s16s32`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_s16_s32(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    offset_c: CblasOffset,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[i16],
    lda: usize,
    ao: i16,
    b: &[i16],
    ldb: usize,
    bo: i16,
    beta: f32,
    c: &mut [i32],
    ldc: usize,
    cb: &[i32],
) -> Result<()> {
    validate_gemm_dims(
        transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc, layout,
    )?;
    let needed_cb = match offset_c {
        CblasOffset::Row => n,
        CblasOffset::Col => m,
        CblasOffset::Fix => 1,
    };
    if cb.len() < needed_cb {
        return Err(Error::InvalidArgument(
            "cb buffer too small for the selected offset mode",
        ));
    }
    unsafe {
        sys::cblas_gemm_s16s16s32(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            offset_c.as_cblas(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            alpha,
            a.as_ptr(),
            dim(lda)?,
            ao as c_short,
            b.as_ptr(),
            dim(ldb)?,
            bo as c_short,
            beta,
            c.as_mut_ptr(),
            dim(ldc)?,
            cb.as_ptr(),
        );
    }
    Ok(())
}

/// `C ← alpha · op(A) · op(B) + beta · C` with all operands in fp16
/// (alpha / beta included). Wraps `cblas_hgemm`.
///
/// Pre-Sapphire-Rapids legacy path; for modern hardware prefer
/// [`gemm_f16_f32`] (fp32 accumulator) for better numerical accuracy
/// at the same fp16 input precision.
#[allow(clippy::too_many_arguments)]
pub fn hgemm(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha_bits: u16,
    a: &[u16],
    lda: usize,
    b: &[u16],
    ldb: usize,
    beta_bits: u16,
    c: &mut [u16],
    ldc: usize,
) -> Result<()> {
    validate_gemm_dims(
        transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc, layout,
    )?;
    unsafe {
        sys::cblas_hgemm(
            layout.as_cblas(),
            transa.as_cblas(),
            transb.as_cblas(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            alpha_bits as c_ushort,
            a.as_ptr(),
            dim(lda)?,
            b.as_ptr(),
            dim(ldb)?,
            beta_bits as c_ushort,
            c.as_mut_ptr(),
            dim(ldc)?,
        );
    }
    Ok(())
}

// =====================================================================
// bf16 helpers
// =====================================================================

/// Convert an `f32` to its bf16 bit pattern (round-to-nearest-even,
/// no NaN handling beyond preserving NaN-ness). Useful for tests and
/// ad-hoc data preparation; production callers typically use the
/// `half` crate.
#[inline]
#[must_use]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN: preserve the high mantissa bit.
        return ((bits >> 16) | 0x0040) as u16;
    }
    // Round-to-nearest-even.
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Convert a bf16 bit pattern back to `f32`.
#[inline]
#[must_use]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
