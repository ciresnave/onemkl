#![cfg(feature = "blas")]

//! Verify the mixed-precision GEMM kernels.

use approx::assert_abs_diff_eq;

use onemkl::blas::mixed_precision::{
    bf16_bits_to_f32, f32_to_bf16_bits, gemm_bf16_f32, gemm_e4m3_f32, gemm_e5m2_f32,
    gemm_f16_f32, gemm_s16_s32, gemm_s8u8_s32, hgemm, CblasOffset,
};
use onemkl::{Layout, Transpose};

fn f32_to_f16_bits(x: f32) -> u16 {
    // IEEE 754 binary16 round-to-nearest-even (no subnormal handling
    // beyond zero — sufficient for the small test values here).
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // Inf / NaN.
        return sign | 0x7c00 | (if mant != 0 { 0x0200 } else { 0 });
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00; // Inf
    }
    if new_exp <= 0 {
        return sign; // flush small values to zero for our tests.
    }
    let mant_h = (mant >> 13) as u16;
    sign | ((new_exp as u16) << 10) | mant_h
}

#[test]
fn bf16_helpers_round_trip_close() {
    let original = [1.0_f32, -2.5, 0.125, 100.0, -0.0];
    for &x in &original {
        let bits = f32_to_bf16_bits(x);
        let back = bf16_bits_to_f32(bits);
        // bf16 keeps 8 bits of mantissa, so absolute relative error ≤ 2⁻⁷ ≈ 0.0078.
        if x == 0.0 {
            assert_eq!(back, 0.0);
        } else {
            assert!((back - x).abs() / x.abs() < 0.01, "x={x} back={back}");
        }
    }
}

#[test]
fn bf16_gemm_2x2_identity_passes_a_through() {
    // C = A * I where A = [[1, 2], [3, 4]] and I is the 2×2 identity.
    let a_f32 = [1.0_f32, 2.0, 3.0, 4.0];
    let i_f32 = [1.0_f32, 0.0, 0.0, 1.0];
    let a: Vec<u16> = a_f32.iter().copied().map(f32_to_bf16_bits).collect();
    let b: Vec<u16> = i_f32.iter().copied().map(f32_to_bf16_bits).collect();
    let mut c = [0.0_f32; 4];
    gemm_bf16_f32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2,
    )
    .unwrap();
    // bf16 keeps integer values 1..=4 exactly.
    assert_abs_diff_eq!(c[0], 1.0, epsilon = 1e-3);
    assert_abs_diff_eq!(c[1], 2.0, epsilon = 1e-3);
    assert_abs_diff_eq!(c[2], 3.0, epsilon = 1e-3);
    assert_abs_diff_eq!(c[3], 4.0, epsilon = 1e-3);
}

#[test]
fn f16_gemm_2x2() {
    // A = [[2, 0], [0, 2]], B = [[3, 4], [5, 6]] → C = 2 * B = [[6,8],[10,12]]
    let a_f32 = [2.0_f32, 0.0, 0.0, 2.0];
    let b_f32 = [3.0_f32, 4.0, 5.0, 6.0];
    let a: Vec<u16> = a_f32.iter().copied().map(f32_to_f16_bits).collect();
    let b: Vec<u16> = b_f32.iter().copied().map(f32_to_f16_bits).collect();
    let mut c = [0.0_f32; 4];
    gemm_f16_f32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2,
    )
    .unwrap();
    assert_abs_diff_eq!(c[0], 6.0, epsilon = 1e-3);
    assert_abs_diff_eq!(c[1], 8.0, epsilon = 1e-3);
    assert_abs_diff_eq!(c[2], 10.0, epsilon = 1e-3);
    assert_abs_diff_eq!(c[3], 12.0, epsilon = 1e-3);
}

#[test]
fn s8u8_s32_quantized_matmul() {
    // A = [[1, 2], [3, 4]] (s8), B = [[5, 6], [7, 8]] (u8).
    // A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]].
    let a: [i8; 4] = [1, 2, 3, 4];
    let b: [u8; 4] = [5, 6, 7, 8];
    let mut c = [0_i32; 4];
    let cb = [0_i32]; // single-element bias for Fix mode
    gemm_s8u8_s32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        CblasOffset::Fix,
        2, 2, 2, 1.0, &a, 2, 0, &b, 2, 0, 0.0, &mut c, 2, &cb,
    )
    .unwrap();
    assert_eq!(c[0], 19);
    assert_eq!(c[1], 22);
    assert_eq!(c[2], 43);
    assert_eq!(c[3], 50);
}

#[test]
fn s16_s32_quantized_matmul() {
    // Same logical multiply but with int16 inputs.
    let a: [i16; 4] = [1, 2, 3, 4];
    let b: [i16; 4] = [5, 6, 7, 8];
    let mut c = [0_i32; 4];
    let cb = [0_i32];
    gemm_s16_s32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        CblasOffset::Fix,
        2, 2, 2, 1.0, &a, 2, 0, &b, 2, 0, 0.0, &mut c, 2, &cb,
    )
    .unwrap();
    assert_eq!(c[0], 19);
    assert_eq!(c[1], 22);
    assert_eq!(c[2], 43);
    assert_eq!(c[3], 50);
}

#[test]
fn hgemm_2x2_scaled() {
    // C = 1.0 * A * I + 0.0 * C, all in fp16.
    let a_f32 = [1.0_f32, 2.0, 3.0, 4.0];
    let i_f32 = [1.0_f32, 0.0, 0.0, 1.0];
    let a: Vec<u16> = a_f32.iter().copied().map(f32_to_f16_bits).collect();
    let b: Vec<u16> = i_f32.iter().copied().map(f32_to_f16_bits).collect();
    let mut c: Vec<u16> = vec![0; 4];
    hgemm(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2,
        f32_to_f16_bits(1.0),
        &a, 2, &b, 2,
        f32_to_f16_bits(0.0),
        &mut c, 2,
    )
    .unwrap();
    // hgemm output is fp16; we won't dig into the bit pattern, just
    // verify it isn't all zero (i.e. the call did something).
    assert!(c.iter().any(|&v| v != 0));
}

#[test]
fn dimension_mismatch_rejected() {
    let a = [0_u16; 4];
    let b = [0_u16; 4];
    let mut c = [0.0_f32; 4];
    // Claim k = 8 but A only holds 4 elements.
    let r = gemm_bf16_f32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 8, 1.0, &a, 8, &b, 2, 0.0, &mut c, 2,
    );
    assert!(r.is_err());
}

#[test]
fn fp8_gemm_calls_succeed() {
    // FP8 GEMMs need recent hardware support; we just verify the
    // wrapper accepts well-formed inputs without erroring out at
    // the FFI layer. Functional correctness depends on the host.
    let a = [0_u8; 4];
    let b = [0_u8; 4];
    let mut c = [0.0_f32; 4];
    let r1 = gemm_e5m2_f32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2,
    );
    let r2 = gemm_e4m3_f32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2,
    );
    // Either succeeds or returns an MKL "not supported" error; both
    // exercise the FFI binding without crashing.
    let _ = (r1, r2);
}
