#![cfg(feature = "blas")]

//! Verify the Compact BLAS pack / unpack / gemm cycle.

use approx::assert_abs_diff_eq;

use onemkl::blas::compact::{gemm, gepack, geunpack, get_size, CompactFormat};
use onemkl::{Layout, Transpose};

#[test]
fn pack_unpack_roundtrip_2x2() {
    // Two 2×2 matrices.
    let a0 = vec![1.0_f64, 2.0, 3.0, 4.0];
    let a1 = vec![5.0_f64, 6.0, 7.0, 8.0];
    let format = CompactFormat::Avx;
    let nm = 2;
    let ld = 2; // leading dim for 2×2 stored row-major

    // Compute compact buffer size and allocate.
    let buf_len = get_size::<f64>(ld, ld, format, nm).unwrap();
    let mut packed = vec![0.0_f64; buf_len];

    // Pack the two matrices.
    let ptrs: [*const f64; 2] = [a0.as_ptr(), a1.as_ptr()];
    unsafe {
        gepack::<f64>(
            Layout::RowMajor, ld, ld,
            &ptrs, ld,
            &mut packed, ld,
            format,
        )
        .unwrap();
    }

    // Unpack back into fresh buffers.
    let mut out0 = vec![0.0_f64; 4];
    let mut out1 = vec![0.0_f64; 4];
    let out_ptrs: [*mut f64; 2] = [out0.as_mut_ptr(), out1.as_mut_ptr()];
    unsafe {
        geunpack::<f64>(
            Layout::RowMajor, ld, ld,
            &out_ptrs, ld,
            &packed, ld,
            format,
        )
        .unwrap();
    }

    assert_eq!(out0, a0);
    assert_eq!(out1, a1);
}

#[test]
fn compact_gemm_two_2x2_matmuls() {
    // Compact GEMM: pack two pairs of 2×2 matrices, multiply, unpack.
    let a0 = vec![1.0_f64, 2.0, 3.0, 4.0];
    let a1 = vec![1.0_f64, 0.0, 0.0, 1.0]; // identity

    let b0 = vec![1.0_f64, 0.0, 0.0, 1.0]; // identity
    let b1 = vec![5.0_f64, 6.0, 7.0, 8.0];

    let format = CompactFormat::Avx;
    let nm = 2;
    let ld = 2;

    let buf_len = get_size::<f64>(ld, ld, format, nm).unwrap();
    let mut a_packed = vec![0.0_f64; buf_len];
    let mut b_packed = vec![0.0_f64; buf_len];
    let mut c_packed = vec![0.0_f64; buf_len];

    let a_ptrs: [*const f64; 2] = [a0.as_ptr(), a1.as_ptr()];
    let b_ptrs: [*const f64; 2] = [b0.as_ptr(), b1.as_ptr()];
    unsafe {
        gepack::<f64>(Layout::RowMajor, ld, ld, &a_ptrs, ld, &mut a_packed, ld, format).unwrap();
        gepack::<f64>(Layout::RowMajor, ld, ld, &b_ptrs, ld, &mut b_packed, ld, format).unwrap();
    }

    gemm::<f64>(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0,
        &a_packed, ld,
        &b_packed, ld,
        0.0, &mut c_packed, ld,
        format, nm,
    )
    .unwrap();

    let mut c0 = vec![0.0_f64; 4];
    let mut c1 = vec![0.0_f64; 4];
    let c_ptrs: [*mut f64; 2] = [c0.as_mut_ptr(), c1.as_mut_ptr()];
    unsafe {
        geunpack::<f64>(Layout::RowMajor, ld, ld, &c_ptrs, ld, &c_packed, ld, format).unwrap();
    }

    // c0 = a0 * I = a0
    assert_abs_diff_eq!(c0[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[2], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[3], 4.0, epsilon = 1e-12);
    // c1 = I * b1 = b1
    assert_abs_diff_eq!(c1[0], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c1[1], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c1[2], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c1[3], 8.0, epsilon = 1e-12);
}

#[test]
fn get_size_returns_positive() {
    let s_avx = get_size::<f64>(4, 4, CompactFormat::Avx, 8).unwrap();
    let s_sse = get_size::<f64>(4, 4, CompactFormat::Sse, 8).unwrap();
    assert!(s_avx > 0);
    assert!(s_sse > 0);
}

#[test]
fn compact_gemm_works_for_f32() {
    let a0 = vec![1.0_f32, 2.0, 3.0, 4.0];
    let b0 = vec![1.0_f32, 0.0, 0.0, 1.0];
    let format = CompactFormat::Avx;
    let nm = 1;
    let ld = 2;

    let buf_len = get_size::<f32>(ld, ld, format, nm).unwrap();
    let mut a_packed = vec![0.0_f32; buf_len];
    let mut b_packed = vec![0.0_f32; buf_len];
    let mut c_packed = vec![0.0_f32; buf_len];

    let a_ptrs: [*const f32; 1] = [a0.as_ptr()];
    let b_ptrs: [*const f32; 1] = [b0.as_ptr()];
    unsafe {
        gepack::<f32>(Layout::RowMajor, ld, ld, &a_ptrs, ld, &mut a_packed, ld, format).unwrap();
        gepack::<f32>(Layout::RowMajor, ld, ld, &b_ptrs, ld, &mut b_packed, ld, format).unwrap();
    }

    gemm::<f32>(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0,
        &a_packed, ld,
        &b_packed, ld,
        0.0, &mut c_packed, ld,
        format, nm,
    )
    .unwrap();

    let mut c0 = vec![0.0_f32; 4];
    let c_ptrs: [*mut f32; 1] = [c0.as_mut_ptr()];
    unsafe {
        geunpack::<f32>(Layout::RowMajor, ld, ld, &c_ptrs, ld, &c_packed, ld, format).unwrap();
    }

    assert_abs_diff_eq!(c0[0], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(c0[3], 4.0, epsilon = 1e-5);
}
