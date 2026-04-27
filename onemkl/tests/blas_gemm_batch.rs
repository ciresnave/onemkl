#![cfg(feature = "blas")]

//! Verify pointer-array batched GEMM (cblas_*gemm_batch).

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::blas::extensions::{gemm_batch, GemmBatchGroup};
use onemkl::{Layout, Transpose};

#[test]
fn three_2x2_gemms_uniform_shape() {
    // Three 2×2 GEMMs, all with the same shape, packed into one group.
    // Each GEMM does C = 1.0 * A * B + 0.0 * C.
    let a0 = [1.0_f64, 2.0, 3.0, 4.0];
    let b0 = [1.0_f64, 0.0, 0.0, 1.0]; // identity
    let mut c0 = [0.0_f64; 4];

    let a1 = [5.0_f64, 6.0, 7.0, 8.0];
    let b1 = [2.0_f64, 0.0, 0.0, 2.0]; // 2 * identity
    let mut c1 = [0.0_f64; 4];

    let a2 = [9.0_f64, 10.0, 11.0, 12.0];
    let b2 = [0.0_f64, 1.0, 1.0, 0.0]; // swap columns
    let mut c2 = [0.0_f64; 4];

    let a_ptrs: [*const f64; 3] = [a0.as_ptr(), a1.as_ptr(), a2.as_ptr()];
    let b_ptrs: [*const f64; 3] = [b0.as_ptr(), b1.as_ptr(), b2.as_ptr()];
    let c_ptrs: [*mut f64; 3] = [c0.as_mut_ptr(), c1.as_mut_ptr(), c2.as_mut_ptr()];

    let mut groups = [GemmBatchGroup::<f64> {
        transa: Transpose::NoTrans,
        transb: Transpose::NoTrans,
        m: 2,
        n: 2,
        k: 2,
        alpha: 1.0,
        lda: 2,
        ldb: 2,
        beta: 0.0,
        ldc: 2,
        group_size: 3,
        a_array: &a_ptrs,
        b_array: &b_ptrs,
        c_array: &c_ptrs,
    }];

    gemm_batch(Layout::RowMajor, &mut groups).unwrap();

    // GEMM 0: A0 * I = A0
    assert_abs_diff_eq!(c0[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[2], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[3], 4.0, epsilon = 1e-12);
    // GEMM 1: A1 * 2I = 2 * A1
    assert_abs_diff_eq!(c1[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c1[1], 12.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c1[2], 14.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c1[3], 16.0, epsilon = 1e-12);
    // GEMM 2: A2 * column-swap = swap A2's columns
    //   A2 = [[9,10],[11,12]]; swap cols → [[10,9],[12,11]]
    assert_abs_diff_eq!(c2[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c2[1], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c2[2], 12.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c2[3], 11.0, epsilon = 1e-12);
}

#[test]
fn two_groups_different_shapes() {
    // Group 0: one 2×2 GEMM. Group 1: one 1×1 GEMM.
    let a0 = [1.0_f64, 2.0, 3.0, 4.0];
    let b0 = [5.0_f64, 6.0, 7.0, 8.0];
    let mut c0 = [0.0_f64; 4];

    // 1×1: a = [3], b = [4], c = 12
    let a1 = [3.0_f64];
    let b1 = [4.0_f64];
    let mut c1 = [0.0_f64; 1];

    let a0_ptrs: [*const f64; 1] = [a0.as_ptr()];
    let b0_ptrs: [*const f64; 1] = [b0.as_ptr()];
    let c0_ptrs: [*mut f64; 1] = [c0.as_mut_ptr()];

    let a1_ptrs: [*const f64; 1] = [a1.as_ptr()];
    let b1_ptrs: [*const f64; 1] = [b1.as_ptr()];
    let c1_ptrs: [*mut f64; 1] = [c1.as_mut_ptr()];

    let mut groups = [
        GemmBatchGroup::<f64> {
            transa: Transpose::NoTrans, transb: Transpose::NoTrans,
            m: 2, n: 2, k: 2,
            alpha: 1.0, lda: 2, ldb: 2, beta: 0.0, ldc: 2,
            group_size: 1,
            a_array: &a0_ptrs, b_array: &b0_ptrs, c_array: &c0_ptrs,
        },
        GemmBatchGroup::<f64> {
            transa: Transpose::NoTrans, transb: Transpose::NoTrans,
            m: 1, n: 1, k: 1,
            alpha: 1.0, lda: 1, ldb: 1, beta: 0.0, ldc: 1,
            group_size: 1,
            a_array: &a1_ptrs, b_array: &b1_ptrs, c_array: &c1_ptrs,
        },
    ];

    gemm_batch(Layout::RowMajor, &mut groups).unwrap();

    // 2×2 result: A0 * B0 = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //                     = [[19, 22], [43, 50]]
    assert_abs_diff_eq!(c0[0], 19.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[1], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[2], 43.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c0[3], 50.0, epsilon = 1e-12);
    // 1×1 result: 3 * 4 = 12
    assert_abs_diff_eq!(c1[0], 12.0, epsilon = 1e-12);
}

#[test]
fn empty_groups_is_noop() {
    let mut groups: [GemmBatchGroup<f64>; 0] = [];
    gemm_batch(Layout::RowMajor, &mut groups).unwrap();
}

#[test]
fn array_length_mismatch_rejected() {
    let a = [1.0_f64; 4];
    let b = [1.0_f64; 4];
    let mut c = [0.0_f64; 4];
    // Mismatched: group_size says 2 but only 1 pointer provided.
    let a_ptrs: [*const f64; 1] = [a.as_ptr()];
    let b_ptrs: [*const f64; 1] = [b.as_ptr()];
    let c_ptrs: [*mut f64; 1] = [c.as_mut_ptr()];
    let mut groups = [GemmBatchGroup::<f64> {
        transa: Transpose::NoTrans, transb: Transpose::NoTrans,
        m: 2, n: 2, k: 2,
        alpha: 1.0, lda: 2, ldb: 2, beta: 0.0, ldc: 2,
        group_size: 2, // claims 2 GEMMs
        a_array: &a_ptrs, b_array: &b_ptrs, c_array: &c_ptrs,
    }];
    let r = gemm_batch(Layout::RowMajor, &mut groups);
    assert!(r.is_err());
}

#[test]
fn complex_z_gemm_batch() {
    // One complex 1×1 GEMM: (1+i) * (1+i) = 2i.
    let a = [Complex64::new(1.0, 1.0)];
    let b = [Complex64::new(1.0, 1.0)];
    let mut c = [Complex64::new(0.0, 0.0)];
    let a_ptrs: [*const Complex64; 1] = [a.as_ptr()];
    let b_ptrs: [*const Complex64; 1] = [b.as_ptr()];
    let c_ptrs: [*mut Complex64; 1] = [c.as_mut_ptr()];
    let mut groups = [GemmBatchGroup::<Complex64> {
        transa: Transpose::NoTrans, transb: Transpose::NoTrans,
        m: 1, n: 1, k: 1,
        alpha: Complex64::new(1.0, 0.0),
        lda: 1, ldb: 1,
        beta: Complex64::new(0.0, 0.0),
        ldc: 1,
        group_size: 1,
        a_array: &a_ptrs, b_array: &b_ptrs, c_array: &c_ptrs,
    }];
    gemm_batch(Layout::RowMajor, &mut groups).unwrap();
    assert_abs_diff_eq!(c[0].re, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[0].im, 2.0, epsilon = 1e-12);
}
