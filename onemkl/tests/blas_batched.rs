//! Verify the batched (`*_batch_strided`) BLAS-like extension wrappers.
//!
//! Each test runs a batch of independent operations and checks the
//! per-batch result against the same operation done one batch at a time.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::blas::extensions::{
    axpy_batch_strided, copy_batch_strided, dgmm_batch_strided, gemm3m_batch_strided,
    gemm_batch_strided, gemv_batch_strided, trsm_batch_strided,
};
use onemkl::{Diag, Layout, Side, Transpose, UpLo};

#[test]
fn axpy_batch_strided_real() {
    // Three independent axpy ops. n=3, batch_size=3, stride = 3 elements.
    let x = vec![
        // batch 0: [1, 1, 1]
        1.0_f64, 1.0, 1.0,
        // batch 1: [2, 2, 2]
        2.0, 2.0, 2.0,
        // batch 2: [3, 3, 3]
        3.0, 3.0, 3.0,
    ];
    let mut y = vec![
        10.0_f64, 10.0, 10.0,
        20.0, 20.0, 20.0,
        30.0, 30.0, 30.0,
    ];
    axpy_batch_strided(3, 1.0, &x, 1, 3, &mut y, 1, 3, 3).unwrap();
    assert_abs_diff_eq!(y[0], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[3], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[6], 33.0, epsilon = 1e-12);
}

#[test]
fn copy_batch_strided_real() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut y = vec![0.0_f64; 6];
    copy_batch_strided(3, &x, 1, 3, &mut y, 1, 3, 2).unwrap();
    assert_eq!(x, y);
}

#[test]
fn gemm_batch_strided_real() {
    // Two independent 2x2 matrix multiplications.
    // A1 = [[1,2],[3,4]], A2 = [[5,6],[7,8]]
    // B1 = [[1,0],[0,1]] (identity), B2 = same
    // C1 = A1*B1 = A1; C2 = A2*B2 = A2.
    let a = vec![
        1.0_f64, 2.0, 3.0, 4.0, // A1 (row-major)
        5.0, 6.0, 7.0, 8.0,    // A2
    ];
    let b = vec![
        1.0_f64, 0.0, 0.0, 1.0, // B1
        1.0, 0.0, 0.0, 1.0,    // B2
    ];
    let mut c = vec![0.0_f64; 8];
    gemm_batch_strided(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        2, 2, 2,
        1.0,
        &a, 2, 4,  // lda = 2, stridea = 4 (4 elements per matrix)
        &b, 2, 4,
        0.0,
        &mut c, 2, 4,
        2,
    )
    .unwrap();
    assert_eq!(c, a);
}

#[test]
fn gemv_batch_strided_real() {
    // Two independent 2x2 matrix-vector multiplies. Each matrix is the
    // identity, so y = x.
    let a = vec![
        1.0_f64, 0.0, 0.0, 1.0, // I
        1.0, 0.0, 0.0, 1.0,    // I
    ];
    let x = vec![
        3.0_f64, 4.0,
        7.0, 9.0,
    ];
    let mut y = vec![0.0_f64; 4];
    gemv_batch_strided(
        Layout::RowMajor,
        Transpose::NoTrans,
        2, 2,
        1.0,
        &a, 2, 4,
        &x, 1, 2,
        0.0,
        &mut y, 1, 2,
        2,
    )
    .unwrap();
    assert_eq!(y, x);
}

#[test]
fn trsm_batch_strided_real() {
    // Two independent triangular solves. Each A is the 2x2 identity
    // (upper, non-unit). Solve A * X = B → X = B.
    let a = vec![
        1.0_f64, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
    ];
    let mut b = vec![
        2.0_f64, 5.0, 3.0, 7.0,   // batch 0: 2x2 RHS [[2,5],[3,7]]
        11.0, 13.0, 17.0, 19.0,  // batch 1
    ];
    let original = b.clone();
    trsm_batch_strided(
        Layout::RowMajor,
        Side::Left,
        UpLo::Upper,
        Transpose::NoTrans,
        Diag::NonUnit,
        2, 2,
        1.0,
        &a, 2, 4,
        &mut b, 2, 4,
        2,
    )
    .unwrap();
    assert_eq!(b, original);
}

#[test]
fn dgmm_batch_strided_real() {
    // C[i] = A[i] * diag(x[i]) (Side::Right).
    // A = identity 2x2; x = [2, 3] → C = diag(2, 3).
    let a = vec![
        1.0_f64, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
    ];
    let x = vec![
        2.0_f64, 3.0,
        4.0, 5.0,
    ];
    let mut c = vec![0.0_f64; 8];
    dgmm_batch_strided(
        Layout::RowMajor,
        Side::Right,
        2, 2,
        &a, 2, 4,
        &x, 1, 2,
        &mut c, 2, 4,
        2,
    )
    .unwrap();
    // Batch 0: diag(2, 3) → [[2, 0], [0, 3]]
    assert_abs_diff_eq!(c[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[3], 3.0, epsilon = 1e-12);
    // Batch 1: diag(4, 5) → [[4, 0], [0, 5]]
    assert_abs_diff_eq!(c[4], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[5], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[6], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[7], 5.0, epsilon = 1e-12);
}

#[test]
fn gemm3m_batch_strided_complex_matches_gemm_batch() {
    use onemkl::blas::extensions::gemm_batch_strided as gemm_batched;

    // Same matrices for both algorithms; should agree to ~1e-10.
    let a: Vec<Complex64> = (0..8)
        .map(|i| Complex64::new((i + 1) as f64 * 0.5, -(i as f64) * 0.25))
        .collect();
    let b: Vec<Complex64> = (0..8)
        .map(|i| Complex64::new(-(i as f64) * 0.1, (i + 2) as f64 * 0.3))
        .collect();
    let mut c_gemm = vec![Complex64::new(0.0, 0.0); 8];
    let mut c_3m = vec![Complex64::new(0.0, 0.0); 8];

    gemm_batched(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        2, 2, 2,
        Complex64::new(1.0, 0.0),
        &a, 2, 4,
        &b, 2, 4,
        Complex64::new(0.0, 0.0),
        &mut c_gemm, 2, 4,
        2,
    )
    .unwrap();
    gemm3m_batch_strided(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        2, 2, 2,
        Complex64::new(1.0, 0.0),
        &a, 2, 4,
        &b, 2, 4,
        Complex64::new(0.0, 0.0),
        &mut c_3m, 2, 4,
        2,
    )
    .unwrap();

    for (g, m) in c_gemm.iter().zip(&c_3m) {
        assert_abs_diff_eq!(g.re, m.re, epsilon = 1e-10);
        assert_abs_diff_eq!(g.im, m.im, epsilon = 1e-10);
    }
}
