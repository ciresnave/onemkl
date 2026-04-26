#![cfg(feature = "blas")]

//! Verify BLAS Level 2 and Level 3 wrappers against hand-computed values.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::blas::level2::gemv;
use onemkl::blas::level3::gemm;
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::{Layout, Transpose};

#[test]
fn gemv_real_row_major() {
    // A is 2x3:
    //   [1 2 3]
    //   [4 5 6]
    let a_buf = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = MatrixRef::new(&a_buf, 2, 3, Layout::RowMajor).unwrap();
    let x = vec![1.0_f64, 1.0, 1.0];
    let mut y = vec![0.0_f64, 0.0];

    // y ← 1 * A * x + 0 * y = [1+2+3, 4+5+6] = [6, 15].
    gemv(Transpose::NoTrans, 1.0, &a, &x, 1, 0.0, &mut y, 1).unwrap();
    assert_abs_diff_eq!(y[0], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 15.0, epsilon = 1e-12);
}

#[test]
fn gemv_real_col_major() {
    // Same logical 2x3 matrix but column-major:
    //   col 0: [1, 4]
    //   col 1: [2, 5]
    //   col 2: [3, 6]
    let a_buf = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a = MatrixRef::new(&a_buf, 2, 3, Layout::ColMajor).unwrap();
    let x = vec![1.0_f64, 1.0, 1.0];
    let mut y = vec![0.0_f64, 0.0];

    gemv(Transpose::NoTrans, 1.0, &a, &x, 1, 0.0, &mut y, 1).unwrap();
    assert_abs_diff_eq!(y[0], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 15.0, epsilon = 1e-12);
}

#[test]
fn gemv_with_alpha_beta() {
    // A = [[1, 2], [3, 4]], x = [1, 1]
    // y ← 2 * A * x + 3 * y, with y = [10, 20]
    // A*x = [3, 7]; result = 2*[3, 7] + 3*[10, 20] = [6+30, 14+60] = [36, 74]
    let a_buf = [1.0_f64, 2.0, 3.0, 4.0];
    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let x = vec![1.0_f64, 1.0];
    let mut y = vec![10.0_f64, 20.0];

    gemv(Transpose::NoTrans, 2.0, &a, &x, 1, 3.0, &mut y, 1).unwrap();
    assert_abs_diff_eq!(y[0], 36.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 74.0, epsilon = 1e-12);
}

#[test]
fn gemv_transposed() {
    // A = [[1, 2], [3, 4]] (row-major)
    // Aᵀ * x with x = [1, 2]: Aᵀ = [[1, 3], [2, 4]] → [1+6, 2+8] = [7, 10]
    let a_buf = [1.0_f64, 2.0, 3.0, 4.0];
    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let x = vec![1.0_f64, 2.0];
    let mut y = vec![0.0_f64, 0.0];

    gemv(Transpose::Trans, 1.0, &a, &x, 1, 0.0, &mut y, 1).unwrap();
    assert_abs_diff_eq!(y[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 10.0, epsilon = 1e-12);
}

#[test]
fn gemm_real_row_major_no_trans() {
    // A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    // B: 3x2 = [[1, 0], [0, 1], [1, 1]]
    // A*B = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4, 5], [10, 11]]
    let a_buf = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_buf = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut c_buf = [0.0_f64; 4];

    let a = MatrixRef::new(&a_buf, 2, 3, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 3, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0,
        &a,
        &b,
        0.0,
        &mut c,
    )
    .unwrap();

    assert_abs_diff_eq!(c_buf[0], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[2], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3], 11.0, epsilon = 1e-12);
}

#[test]
fn gemm_real_col_major_matches_row_major() {
    // Same 2x2 * 2x2 multiplication done both ways.
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

    // Row-major.
    let a_rm = [1.0_f64, 2.0, 3.0, 4.0];
    let b_rm = [5.0_f64, 6.0, 7.0, 8.0];
    let mut c_rm = [0.0_f64; 4];
    {
        let a = MatrixRef::new(&a_rm, 2, 2, Layout::RowMajor).unwrap();
        let b = MatrixRef::new(&b_rm, 2, 2, Layout::RowMajor).unwrap();
        let mut c = MatrixMut::new(&mut c_rm, 2, 2, Layout::RowMajor).unwrap();
        gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        )
        .unwrap();
    }
    assert_abs_diff_eq!(c_rm[0], 19.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_rm[1], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_rm[2], 43.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_rm[3], 50.0, epsilon = 1e-12);

    // Column-major: same logical matrices stored col-by-col.
    let a_cm = [1.0_f64, 3.0, 2.0, 4.0]; // cols (1,3) (2,4)
    let b_cm = [5.0_f64, 7.0, 6.0, 8.0]; // cols (5,7) (6,8)
    let mut c_cm = [0.0_f64; 4];
    {
        let a = MatrixRef::new(&a_cm, 2, 2, Layout::ColMajor).unwrap();
        let b = MatrixRef::new(&b_cm, 2, 2, Layout::ColMajor).unwrap();
        let mut c = MatrixMut::new(&mut c_cm, 2, 2, Layout::ColMajor).unwrap();
        gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        )
        .unwrap();
    }
    // Column-major C: c[0] = (0,0), c[1] = (1,0), c[2] = (0,1), c[3] = (1,1)
    assert_abs_diff_eq!(c_cm[0], 19.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_cm[1], 43.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_cm[2], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_cm[3], 50.0, epsilon = 1e-12);
}

#[test]
fn gemm_complex_with_conj_trans() {
    // A: 1x2 row-major = [1+i, 2-i]
    // We compute Aᴴ * A (a 2x2 outer-product-like result).
    //
    //   Aᴴ = [[1-i], [2+i]]   (conjugate-transpose of A)
    //   Aᴴ * A = [[(1-i)(1+i), (1-i)(2-i)],
    //             [(2+i)(1+i), (2+i)(2-i)]]
    //          = [[1+1,        2-i-2i+i²],     [(1-i)(2-i)= 2-i-2i+i²= 2-3i-1=1-3i]
    //             [2+i+2i+i²,  4 - i²       ]]
    //          = [[2, 1-3i],
    //             [1+3i, 5]]
    let a_buf = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
    let mut c_buf = [Complex64::new(0.0, 0.0); 4];

    let a_for_lhs = MatrixRef::new(&a_buf, 1, 2, Layout::RowMajor).unwrap();
    let a_for_rhs = MatrixRef::new(&a_buf, 1, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    // op(A_lhs) = Aᴴ → 2x1
    // op(A_rhs) = A   → 1x2
    // result    = 2x2
    gemm(
        Transpose::ConjTrans,
        Transpose::NoTrans,
        Complex64::new(1.0, 0.0),
        &a_for_lhs,
        &a_for_rhs,
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    assert_abs_diff_eq!(c_buf[0].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[0].im, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1].im, -3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[2].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[2].im, 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3].re, 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3].im, 0.0, epsilon = 1e-12);
}

#[test]
fn gemm_layout_mismatch_rejected() {
    let a_buf = [1.0_f64; 4];
    let b_buf = [1.0_f64; 4];
    let mut c_buf = [0.0_f64; 4];
    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 2, 2, Layout::ColMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    let err = gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0,
        &a,
        &b,
        0.0,
        &mut c,
    );
    assert!(err.is_err());
}
