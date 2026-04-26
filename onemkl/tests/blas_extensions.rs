#![cfg(feature = "blas")]

//! Verify the BLAS-like extension wrappers against hand-computed values.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::blas::extensions::{axpby, axpby_inc, imatcopy, omatadd, omatcopy};
use onemkl::{Layout, Transpose};

#[test]
fn axpby_real() {
    // y ← 2*x + 3*y
    let x = vec![1.0_f64, 2.0, 3.0];
    let mut y = vec![10.0_f64, 20.0, 30.0];
    axpby(2.0, &x, 3.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 32.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 64.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 96.0, epsilon = 1e-12);
}

#[test]
fn axpby_complex() {
    let x = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let mut y = vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)];
    // α = i, β = 1 → y ← i*x + y
    // i*x = [i, -1]; y_new = [i + i, -1 + 1] = [2i, 0]
    axpby(
        Complex64::new(0.0, 1.0),
        &x,
        Complex64::new(1.0, 0.0),
        &mut y,
    )
    .unwrap();
    assert_abs_diff_eq!(y[0].re, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[0].im, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1].re, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1].im, 0.0, epsilon = 1e-12);
}

#[test]
fn axpby_strided() {
    // Stride 2 over x and y.
    let x = vec![1.0_f64, 99.0, 2.0, 99.0, 3.0];
    let mut y = vec![10.0_f64, 99.0, 20.0, 99.0, 30.0];
    axpby_inc(1.0, &x, 2, 1.0, &mut y, 2).unwrap();
    assert_abs_diff_eq!(y[0], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[4], 33.0, epsilon = 1e-12);
}

#[test]
fn omatcopy_transpose_real() {
    // A row-major 2x3:
    //   [1 2 3]
    //   [4 5 6]
    // op(A) with Trans = 3x2:
    //   [1 4]
    //   [2 5]
    //   [3 6]
    // α = 1
    let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut b = [0.0_f64; 6];
    omatcopy(
        Layout::RowMajor,
        Transpose::Trans,
        2, 3, // rows × cols of A
        1.0,
        &a, 3,
        &mut b, 2,
    )
    .unwrap();
    // Output (3x2 row-major): each row stride = ldb = 2.
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[3], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[4], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[5], 6.0, epsilon = 1e-12);
}

#[test]
fn omatcopy_no_trans_with_alpha() {
    // 2x2 row-major, just scaling.
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let mut b = [0.0_f64; 4];
    omatcopy(
        Layout::RowMajor,
        Transpose::NoTrans,
        2, 2,
        2.0,
        &a, 2,
        &mut b, 2,
    )
    .unwrap();
    assert_abs_diff_eq!(b[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[2], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[3], 8.0, epsilon = 1e-12);
}

#[test]
fn omatcopy_conj_trans_complex() {
    // 1x2 row-major: [1+i, 2-i] → output 2x1 = conjugate-transpose:
    //   [1-i]
    //   [2+i]
    let a = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
    let mut b = [Complex64::new(0.0, 0.0); 2];
    omatcopy(
        Layout::RowMajor,
        Transpose::ConjTrans,
        1, 2,
        Complex64::new(1.0, 0.0),
        &a, 2,
        &mut b, 1,
    )
    .unwrap();
    assert_abs_diff_eq!(b[0].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[0].im, -1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1].im, 1.0, epsilon = 1e-12);
}

#[test]
fn imatcopy_in_place_transpose_square() {
    // Square 2x2 transpose in place.
    let mut ab = [1.0_f64, 2.0, 3.0, 4.0];
    imatcopy(
        Layout::RowMajor,
        Transpose::Trans,
        2, 2,
        1.0,
        &mut ab,
        2, 2,
    )
    .unwrap();
    // Original (row-major):  [[1, 2], [3, 4]]
    // Transposed:            [[1, 3], [2, 4]]
    assert_abs_diff_eq!(ab[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ab[1], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ab[2], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ab[3], 4.0, epsilon = 1e-12);
}

#[test]
fn omatadd_two_matrices() {
    // C ← 1 * A + 1 * B with A, B both 2x2 row-major, no transpose.
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let b = [10.0_f64, 20.0, 30.0, 40.0];
    let mut c = [0.0_f64; 4];
    omatadd(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        2, 2,
        1.0, &a, 2,
        1.0, &b, 2,
        &mut c, 2,
    )
    .unwrap();
    assert_abs_diff_eq!(c[0], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], 33.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[3], 44.0, epsilon = 1e-12);
}

#[test]
fn omatadd_with_transpose() {
    // A = [[1, 2], [3, 4]] (row-major); Aᵀ = [[1, 3], [2, 4]]
    // B = [[10, 20], [30, 40]]
    // C ← Aᵀ + B = [[11, 23], [32, 44]]
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let b = [10.0_f64, 20.0, 30.0, 40.0];
    let mut c = [0.0_f64; 4];
    omatadd(
        Layout::RowMajor,
        Transpose::Trans,
        Transpose::NoTrans,
        2, 2,
        1.0, &a, 2,
        1.0, &b, 2,
        &mut c, 2,
    )
    .unwrap();
    assert_abs_diff_eq!(c[0], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 23.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], 32.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[3], 44.0, epsilon = 1e-12);
}
