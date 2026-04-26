#![cfg(feature = "lapack")]

//! Verify the packed LAPACK driver wrappers.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::lapack::{hpsv, hptrf, hptrs, ppsv, pptrf, pptrs, spsv, sptrf, sptrs};
use onemkl::{Layout, UpLo};

#[test]
fn ppsv_pd_packed() {
    // SPD matrix 3x3:
    //   [[ 4, -1,  0],
    //    [-1,  4, -1],
    //    [ 0, -1,  4]]
    // Column-major upper packed: ap = [a00, a01, a11, a02, a12, a22]
    //                                = [4, -1, 4, 0, -1, 4]
    let mut ap = vec![4.0_f64, -1.0, 4.0, 0.0, -1.0, 4.0];
    let mut b = vec![3.0_f64, 2.0, 3.0];
    ppsv::<f64>(Layout::ColMajor, UpLo::Upper, 3, 1, &mut ap, &mut b, 3).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn pptrf_then_pptrs() {
    let mut ap = vec![4.0_f64, -1.0, 4.0, 0.0, -1.0, 4.0];
    pptrf::<f64>(Layout::ColMajor, UpLo::Upper, 3, &mut ap).unwrap();
    let mut b = vec![3.0_f64, 2.0, 3.0];
    pptrs::<f64>(Layout::ColMajor, UpLo::Upper, 3, 1, &ap, &mut b, 3).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn spsv_symmetric_packed() {
    // Symmetric (non-PD) 2x2: [[1, 2], [2, 1]] (eigenvalues 3, -1)
    // Column-major upper packed: [a00, a01, a11] = [1, 2, 1]
    // x_true = [1, 1] → b = [3, 3]
    let mut ap = vec![1.0_f64, 2.0, 1.0];
    let mut b = vec![3.0_f64, 3.0];
    let mut ipiv = vec![0_i32; 2];
    spsv::<f64>(Layout::ColMajor, UpLo::Upper, 2, 1, &mut ap, &mut ipiv, &mut b, 2)
        .unwrap();
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-12);
}

#[test]
fn sptrf_then_sptrs() {
    let mut ap = vec![1.0_f64, 2.0, 1.0];
    let mut ipiv = vec![0_i32; 2];
    sptrf::<f64>(Layout::ColMajor, UpLo::Upper, 2, &mut ap, &mut ipiv).unwrap();
    let mut b = vec![3.0_f64, 3.0];
    sptrs::<f64>(Layout::ColMajor, UpLo::Upper, 2, 1, &ap, &ipiv, &mut b, 2).unwrap();
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-12);
}

#[test]
fn hpsv_hermitian_packed() {
    // Hermitian: [[2, 1+i], [1-i, 3]]
    // Column-major upper packed: [a00, a01, a11] = [2, 1+i, 3]
    // x_true = [1, 1] → b = A * [1, 1] = [2 + (1+i), (1-i) + 3] = [3+i, 4-i]
    let mut ap = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
    ];
    let mut b = vec![Complex64::new(3.0, 1.0), Complex64::new(4.0, -1.0)];
    let mut ipiv = vec![0_i32; 2];
    hpsv::<Complex64>(
        Layout::ColMajor, UpLo::Upper, 2, 1, &mut ap, &mut ipiv, &mut b, 2,
    )
    .unwrap();
    assert_abs_diff_eq!(b[0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1].im, 0.0, epsilon = 1e-10);
}

#[test]
fn hptrf_then_hptrs() {
    let mut ap = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
    ];
    let mut ipiv = vec![0_i32; 2];
    hptrf::<Complex64>(Layout::ColMajor, UpLo::Upper, 2, &mut ap, &mut ipiv).unwrap();
    let mut b = vec![Complex64::new(3.0, 1.0), Complex64::new(4.0, -1.0)];
    hptrs::<Complex64>(
        Layout::ColMajor, UpLo::Upper, 2, 1, &ap, &ipiv, &mut b, 2,
    )
    .unwrap();
    assert_abs_diff_eq!(b[0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1].re, 1.0, epsilon = 1e-10);
}

#[test]
fn invalid_packed_buffer_rejected() {
    let mut ap = vec![1.0_f64; 1]; // too short for n=3 (needs 6)
    let mut b = vec![0.0_f64; 3];
    let r = ppsv::<f64>(Layout::ColMajor, UpLo::Upper, 3, 1, &mut ap, &mut b, 3);
    assert!(r.is_err());
}
