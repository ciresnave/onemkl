#![cfg(feature = "lapack")]

//! Verify the banded LAPACK driver wrappers.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::lapack::{
    gbsv, gbtrf, gbtrs, gtsv, gttrf, gttrs, pbsv, pbtrf, pbtrs, ptsv, pttrf,
    pttrs_complex, pttrs_real,
};
use onemkl::{Layout, Transpose, UpLo};

#[test]
fn gbsv_solves_tridiagonal() {
    // 4x4 tridiagonal with kl = ku = 1:
    //   [[ 4, -1,  0,  0],
    //    [-1,  4, -1,  0],
    //    [ 0, -1,  4, -1],
    //    [ 0,  0, -1,  4]]
    // x_true = [1, 1, 1, 1] → b = [3, 2, 2, 3].
    //
    // CBLAS column-major banded format with kl=1, ku=1, ldab = 2*kl + ku + 1 = 4:
    //   col j (length ldab=4) holds rows [j-ku, j-ku+1, ..., j+kl] padded with junk.
    //
    // Layout in column-major order, ldab=4:
    //   col 0: [_, _, 4, -1]
    //   col 1: [_, -1, 4, -1]
    //   col 2: [_, -1, 4, -1]
    //   col 3: [_, -1, 4, _ ]
    let _ = (4, 1, 1); // n, kl, ku
    let mut ab = vec![
        0.0_f64, 0.0,  4.0, -1.0,  // col 0
        0.0,    -1.0,  4.0, -1.0,  // col 1
        0.0,    -1.0,  4.0, -1.0,  // col 2
        0.0,    -1.0,  4.0,  0.0,  // col 3
    ];
    let mut b = vec![3.0_f64, 2.0, 2.0, 3.0];
    let mut ipiv = vec![0_i32; 4];
    gbsv::<f64>(Layout::ColMajor, 4, 1, 1, 1, &mut ab, 4, &mut ipiv, &mut b, 4).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn gbtrf_then_gbtrs() {
    // Same matrix as above; factor then solve in two steps.
    let mut ab = vec![
        0.0_f64, 0.0,  4.0, -1.0,
        0.0,    -1.0,  4.0, -1.0,
        0.0,    -1.0,  4.0, -1.0,
        0.0,    -1.0,  4.0,  0.0,
    ];
    let mut ipiv = vec![0_i32; 4];
    gbtrf::<f64>(Layout::ColMajor, 4, 4, 1, 1, &mut ab, 4, &mut ipiv).unwrap();

    let mut b = vec![3.0_f64, 2.0, 2.0, 3.0];
    gbtrs::<f64>(
        Layout::ColMajor, Transpose::NoTrans,
        4, 1, 1, 1, &ab, 4, &ipiv, &mut b, 4,
    )
    .unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn gtsv_tridiagonal_3x3() {
    // 3x3 tridiagonal:
    //   [[ 4, -1,  0],
    //    [-1,  4, -1],
    //    [ 0, -1,  4]]
    // x_true = [1, 1, 1] → b = [3, 2, 3].
    let mut dl = vec![-1.0_f64, -1.0]; // sub-diagonal length n-1 = 2
    let mut d = vec![4.0_f64, 4.0, 4.0]; // diagonal length n = 3
    let mut du = vec![-1.0_f64, -1.0]; // super-diagonal length n-1 = 2
    let mut b = vec![3.0_f64, 2.0, 3.0];
    gtsv::<f64>(Layout::ColMajor, 3, 1, &mut dl, &mut d, &mut du, &mut b, 3).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn gttrf_then_gttrs() {
    let mut dl = vec![-1.0_f64, -1.0];
    let mut d = vec![4.0_f64, 4.0, 4.0];
    let mut du = vec![-1.0_f64, -1.0];
    let mut du2 = vec![0.0_f64; 1]; // n-2 = 1
    let mut ipiv = vec![0_i32; 3];
    gttrf::<f64>(3, &mut dl, &mut d, &mut du, &mut du2, &mut ipiv).unwrap();

    let mut b = vec![3.0_f64, 2.0, 3.0];
    gttrs::<f64>(
        Layout::ColMajor, Transpose::NoTrans, 3, 1,
        &dl, &d, &du, &du2, &ipiv, &mut b, 3,
    )
    .unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn pbsv_pd_banded() {
    // SPD banded with kd = 1, n = 4.
    // Same matrix as gbsv test (which is SPD).
    // For pbsv, only the upper triangle is stored. ldab = kd + 1 = 2.
    // Column-major upper banded storage: col j holds [a(j-kd, j), ..., a(j, j)]
    //   col 0: [_, 4]
    //   col 1: [-1, 4]
    //   col 2: [-1, 4]
    //   col 3: [-1, 4]
    let mut ab = vec![
        0.0_f64,  4.0,  // col 0
        -1.0,     4.0,  // col 1
        -1.0,     4.0,  // col 2
        -1.0,     4.0,  // col 3
    ];
    let mut b = vec![3.0_f64, 2.0, 2.0, 3.0];
    pbsv::<f64>(Layout::ColMajor, UpLo::Upper, 4, 1, 1, &mut ab, 2, &mut b, 4).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn pbtrf_then_pbtrs() {
    let mut ab = vec![
        0.0_f64,  4.0,
        -1.0,     4.0,
        -1.0,     4.0,
        -1.0,     4.0,
    ];
    pbtrf::<f64>(Layout::ColMajor, UpLo::Upper, 4, 1, &mut ab, 2).unwrap();

    let mut b = vec![3.0_f64, 2.0, 2.0, 3.0];
    pbtrs::<f64>(Layout::ColMajor, UpLo::Upper, 4, 1, 1, &ab, 2, &mut b, 4).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn ptsv_pd_tridiagonal() {
    // Same SPD tridiagonal: d = [4,4,4], e = [-1,-1].
    let mut d = vec![4.0_f64, 4.0, 4.0];
    let mut e = vec![-1.0_f64, -1.0];
    let mut b = vec![3.0_f64, 2.0, 3.0];
    ptsv::<f64>(Layout::ColMajor, 3, 1, &mut d, &mut e, &mut b, 3).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn pttrf_then_pttrs_real() {
    let mut d = vec![4.0_f64, 4.0, 4.0];
    let mut e = vec![-1.0_f64, -1.0];
    pttrf::<f64>(3, &mut d, &mut e).unwrap();

    let mut b = vec![3.0_f64, 2.0, 3.0];
    pttrs_real::<f64>(Layout::ColMajor, 3, 1, &d, &e, &mut b, 3).unwrap();
    for x in &b {
        assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn pttrs_complex_works() {
    // Complex Hermitian PD tridiagonal: d = [4, 4, 4] (real diag),
    // e = [1+i, 1+i] (complex off-diag).
    // For complex Hermitian, x_true = [1, 1, 1] gives:
    //   row 0: 4*1 + (1+i)*1     = 5 + i
    //   row 1: conj(1+i)*1 + 4*1 + (1+i)*1 = (1-i) + 4 + (1+i) = 6
    //   row 2: conj(1+i)*1 + 4*1 = 5 - i
    let mut d = vec![4.0_f64, 4.0, 4.0];
    let mut e = vec![Complex64::new(1.0, 1.0), Complex64::new(1.0, 1.0)];
    pttrf::<Complex64>(3, &mut d, &mut e).unwrap();

    let mut b = vec![
        Complex64::new(5.0, 1.0),
        Complex64::new(6.0, 0.0),
        Complex64::new(5.0, -1.0),
    ];
    pttrs_complex::<Complex64>(Layout::ColMajor, UpLo::Upper, 3, 1, &d, &e, &mut b, 3)
        .unwrap();
    for x in &b {
        assert_abs_diff_eq!(x.re, 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(x.im, 0.0, epsilon = 1e-9);
    }
}
