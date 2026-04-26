#![cfg(feature = "feast")]

//! Verify FEAST extended eigensolver wrappers.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::feast::{eigh_complex_dense, eigh_real_dense};
use onemkl::UpLo;

#[test]
fn diagonal_real_finds_in_range() {
    // 3x3 diagonal A = diag(1, 2, 5).
    // Look for eigenvalues in [1.5, 4.5] → should find 2 only.
    let a = vec![
        1.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 5.0,
    ];
    let res = eigh_real_dense(UpLo::Upper, &a, 3, 3, 1.5, 4.5, 3).unwrap();
    assert_eq!(res.m, 1);
    assert_abs_diff_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-8);
}

#[test]
fn diagonal_real_finds_multiple() {
    // diag(1, 2, 3, 4, 5); range [0, 3.5] → expect 1, 2, 3.
    let a = vec![
        1.0_f64, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 5.0,
    ];
    let res = eigh_real_dense(UpLo::Upper, &a, 5, 5, 0.0, 3.5, 5).unwrap();
    assert_eq!(res.m, 3);
    let mut found = res.eigenvalues.clone();
    found.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_abs_diff_eq!(found[0], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(found[1], 2.0, epsilon = 1e-8);
    assert_abs_diff_eq!(found[2], 3.0, epsilon = 1e-8);
}

#[test]
fn complex_hermitian_finds_in_range() {
    // Hermitian: [[2, 1+i], [1-i, 3]] — eigenvalues 1 and 4.
    // Range [0.5, 1.5] → only 1.
    let a = vec![
        Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0),
        Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0),
    ];
    let res = eigh_complex_dense(UpLo::Upper, &a, 2, 2, 0.5, 1.5, 2).unwrap();
    assert_eq!(res.m, 1);
    assert_abs_diff_eq!(res.eigenvalues[0], 1.0, epsilon = 1e-8);
}

#[test]
fn empty_range_returns_zero() {
    let a = vec![1.0_f64, 0.0, 0.0, 2.0];
    let res = eigh_real_dense(UpLo::Upper, &a, 2, 2, 100.0, 200.0, 2).unwrap();
    assert_eq!(res.m, 0);
    assert!(res.eigenvalues.is_empty());
}
