#![cfg(feature = "lapack")]

//! Verify generalized eigenvalue LAPACK wrappers.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::lapack::{ggev_complex, ggev_real, hegv, sygv, GeneralizedEigenType, Job};
use onemkl::matrix::MatrixMut;
use onemkl::{Layout, UpLo};

#[test]
fn sygv_real_diagonal() {
    // A = diag(2, 4); B = diag(1, 1) (identity).
    // A x = lambda B x → eigenvalues are 2, 4.
    let mut a = vec![2.0_f64, 0.0, 0.0, 4.0];
    let mut b = vec![1.0_f64, 0.0, 0.0, 1.0];
    let mut w = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 2, Layout::RowMajor).unwrap();
    sygv(
        GeneralizedEigenType::AxLambdaBx,
        Job::None,
        UpLo::Upper,
        &mut a_view,
        &mut b_view,
        &mut w,
    )
    .unwrap();
    // Eigenvalues sorted ascending.
    assert_abs_diff_eq!(w[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 4.0, epsilon = 1e-12);
}

#[test]
fn sygv_with_general_b() {
    // A = [[2, 0], [0, 4]], B = [[2, 0], [0, 8]].
    // A x = λ B x → λ_i = a_ii / b_ii = 1, 0.5.
    // Sorted ascending → 0.5, 1.0.
    let mut a = vec![2.0_f64, 0.0, 0.0, 4.0];
    let mut b = vec![2.0_f64, 0.0, 0.0, 8.0];
    let mut w = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 2, Layout::RowMajor).unwrap();
    sygv(
        GeneralizedEigenType::AxLambdaBx,
        Job::None,
        UpLo::Upper,
        &mut a_view,
        &mut b_view,
        &mut w,
    )
    .unwrap();
    assert_abs_diff_eq!(w[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 1.0, epsilon = 1e-12);
}

#[test]
fn hegv_hermitian_definite() {
    // A = Hermitian [[2, 1+i], [1-i, 3]]
    // B = identity 2x2
    // A x = λ B x is the same as A's eigenvalues: 1 and 4 (computed earlier in heev).
    let mut a = vec![
        Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0),
        Complex64::new(99.0, 99.0), Complex64::new(3.0, 0.0),
    ];
    let mut b = vec![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(99.0, 99.0), Complex64::new(1.0, 0.0),
    ];
    let mut w = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 2, Layout::RowMajor).unwrap();
    hegv(
        GeneralizedEigenType::AxLambdaBx,
        Job::None,
        UpLo::Upper,
        &mut a_view,
        &mut b_view,
        &mut w,
    )
    .unwrap();
    assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 4.0, epsilon = 1e-12);
}

#[test]
fn ggev_real_diagonal() {
    // A = diag(2, 4); B = identity.
    // alphar / beta = 2, 4. alphai = 0.
    let mut a = vec![2.0_f64, 0.0, 0.0, 4.0];
    let mut b = vec![1.0_f64, 0.0, 0.0, 1.0];
    let mut alphar = vec![0.0_f64; 2];
    let mut alphai = vec![0.0_f64; 2];
    let mut beta = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 2, Layout::RowMajor).unwrap();
    ggev_real(
        Job::None,
        Job::None,
        &mut a_view,
        &mut b_view,
        &mut alphar,
        &mut alphai,
        &mut beta,
        None,
        None,
    )
    .unwrap();
    // Eigenvalues are alphar/beta. Sort to compare regardless of order.
    let mut eigs: Vec<f64> = alphar.iter().zip(&beta).map(|(a, b)| a / b).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_abs_diff_eq!(eigs[0], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(eigs[1], 4.0, epsilon = 1e-10);
    for v in &alphai {
        assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn ggev_complex_diagonal() {
    // A = diag(2+i, 3-i); B = identity.
    // alpha/beta = 2+i, 3-i.
    let mut a = vec![
        Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(3.0, -1.0),
    ];
    let mut b = vec![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
    ];
    let mut alpha = vec![Complex64::new(0.0, 0.0); 2];
    let mut beta = vec![Complex64::new(0.0, 0.0); 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 2, Layout::RowMajor).unwrap();
    ggev_complex(
        Job::None,
        Job::None,
        &mut a_view,
        &mut b_view,
        &mut alpha,
        &mut beta,
        None,
        None,
    )
    .unwrap();
    // Each eigenvalue = alpha[i] / beta[i].
    let eig0 = alpha[0] / beta[0];
    let eig1 = alpha[1] / beta[1];
    let has_2pi = (eig0.re - 2.0).abs() < 1e-10 && (eig0.im - 1.0).abs() < 1e-10
        || (eig1.re - 2.0).abs() < 1e-10 && (eig1.im - 1.0).abs() < 1e-10;
    let has_3mi = (eig0.re - 3.0).abs() < 1e-10 && (eig0.im + 1.0).abs() < 1e-10
        || (eig1.re - 3.0).abs() < 1e-10 && (eig1.im + 1.0).abs() < 1e-10;
    assert!(has_2pi);
    assert!(has_3mi);
}
