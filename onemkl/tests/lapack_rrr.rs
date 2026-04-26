#![cfg(feature = "lapack")]

//! Verify RRR and divide-and-conquer eigensolvers.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::lapack::{heevd, heevr, syevd, syevr, EigenRange, Job, RrrEigResult};
use onemkl::matrix::MatrixMut;
use onemkl::{Layout, UpLo};

#[test]
fn syevd_diagonal() {
    let mut a = vec![3.0_f64, 0.0, 0.0, 1.0];
    let mut w = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    syevd::<f64>(Job::None, UpLo::Upper, &mut a_view, &mut w).unwrap();
    assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 3.0, epsilon = 1e-12);
}

#[test]
fn syevr_all_eigenvalues() {
    // diag(5, 2, 1) → sorted ascending → 1, 2, 5.
    let mut a = vec![
        5.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let mut a_view = MatrixMut::new(&mut a, 3, 3, Layout::RowMajor).unwrap();
    let res: RrrEigResult<f64, f64> =
        syevr::<f64>(Job::None, EigenRange::All, UpLo::Upper, &mut a_view, 0.0).unwrap();
    assert_eq!(res.m, 3);
    assert_abs_diff_eq!(res.eigenvalues[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(res.eigenvalues[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(res.eigenvalues[2], 5.0, epsilon = 1e-12);
}

#[test]
fn syevr_value_range() {
    // diag(1, 2, 5, 8); request only those in (1.5, 6.0] → 2, 5.
    let mut a = vec![
        1.0_f64, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 8.0,
    ];
    let mut a_view = MatrixMut::new(&mut a, 4, 4, Layout::RowMajor).unwrap();
    let res = syevr::<f64>(
        Job::None,
        EigenRange::Values { vl: 1.5, vu: 6.0 },
        UpLo::Upper,
        &mut a_view,
        0.0,
    )
    .unwrap();
    assert_eq!(res.m, 2);
    assert_abs_diff_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(res.eigenvalues[1], 5.0, epsilon = 1e-12);
}

#[test]
fn syevr_index_range() {
    // diag(1, 2, 5, 8); request the 2nd and 3rd smallest (1-based) → 2, 5.
    let mut a = vec![
        1.0_f64, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 8.0,
    ];
    let mut a_view = MatrixMut::new(&mut a, 4, 4, Layout::RowMajor).unwrap();
    let res = syevr::<f64>(
        Job::None,
        EigenRange::Indices { il: 2, iu: 3 },
        UpLo::Upper,
        &mut a_view,
        0.0,
    )
    .unwrap();
    assert_eq!(res.m, 2);
    assert_abs_diff_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(res.eigenvalues[1], 5.0, epsilon = 1e-12);
}

#[test]
fn syevr_with_eigenvectors() {
    // 2x2 SPD; both vectors requested.
    let mut a = vec![2.0_f64, 1.0, 1.0, 2.0];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let res = syevr::<f64>(
        Job::Compute,
        EigenRange::All,
        UpLo::Upper,
        &mut a_view,
        0.0,
    )
    .unwrap();
    assert_eq!(res.m, 2);
    // Eigenvalues 1 and 3 (since A = [[2,1],[1,2]]).
    assert_abs_diff_eq!(res.eigenvalues[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(res.eigenvalues[1], 3.0, epsilon = 1e-12);
    assert_eq!(res.eigenvectors.len(), 4); // 2x2 column-major
    assert_eq!(res.isuppz.len(), 4); // 2 * m
}

#[test]
fn heevd_hermitian() {
    // Hermitian [[2, 1+i], [1-i, 3]] — eigenvalues 1, 4.
    let mut a = vec![
        Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0),
        Complex64::new(99.0, 99.0), Complex64::new(3.0, 0.0),
    ];
    let mut w = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    heevd::<Complex64>(Job::None, UpLo::Upper, &mut a_view, &mut w).unwrap();
    assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 4.0, epsilon = 1e-12);
}

#[test]
fn heevr_range_value() {
    // Same Hermitian; request range (0.5, 2.0] → only 1.
    let mut a = vec![
        Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0),
        Complex64::new(99.0, 99.0), Complex64::new(3.0, 0.0),
    ];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let res = heevr::<Complex64>(
        Job::None,
        EigenRange::Values { vl: 0.5, vu: 2.0 },
        UpLo::Upper,
        &mut a_view,
        0.0,
    )
    .unwrap();
    assert_eq!(res.m, 1);
    assert_abs_diff_eq!(res.eigenvalues[0], 1.0, epsilon = 1e-12);
}
