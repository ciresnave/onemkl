#![cfg(feature = "dss")]

//! Verify DSS direct sparse solver wrappers.

use approx::assert_abs_diff_eq;

use onemkl::dss::{Definite, Dss, IndexBase, Symmetry};

#[test]
fn solve_real_3x3_spd() {
    // SPD matrix:
    //   [[ 4, -1,  0],
    //    [-1,  4, -1],
    //    [ 0, -1,  4]]
    // 1-based CSR upper triangle (DSS convention for symmetric):
    let row_ptr = vec![1_i32, 3, 5, 6];
    let col_idx = vec![1_i32, 2, 2, 3, 3];
    let values = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut dss = Dss::<f64>::new().unwrap();
    dss.define_structure(Symmetry::Symmetric, 3, 3, &row_ptr, &col_idx, IndexBase::One)
        .unwrap();
    dss.reorder().unwrap();
    dss.factor_real(Definite::PositiveDefinite, &values).unwrap();

    // x_true = [1, 1, 1] → b = A * x = [3, 2, 3]
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];
    dss.solve_real(&b, 1, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
}

#[test]
fn solve_two_rhs() {
    let row_ptr = vec![1_i32, 3, 5, 6];
    let col_idx = vec![1_i32, 2, 2, 3, 3];
    let values = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut dss = Dss::<f64>::new().unwrap();
    dss.define_structure(Symmetry::Symmetric, 3, 3, &row_ptr, &col_idx, IndexBase::One)
        .unwrap();
    dss.reorder().unwrap();
    dss.factor_real(Definite::PositiveDefinite, &values).unwrap();

    // Two RHS, column-major.
    // X_true = [[1, 2], [1, 2], [1, 2]]
    // B = A * X_true = [[3, 6], [2, 4], [3, 6]]
    let b = vec![3.0_f64, 2.0, 3.0, 6.0, 4.0, 6.0];
    let mut x = vec![0.0_f64; 6];
    dss.solve_real(&b, 2, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[3], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[4], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[5], 2.0, epsilon = 1e-10);
}

#[test]
fn statistics_after_factor() {
    // Tridiagonal: tridiag(-1, 4, -1); determinant of 3x3 case is
    // 4 * (4*4 - 1) - (-1) * (-4 - 0) = 4*15 - 4 = 56.
    //
    // 1-based upper-triangular CSR.
    let row_ptr = vec![1_i32, 3, 5, 6];
    let col_idx = vec![1_i32, 2, 2, 3, 3];
    let values = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut dss = Dss::<f64>::new().unwrap();
    dss.define_structure(Symmetry::Symmetric, 3, 3, &row_ptr, &col_idx, IndexBase::One)
        .unwrap();
    dss.reorder().unwrap();
    dss.factor_real(Definite::Indefinite, &values).unwrap();

    // Time / memory / flops are non-negative but otherwise environment-dependent.
    assert!(dss.factor_time().unwrap() >= 0.0);
    assert!(dss.peak_memory_kb().unwrap() >= 0.0);
    assert!(dss.factor_memory_kb().unwrap() >= 0.0);
    assert!(dss.flops().unwrap() >= 0.0);

    let det = dss.determinant().unwrap();
    let recovered = det.mantissa * 10.0_f64.powf(det.pow);
    assert_abs_diff_eq!(recovered, 56.0, epsilon = 1e-8);
}

#[test]
fn inertia_for_indefinite_matrix() {
    // Tridiagonal mostly-positive matrix that is still indefinite. Use
    // diag(2, 0, -3) connected via small off-diagonals so DSS doesn't
    // shortcut to a diagonal-matrix path.
    //   [[ 2,  1,  0],
    //    [ 1,  0,  1],
    //    [ 0,  1, -3]]
    // Eigenvalues are roughly { 2.30, -3.13, -0.17 } → 1 positive, 2 negative.
    let row_ptr = vec![1_i32, 3, 5, 6];
    let col_idx = vec![1_i32, 2, 2, 3, 3];
    let values = vec![2.0_f64, 1.0, 0.0, 1.0, -3.0];

    let mut dss = Dss::<f64>::new().unwrap();
    dss.define_structure(Symmetry::Symmetric, 3, 3, &row_ptr, &col_idx, IndexBase::One)
        .unwrap();
    dss.reorder().unwrap();
    dss.factor_real(Definite::Indefinite, &values).unwrap();

    let inertia = dss.inertia().unwrap();
    assert_eq!(inertia.positive + inertia.negative + inertia.zero, 3);
    assert_eq!(inertia.positive, 1);
    assert_eq!(inertia.negative, 2);
    assert_eq!(inertia.zero, 0);
}

