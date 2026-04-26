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

