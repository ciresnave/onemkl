#![cfg(feature = "sparse")]

//! Verify the sparse iterative-solver primitives — dotmv and the
//! Gauss–Seidel sweep.

use approx::assert_abs_diff_eq;

use onemkl::sparse::{
    DiagType, FillMode, IndexBase, MatrixDescr, MatrixType, Operation, SparseMatrix,
};

#[test]
fn dotmv_returns_x_dot_y() {
    // 3×3 diagonal A = diag(1, 2, 3); x = [1, 1, 1].
    // A * x = [1, 2, 3]; xᵀ · y = 6.
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![1.0_f64, 2.0, 3.0];
    let a =
        SparseMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    let d = a
        .dot_mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(d, 6.0, epsilon = 1e-12);
}

#[test]
fn dotmv_with_alpha_beta() {
    // y = 2 * A * x + 3 * y_0; d = x · y_new.
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![1.0_f64, 1.0, 1.0]; // identity
    let a =
        SparseMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    let x = [1.0_f64, 2.0, 3.0];
    let mut y = [4.0_f64, 5.0, 6.0];
    let d = a
        .dot_mv(Operation::NoTrans, 2.0, MatrixType::General, &x, 3.0, &mut y)
        .unwrap();
    // y_new = 2 * I * x + 3 * y_0 = 2*x + 3*y_0 = [2+12, 4+15, 6+18] = [14, 19, 24]
    assert_abs_diff_eq!(y[0], 14.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 19.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 24.0, epsilon = 1e-12);
    // d = 1*14 + 2*19 + 3*24 = 14 + 38 + 72 = 124
    assert_abs_diff_eq!(d, 124.0, epsilon = 1e-12);
}

#[test]
fn symgs_solves_diagonal_system() {
    // A = diag(2, 3, 4); b = [4, 6, 8]; expect x = [2, 2, 2].
    // Single Gauss–Seidel sweep on a diagonal system is exact.
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![2.0_f64, 3.0, 4.0];
    let a =
        SparseMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    let b = [4.0_f64, 6.0, 8.0];
    let mut x = [0.0_f64; 3];
    a.symgs(
        Operation::NoTrans,
        MatrixDescr::triangular(FillMode::Lower, DiagType::NonUnit),
        1.0,
        &b,
        &mut x,
    )
    .unwrap();
    assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[2], 2.0, epsilon = 1e-12);
}

#[test]
fn symgs_mv_combines_sweep_and_matvec() {
    // Same diagonal system; verify x is solved AND y = A * x = b.
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![2.0_f64, 3.0, 4.0];
    let a =
        SparseMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    let b = [4.0_f64, 6.0, 8.0];
    let mut x = [0.0_f64; 3];
    let mut y = [0.0_f64; 3];
    a.symgs_mv(
        Operation::NoTrans,
        MatrixDescr::triangular(FillMode::Lower, DiagType::NonUnit),
        1.0,
        &b,
        &mut x,
        &mut y,
    )
    .unwrap();
    // x recovered to [2, 2, 2]; y = A * x = b.
    assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[2], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[0], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 8.0, epsilon = 1e-12);
}
