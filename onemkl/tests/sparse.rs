#![cfg(feature = "sparse")]

//! Verify the Inspector-Executor Sparse BLAS wrappers.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::sparse::{
    CsrMatrix, DenseLayout, DiagType, FillMode, IndexBase, MatrixDescr, MatrixType,
    Operation,
};

#[test]
fn diagonal_csr_mv_real() {
    // 3x3 diagonal: diag(1, 2, 3) — y = A * [1, 1, 1] = [1, 2, 3].
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![1.0_f64, 2.0, 3.0];
    let a = CsrMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 3.0, epsilon = 1e-12);
}

#[test]
fn dense_2x3_csr_mv_with_alpha_beta() {
    // A (dense view, row-major):
    //   [[1, 0, 2],
    //    [0, 3, 0]]
    // CSR: row_ptr=[0,2,3], col_idx=[0,2,1], values=[1,2,3]
    let row_ptr = vec![0, 2, 3];
    let col_idx = vec![0, 2, 1];
    let values = vec![1.0_f64, 2.0, 3.0];
    let a = CsrMatrix::from_csr(2, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    // y starts at [10, 20]; y ← 2 * A * [1, 1, 1] + 3 * y
    //   A * [1,1,1] = [3, 3]; result = 2*[3,3] + 3*[10,20] = [6+30, 6+60] = [36, 66]
    let x = [1.0_f64; 3];
    let mut y = [10.0_f64, 20.0];
    a.mv(Operation::NoTrans, 2.0, MatrixType::General, &x, 3.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 36.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 66.0, epsilon = 1e-12);
}

#[test]
fn transposed_mv() {
    // Same A as above, 2x3.
    // Aᵀ is 3x2; Aᵀ * [1, 1] = [1, 3, 2]
    let row_ptr = vec![0, 2, 3];
    let col_idx = vec![0, 2, 1];
    let values = vec![1.0_f64, 2.0, 3.0];
    let a = CsrMatrix::from_csr(2, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    let x = [1.0_f64, 1.0];
    let mut y = [0.0_f64; 3];
    a.mv(Operation::Trans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 2.0, epsilon = 1e-12);
}

#[test]
fn complex_mv() {
    // 1x1 sparse with single entry 2+3i; y = (2+3i) * (1+i) = -1 + 5i
    let row_ptr = vec![0, 1];
    let col_idx = vec![0];
    let values = vec![Complex64::new(2.0, 3.0)];
    let a = CsrMatrix::from_csr(1, 1, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    let x = [Complex64::new(1.0, 1.0)];
    let mut y = [Complex64::new(0.0, 0.0)];
    a.mv(
        Operation::NoTrans,
        Complex64::new(1.0, 0.0),
        MatrixType::General,
        &x,
        Complex64::new(0.0, 0.0),
        &mut y,
    )
    .unwrap();
    assert_abs_diff_eq!(y[0].re, -1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[0].im, 5.0, epsilon = 1e-12);
}

#[test]
fn mm_two_columns() {
    // 2x2 identity matrix. mm with X = [[1, 2], [3, 4]] gives Y = X.
    let row_ptr = vec![0, 1, 2];
    let col_idx = vec![0, 1];
    let values = vec![1.0_f64, 1.0];
    let a = CsrMatrix::from_csr(2, 2, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    let x = [1.0_f64, 2.0, 3.0, 4.0]; // row-major 2x2
    let mut y = [0.0_f64; 4];
    a.mm(
        Operation::NoTrans,
        1.0,
        MatrixType::General,
        DenseLayout::RowMajor,
        &x,
        2,    // columns of X/Y
        2,    // ldx (leading dim of row-major 2x2)
        0.0,
        &mut y,
        2,    // ldy
    )
    .unwrap();
    assert_eq!(y, x);
}

#[test]
fn trsv_lower_triangular() {
    // L = [[1, 0], [2, 3]]; solve L * y = [1, 5] → y = [1, 1]
    let row_ptr = vec![0, 1, 3];
    let col_idx = vec![0, 0, 1];
    let values = vec![1.0_f64, 2.0, 3.0];
    let a = CsrMatrix::from_csr(2, 2, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    let x = [1.0_f64, 5.0];
    let mut y = [0.0_f64; 2];
    a.trsv(
        Operation::NoTrans,
        1.0,
        MatrixDescr::triangular(FillMode::Lower, DiagType::NonUnit),
        &x,
        &mut y,
    )
    .unwrap();
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 1.0, epsilon = 1e-12);
}

#[test]
fn optimize_succeeds() {
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![1.0_f64, 2.0, 3.0];
    let a = CsrMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    a.optimize().unwrap();
}

#[test]
fn invalid_row_ptr_length_rejected() {
    let row_ptr = vec![0, 1, 2]; // wrong length for 3 rows (should be 4)
    let col_idx = vec![0, 1];
    let values = vec![1.0_f64, 2.0];
    let r = CsrMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values);
    assert!(r.is_err());
}
