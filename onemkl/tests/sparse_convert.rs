#![cfg(feature = "sparse")]

//! Verify Sparse BLAS copy / convert / order / hint setters.

use approx::assert_abs_diff_eq;

use onemkl::sparse::{
    BlockLayout, DenseLayout, IndexBase, MatrixDescr, MatrixType, Operation, SparseMatrix,
};

fn build_test_csr() -> SparseMatrix<f64> {
    // 2 × 3 matrix:
    //   [[1, 0, 2],
    //    [0, 3, 0]]
    let row_ptr = vec![0, 2, 3];
    let col_idx = vec![0, 2, 1];
    let values = vec![1.0_f64, 2.0, 3.0];
    SparseMatrix::from_csr(2, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap()
}

#[test]
fn copy_preserves_mv() {
    let a = build_test_csr();
    let b = a.copy(MatrixDescr::general()).unwrap();

    let x = [1.0_f64; 3];
    let mut ya = [0.0_f64; 2];
    let mut yb = [0.0_f64; 2];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut ya).unwrap();
    b.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut yb).unwrap();
    assert_eq!(ya, yb);
}

#[test]
fn convert_csr_to_csc_then_back() {
    let a = build_test_csr();
    let csc = a.convert_csc(Operation::NoTrans).unwrap();
    let csr = csc.convert_csr(Operation::NoTrans).unwrap();

    let x = [1.0_f64; 3];
    let mut ya = [0.0_f64; 2];
    let mut yc = [0.0_f64; 2];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut ya).unwrap();
    csr.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut yc).unwrap();
    for (a, b) in ya.iter().zip(&yc) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
    }
}

#[test]
fn convert_csr_to_coo() {
    let a = build_test_csr();
    let coo = a.convert_coo(Operation::NoTrans).unwrap();
    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 2];
    coo.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    // A * [1,1,1] = [1+2, 3] = [3, 3]
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
}

#[test]
fn convert_csr_with_transpose() {
    let a = build_test_csr();
    // a is 2×3; transposed is 3×2.
    let at = a.convert_csr(Operation::Trans).unwrap();
    assert_eq!(at.rows(), 3);
    assert_eq!(at.cols(), 2);
    // (Aᵀ) * [1, 1] = column sums of A = [1, 3, 2].
    let x = [1.0_f64, 1.0];
    let mut y = [0.0_f64; 3];
    at.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 2.0, epsilon = 1e-12);
}

#[test]
fn convert_to_bsr_block_diagonal() {
    // Build a 4×4 block-diagonal CSR with two 2×2 blocks, then convert
    // to BSR with block_size=2 and verify mv against the original CSR.
    //   [[1, 2, 0, 0],
    //    [3, 4, 0, 0],
    //    [0, 0, 5, 6],
    //    [0, 0, 7, 8]]
    let row_ptr = vec![0, 2, 4, 6, 8];
    let col_idx = vec![0, 1, 0, 1, 2, 3, 2, 3];
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let csr =
        SparseMatrix::from_csr(4, 4, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    let bsr = csr
        .convert_bsr(2, BlockLayout::RowMajor, Operation::NoTrans)
        .unwrap();

    let x = [1.0_f64; 4];
    let mut y_csr = [0.0_f64; 4];
    let mut y_bsr = [0.0_f64; 4];
    csr.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y_csr).unwrap();
    bsr.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y_bsr).unwrap();
    for (a, b) in y_csr.iter().zip(&y_bsr) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
    }
}

#[test]
fn order_succeeds() {
    // Build a CSR with deliberately unsorted column indices in each
    // row, then call order() to sort them. mv should still produce the
    // same result.
    let row_ptr = vec![0, 2, 3];
    // Row 0 columns out of order: [2, 0] instead of [0, 2].
    let col_idx = vec![2, 0, 1];
    let values = vec![2.0_f64, 1.0, 3.0];
    let a =
        SparseMatrix::from_csr(2, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    a.order().unwrap();

    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 2];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
}

#[test]
fn mv_hint_then_optimize() {
    let a = build_test_csr();
    a.set_mv_hint(Operation::NoTrans, MatrixType::General, 100)
        .unwrap();
    a.optimize().unwrap();
    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 2];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
}

#[test]
fn mm_hint_then_optimize() {
    let a = build_test_csr();
    a.set_mm_hint(
        Operation::NoTrans,
        MatrixType::General,
        DenseLayout::RowMajor,
        4,
        50,
    )
    .unwrap();
    a.optimize().unwrap();
}

#[test]
fn sv_hint_then_optimize() {
    // Build an upper-triangular 2×2 for trsv hint.
    let row_ptr = vec![0, 2, 3];
    let col_idx = vec![0, 1, 1];
    let values = vec![2.0_f64, 1.0, 3.0];
    let a =
        SparseMatrix::from_csr(2, 2, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    a.set_sv_hint(
        Operation::NoTrans,
        MatrixDescr::triangular(
            onemkl::sparse::FillMode::Upper,
            onemkl::sparse::DiagType::NonUnit,
        ),
        20,
    )
    .unwrap();
    a.optimize().unwrap();
}
