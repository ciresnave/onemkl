#![cfg(feature = "sparse")]

//! Verify COO, CSC, and BSR sparse-matrix construction.

use approx::assert_abs_diff_eq;

use onemkl::sparse::{
    BlockLayout, IndexBase, MatrixType, Operation, SparseMatrix,
};

#[test]
fn coo_diagonal_mv() {
    // 3x3 diag(1, 2, 3) in COO: (0,0)=1, (1,1)=2, (2,2)=3.
    let row_indx = vec![0, 1, 2];
    let col_indx = vec![0, 1, 2];
    let values = vec![1.0_f64, 2.0, 3.0];
    let a = SparseMatrix::from_coo(3, 3, IndexBase::Zero, row_indx, col_indx, values)
        .unwrap();

    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 3.0, epsilon = 1e-12);
}

#[test]
fn coo_unordered_entries() {
    // 2x3:
    //   [[1, 0, 2],
    //    [0, 3, 0]]
    // Specify entries out of order to verify COO doesn't require sorting.
    let row_indx = vec![1, 0, 0];
    let col_indx = vec![1, 2, 0];
    let values = vec![3.0_f64, 2.0, 1.0];
    let a = SparseMatrix::from_coo(2, 3, IndexBase::Zero, row_indx, col_indx, values)
        .unwrap();

    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 2];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12); // 1 + 2
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
}

#[test]
fn csc_dense_2x3() {
    // A =
    //   [[1, 0, 2],
    //    [0, 3, 0]]
    // CSC: walk columns. col 0: (0, 1). col 1: (1, 3). col 2: (0, 2).
    // col_ptr = [0, 1, 2, 3]; row_indx = [0, 1, 0]; values = [1, 3, 2].
    let col_ptr = vec![0, 1, 2, 3];
    let row_indx = vec![0, 1, 0];
    let values = vec![1.0_f64, 3.0, 2.0];
    let a = SparseMatrix::from_csc(2, 3, IndexBase::Zero, col_ptr, row_indx, values)
        .unwrap();

    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 2];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12); // 1 + 2
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
}

#[test]
fn csc_transposed_mv() {
    // Same A as csc_dense_2x3; Aᵀ * [1, 1] = [1, 3, 2].
    let col_ptr = vec![0, 1, 2, 3];
    let row_indx = vec![0, 1, 0];
    let values = vec![1.0_f64, 3.0, 2.0];
    let a = SparseMatrix::from_csc(2, 3, IndexBase::Zero, col_ptr, row_indx, values)
        .unwrap();

    let x = [1.0_f64, 1.0];
    let mut y = [0.0_f64; 3];
    a.mv(Operation::Trans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 2.0, epsilon = 1e-12);
}

#[test]
fn bsr_block_diagonal_mv() {
    // 4x4 block-diagonal with two 2x2 blocks:
    //   B0 = [[1, 2], [3, 4]],  B1 = [[5, 6], [7, 8]]
    // 2 block-rows, 1 block per row. row_ptr = [0, 1, 2]; col_idx = [0, 1].
    // Values stored row-major within each block: [1, 2, 3, 4, 5, 6, 7, 8].
    let row_ptr = vec![0, 1, 2];
    let col_idx = vec![0, 1];
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = SparseMatrix::from_bsr(
        4,
        4,
        2,
        BlockLayout::RowMajor,
        IndexBase::Zero,
        row_ptr,
        col_idx,
        values,
    )
    .unwrap();

    // A * [1, 1, 1, 1] = [1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]
    let x = [1.0_f64; 4];
    let mut y = [0.0_f64; 4];
    a.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[3], 15.0, epsilon = 1e-12);
}

#[test]
fn invalid_coo_length_rejected() {
    // Mismatched row/col/value lengths.
    let row_indx = vec![0, 1];
    let col_indx = vec![0, 1, 2];
    let values = vec![1.0_f64, 2.0];
    let r = SparseMatrix::from_coo(3, 3, IndexBase::Zero, row_indx, col_indx, values);
    assert!(r.is_err());
}

#[test]
fn invalid_csc_col_ptr_rejected() {
    // col_ptr has wrong length (must be cols + 1 = 4).
    let col_ptr = vec![0, 1, 2];
    let row_indx = vec![0, 1];
    let values = vec![1.0_f64, 2.0];
    let r = SparseMatrix::from_csc(3, 3, IndexBase::Zero, col_ptr, row_indx, values);
    assert!(r.is_err());
}

#[test]
fn invalid_bsr_block_size_rejected() {
    // 3x3 with block_size=2 is invalid (3 not a multiple of 2).
    let r = SparseMatrix::from_bsr(
        3,
        3,
        2,
        BlockLayout::RowMajor,
        IndexBase::Zero,
        vec![0, 1],
        vec![0],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    );
    assert!(r.is_err());
}
