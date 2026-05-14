#![cfg(feature = "sparse")]

//! Verify sparse-matrix arithmetic: add / spmm / spmmd.

use approx::assert_abs_diff_eq;

use onemkl::sparse::{
    DenseLayout, FillMode, IndexBase, MatrixDescr, MatrixType, Operation, SparseMatrix,
};

fn diagonal_3x3<T: onemkl::sparse::SparseScalar + Copy>(values: [T; 3]) -> SparseMatrix<T> {
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    SparseMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values.to_vec()).unwrap()
}

#[test]
fn add_two_diagonal_matrices() {
    let a = diagonal_3x3([1.0_f64, 2.0, 3.0]);
    let b = diagonal_3x3([10.0_f64, 20.0, 30.0]);
    // C = A + 1 * B = diag(11, 22, 33).
    let c = a.add(Operation::NoTrans, 1.0, &b).unwrap();

    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 33.0, epsilon = 1e-12);
}

#[test]
fn add_with_alpha_scaling() {
    // C = A + 0.5 * B.
    let a = diagonal_3x3([2.0_f64, 4.0, 6.0]);
    let b = diagonal_3x3([2.0_f64, 4.0, 6.0]);
    let c = a.add(Operation::NoTrans, 0.5, &b).unwrap();
    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 9.0, epsilon = 1e-12);
}

#[test]
fn spmm_diagonal_times_diagonal() {
    let a = diagonal_3x3([1.0_f64, 2.0, 3.0]);
    let b = diagonal_3x3([10.0_f64, 20.0, 30.0]);
    let c = a.spmm(Operation::NoTrans, &b).unwrap();
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 3);

    // diag(1,2,3) * diag(10,20,30) = diag(10, 40, 90).
    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 40.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 90.0, epsilon = 1e-12);
}

#[test]
fn spmm_general_2x3_times_3x2() {
    // A (2×3): [[1, 0, 2], [0, 3, 0]]
    // B (3×2): [[1, 0], [0, 1], [4, 0]]
    // A * B = [[1+8, 0], [0, 3]] = [[9, 0], [0, 3]]
    let a_row_ptr = vec![0, 2, 3];
    let a_col_idx = vec![0, 2, 1];
    let a_values = vec![1.0_f64, 2.0, 3.0];
    let a =
        SparseMatrix::from_csr(2, 3, IndexBase::Zero, a_row_ptr, a_col_idx, a_values).unwrap();
    let b_row_ptr = vec![0, 1, 2, 3];
    let b_col_idx = vec![0, 1, 0];
    let b_values = vec![1.0_f64, 1.0, 4.0];
    let b =
        SparseMatrix::from_csr(3, 2, IndexBase::Zero, b_row_ptr, b_col_idx, b_values).unwrap();

    let c = a.spmm(Operation::NoTrans, &b).unwrap();
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);

    let x = [1.0_f64, 1.0];
    let mut y = [0.0_f64; 2];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 3.0, epsilon = 1e-12);
}

#[test]
fn spmmd_diagonal_times_diagonal_to_dense() {
    let a = diagonal_3x3([1.0_f64, 2.0, 3.0]);
    let b = diagonal_3x3([10.0_f64, 20.0, 30.0]);
    // 3×3 dense output, row-major, ldc = 3.
    let dense = a.spmmd(Operation::NoTrans, &b, DenseLayout::RowMajor, 3).unwrap();
    assert_eq!(dense.len(), 9);
    // Diagonal entries 10, 40, 90 at (0,0), (1,1), (2,2); rest zero.
    assert_abs_diff_eq!(dense[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(dense[4], 40.0, epsilon = 1e-12);
    assert_abs_diff_eq!(dense[8], 90.0, epsilon = 1e-12);
    for &i in &[1_usize, 2, 3, 5, 6, 7] {
        assert_abs_diff_eq!(dense[i], 0.0, epsilon = 1e-12);
    }
}

#[test]
fn spmm_with_transpose() {
    // A is 2×3. A^T * A is 3×3.
    let a_row_ptr = vec![0, 2, 3];
    let a_col_idx = vec![0, 2, 1];
    let a_values = vec![1.0_f64, 2.0, 3.0];
    let a =
        SparseMatrix::from_csr(2, 3, IndexBase::Zero, a_row_ptr, a_col_idx, a_values).unwrap();
    let c = a.spmm(Operation::Trans, &a).unwrap();
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 3);

    // Aᵀ A diagonal entries are column-norms-squared:
    //   col 0 has [1, 0]    → 1
    //   col 1 has [0, 3]    → 9
    //   col 2 has [2, 0]    → 4
    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    // Sum of row 0 of (AᵀA) = 1 + 0 + 2 = 3 (since col 0 ⋅ col 1 = 0,
    // col 0 ⋅ col 2 = 1*2 = 2).
    assert_abs_diff_eq!(y[0], 1.0 + 0.0 + 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 0.0 + 9.0 + 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 2.0 + 0.0 + 4.0, epsilon = 1e-12);
}

#[test]
fn syrk_gram_matrix() {
    // A = [[1, 0, 2], [0, 3, 0]] (2×3); A·Aᵀ should be diag(1+4, 9) = diag(5, 9).
    let a_row_ptr = vec![0, 2, 3];
    let a_col_idx = vec![0, 2, 1];
    let a_values = vec![1.0_f64, 2.0, 3.0];
    let a =
        SparseMatrix::from_csr(2, 3, IndexBase::Zero, a_row_ptr, a_col_idx, a_values).unwrap();

    let c = a.syrk(Operation::NoTrans).unwrap();
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);

    // Probe via mv with [1, 1] — only the **upper-triangular** part of
    // the symmetric output is populated, so applying it as a general
    // matrix to [1, 1] gives row sums [5, 9] (diag + off-diag-once).
    let x = [1.0_f64, 1.0];
    let mut y = [0.0_f64, 0.0];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 9.0, epsilon = 1e-12);
}

#[test]
fn syrk_transpose_columns_squared() {
    // Same A as the spmm test — Aᵀ A is 3×3 with column norms on the
    // diagonal (1, 9, 4) and off-diag entries from column inner products.
    let a_row_ptr = vec![0, 2, 3];
    let a_col_idx = vec![0, 2, 1];
    let a_values = vec![1.0_f64, 2.0, 3.0];
    let a =
        SparseMatrix::from_csr(2, 3, IndexBase::Zero, a_row_ptr, a_col_idx, a_values).unwrap();

    let c = a.syrk(Operation::Trans).unwrap();
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 3);

    let x = [1.0_f64, 1.0, 1.0];
    let mut y = [0.0_f64, 0.0, 0.0];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    // Upper-triangular storage: row 0 has (1, 0, 2), row 1 has (9, 0),
    // row 2 has (4). Row sums applied to [1, 1, 1] = [3, 9, 4].
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 4.0, epsilon = 1e-12);
}

#[test]
fn sypr_triple_product_with_identity() {
    // A = [[1, 1], [0, 1]] (upper triangular 2×2 stored as CSR).
    // B = identity 2×2 (symmetric). Then A · B · Aᵀ = A · Aᵀ = [[2, 1], [1, 1]].
    let a = SparseMatrix::from_csr(
        2,
        2,
        IndexBase::Zero,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();
    let b = SparseMatrix::from_csr(
        2,
        2,
        IndexBase::Zero,
        vec![0, 1, 2],
        vec![0, 1],
        vec![1.0_f64, 1.0],
    )
    .unwrap();
    let descr_b = MatrixDescr::symmetric(FillMode::Upper);

    let c = a.sypr(Operation::NoTrans, &b, descr_b).unwrap();
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);

    // Only the upper triangle is materialized; applying to [1, 1]
    // gives row sums [2 + 1, 1] = [3, 1].
    let x = [1.0_f64, 1.0];
    let mut y = [0.0_f64, 0.0];
    c.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y)
        .unwrap();
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 1.0, epsilon = 1e-12);
}
