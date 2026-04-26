//! Verify Sparse QR factorization + solve.

use approx::assert_abs_diff_eq;

use onemkl::sparse::{CsrMatrix, DenseLayout, IndexBase, MatrixType, Operation};

#[test]
fn qr_solve_overdetermined_system() {
    // 4x2 sparse matrix:
    //   [[1, 0],
    //    [0, 1],
    //    [1, 1],
    //    [1, 0]]
    // CSR (zero-based): row_ptr = [0, 1, 2, 4, 5]; col_idx = [0, 1, 0, 1, 0]
    let row_ptr = vec![0_i32, 1, 2, 4, 5];
    let col_idx = vec![0_i32, 1, 0, 1, 0];
    let values = vec![1.0_f64, 1.0, 1.0, 1.0, 1.0];
    let a = CsrMatrix::from_csr(4, 2, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    a.qr_factor(MatrixType::General).unwrap();

    // Pick x_true = [1, 2], b = A * x_true = [1, 2, 3, 1].
    let b = [1.0_f64, 2.0, 3.0, 1.0];
    let mut x = [0.0_f64; 2];
    a.qr_solve(
        Operation::NoTrans,
        DenseLayout::ColMajor,
        &b,
        1,
        4, // ldx (rows of full residual, 4 with NoTrans on m=4 matrix)
        &mut x,
        4,
    )
    .unwrap();
    // Least-squares solution of an exact system should still be x_true.
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-8);
}
