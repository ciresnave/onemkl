//! Solve `A * x = b` with Conjugate Gradient, where `A` is a sparse
//! SPD matrix. Uses the closure-driven `solve_cg` from `iss` and the
//! `SparseMatrix::mv` for the matrix-vector product.
//!
//! Run with `cargo run --example sparse_cg`.

use onemkl::iss::{solve_cg, IssOptions, IssStopReason};
use onemkl::sparse::{IndexBase, MatrixType, Operation, SparseMatrix};

fn main() {
    // 5×5 SPD pentadiagonal A, the discrete Laplacian on a line:
    //   diag = 4, off-diag = -1.
    let row_ptr = vec![0, 2, 5, 8, 11, 13];
    let col_idx = vec![
        0, 1,
        0, 1, 2,
        1, 2, 3,
        2, 3, 4,
        3, 4,
    ];
    let values = vec![
        4.0_f64, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0,
    ];
    let a =
        SparseMatrix::from_csr(5, 5, IndexBase::Zero, row_ptr, col_idx, values).unwrap();

    // x_true = [1, 2, 3, 4, 5]; b = A * x_true = [2, 4, 6, 8, 16].
    let b = [2.0_f64, 4.0, 6.0, 8.0, 16.0];
    let mut x = [0.0_f64; 5];

    let result = solve_cg(
        &b,
        &mut x,
        IssOptions::default(),
        |v: &[f64], out: &mut [f64]| {
            a.mv(Operation::NoTrans, 1.0, MatrixType::General, v, 0.0, out)
                .unwrap();
        },
    )
    .unwrap();

    println!(
        "CG converged in {} iterations (reason: {:?}, residual: {:.2e})",
        result.iterations, result.stop_reason, result.final_residual_norm,
    );
    println!("Solution x = {:?}", x);
    assert_eq!(result.stop_reason, IssStopReason::Converged);
}
