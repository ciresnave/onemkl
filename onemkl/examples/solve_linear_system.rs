//! Solve a 3×3 linear system `A * x = b` using LAPACK's `gesv`
//! (LU factorization + forward / backward substitution).
//!
//! Run with `cargo run --example solve_linear_system`.

use onemkl::lapack::gesv;
use onemkl::matrix::MatrixMut;
use onemkl::Layout;

fn main() {
    // A = [[ 2,  1,  1],
    //      [ 1,  3,  2],
    //      [ 1,  0,  0]]
    // b = [4, 5, 6]
    // Solution by hand: x = [6, 15, -23].
    let mut a = vec![
        2.0_f64, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    ];
    let mut b = vec![4.0_f64, 5.0, 6.0];
    let mut ipiv = vec![0_i32; 3];

    let mut a_view = MatrixMut::new(&mut a, 3, 3, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 3, 1, Layout::RowMajor).unwrap();
    gesv::<f64>(&mut a_view, &mut ipiv, &mut b_view).unwrap();

    println!("Solution x = {:?}", b);
}
