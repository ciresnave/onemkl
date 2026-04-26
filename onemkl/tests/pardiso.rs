#![cfg(feature = "pardiso")]

//! Verify PARDISO direct sparse solver wrappers.

use approx::assert_abs_diff_eq;

use onemkl::pardiso::{IndexBase, Pardiso, PardisoMatrixType};

#[test]
fn solve_real_spd_3x3() {
    // SPD matrix:
    //   [[ 4, -1,  0],
    //    [-1,  4, -1],
    //    [ 0, -1,  4]]
    // PARDISO needs upper triangle for SPD storage.
    // CSR (1-based, upper triangle only): row 0 → cols [1,2], row 1 → cols [2,3], row 2 → col [3]
    //
    // ia (row pointers, 1-based, length n+1=4):
    //   row 0 starts at index 1, row 1 at 3, row 2 at 5, end at 6
    let ia = vec![1_i32, 3, 5, 6];
    // ja (column indices, 1-based, length nnz=5)
    let ja = vec![1_i32, 2, 2, 3, 3];
    // values (matching ja order)
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    // RHS: A * x_true with x_true = [1, 1, 1] = [3, 2, 3]
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
}

#[test]
fn solve_real_unsymmetric_3x3() {
    // Asymmetric matrix:
    //   [[2, 1, 0],
    //    [0, 3, 1],
    //    [1, 0, 4]]
    // Stored as full CSR (1-based).
    // row 0: cols [1, 2], values [2, 1]
    // row 1: cols [2, 3], values [3, 1]
    // row 2: cols [1, 3], values [1, 4]
    let ia = vec![1_i32, 3, 5, 7];
    let ja = vec![1_i32, 2, 2, 3, 1, 3];
    let a = vec![2.0_f64, 1.0, 3.0, 1.0, 1.0, 4.0];
    // x_true = [1, 1, 1]; b = A * x_true = [3, 4, 5]
    let b = vec![3.0_f64, 4.0, 5.0];
    let mut x = vec![0.0_f64; 3];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealUnsym)
        .with_indexing(IndexBase::One);
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
}

#[test]
fn solve_zero_indexed_csr() {
    // Same SPD as the first test but using 0-based CSR.
    let ia = vec![0_i32, 2, 4, 5];
    let ja = vec![0_i32, 1, 1, 2, 2];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::Zero);
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
}

#[test]
fn solve_two_rhs() {
    // Same SPD; two right-hand sides at once.
    //   B = [[3, 6], [2, 4], [3, 6]] (column-major: [3,2,3,6,4,6])
    //   X_true = [[1, 2], [1, 2], [1, 2]]
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    let b = vec![3.0_f64, 2.0, 3.0, 6.0, 4.0, 6.0];
    let mut x = vec![0.0_f64; 6];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.solve_multi(3, 2, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[3], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[4], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[5], 2.0, epsilon = 1e-10);
}

#[test]
fn second_solve_uses_cached_factorization() {
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);

    // First solve: factorizes.
    let b1 = vec![3.0_f64, 2.0, 3.0];
    let mut x1 = vec![0.0_f64; 3];
    solver.solve(3, &a, &ia, &ja, &b1, &mut x1).unwrap();
    assert_abs_diff_eq!(x1[0], 1.0, epsilon = 1e-10);

    // Second solve: should reuse the existing factorization (phase 33 only).
    let b2 = vec![6.0_f64, 4.0, 6.0];
    let mut x2 = vec![0.0_f64; 3];
    solver.solve(3, &a, &ia, &ja, &b2, &mut x2).unwrap();
    assert_abs_diff_eq!(x2[0], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x2[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x2[2], 2.0, epsilon = 1e-10);
}

#[test]
fn analyze_then_solve() {
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.analyze_and_factorize(3, &a, &ia, &ja).unwrap();

    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
}
