#![cfg(feature = "lapack")]

//! Verify LAPACK auxiliary routines (lacpy / lange / gecon / laswp / larfg).

use approx::assert_abs_diff_eq;

use onemkl::lapack::{
    gecon, getrf, lacpy, lange, larfg, laswp, LacpyPart, MatrixNorm,
};
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::Layout;

#[test]
fn lacpy_full_matrix() {
    let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut b = vec![0.0_f64; 6];
    let a_ref = MatrixRef::new(&a, 2, 3, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 3, Layout::RowMajor).unwrap();
    lacpy(LacpyPart::Full, &a_ref, &mut b_view).unwrap();
    assert_eq!(b, a);
}

#[test]
fn lacpy_upper_triangle_only() {
    // 3×3 source: rows are [1, 2, 3], [4, 5, 6], [7, 8, 9].
    let a = vec![
        1.0_f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let mut b = vec![100.0_f64; 9];
    let a_ref = MatrixRef::new(&a, 3, 3, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 3, 3, Layout::RowMajor).unwrap();
    lacpy(LacpyPart::Upper, &a_ref, &mut b_view).unwrap();
    // Upper triangle (including diagonal) is copied.
    assert_eq!(b[0], 1.0); // (0,0)
    assert_eq!(b[1], 2.0); // (0,1)
    assert_eq!(b[2], 3.0); // (0,2)
    assert_eq!(b[4], 5.0); // (1,1)
    assert_eq!(b[5], 6.0); // (1,2)
    assert_eq!(b[8], 9.0); // (2,2)
    // The lower triangle is unspecified — LAPACK doesn't promise to
    // leave it alone or to zero it. We don't assert anything about
    // b[3], b[6], b[7].
}

#[test]
fn lange_one_norm() {
    // 2×3 with column-sum-of-abs:
    //   col 0: |1| + |4| = 5
    //   col 1: |2| + |5| = 7
    //   col 2: |3| + |6| = 9   ← max
    let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_ref = MatrixRef::new(&a, 2, 3, Layout::RowMajor).unwrap();
    let n = lange::<f64>(MatrixNorm::One, &a_ref).unwrap();
    assert_abs_diff_eq!(n, 9.0, epsilon = 1e-12);
}

#[test]
fn lange_infinity_norm() {
    // Same matrix; row-sum-of-abs:
    //   row 0: 1 + 2 + 3 = 6
    //   row 1: 4 + 5 + 6 = 15  ← max
    let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_ref = MatrixRef::new(&a, 2, 3, Layout::RowMajor).unwrap();
    let n = lange::<f64>(MatrixNorm::Infinity, &a_ref).unwrap();
    assert_abs_diff_eq!(n, 15.0, epsilon = 1e-12);
}

#[test]
fn lange_max_abs_element() {
    let a = vec![1.0_f64, -7.0, 3.0, 4.0, 5.0, 6.0];
    let a_ref = MatrixRef::new(&a, 2, 3, Layout::RowMajor).unwrap();
    let n = lange::<f64>(MatrixNorm::Max, &a_ref).unwrap();
    assert_abs_diff_eq!(n, 7.0, epsilon = 1e-12);
}

#[test]
fn gecon_well_conditioned_matrix() {
    // Identity has condition number 1, so 1 / κ = 1.
    let mut a = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let a_snapshot = a.clone();
    let snapshot_ref = MatrixRef::new(&a_snapshot, 3, 3, Layout::RowMajor).unwrap();
    let anorm = lange::<f64>(MatrixNorm::One, &snapshot_ref).unwrap();
    // LU factorize in-place.
    {
        let mut a_view = MatrixMut::new(&mut a, 3, 3, Layout::RowMajor).unwrap();
        let mut ipiv = vec![0_i32; 3];
        getrf::<f64>(&mut a_view, &mut ipiv).unwrap();
    }
    // gecon expects a MatrixRef of the LU-factored matrix.
    let lu_ref = MatrixRef::new(&a, 3, 3, Layout::RowMajor).unwrap();
    let rcond = gecon::<f64>(MatrixNorm::One, &lu_ref, anorm).unwrap();
    assert_abs_diff_eq!(rcond, 1.0, epsilon = 1e-10);
}

#[test]
fn laswp_swaps_two_rows() {
    // 3×2 matrix:
    //   row 0: [1, 2]
    //   row 1: [3, 4]
    //   row 2: [5, 6]
    let mut a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut a_view = MatrixMut::new(&mut a, 3, 2, Layout::RowMajor).unwrap();
    // Swap row 1 with row 3 (1-based ipiv[0] = 3).
    let ipiv = vec![3_i32, 2, 3];
    laswp::<f64>(&mut a_view, 1, 1, &ipiv, 1).unwrap();
    // After swapping rows 1 and 3:
    //   row 0: [5, 6] (was row 2)
    //   row 1: [3, 4]
    //   row 2: [1, 2] (was row 0)
    assert_abs_diff_eq!(a[0], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a[1], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a[4], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a[5], 2.0, epsilon = 1e-12);
}

#[test]
fn larfg_zeroes_out_the_tail() {
    // Build a Householder reflector that zeros out [3, 4] when applied
    // to leading α = 0. After larfg the new α should equal the
    // 2-norm of the original (α, x) up to sign.
    let mut alpha = 0.0_f64;
    let mut x = vec![3.0_f64, 4.0];
    let result = larfg::<f64>(3, &mut alpha, &mut x, 1).unwrap();
    // ‖(0, 3, 4)‖₂ = 5.
    assert_abs_diff_eq!(result.alpha.abs(), 5.0, epsilon = 1e-12);
    assert_eq!(alpha, result.alpha);
    // tau is in [1, 2] for valid reflectors with this geometry.
    assert!(result.tau >= 1.0 && result.tau <= 2.0);
}

#[test]
fn larfg_handles_already_canonical() {
    // If x is all zeros, the reflector is trivial: alpha unchanged,
    // tau = 0.
    let mut alpha = 7.0_f64;
    let mut x = vec![0.0_f64, 0.0];
    let result = larfg::<f64>(3, &mut alpha, &mut x, 1).unwrap();
    assert_abs_diff_eq!(alpha, 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result.tau, 0.0, epsilon = 1e-12);
}
