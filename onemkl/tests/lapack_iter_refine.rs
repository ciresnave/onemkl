#![cfg(feature = "lapack")]

//! Verify mixed-precision iterative-refinement solvers
//! (`?sgesv` / `?sposv` family).

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::lapack::{
    iter_refine_gesv_c64, iter_refine_gesv_f64, iter_refine_posv_c64,
    iter_refine_posv_f64,
};
use onemkl::matrix::MatrixMut;
use onemkl::{Layout, UpLo};

#[test]
fn iter_refine_gesv_f64_solves_2x2() {
    // A = [[2, 1], [1, 3]] (row-major); B = [3; 4]
    // True solution x = [1; 1].
    let mut a = [2.0_f64, 1.0, 1.0, 3.0];
    let mut b = [3.0_f64, 4.0];
    let mut x = [0.0_f64, 0.0];
    let mut ipiv = vec![0_i32; 2];

    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
    let mut x_view = MatrixMut::new(&mut x, 2, 1, Layout::RowMajor).unwrap();

    let outcome = iter_refine_gesv_f64(
        &mut a_view,
        &mut ipiv,
        &mut b_view,
        &mut x_view,
    )
    .unwrap();

    // iter can be negative (fell back to plain double) for this tiny
    // matrix, but the solution must still be correct.
    let _ = outcome.iter;
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
}

#[test]
fn iter_refine_posv_f64_solves_spd() {
    // A is SPD: [[4, 1], [1, 3]]; B = [5; 7]
    // x ≈ [0.7272727, 2.0909091]
    let mut a = [4.0_f64, 1.0, 1.0, 3.0];
    let mut b = [5.0_f64, 7.0];
    let mut x = [0.0_f64, 0.0];

    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
    let mut x_view = MatrixMut::new(&mut x, 2, 1, Layout::RowMajor).unwrap();

    iter_refine_posv_f64(
        UpLo::Upper,
        &mut a_view,
        &mut b_view,
        &mut x_view,
    )
    .unwrap();

    let det = 4.0 * 3.0 - 1.0 * 1.0;
    let expected_x0 = (3.0 * 5.0 - 1.0 * 7.0) / det;
    let expected_x1 = (-1.0 * 5.0 + 4.0 * 7.0) / det;
    assert_abs_diff_eq!(x[0], expected_x0, epsilon = 1e-9);
    assert_abs_diff_eq!(x[1], expected_x1, epsilon = 1e-9);
}

#[test]
fn iter_refine_gesv_c64_solves_2x2_complex() {
    // A = [[2+0i, 0+1i], [0-1i, 2+0i]] (Hermitian PD as a real test);
    // B = [1+0i, 0+0i]ᵀ.
    let mut a = [
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(2.0, 0.0),
    ];
    let mut b = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let mut x = [Complex64::new(0.0, 0.0); 2];
    let mut ipiv = vec![0_i32; 2];

    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
    let mut x_view = MatrixMut::new(&mut x, 2, 1, Layout::RowMajor).unwrap();

    iter_refine_gesv_c64(
        &mut a_view,
        &mut ipiv,
        &mut b_view,
        &mut x_view,
    )
    .unwrap();

    // det = 4 - (0 + i)(0 - i) = 4 - 1 = 3
    // x = (1/3) * [2, i] · [1, 0]ᵀ = [2/3, i/3]
    assert_abs_diff_eq!(x[0].re, 2.0 / 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1].re, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1].im, 1.0 / 3.0, epsilon = 1e-10);
}

#[test]
fn iter_refine_posv_c64_solves_hermitian_pd() {
    // Same matrix as the complex gesv test; it's Hermitian PD.
    let mut a = [
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(2.0, 0.0),
    ];
    let mut b = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let mut x = [Complex64::new(0.0, 0.0); 2];

    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
    let mut x_view = MatrixMut::new(&mut x, 2, 1, Layout::RowMajor).unwrap();

    iter_refine_posv_c64(
        UpLo::Upper,
        &mut a_view,
        &mut b_view,
        &mut x_view,
    )
    .unwrap();

    assert_abs_diff_eq!(x[0].re, 2.0 / 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1].im, 1.0 / 3.0, epsilon = 1e-10);
}
