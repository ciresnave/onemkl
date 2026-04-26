#![cfg(feature = "iss")]

//! Verify CG and FGMRES iterative solvers.

use approx::assert_abs_diff_eq;

use onemkl::iss::{solve_cg, solve_fgmres, IssOptions, IssStopReason};

#[test]
fn cg_solves_3x3_spd() {
    // A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]; b = [3, 2, 3].
    // x_true = [1, 1, 1].
    let b = [3.0_f64, 2.0, 3.0];
    let mut x = [0.0_f64; 3];
    let mat_vec = |v: &[f64], out: &mut [f64]| {
        out[0] = 4.0 * v[0] - v[1];
        out[1] = -v[0] + 4.0 * v[1] - v[2];
        out[2] = -v[1] + 4.0 * v[2];
    };
    let res = solve_cg(&b, &mut x, IssOptions::default(), mat_vec).unwrap();
    assert!(res.iterations <= 3, "CG took {} iterations on 3x3 SPD", res.iterations);
    assert_eq!(res.stop_reason, IssStopReason::Converged);
    assert!(res.final_residual_norm <= res.initial_residual_norm);
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-8);
}

#[test]
fn cg_solves_diagonal() {
    // diag(2, 2, 2, 2) is trivially SPD.
    let b = [4.0_f64, 6.0, 8.0, 10.0];
    let mut x = [0.0_f64; 4];
    let mat_vec = |v: &[f64], out: &mut [f64]| {
        for i in 0..4 {
            out[i] = 2.0 * v[i];
        }
    };
    let _ = solve_cg(&b, &mut x, IssOptions::default(), mat_vec).unwrap();
    for (xi, bi) in x.iter().zip(&b) {
        assert_abs_diff_eq!(*xi * 2.0, *bi, epsilon = 1e-8);
    }
}

#[test]
fn fgmres_solves_nonsymmetric_3x3() {
    // A = [[2, 1, 0], [0, 3, 1], [1, 0, 4]]; b = [3, 4, 5].
    // x_true = [1, 1, 1].
    let mut b = vec![3.0_f64, 4.0, 5.0];
    let mut x = vec![0.0_f64; 3];
    let mat_vec = |v: &[f64], out: &mut [f64]| {
        out[0] = 2.0 * v[0] + v[1];
        out[1] = 3.0 * v[1] + v[2];
        out[2] = v[0] + 4.0 * v[2];
    };
    let opts = IssOptions {
        relative_tolerance: 1e-10,
        absolute_tolerance: 0.0,
        max_iterations: 100,
        restart_length: 10,
    };
    let res = solve_fgmres(&mut b, &mut x, opts, mat_vec).unwrap();
    assert_eq!(res.stop_reason, IssStopReason::Converged);
    assert!(res.final_residual_norm <= res.initial_residual_norm);
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-6);
}

#[test]
fn cg_rejects_mismatched_dimensions() {
    let b = [1.0_f64; 3];
    let mut x = [0.0_f64; 4];
    let mat_vec = |_v: &[f64], _out: &mut [f64]| {};
    let r = solve_cg(&b, &mut x, IssOptions::default(), mat_vec);
    assert!(r.is_err());
}
