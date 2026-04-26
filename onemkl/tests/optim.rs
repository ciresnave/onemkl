//! Verify trust-region NLLS solvers with closure callbacks.

use approx::assert_abs_diff_eq;

use onemkl::optim::{solve_trnls, solve_trnls_bounded, TrnlsOptions};

#[test]
fn fits_a_line_through_two_points() {
    // Fit y = a + b*x to points (1, 2) and (2, 3). Exact solution
    // is a = 1, b = 1.
    // Variables x = [a, b]; residuals:
    //   F[0] = a + 1*b - 2
    //   F[1] = a + 2*b - 3
    let mut params = vec![0.0_f64, 0.0]; // initial guess [a, b] = [0, 0]
    let residual = |x: &[f64], f: &mut [f64]| {
        f[0] = x[0] + 1.0 * x[1] - 2.0;
        f[1] = x[0] + 2.0 * x[1] - 3.0;
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| {
        // Column-major m × n with m=2, n=2.
        // ∂F[0]/∂a = 1, ∂F[1]/∂a = 1   (column 0)
        // ∂F[0]/∂b = 1, ∂F[1]/∂b = 2   (column 1)
        j[0] = 1.0;
        j[1] = 1.0;
        j[2] = 1.0;
        j[3] = 2.0;
    };

    let result = solve_trnls(2, 2, &mut params, TrnlsOptions::default(), residual, jacobian)
        .unwrap();
    assert!(result.iterations > 0);
    assert_abs_diff_eq!(params[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(params[1], 1.0, epsilon = 1e-6);
}

#[test]
fn fits_quadratic_to_three_points() {
    // y = a + b*t + c*t² through (0,1), (1,2), (2,5).
    // Solve: a = 1, b = 0, c = 1.
    let mut params = vec![0.0_f64; 3];
    let residual = |x: &[f64], f: &mut [f64]| {
        let (a, b, c) = (x[0], x[1], x[2]);
        f[0] = a + 0.0 * b + 0.0 * c - 1.0;
        f[1] = a + 1.0 * b + 1.0 * c - 2.0;
        f[2] = a + 2.0 * b + 4.0 * c - 5.0;
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| {
        // 3 rows × 3 cols, column-major.
        // col 0: ∂F/∂a = [1, 1, 1]
        j[0] = 1.0; j[1] = 1.0; j[2] = 1.0;
        // col 1: ∂F/∂b = [0, 1, 2]
        j[3] = 0.0; j[4] = 1.0; j[5] = 2.0;
        // col 2: ∂F/∂c = [0, 1, 4]
        j[6] = 0.0; j[7] = 1.0; j[8] = 4.0;
    };

    let result = solve_trnls(3, 3, &mut params, TrnlsOptions::default(), residual, jacobian)
        .unwrap();
    assert!(result.iterations > 0);
    assert_abs_diff_eq!(params[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(params[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(params[2], 1.0, epsilon = 1e-6);
}

#[test]
fn bounded_solver_respects_bounds() {
    // Same line-fitting problem, but constrain a ≥ 0.5.
    // Unconstrained answer is a = 1; should still get there.
    let mut params = vec![0.0_f64, 0.0];
    let lower = vec![0.5_f64, -10.0];
    let upper = vec![10.0_f64, 10.0];
    let residual = |x: &[f64], f: &mut [f64]| {
        f[0] = x[0] + 1.0 * x[1] - 2.0;
        f[1] = x[0] + 2.0 * x[1] - 3.0;
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| {
        j[0] = 1.0; j[1] = 1.0;
        j[2] = 1.0; j[3] = 2.0;
    };
    let _result = solve_trnls_bounded(
        2, 2, &mut params, &lower, &upper,
        TrnlsOptions::default(), residual, jacobian,
    ).unwrap();
    assert!(params[0] >= 0.5 - 1e-9);
    assert_abs_diff_eq!(params[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(params[1], 1.0, epsilon = 1e-6);
}
