#![cfg(feature = "iss")]

//! Verify the user-driven RCI surface (CgSession / FgmresSession).

use approx::assert_abs_diff_eq;

use onemkl::iss::{CgSession, FgmresSession, IssOptions, IssStopReason, RciAction};
use onemkl::Error;

#[test]
fn cg_session_solves_3x3_spd() {
    // A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]; b = [3, 2, 3].
    let b = [3.0_f64, 2.0, 3.0];
    let mut x = [0.0_f64; 3];
    let opts = IssOptions::default();
    let mut session = CgSession::new(3, &b, &x, opts, false).unwrap();

    let mut steps = 0;
    loop {
        match session.step(&b, &mut x).unwrap() {
            RciAction::Converged => break,
            RciAction::NeedMatVec { src, dst } => {
                dst[0] = 4.0 * src[0] - src[1];
                dst[1] = -src[0] + 4.0 * src[1] - src[2];
                dst[2] = -src[1] + 4.0 * src[2];
            }
            RciAction::NeedPrecondition { .. } => unreachable!(),
            RciAction::StoppingTest => continue,
            RciAction::Other(code) => panic!("unexpected RCI code {code}"),
        }
        steps += 1;
        if steps > 100 {
            panic!("CG did not converge in 100 outer steps");
        }
    }

    let res = session.finish(&b, &x).unwrap();
    assert_eq!(res.stop_reason, IssStopReason::Converged);
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-8);
}

#[test]
fn cg_session_with_preconditioner() {
    // Same SPD matrix, but with a Jacobi (diagonal) preconditioner.
    // M = diag(A) = diag(4, 4, 4); M⁻¹ * v = v / 4.
    let b = [3.0_f64, 2.0, 3.0];
    let mut x = [0.0_f64; 3];
    let opts = IssOptions::default();
    let mut session = CgSession::new(3, &b, &x, opts, true).unwrap();

    loop {
        match session.step(&b, &mut x).unwrap() {
            RciAction::Converged => break,
            RciAction::NeedMatVec { src, dst } => {
                dst[0] = 4.0 * src[0] - src[1];
                dst[1] = -src[0] + 4.0 * src[1] - src[2];
                dst[2] = -src[1] + 4.0 * src[2];
            }
            RciAction::NeedPrecondition { src, dst } => {
                for i in 0..3 {
                    dst[i] = src[i] / 4.0;
                }
            }
            RciAction::StoppingTest => continue,
            RciAction::Other(code) => panic!("unexpected RCI code {code}"),
        }
    }

    session.finish(&b, &x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-8);
}

#[test]
fn fgmres_session_solves_3x3() {
    // A = [[2, 1, 0], [0, 3, 1], [1, 0, 4]]; b = [3, 4, 5]; x_true = [1,1,1].
    let mut b = vec![3.0_f64, 4.0, 5.0];
    let mut x = vec![0.0_f64; 3];
    let opts = IssOptions {
        relative_tolerance: 1e-10,
        absolute_tolerance: 0.0,
        max_iterations: 100,
        restart_length: 10,
    };
    let mut session = FgmresSession::new(3, &b, &x, opts).unwrap();

    let mut steps = 0;
    loop {
        match session.step(&mut b, &mut x).unwrap() {
            RciAction::Converged => break,
            RciAction::NeedMatVec { src, dst } => {
                dst[0] = 2.0 * src[0] + src[1];
                dst[1] = 3.0 * src[1] + src[2];
                dst[2] = src[0] + 4.0 * src[2];
            }
            RciAction::StoppingTest => continue,
            RciAction::NeedPrecondition { .. } => unreachable!(),
            RciAction::Other(code) => panic!("unexpected RCI code {code}"),
        }
        steps += 1;
        if steps > 100 {
            panic!("FGMRES did not converge in 100 outer steps");
        }
    }

    let res = session.finish(&mut b, &mut x).unwrap();
    assert_eq!(res.stop_reason, IssStopReason::Converged);
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-6);
}

#[test]
fn cg_session_rejects_dimension_mismatch() {
    let b = [1.0_f64; 3];
    let x = [0.0_f64; 4];
    let r = CgSession::new(3, &b, &x, IssOptions::default(), false);
    assert!(matches!(r, Err(Error::InvalidArgument(_))));
}

#[test]
fn cg_session_rejects_unrequested_precondition() {
    // Build session WITHOUT preconditioning, then try to handle a
    // precondition request — but we configure ipar[10] = 0, so MKL
    // shouldn't ask. This test just verifies the wiring: the
    // preconditioned flag controls whether a NeedPrecondition would
    // be returned.
    let b = [3.0_f64, 2.0, 3.0];
    let mut x = [0.0_f64; 3];
    let mut session = CgSession::new(3, &b, &x, IssOptions::default(), false).unwrap();
    // Simulate that MKL never asks for preconditioning by just checking
    // the typical CG run doesn't return NeedPrecondition.
    loop {
        match session.step(&b, &mut x).unwrap() {
            RciAction::Converged => break,
            RciAction::NeedMatVec { src, dst } => {
                dst[0] = 4.0 * src[0] - src[1];
                dst[1] = -src[0] + 4.0 * src[1] - src[2];
                dst[2] = -src[1] + 4.0 * src[2];
            }
            RciAction::StoppingTest => continue,
            RciAction::NeedPrecondition { .. } => {
                panic!("CG should not request preconditioning when not enabled");
            }
            RciAction::Other(code) => panic!("unexpected RCI code {code}"),
        }
    }
    session.finish(&b, &x).unwrap();
}
