#![cfg(feature = "preconditioners")]

//! Verify ILU0 / ILUT preconditioners.

use approx::assert_abs_diff_eq;

use onemkl::preconditioners::{apply_ilu, ilu0, ilut};

#[test]
fn ilu0_preserves_sparsity() {
    // 3x3 SPD tridiagonal in 1-based CSR (full storage):
    //   [[ 4, -1,  0],
    //    [-1,  4, -1],
    //    [ 0, -1,  4]]
    let ia = vec![1_i32, 3, 6, 8];
    let ja = vec![1_i32, 2, 1, 2, 3, 2, 3];
    let a = vec![4.0_f64, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];

    let alu = ilu0(3, &a, &ia, &ja).unwrap();
    // Output has the same sparsity pattern as input.
    assert_eq!(alu.len(), a.len());
    // ILU0 of an SPD matrix is exact for tridiagonal — diagonal entries
    // become the U pivots.
    // For this matrix the exact LU has u_00 = 4, l_10 = -0.25, u_11 = 4 - 0.25,
    // l_21 = -1/(4-0.25), u_22 = 4 - l_21*1.
    assert_abs_diff_eq!(alu[0], 4.0, epsilon = 1e-12);
}

#[test]
fn ilut_returns_factor() {
    // 4x4 SPD diagonal-dominant pentadiagonal:
    //   [[ 4, -1,  0,  0],
    //    [-1,  4, -1,  0],
    //    [ 0, -1,  4, -1],
    //    [ 0,  0, -1,  4]]
    let ia = vec![1_i32, 3, 6, 9, 11];
    let ja = vec![
        1_i32, 2,
        1, 2, 3,
        2, 3, 4,
        3, 4,
    ];
    let a = vec![
        4.0_f64, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0,
    ];
    let res = ilut(4, &a, &ia, &ja, 1e-6, 5).unwrap();
    assert_eq!(res.ialut.len(), 5);
    // Factor must keep at least the n diagonal entries.
    assert!(res.alut.len() >= 4);
}

#[test]
fn invalid_ia_length_rejected() {
    let r = ilu0(3, &[1.0_f64], &[0_i32, 1], &[0_i32]);
    assert!(r.is_err());
}

#[test]
fn apply_ilu0_round_trips_through_a() {
    // 3×3 tridiagonal A = [[ 4, -1,  0], [-1, 4, -1], [0, -1, 4]].
    // ILU(0) is exact on a tridiagonal SPD matrix, so M = A and
    // M⁻¹ * (A * x) = x for any x.
    let ia = vec![1_i32, 3, 6, 8];
    let ja = vec![1_i32, 2, 1, 2, 3, 2, 3];
    let a = vec![4.0_f64, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let alu = ilu0(3, &a, &ia, &ja).unwrap();

    // Choose x = [1, 2, 3]; compute A * x by hand.
    //   A * [1,2,3] = [4-2, -1+8-3, -2+12] = [2, 4, 10]
    let v = vec![2.0_f64, 4.0, 10.0];
    let recovered = apply_ilu(3, &alu, &ia, &ja, &v).unwrap();
    assert_abs_diff_eq!(recovered[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(recovered[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(recovered[2], 3.0, epsilon = 1e-10);
}

#[test]
fn apply_ilu_validates_dimensions() {
    let ia = vec![1_i32, 2];
    let ja = vec![1_i32];
    let alu = vec![1.0_f64];
    let v_wrong = vec![1.0_f64, 2.0];
    let r = apply_ilu(1, &alu, &ia, &ja, &v_wrong);
    assert!(r.is_err());
}
