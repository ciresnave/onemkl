#![cfg(feature = "preconditioners")]

//! Verify ILU0 / ILUT preconditioners.

use approx::assert_abs_diff_eq;

use onemkl::preconditioners::{ilu0, ilut};

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
