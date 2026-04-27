#![cfg(feature = "feast")]

//! Verify FEAST CSR, banded, and generalized variants.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::feast::{
    eigh_complex_banded, eigh_complex_csr, eigh_real_banded, eigh_real_csr,
    gen_eigh_complex_dense, gen_eigh_real_dense,
};
use onemkl::UpLo;

#[test]
fn real_csr_diagonal() {
    // diag(1, 2, 5) in 1-based CSR upper triangle.
    let isa = vec![1_i32, 2, 3, 4];
    let jsa = vec![1_i32, 2, 3];
    let sa = vec![1.0_f64, 2.0, 5.0];
    let res = eigh_real_csr::<f64>(UpLo::Upper, 3, &sa, &isa, &jsa, 1.5, 4.5, 3).unwrap();
    assert_eq!(res.m, 1);
    assert_abs_diff_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-8);
}

#[test]
fn complex_csr_hermitian() {
    // Hermitian [[2, 1+i], [1-i, 3]] — eigenvalues 1 and 4.
    // 1-based upper-triangular CSR.
    let isa = vec![1_i32, 3, 4];
    let jsa = vec![1_i32, 2, 2];
    let sa = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
    ];
    let res = eigh_complex_csr::<Complex64>(UpLo::Upper, 2, &sa, &isa, &jsa, 0.5, 1.5, 2)
        .unwrap();
    assert_eq!(res.m, 1);
    assert_abs_diff_eq!(res.eigenvalues[0], 1.0, epsilon = 1e-8);
}

#[test]
fn real_banded_tridiagonal() {
    // Symmetric tridiagonal A:
    //   [[2, -1, 0],
    //    [-1, 2, -1],
    //    [ 0, -1, 2]]
    // Eigenvalues: 2 - sqrt(2), 2, 2 + sqrt(2) ≈ 0.586, 2.0, 3.414.
    // kla = 1 (one super-diagonal). LAPACK upper band format,
    // column-major, lda = kla + 1 = 2:
    //   col 0: [unused,         a(0,0)=2]   → [_, 2]
    //   col 1: [a(0,1)=-1,      a(1,1)=2]   → [-1, 2]
    //   col 2: [a(1,2)=-1,      a(2,2)=2]   → [-1, 2]
    let kla = 1;
    let lda = kla + 1; // 2
    let a = vec![
        0.0_f64, 2.0,   // col 0
        -1.0, 2.0,      // col 1
        -1.0, 2.0,      // col 2
    ];
    let res = eigh_real_banded::<f64>(UpLo::Upper, 3, kla, &a, lda, 1.5, 2.5, 3).unwrap();
    assert_eq!(res.m, 1);
    assert_abs_diff_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-8);
}

#[test]
fn complex_banded_hermitian() {
    // Hermitian [[2, 1+i], [1-i, 3]] in band format with kla = 1, upper.
    // For UPPER format with band size kla=1:
    //   row 0 (super-diagonal): a(1,2) = 1+i
    //   row 1 (diagonal):       a(1,1)=2, a(2,2)=3
    // Stored column-major: col 0 = [unused, 2], col 1 = [1+i, 3].
    let kla = 1;
    let lda = kla + 1; // 2
    let a = vec![
        Complex64::new(0.0, 0.0), // unused (super-diagonal of column 0)
        Complex64::new(2.0, 0.0), // diagonal (1,1)
        Complex64::new(1.0, 1.0), // super-diagonal (1,2)
        Complex64::new(3.0, 0.0), // diagonal (2,2)
    ];
    let res = eigh_complex_banded::<Complex64>(
        UpLo::Upper, 2, kla, &a, lda, 0.5, 1.5, 2,
    )
    .unwrap();
    assert_eq!(res.m, 1);
    assert_abs_diff_eq!(res.eigenvalues[0], 1.0, epsilon = 1e-8);
}

#[test]
fn real_generalized_dense() {
    // A = diag(2, 8); B = diag(1, 4).
    // A x = λ B x → λ = a_ii / b_ii = 2, 2.
    // Range [1.5, 2.5] → both eigenvalues fall in range.
    let a = vec![2.0_f64, 0.0, 0.0, 8.0];
    let b = vec![1.0_f64, 0.0, 0.0, 4.0];
    let res = gen_eigh_real_dense::<f64>(
        UpLo::Upper, 2, &a, 2, &b, 2, 1.5, 2.5, 2,
    )
    .unwrap();
    assert_eq!(res.m, 2);
    assert_abs_diff_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-8);
    assert_abs_diff_eq!(res.eigenvalues[1], 2.0, epsilon = 1e-8);
}

#[test]
fn complex_generalized_dense() {
    // A = diag(2, 8); B = diag(1, 4) — same trick, complex storage.
    let a = vec![
        Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(8.0, 0.0),
    ];
    let b = vec![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(4.0, 0.0),
    ];
    let res = gen_eigh_complex_dense::<Complex64>(
        UpLo::Upper, 2, &a, 2, &b, 2, 1.5, 2.5, 2,
    )
    .unwrap();
    assert_eq!(res.m, 2);
    assert_abs_diff_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-8);
    assert_abs_diff_eq!(res.eigenvalues[1], 2.0, epsilon = 1e-8);
}
