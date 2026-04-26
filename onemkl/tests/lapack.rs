//! Verify LAPACK wrappers against hand-computed reference values.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::lapack::{
    geev_complex, geev_real, gels, gelsd, gesdd, gesvd, geqrf, gesv, getrf,
    getri, getrs, heev, orgqr, posv, potrf, potrs, syev, sysv, Job,
};
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::{Layout, Transpose, UpLo};

#[test]
fn gesv_solves_2x2_system() {
    // A = [[2, 1], [1, 3]] (row-major); b = [3, 4]
    // Solution: x ≈ [1, 1] (since A * [1,1]ᵀ = [3, 4])
    let mut a = [2.0_f64, 1.0, 1.0, 3.0];
    let mut b = [3.0_f64, 4.0];
    let mut ipiv = vec![0_i32; 2];

    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
    gesv(&mut a_view, &mut ipiv, &mut b_view).unwrap();
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-12);
}

#[test]
fn getrf_then_getrs() {
    // Same system. Two-step.
    let mut a = [2.0_f64, 1.0, 1.0, 3.0];
    let mut b = [3.0_f64, 4.0];
    let mut ipiv = vec![0_i32; 2];

    {
        let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
        getrf(&mut a_view, &mut ipiv).unwrap();
    }
    {
        let a_view = MatrixRef::new(&a, 2, 2, Layout::RowMajor).unwrap();
        let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
        getrs(Transpose::NoTrans, &a_view, &ipiv, &mut b_view).unwrap();
    }
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-12);
}

#[test]
fn getri_inverts() {
    // A = [[2, 1], [1, 3]], A⁻¹ * [3, 4]ᵀ = [1, 1].
    let mut a = [2.0_f64, 1.0, 1.0, 3.0];
    let mut ipiv = vec![0_i32; 2];

    {
        let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
        getrf(&mut a_view, &mut ipiv).unwrap();
        getri(&mut a_view, &ipiv).unwrap();
    }
    // det = 5, so A⁻¹ = [[3, -1], [-1, 2]] / 5.
    assert_abs_diff_eq!(a[0], 3.0 / 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a[1], -1.0 / 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a[2], -1.0 / 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a[3], 2.0 / 5.0, epsilon = 1e-12);
}

#[test]
fn posv_positive_definite() {
    // SPD: A = [[4, 1], [1, 3]]; b = [9, 7]
    // Solution to A*x = b: x = [2, 1.6...] — let's compute: (4*2+1*y=9)→y=1; check
    //   row 0: 4*2 + 1*1 = 9 ✓
    //   row 1: 1*2 + 3*1 = 5 ≠ 7
    // Pick again: A*x = b with x = [2, 5/3]:
    //   4*2 + 5/3 = 8 + 1.67 = 9.67 ≠ 9
    // Use: x = [(9*3 - 7)/(4*3 - 1), (4*7 - 9)/11] = [20/11, 19/11]
    let mut a = [4.0_f64, 1.0, 1.0, 3.0];
    let mut b = [9.0_f64, 7.0];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
    posv(UpLo::Upper, &mut a_view, &mut b_view).unwrap();
    assert_abs_diff_eq!(b[0], 20.0 / 11.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1], 19.0 / 11.0, epsilon = 1e-10);
}

#[test]
fn potrf_then_potrs() {
    let mut a = [4.0_f64, 1.0, 99.0, 3.0]; // upper used; lower overwritten arbitrary
    let mut b = [9.0_f64, 7.0];
    {
        let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
        potrf(UpLo::Upper, &mut a_view).unwrap();
    }
    {
        let a_view = MatrixRef::new(&a, 2, 2, Layout::RowMajor).unwrap();
        let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
        potrs(UpLo::Upper, &a_view, &mut b_view).unwrap();
    }
    assert_abs_diff_eq!(b[0], 20.0 / 11.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1], 19.0 / 11.0, epsilon = 1e-10);
}

#[test]
fn sysv_symmetric() {
    // Indefinite symmetric A = [[1, 2], [2, 1]] (eigenvalues 3, -1).
    // b = [3, 3]; solve A*x = b: x = [1, 1] (since [1,2]·[1,1]=3, [2,1]·[1,1]=3)
    let mut a = [1.0_f64, 2.0, 2.0, 1.0];
    let mut b = [3.0_f64, 3.0];
    let mut ipiv = vec![0_i32; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 2, 1, Layout::RowMajor).unwrap();
    sysv(UpLo::Upper, &mut a_view, &mut ipiv, &mut b_view).unwrap();
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-12);
}

#[test]
fn syev_eigenvalues() {
    // Real symmetric A = [[2, 1], [1, 2]] has eigenvalues 1 and 3.
    let mut a = [2.0_f64, 1.0, 1.0, 2.0];
    let mut w = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    syev(Job::None, UpLo::Upper, &mut a_view, &mut w).unwrap();
    // Eigenvalues are returned in ascending order.
    assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 3.0, epsilon = 1e-12);
}

#[test]
fn syev_with_eigenvectors() {
    // 3x3 diagonal — eigenvalues are the diagonal entries.
    let mut a = [
        5.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let mut w = vec![0.0_f64; 3];
    let mut a_view = MatrixMut::new(&mut a, 3, 3, Layout::RowMajor).unwrap();
    syev(Job::Compute, UpLo::Upper, &mut a_view, &mut w).unwrap();
    // Sorted ascending → 1, 2, 5.
    assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[2], 5.0, epsilon = 1e-12);
}

#[test]
fn heev_hermitian() {
    // Hermitian A:
    //   [[2, 1+i],
    //    [1-i, 3]]
    // Eigenvalues: trace = 5, det = 6 - |1+i|^2 = 6 - 2 = 4
    // λ² - 5λ + 4 = 0 → λ = 1, 4
    let mut a = [
        Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0),
        Complex64::new(99.0, 99.0), Complex64::new(3.0, 0.0),
    ];
    let mut w = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    heev(Job::None, UpLo::Upper, &mut a_view, &mut w).unwrap();
    assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w[1], 4.0, epsilon = 1e-12);
}

#[test]
fn geev_real_general() {
    // Diagonal A — eigenvalues are the diagonal entries (no imaginary part).
    let mut a = [
        5.0_f64, 0.0,
        0.0, 2.0,
    ];
    let mut wr = vec![0.0_f64; 2];
    let mut wi = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    geev_real(Job::None, Job::None, &mut a_view, &mut wr, &mut wi, None, None).unwrap();
    // The order isn't guaranteed for geev. Sort.
    let mut eigs = wr.clone();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_abs_diff_eq!(eigs[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(eigs[1], 5.0, epsilon = 1e-12);
    for v in &wi {
        assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn geev_complex_general() {
    // Diagonal complex A.
    let mut a = [
        Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(3.0, -1.0),
    ];
    let mut w = vec![Complex64::new(0.0, 0.0); 2];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    geev_complex(Job::None, Job::None, &mut a_view, &mut w, None, None).unwrap();
    // Should contain both diagonal values, in some order.
    let has_first = w.iter().any(|z| (z.re - 2.0).abs() < 1e-12 && (z.im - 1.0).abs() < 1e-12);
    let has_second = w.iter().any(|z| (z.re - 3.0).abs() < 1e-12 && (z.im + 1.0).abs() < 1e-12);
    assert!(has_first && has_second);
}

#[test]
fn geqrf_then_orgqr_reconstructs_q() {
    // 3x2 matrix; QR factor then reconstruct Q.
    // We then check QᵀQ = I (Q should have orthonormal columns).
    let mut a = [
        1.0_f64, 0.0,
        1.0, 1.0,
        1.0, 2.0,
    ];
    let mut tau = vec![0.0_f64; 2];
    {
        let mut a_view = MatrixMut::new(&mut a, 3, 2, Layout::RowMajor).unwrap();
        geqrf(&mut a_view, &mut tau).unwrap();
    }
    // Reconstruct Q (3x2 — first 2 columns of full Q).
    {
        let mut a_view = MatrixMut::new(&mut a, 3, 2, Layout::RowMajor).unwrap();
        orgqr(&mut a_view, &tau, 2).unwrap();
    }
    // Now `a` holds Q; check QᵀQ = I.
    // Q[0] = [a[0], a[1]], Q[1] = [a[2], a[3]], Q[2] = [a[4], a[5]]
    // (QᵀQ)[i][j] = Σ_k Q[k][i] * Q[k][j]
    let q = a;
    let dot00 = q[0]*q[0] + q[2]*q[2] + q[4]*q[4];
    let dot11 = q[1]*q[1] + q[3]*q[3] + q[5]*q[5];
    let dot01 = q[0]*q[1] + q[2]*q[3] + q[4]*q[5];
    assert_abs_diff_eq!(dot00, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(dot11, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(dot01, 0.0, epsilon = 1e-10);
}

#[test]
fn gels_least_squares() {
    // Overdetermined: 3x2, A * x ≈ b.
    // A = [[1, 1], [1, 2], [1, 3]]
    // b = [1, 2, 3] → exact fit at x = [0, 1].
    let mut a = [1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0];
    let mut b = [1.0_f64, 2.0, 3.0];
    let mut a_view = MatrixMut::new(&mut a, 3, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 3, 1, Layout::RowMajor).unwrap();
    gels(Transpose::NoTrans, &mut a_view, &mut b_view).unwrap();
    // First 2 rows of b hold the solution.
    assert_abs_diff_eq!(b[0], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
}

#[test]
fn gelsd_least_squares_via_svd() {
    let mut a = [1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0];
    let mut b = [1.0_f64, 2.0, 3.0];
    let mut s = vec![0.0_f64; 2];
    let mut a_view = MatrixMut::new(&mut a, 3, 2, Layout::RowMajor).unwrap();
    let mut b_view = MatrixMut::new(&mut b, 3, 1, Layout::RowMajor).unwrap();
    let rank = gelsd(&mut a_view, &mut b_view, &mut s, -1.0).unwrap();
    assert_eq!(rank, 2);
    assert_abs_diff_eq!(b[0], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
}

#[test]
fn gesdd_returns_singular_values() {
    // Simple 2x2 diagonal: singular values are absolute diagonal entries.
    let mut a = [3.0_f64, 0.0, 0.0, -4.0];
    let mut s = vec![0.0_f64; 2];
    let mut u_buf = vec![0.0_f64; 4];
    let mut vt_buf = vec![0.0_f64; 4];
    {
        let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
        let mut u = MatrixMut::new(&mut u_buf, 2, 2, Layout::RowMajor).unwrap();
        let mut vt = MatrixMut::new(&mut vt_buf, 2, 2, Layout::RowMajor).unwrap();
        gesdd(Job::All, &mut a_view, &mut s, Some(&mut u), Some(&mut vt))
            .unwrap();
    }
    // Singular values are 4 and 3 (descending order).
    assert_abs_diff_eq!(s[0], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[1], 3.0, epsilon = 1e-12);
}

#[test]
fn gesvd_returns_singular_values() {
    let mut a = [3.0_f64, 0.0, 0.0, -4.0];
    let mut s = vec![0.0_f64; 2];
    let mut superb = vec![0.0_f64; 1];
    let mut a_view = MatrixMut::new(&mut a, 2, 2, Layout::RowMajor).unwrap();
    gesvd(Job::None, Job::None, &mut a_view, &mut s, None, None, &mut superb).unwrap();
    assert_abs_diff_eq!(s[0], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[1], 3.0, epsilon = 1e-12);
}
