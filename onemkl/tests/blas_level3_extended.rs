#![cfg(feature = "blas")]

//! Verify the expanded BLAS Level 3 surface against hand-computed values.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::blas::level3::{
    gemm3m, gemmt, hemm, her2k, herk, symm, syr2k, syrk, trmm, trsm,
};
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::{Diag, Layout, Side, Transpose, UpLo};

#[test]
fn symm_left_real() {
    // A symmetric (upper) 2x2 = [[2, 1], [_, 3]] (lower not read)
    // B 2x2 = [[1, 0], [0, 1]]; α=1, β=0 → C = A * B = full A.
    let a_buf = [2.0_f64, 1.0, 99.0, 3.0];
    let b_buf = [1.0_f64, 0.0, 0.0, 1.0];
    let mut c_buf = [0.0_f64; 4];

    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    symm(Side::Left, UpLo::Upper, 1.0, &a, &b, 0.0, &mut c).unwrap();

    // C row-major: [a00 a01 / a10 a11] = [2 1 / 1 3]
    assert_abs_diff_eq!(c_buf[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[2], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3], 3.0, epsilon = 1e-12);
}

#[test]
fn syrk_no_trans_real() {
    // A = [[1, 2], [3, 4]] (2x2, NoTrans)
    // C = A * Aᵀ = [[1+4, 3+8], [3+8, 9+16]] = [[5, 11], [11, 25]]
    // syrk only writes upper triangle.
    let a_buf = [1.0_f64, 2.0, 3.0, 4.0];
    let mut c_buf = [0.0_f64; 4];

    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    syrk(UpLo::Upper, Transpose::NoTrans, 1.0, &a, 0.0, &mut c).unwrap();

    assert_abs_diff_eq!(c_buf[0], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1], 11.0, epsilon = 1e-12);
    // Lower triangle (c_buf[2]) untouched.
    assert_abs_diff_eq!(c_buf[2], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3], 25.0, epsilon = 1e-12);
}

#[test]
fn syr2k_no_trans_real() {
    // A=[[1,0],[0,1]], B=[[0,1],[1,0]]
    // C = α*(A*Bᵀ + B*Aᵀ) + 0
    // A*Bᵀ = [[0,1],[1,0]]; B*Aᵀ = [[0,1],[1,0]]
    // Sum = [[0,2],[2,0]]; only upper written.
    let a_buf = [1.0_f64, 0.0, 0.0, 1.0];
    let b_buf = [0.0_f64, 1.0, 1.0, 0.0];
    let mut c_buf = [0.0_f64; 4];

    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    syr2k(UpLo::Upper, Transpose::NoTrans, 1.0, &a, &b, 0.0, &mut c).unwrap();
    assert_abs_diff_eq!(c_buf[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3], 0.0, epsilon = 1e-12);
}

#[test]
fn trmm_left_upper_no_trans() {
    // A = [[2, 1], [_, 3]] (upper triangular, lower not read)
    // B = [[1, 0], [0, 1]]
    // B ← A * B = A
    let a_buf = [2.0_f64, 1.0, 99.0, 3.0];
    let mut b_buf = [1.0_f64, 0.0, 0.0, 1.0];

    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut b = MatrixMut::new(&mut b_buf, 2, 2, Layout::RowMajor).unwrap();

    trmm(
        Side::Left,
        UpLo::Upper,
        Transpose::NoTrans,
        Diag::NonUnit,
        1.0_f64,
        &a,
        &mut b,
    )
    .unwrap();

    assert_abs_diff_eq!(b_buf[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b_buf[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b_buf[2], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b_buf[3], 3.0, epsilon = 1e-12);
}

#[test]
fn trsm_inverse_of_trmm() {
    // After trmm above, B held = A. trsm should recover identity.
    // We'll just do trmm-then-trsm and check we round-trip.
    let a_buf = [2.0_f64, 1.0, 99.0, 3.0];
    let mut b_buf = [1.0_f64, 0.0, 0.0, 1.0];

    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    {
        let mut b = MatrixMut::new(&mut b_buf, 2, 2, Layout::RowMajor).unwrap();
        trmm(
            Side::Left,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            1.0_f64,
            &a,
            &mut b,
        )
        .unwrap();
    }
    {
        let mut b = MatrixMut::new(&mut b_buf, 2, 2, Layout::RowMajor).unwrap();
        trsm(
            Side::Left,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            1.0_f64,
            &a,
            &mut b,
        )
        .unwrap();
    }
    // Round-trip should give back the identity.
    assert_abs_diff_eq!(b_buf[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b_buf[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b_buf[2], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b_buf[3], 1.0, epsilon = 1e-12);
}

#[test]
fn gemmt_writes_only_one_triangle() {
    // 2x2 gemm of A = [[1, 1], [1, 1]] with itself = [[2, 2], [2, 2]]
    // gemmt with UpLo::Upper writes upper only; lower stays at original 99.
    let a_buf = [1.0_f64, 1.0, 1.0, 1.0];
    let b_buf = [1.0_f64, 1.0, 1.0, 1.0];
    let mut c_buf = [99.0_f64; 4];

    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    gemmt(
        UpLo::Upper,
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0,
        &a,
        &b,
        0.0,
        &mut c,
    )
    .unwrap();

    // Upper triangle (row-major: [0]=(0,0), [1]=(0,1), [3]=(1,1)) gets overwritten.
    assert_abs_diff_eq!(c_buf[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3], 2.0, epsilon = 1e-12);
    // Lower-triangle off-diagonal (row-major c_buf[2]) is the (1,0) entry → not written.
    assert_abs_diff_eq!(c_buf[2], 99.0, epsilon = 1e-12);
}

#[test]
fn hemm_complex_matches_expanded_full() {
    // 2x2 Hermitian A (upper stored): [[2, 1+i], [_, 3]]
    // B = identity → C = A_full = [[2, 1+i], [1-i, 3]]
    let a_buf = [
        Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0),
        Complex64::new(99.0, 99.0), Complex64::new(3.0, 0.0),
    ];
    let b_buf = [
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
    ];
    let mut c_buf = [Complex64::new(0.0, 0.0); 4];

    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    hemm(
        Side::Left,
        UpLo::Upper,
        Complex64::new(1.0, 0.0),
        &a,
        &b,
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    assert_abs_diff_eq!(c_buf[0].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[0].im, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1].im, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[2].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[2].im, -1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3].re, 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3].im, 0.0, epsilon = 1e-12);
}

#[test]
fn herk_real_alpha_beta() {
    // A = [[1+i, 2]] (1x2 row-major), trans = ConjTrans → op(A) = Aᴴ : 2x1
    // C = α * Aᴴ * A + β * 0 = α * (Aᴴ * A)
    // Aᴴ A = [[(1-i)(1+i), (1-i)*2], [conj((1-i)*2), 4]] = [[2, 2-2i],[_, 4]]
    let a_buf = [Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
    let mut c_buf = [Complex64::new(0.0, 0.0); 4];

    let a = MatrixRef::new(&a_buf, 1, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    herk(UpLo::Upper, Transpose::ConjTrans, 1.0_f64, &a, 0.0_f64, &mut c).unwrap();

    assert_abs_diff_eq!(c_buf[0].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[0].im, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1].im, -2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3].re, 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3].im, 0.0, epsilon = 1e-12);
}

#[test]
fn herk_rejects_plain_trans() {
    let a_buf = [Complex64::new(1.0, 0.0); 4];
    let mut c_buf = [Complex64::new(0.0, 0.0); 4];
    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    let r = herk(UpLo::Upper, Transpose::Trans, 1.0_f64, &a, 0.0_f64, &mut c);
    assert!(r.is_err());
}

#[test]
fn her2k_with_real_beta() {
    // 2x2 trivially zero out and update.
    // A = [[1, 0], [0, 0]] (row-major); B = [[0, 1], [0, 0]];
    // trans = NoTrans, k = 2 (cols of A/B).
    // op(A) op(B)ᴴ = A * Bᴴ
    //   Bᴴ = [[0, 0], [1, 0]] (transpose+conj of all-real B)
    //   A * Bᴴ = [[0, 0], [0, 0]]
    // op(B) op(A)ᴴ = B * Aᴴ = [[0, 0], [0, 0]]   (similar)
    // So C ← α * 0 + β * C₀.  Pick C₀ that exercises β.
    let a_buf = [
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
    ];
    let b_buf = [
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
    ];
    let mut c_buf = [
        Complex64::new(5.0, 0.0), Complex64::new(7.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(11.0, 0.0),
    ];
    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_buf, 2, 2, Layout::RowMajor).unwrap();

    her2k(
        UpLo::Upper,
        Transpose::NoTrans,
        Complex64::new(1.0, 0.0),
        &a,
        &b,
        2.0_f64,
        &mut c,
    )
    .unwrap();

    // The Hermitian update from this particular A,B is also non-zero in
    // upper triangle: A*Bᴴ + B*Aᴴ — recompute carefully.
    //   Bᴴ = [[0,0],[1,0]], Aᴴ = [[1,0],[0,0]]
    //   A * Bᴴ = [[0,0],[0,0]]   (because A only has (0,0)=1)
    //   B * Aᴴ = [[1,0],[0,0]]   (B has (0,1)=1; B*Aᴴ pulls Aᴴ row 1)
    // Wait: B is [[0,1],[0,0]], so B[0]=[0,1], B[1]=[0,0].
    //   B * Aᴴ row 0 = B[0,:] · Aᴴ[:,0..2] = [0*1+1*0, 0*0+1*0] = [0,0]
    //   B * Aᴴ row 1 = [0,0]
    // So both terms are zero; result is purely β * C₀.
    // Upper triangle: c[0]=2*5=10, c[1]=2*7=14, c[3]=2*11=22.
    assert_abs_diff_eq!(c_buf[0].re, 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[1].re, 14.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c_buf[3].re, 22.0, epsilon = 1e-12);
    // Lower untouched.
    assert_abs_diff_eq!(c_buf[2].re, 0.0, epsilon = 1e-12);
}

#[test]
fn gemm3m_matches_gemm() {
    use onemkl::blas::level3::gemm;

    // Random-ish but deterministic complex matrices.
    let a_buf: Vec<Complex64> = (0..6)
        .map(|i| Complex64::new(i as f64, (i + 1) as f64 * 0.5))
        .collect();
    let b_buf: Vec<Complex64> = (0..6)
        .map(|i| Complex64::new(-(i as f64) * 0.25, i as f64 * 0.75))
        .collect();
    let mut c_gemm = vec![Complex64::new(0.0, 0.0); 4];
    let mut c_gemm3m = vec![Complex64::new(0.0, 0.0); 4];

    let a = MatrixRef::new(&a_buf, 2, 3, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_buf, 3, 2, Layout::RowMajor).unwrap();

    {
        let mut c = MatrixMut::new(&mut c_gemm, 2, 2, Layout::RowMajor).unwrap();
        gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            Complex64::new(1.0, 0.0),
            &a,
            &b,
            Complex64::new(0.0, 0.0),
            &mut c,
        )
        .unwrap();
    }
    {
        let mut c = MatrixMut::new(&mut c_gemm3m, 2, 2, Layout::RowMajor).unwrap();
        gemm3m(
            Transpose::NoTrans,
            Transpose::NoTrans,
            Complex64::new(1.0, 0.0),
            &a,
            &b,
            Complex64::new(0.0, 0.0),
            &mut c,
        )
        .unwrap();
    }

    // gemm3m has slightly different rounding but should be close.
    for (g, g3) in c_gemm.iter().zip(&c_gemm3m) {
        assert_abs_diff_eq!(g.re, g3.re, epsilon = 1e-10);
        assert_abs_diff_eq!(g.im, g3.im, epsilon = 1e-10);
    }
}
