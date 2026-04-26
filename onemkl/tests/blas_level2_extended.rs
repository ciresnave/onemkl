#![cfg(feature = "blas")]

//! Verify the expanded BLAS Level 2 surface against hand-computed values.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::blas::level2::{
    ger, gerc, geru, hemv, her, her2, hpmv, hpr, hpr2, sbmv, spmv, spr, spr2, symv,
    syr, syr2, tpmv, tpsv, trmv, trsv,
};
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::{Diag, Layout, Transpose, UpLo};

#[test]
fn ger_outer_product() {
    // A starts as zeros; A ← 1 * x * yᵀ + 0
    // x = [1, 2, 3], y = [4, 5]
    // Result: 3x2 matrix with rows = [4 5], [8 10], [12 15]
    let mut a_buf = vec![0.0_f64; 6];
    {
        let mut a = MatrixMut::new(&mut a_buf, 3, 2, Layout::RowMajor).unwrap();
        ger(1.0, &[1.0_f64, 2.0, 3.0], 1, &[4.0_f64, 5.0], 1, &mut a).unwrap();
    }
    assert_eq!(a_buf, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
}

#[test]
fn gerc_geru_distinguish_conjugation() {
    // x = [1+i], y = [3+2i]
    // gerc: A ← x * yᴴ = (1+i)(3-2i) = 3 - 2i + 3i + 2 = 5 + i
    // geru: A ← x * yᵀ = (1+i)(3+2i) = 3 + 2i + 3i - 2 = 1 + 5i
    let x = vec![Complex64::new(1.0, 1.0)];
    let y = vec![Complex64::new(3.0, 2.0)];

    let mut a_buf = vec![Complex64::new(0.0, 0.0); 1];
    {
        let mut a = MatrixMut::new(&mut a_buf, 1, 1, Layout::RowMajor).unwrap();
        gerc(Complex64::new(1.0, 0.0), &x, 1, &y, 1, &mut a).unwrap();
    }
    assert_abs_diff_eq!(a_buf[0].re, 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[0].im, 1.0, epsilon = 1e-12);

    let mut a_buf = vec![Complex64::new(0.0, 0.0); 1];
    {
        let mut a = MatrixMut::new(&mut a_buf, 1, 1, Layout::RowMajor).unwrap();
        geru(Complex64::new(1.0, 0.0), &x, 1, &y, 1, &mut a).unwrap();
    }
    assert_abs_diff_eq!(a_buf[0].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[0].im, 5.0, epsilon = 1e-12);
}

#[test]
fn trmv_upper_no_trans() {
    // A = [[1, 2, 3], [0, 4, 5], [0, 0, 6]] (upper triangular)
    // x = [1, 1, 1]
    // A * x = [6, 9, 6]
    let a_buf = [1.0_f64, 2.0, 3.0, 99.0, 4.0, 5.0, 99.0, 99.0, 6.0];
    let a = MatrixRef::new(&a_buf, 3, 3, Layout::RowMajor).unwrap();
    let mut x = vec![1.0_f64, 1.0, 1.0];
    trmv(UpLo::Upper, Transpose::NoTrans, Diag::NonUnit, &a, &mut x, 1).unwrap();
    assert_abs_diff_eq!(x[0], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[2], 6.0, epsilon = 1e-12);
}

#[test]
fn trsv_upper_no_trans_inverse() {
    // Same A; solving A * x = b with b = [6, 9, 6] should give x = [1, 1, 1].
    let a_buf = [1.0_f64, 2.0, 3.0, 99.0, 4.0, 5.0, 99.0, 99.0, 6.0];
    let a = MatrixRef::new(&a_buf, 3, 3, Layout::RowMajor).unwrap();
    let mut b = vec![6.0_f64, 9.0, 6.0];
    trsv(UpLo::Upper, Transpose::NoTrans, Diag::NonUnit, &a, &mut b, 1).unwrap();
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[2], 1.0, epsilon = 1e-12);
}

#[test]
fn trmv_lower_unit_diag() {
    // A is lower triangular with implicit unit diagonal:
    //   [[1, _, _], [2, 1, _], [3, 4, 1]]   (the _ entries are not read)
    // x = [1, 2, 3]
    // A * x = [1, 2*1 + 2 = 4, 3*1 + 4*2 + 3 = 14]
    let a_buf = [
        99.0_f64, 99.0, 99.0,  // diagonal "1"s implicit when Diag::Unit
        2.0, 99.0, 99.0,
        3.0, 4.0, 99.0,
    ];
    let a = MatrixRef::new(&a_buf, 3, 3, Layout::RowMajor).unwrap();
    let mut x = vec![1.0_f64, 2.0, 3.0];
    trmv(UpLo::Lower, Transpose::NoTrans, Diag::Unit, &a, &mut x, 1).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[2], 14.0, epsilon = 1e-12);
}

#[test]
fn symv_real_upper() {
    // Symmetric matrix (only upper read):
    //   [[1, 2, 3],
    //    [_, 4, 5],
    //    [_, _, 6]]
    // x = [1, 1, 1]
    // A * x = [1+2+3, 2+4+5, 3+5+6] = [6, 11, 14]
    let a_buf = [1.0_f64, 2.0, 3.0, 99.0, 4.0, 5.0, 99.0, 99.0, 6.0];
    let a = MatrixRef::new(&a_buf, 3, 3, Layout::RowMajor).unwrap();
    let x = vec![1.0_f64, 1.0, 1.0];
    let mut y = vec![0.0_f64; 3];
    symv(UpLo::Upper, 1.0, &a, &x, 1, 0.0, &mut y, 1).unwrap();
    assert_abs_diff_eq!(y[0], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 14.0, epsilon = 1e-12);
}

#[test]
fn syr_rank1_update() {
    // A is 2x2 symmetric, only upper used. Start with A = 0.
    // x = [1, 2]; α = 2 → A_upper ← 2 * x * xᵀ = 2 * [[1,2],[2,4]] = [[2,4],[_,8]]
    let mut a_buf = vec![0.0_f64; 4];
    {
        let mut a = MatrixMut::new(&mut a_buf, 2, 2, Layout::RowMajor).unwrap();
        syr(UpLo::Upper, 2.0, &[1.0_f64, 2.0], 1, &mut a).unwrap();
    }
    assert_abs_diff_eq!(a_buf[0], 2.0, epsilon = 1e-12); // (0,0)
    assert_abs_diff_eq!(a_buf[1], 4.0, epsilon = 1e-12); // (0,1)
    // (1,0) is in the lower triangle → not touched.
    assert_abs_diff_eq!(a_buf[3], 8.0, epsilon = 1e-12); // (1,1)
}

#[test]
fn syr2_rank2_update() {
    // 2x2 symmetric upper; A starts at zero.
    // x = [1, 0], y = [0, 1]; α = 1
    // A ← α * (x yᵀ + y xᵀ) = [[0, 1], [_, 0]] (upper only)
    let mut a_buf = vec![0.0_f64; 4];
    {
        let mut a = MatrixMut::new(&mut a_buf, 2, 2, Layout::RowMajor).unwrap();
        syr2(UpLo::Upper, 1.0, &[1.0_f64, 0.0], 1, &[0.0_f64, 1.0], 1, &mut a).unwrap();
    }
    assert_abs_diff_eq!(a_buf[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[3], 0.0, epsilon = 1e-12);
}

#[test]
fn hemv_complex_upper() {
    // Hermitian matrix (upper stored):
    //   [[2, 1+i],
    //    [_, 3]]
    // x = [1, 1]
    // A * x = [2 + 1+i, conj(1+i) + 3] = [3+i, 4-i]
    let a_buf = [
        Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0),
        Complex64::new(99.0, 99.0), Complex64::new(3.0, 0.0),
    ];
    let a = MatrixRef::new(&a_buf, 2, 2, Layout::RowMajor).unwrap();
    let x = vec![Complex64::new(1.0, 0.0); 2];
    let mut y = vec![Complex64::new(0.0, 0.0); 2];
    hemv(
        UpLo::Upper,
        Complex64::new(1.0, 0.0),
        &a,
        &x,
        1,
        Complex64::new(0.0, 0.0),
        &mut y,
        1,
    )
    .unwrap();
    assert_abs_diff_eq!(y[0].re, 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[0].im, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1].re, 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1].im, -1.0, epsilon = 1e-12);
}

#[test]
fn her_rank1_with_real_alpha() {
    // A starts zero; x = [1+i, 2]; α = 1
    // A ← α * x * xᴴ = [[(1+i)(1-i), (1+i)*2], [conj((1+i)*2), 4]]
    //                = [[2, 2+2i], [_, 4]]   (only upper stored)
    let mut a_buf = vec![Complex64::new(0.0, 0.0); 4];
    {
        let mut a = MatrixMut::new(&mut a_buf, 2, 2, Layout::RowMajor).unwrap();
        her(
            UpLo::Upper,
            1.0,
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            1,
            &mut a,
        )
        .unwrap();
    }
    assert_abs_diff_eq!(a_buf[0].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[0].im, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[1].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[1].im, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[3].re, 4.0, epsilon = 1e-12);
}

#[test]
fn her2_rank2_with_complex_alpha() {
    // A starts zero; x = [1, 0], y = [0, 1]; α = 1+0i
    // A ← α * x yᴴ + conj(α) * y xᴴ
    //   = (1+0i) * [[0, 1], [0, 0]] + (1+0i) * [[0, 0], [1, 0]]   (yᴴ for y=[0,1] is [0,1])
    //   = [[0, 1], [_, 0]]   (upper triangle only)
    let mut a_buf = vec![Complex64::new(0.0, 0.0); 4];
    {
        let mut a = MatrixMut::new(&mut a_buf, 2, 2, Layout::RowMajor).unwrap();
        her2(
            UpLo::Upper,
            Complex64::new(1.0, 0.0),
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            1,
            &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            1,
            &mut a,
        )
        .unwrap();
    }
    assert_abs_diff_eq!(a_buf[0].re, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[1].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[1].im, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(a_buf[3].re, 0.0, epsilon = 1e-12);
}

#[test]
fn sbmv_with_one_offdiagonal() {
    // 3x3 symmetric tridiagonal:
    //   [[1, 2, 0],
    //    [2, 3, 4],
    //    [0, 4, 5]]
    // Stored in band (upper, k=1), row-major: each row is [super, diagonal].
    // Layout for sbmv with row-major and Upper: A[i, j] for j-i = 0..k
    // CBLAS row-major sbmv with Upper: A_band[i, k - j + i] = original[i,j]?
    // (Actually, the storage convention is non-trivial. We use the "expected"
    // value via column-major to keep it simple here.)
    // Use column-major storage, which has the simpler convention:
    //   For Upper, k=1: row 0 = [_, 2, 4]; row 1 = [1, 3, 5]
    // (column-major lda = k+1 = 2; lda*n = 6 elements)
    let band = [
        99.0_f64, 1.0,  // col 0: garbage diag-1, then diag
        2.0, 3.0,       // col 1: super, diag
        4.0, 5.0,       // col 2: super, diag
    ];
    let a = MatrixRef::with_lda(&band, 2, 3, Layout::ColMajor, 2).unwrap();
    let x = vec![1.0_f64, 1.0, 1.0];
    let mut y = vec![0.0_f64; 3];
    sbmv(UpLo::Upper, 3, 1, 1.0, &a, &x, 1, 0.0, &mut y, 1).unwrap();
    // Row sums of full matrix: 1+2 = 3, 2+3+4 = 9, 4+5 = 9
    assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 9.0, epsilon = 1e-12);
}

#[test]
fn spmv_packed_symmetric() {
    // Symmetric matrix A:
    //   [[1, 2, 3],
    //    [2, 4, 5],
    //    [3, 5, 6]]
    // CBLAS column-major upper packed: ap = [a00, a01, a11, a02, a12, a22]
    //                                       = [1, 2, 4, 3, 5, 6]
    let ap = vec![1.0_f64, 2.0, 4.0, 3.0, 5.0, 6.0];
    let x = vec![1.0_f64, 1.0, 1.0];
    let mut y = vec![0.0_f64; 3];
    spmv(Layout::ColMajor, UpLo::Upper, 3, 1.0, &ap, &x, 1, 0.0, &mut y, 1).unwrap();
    assert_abs_diff_eq!(y[0], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], 11.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], 14.0, epsilon = 1e-12);
}

#[test]
fn spr_packed_rank1() {
    // ap initially zeros (length 3 for n=2), x = [1, 2], α = 1
    // After: A_upper = [[1, 2], [_, 4]]
    // Column-major upper packed for n=2: [a00, a01, a11]
    let mut ap = vec![0.0_f64; 3];
    spr(Layout::ColMajor, UpLo::Upper, 2, 1.0, &[1.0_f64, 2.0], 1, &mut ap).unwrap();
    assert_abs_diff_eq!(ap[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[2], 4.0, epsilon = 1e-12);
}

#[test]
fn spr2_packed_rank2() {
    // n=2, x = [1, 0], y = [0, 1], α = 1
    // A ← α * (x yᵀ + y xᵀ) = [[0, 1], [1, 0]]
    // Column-major upper packed: [a00, a01, a11] = [0, 1, 0]
    let mut ap = vec![0.0_f64; 3];
    spr2(
        Layout::ColMajor,
        UpLo::Upper,
        2,
        1.0,
        &[1.0_f64, 0.0],
        1,
        &[0.0_f64, 1.0],
        1,
        &mut ap,
    )
    .unwrap();
    assert_abs_diff_eq!(ap[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[2], 0.0, epsilon = 1e-12);
}

#[test]
fn tpmv_packed_triangular() {
    // Upper triangular A = [[1, 2, 3], [_, 4, 5], [_, _, 6]]
    // Column-major upper packed: [1, 2, 4, 3, 5, 6]
    let ap = vec![1.0_f64, 2.0, 4.0, 3.0, 5.0, 6.0];
    let mut x = vec![1.0_f64, 1.0, 1.0];
    tpmv(
        Layout::ColMajor,
        UpLo::Upper,
        Transpose::NoTrans,
        Diag::NonUnit,
        3,
        &ap,
        &mut x,
        1,
    )
    .unwrap();
    assert_abs_diff_eq!(x[0], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[2], 6.0, epsilon = 1e-12);
}

#[test]
fn tpsv_inverse_of_tpmv() {
    // Same matrix as tpmv test; solve from b = [6, 9, 6] back to x = [1, 1, 1].
    let ap = vec![1.0_f64, 2.0, 4.0, 3.0, 5.0, 6.0];
    let mut b = vec![6.0_f64, 9.0, 6.0];
    tpsv(
        Layout::ColMajor,
        UpLo::Upper,
        Transpose::NoTrans,
        Diag::NonUnit,
        3,
        &ap,
        &mut b,
        1,
    )
    .unwrap();
    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b[2], 1.0, epsilon = 1e-12);
}

#[test]
fn hpmv_packed_hermitian() {
    // Hermitian matrix:
    //   [[2, 1+i],
    //    [1-i, 3]]
    // Column-major upper packed for n=2: [a00, a01, a11] = [2, 1+i, 3]
    let ap = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
    ];
    let x = vec![Complex64::new(1.0, 0.0); 2];
    let mut y = vec![Complex64::new(0.0, 0.0); 2];
    hpmv(
        Layout::ColMajor,
        UpLo::Upper,
        2,
        Complex64::new(1.0, 0.0),
        &ap,
        &x,
        1,
        Complex64::new(0.0, 0.0),
        &mut y,
        1,
    )
    .unwrap();
    // Same expected values as the dense `hemv` test.
    assert_abs_diff_eq!(y[0].re, 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[0].im, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1].re, 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1].im, -1.0, epsilon = 1e-12);
}

#[test]
fn hpr_real_alpha() {
    // ap zero; x = [1+i, 2]; α = 1 (real)
    // After: A_upper packed = [|1+i|^2, (1+i)*conj(2), |2|^2] = [2, 2+2i, 4]
    // Column-major upper packed for n=2: [a00, a01, a11]
    let mut ap = vec![Complex64::new(0.0, 0.0); 3];
    hpr(
        Layout::ColMajor,
        UpLo::Upper,
        2,
        1.0,
        &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
        1,
        &mut ap,
    )
    .unwrap();
    assert_abs_diff_eq!(ap[0].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[0].im, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[1].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[1].im, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[2].re, 4.0, epsilon = 1e-12);
}

#[test]
fn hpr2_complex_alpha() {
    // ap zero; x = [1, 0], y = [0, 1]; α = 1+0i
    // A ← x yᴴ + y xᴴ; column-major upper packed → [0, 1, 0]
    let mut ap = vec![Complex64::new(0.0, 0.0); 3];
    hpr2(
        Layout::ColMajor,
        UpLo::Upper,
        2,
        Complex64::new(1.0, 0.0),
        &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        1,
        &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        1,
        &mut ap,
    )
    .unwrap();
    assert_abs_diff_eq!(ap[0].re, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[1].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ap[2].re, 0.0, epsilon = 1e-12);
}
