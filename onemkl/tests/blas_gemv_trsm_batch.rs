#![cfg(feature = "blas")]

//! Verify pointer-array batched GEMV and TRSM (cblas_*gemv_batch /
//! cblas_*trsm_batch).

use approx::assert_abs_diff_eq;

use onemkl::blas::extensions::{
    gemv_batch, trsm_batch, GemvBatchGroup, TrsmBatchGroup,
};
use onemkl::{Diag, Layout, Side, Transpose, UpLo};

#[test]
fn gemv_batch_two_uniform_3x3() {
    // Two 3×3 matrices, each times a length-3 vector.
    let a0 = [
        1.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ];
    let x0 = [1.0_f64, 1.0, 1.0];
    let mut y0 = [0.0_f64; 3];

    let a1 = [
        2.0_f64, 0.0, 0.0,
        0.0, 4.0, 0.0,
        0.0, 0.0, 6.0,
    ];
    let x1 = [1.0_f64, 1.0, 1.0];
    let mut y1 = [0.0_f64; 3];

    let a_ptrs: [*const f64; 2] = [a0.as_ptr(), a1.as_ptr()];
    let x_ptrs: [*const f64; 2] = [x0.as_ptr(), x1.as_ptr()];
    let y_ptrs: [*mut f64; 2] = [y0.as_mut_ptr(), y1.as_mut_ptr()];

    let mut groups = [GemvBatchGroup::<f64> {
        trans: Transpose::NoTrans,
        m: 3,
        n: 3,
        alpha: 1.0,
        lda: 3,
        incx: 1,
        beta: 0.0,
        incy: 1,
        group_size: 2,
        a_array: &a_ptrs,
        x_array: &x_ptrs,
        y_array: &y_ptrs,
    }];

    gemv_batch(Layout::RowMajor, &mut groups).unwrap();

    // y0 = diag(1,2,3) * [1,1,1] = [1,2,3]
    assert_abs_diff_eq!(y0[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y0[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y0[2], 3.0, epsilon = 1e-12);
    // y1 = diag(2,4,6) * [1,1,1] = [2,4,6]
    assert_abs_diff_eq!(y1[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y1[1], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y1[2], 6.0, epsilon = 1e-12);
}

#[test]
fn trsm_batch_two_2x2_lower() {
    // Solve L * X = B for two 2×2 systems with lower-triangular L.
    // L = [[2, 0], [1, 3]], B0 = [[2, 4], [4, 11]]; solution X0 = [[1, 2], [1, 3]].
    // L = [[1, 0], [2, 1]], B1 = [[1, 1], [3, 4]]; solution X1 = [[1, 1], [1, 2]].
    let l0 = [2.0_f64, 0.0, 1.0, 3.0];
    let mut b0 = [2.0_f64, 4.0, 4.0, 11.0];

    let l1 = [1.0_f64, 0.0, 2.0, 1.0];
    let mut b1 = [1.0_f64, 1.0, 3.0, 4.0];

    let a_ptrs: [*const f64; 2] = [l0.as_ptr(), l1.as_ptr()];
    let b_ptrs: [*mut f64; 2] = [b0.as_mut_ptr(), b1.as_mut_ptr()];

    let mut groups = [TrsmBatchGroup::<f64> {
        side: Side::Left,
        uplo: UpLo::Lower,
        transa: Transpose::NoTrans,
        diag: Diag::NonUnit,
        m: 2,
        n: 2,
        alpha: 1.0,
        lda: 2,
        ldb: 2,
        group_size: 2,
        a_array: &a_ptrs,
        b_array: &b_ptrs,
    }];

    trsm_batch(Layout::RowMajor, &mut groups).unwrap();

    // X0:
    assert_abs_diff_eq!(b0[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b0[1], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b0[2], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b0[3], 3.0, epsilon = 1e-12);
    // X1:
    assert_abs_diff_eq!(b1[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b1[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b1[2], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(b1[3], 2.0, epsilon = 1e-12);
}

#[test]
fn empty_groups_noop() {
    let mut gv: [GemvBatchGroup<f64>; 0] = [];
    let mut ts: [TrsmBatchGroup<f64>; 0] = [];
    gemv_batch(Layout::RowMajor, &mut gv).unwrap();
    trsm_batch(Layout::RowMajor, &mut ts).unwrap();
}

#[test]
fn gemv_batch_length_mismatch() {
    let a = [1.0_f64; 4];
    let x = [1.0_f64; 2];
    let mut y = [0.0_f64; 2];
    let a_ptrs: [*const f64; 1] = [a.as_ptr()];
    let x_ptrs: [*const f64; 1] = [x.as_ptr()];
    let y_ptrs: [*mut f64; 1] = [y.as_mut_ptr()];
    let mut groups = [GemvBatchGroup::<f64> {
        trans: Transpose::NoTrans,
        m: 2, n: 2,
        alpha: 1.0, lda: 2, incx: 1, beta: 0.0, incy: 1,
        group_size: 2, // mismatch
        a_array: &a_ptrs, x_array: &x_ptrs, y_array: &y_ptrs,
    }];
    assert!(gemv_batch(Layout::RowMajor, &mut groups).is_err());
}
