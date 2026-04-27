#![cfg(feature = "blas")]

//! Verify the pack / compute GEMM API.

use approx::assert_abs_diff_eq;

use onemkl::blas::packed::{
    gemm_compute_packed_a, gemm_compute_packed_b, PackIdentifier, PackedMatrix,
};
use onemkl::{Layout, Transpose};

#[test]
fn pack_a_then_compute_2x2() {
    // C = (1.0 * A) * B + 0.0 * C
    //   A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] → [[19, 22], [43, 50]]
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let b = [5.0_f64, 6.0, 7.0, 8.0];
    let mut c = [0.0_f64; 4];

    let packed_a = PackedMatrix::<f64>::pack_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2,
    )
    .unwrap();
    assert_eq!(packed_a.identifier(), PackIdentifier::A);

    gemm_compute_packed_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2,
        &packed_a, &b, 2, 0.0, &mut c, 2,
    )
    .unwrap();

    assert_abs_diff_eq!(c[0], 19.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], 43.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[3], 50.0, epsilon = 1e-12);
}

#[test]
fn pack_b_then_compute_2x2() {
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let b = [5.0_f64, 6.0, 7.0, 8.0];
    let mut c = [0.0_f64; 4];

    let packed_b = PackedMatrix::<f64>::pack_b(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, 1.0, &b, 2,
    )
    .unwrap();
    assert_eq!(packed_b.identifier(), PackIdentifier::B);

    gemm_compute_packed_b(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2,
        &a, 2, &packed_b, 0.0, &mut c, 2,
    )
    .unwrap();

    assert_abs_diff_eq!(c[0], 19.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], 43.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[3], 50.0, epsilon = 1e-12);
}

#[test]
fn pack_a_reused_across_many_b_matrices() {
    // Pack A once, multiply by 10 different B matrices.
    let a = [1.0_f64, 0.0, 0.0, 1.0]; // identity
    let packed_a = PackedMatrix::<f64>::pack_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2,
    )
    .unwrap();

    for trial in 0..10_i32 {
        let b: [f64; 4] = [
            trial as f64,
            (trial + 1) as f64,
            (trial + 2) as f64,
            (trial + 3) as f64,
        ];
        let mut c = [0.0_f64; 4];
        gemm_compute_packed_a(
            Layout::RowMajor, Transpose::NoTrans,
            2, 2, 2,
            &packed_a, &b, 2, 0.0, &mut c, 2,
        )
        .unwrap();
        // I * B = B
        for (i, &v) in c.iter().enumerate() {
            assert_abs_diff_eq!(v, b[i], epsilon = 1e-12);
        }
    }
}

#[test]
fn pack_with_alpha_baked_in() {
    // Pack A with alpha = 2.0, then compute against identity B.
    // Result should be 2.0 * A.
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    let mut c = [0.0_f64; 4];

    let packed_a = PackedMatrix::<f64>::pack_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, 2.0, &a, 2,
    )
    .unwrap();

    gemm_compute_packed_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2,
        &packed_a, &identity, 2, 0.0, &mut c, 2,
    )
    .unwrap();

    assert_abs_diff_eq!(c[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[3], 8.0, epsilon = 1e-12);
}

#[test]
fn shape_mismatch_rejected() {
    let a = [1.0_f64; 4];
    let b = [1.0_f64; 4];
    let mut c = [0.0_f64; 4];
    let packed_a = PackedMatrix::<f64>::pack_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2,
    )
    .unwrap();
    // Try to compute with a different shape.
    let r = gemm_compute_packed_a(
        Layout::RowMajor, Transpose::NoTrans,
        4, 4, 4, &packed_a, &b, 4, 0.0, &mut c, 4,
    );
    assert!(r.is_err());
}

#[test]
fn pack_b_mistakenly_used_as_a_rejected() {
    let a = [1.0_f64; 4];
    let b = [1.0_f64; 4];
    let mut c = [0.0_f64; 4];
    let packed_b = PackedMatrix::<f64>::pack_b(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, 1.0, &b, 2,
    )
    .unwrap();
    // Try to use a B-packed matrix as A.
    let r = gemm_compute_packed_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, &packed_b, &a, 2, 0.0, &mut c, 2,
    );
    assert!(r.is_err());
}

#[test]
fn pack_works_for_f32() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let mut c = [0.0_f32; 4];

    let packed_a = PackedMatrix::<f32>::pack_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2,
    )
    .unwrap();
    gemm_compute_packed_a(
        Layout::RowMajor, Transpose::NoTrans,
        2, 2, 2,
        &packed_a, &b, 2, 0.0, &mut c, 2,
    )
    .unwrap();
    assert_abs_diff_eq!(c[0], 19.0, epsilon = 1e-5);
    assert_abs_diff_eq!(c[3], 50.0, epsilon = 1e-5);
}
