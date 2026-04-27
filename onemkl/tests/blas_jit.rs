#![cfg(feature = "blas")]

//! Verify the JIT GEMM API.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::blas::jit::JitGemm;
use onemkl::{Layout, Transpose};

#[test]
fn jit_dgemm_2x2() {
    // C = A * B where A = [[1,2],[3,4]], B = [[5,6],[7,8]] gives
    // [[19, 22], [43, 50]].
    let plan = JitGemm::<f64>::new(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0, 2, 2, 0.0, 2,
    )
    .unwrap();
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let b = [5.0_f64, 6.0, 7.0, 8.0];
    let mut c = [0.0_f64; 4];
    plan.execute(&a, &b, &mut c).unwrap();
    assert_abs_diff_eq!(c[0], 19.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 22.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], 43.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[3], 50.0, epsilon = 1e-12);
}

#[test]
fn jit_dgemm_runs_many_times() {
    // Build the kernel once and invoke it many times with different
    // data — the canonical use case for JIT.
    let plan = JitGemm::<f64>::new(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        4, 4, 4, 1.0, 4, 4, 0.0, 4,
    )
    .unwrap();
    let identity = [
        1.0_f64, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for trial in 0..10 {
        let a: Vec<f64> = (0..16).map(|i| (i + trial) as f64).collect();
        let mut c = vec![0.0_f64; 16];
        plan.execute(&a, &identity, &mut c).unwrap();
        // A * I = A
        for (i, &v) in c.iter().enumerate() {
            assert_abs_diff_eq!(v, a[i], epsilon = 1e-12);
        }
    }
}

#[test]
fn jit_sgemm_with_alpha_beta() {
    // C ← 2 * A * B + 3 * C, baked in at JIT time.
    let plan = JitGemm::<f32>::new(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 2.0, 2, 2, 3.0, 2,
    )
    .unwrap();
    let a = [1.0_f32, 0.0, 0.0, 1.0]; // identity
    let b = [4.0_f32, 5.0, 6.0, 7.0];
    let mut c = [10.0_f32, 20.0, 30.0, 40.0];
    plan.execute(&a, &b, &mut c).unwrap();
    // A is identity so A*B = B; result = 2*B + 3*C_original.
    assert_abs_diff_eq!(c[0], 2.0 * 4.0 + 3.0 * 10.0, epsilon = 1e-5);
    assert_abs_diff_eq!(c[1], 2.0 * 5.0 + 3.0 * 20.0, epsilon = 1e-5);
    assert_abs_diff_eq!(c[2], 2.0 * 6.0 + 3.0 * 30.0, epsilon = 1e-5);
    assert_abs_diff_eq!(c[3], 2.0 * 7.0 + 3.0 * 40.0, epsilon = 1e-5);
}

#[test]
fn jit_zgemm_complex() {
    // (1+i) * (1-i) = 2 in 1×1.
    let plan = JitGemm::<Complex64>::new(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        1, 1, 1,
        Complex64::new(1.0, 0.0), 1, 1,
        Complex64::new(0.0, 0.0), 1,
    )
    .unwrap();
    let a = [Complex64::new(1.0, 1.0)];
    let b = [Complex64::new(1.0, -1.0)];
    let mut c = [Complex64::new(0.0, 0.0)];
    plan.execute(&a, &b, &mut c).unwrap();
    assert_abs_diff_eq!(c[0].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[0].im, 0.0, epsilon = 1e-12);
}

#[test]
fn jit_rejects_undersized_buffers() {
    let plan = JitGemm::<f64>::new(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        2, 2, 2, 1.0, 2, 2, 0.0, 2,
    )
    .unwrap();
    let a = [1.0_f64, 2.0]; // too small
    let b = [1.0_f64; 4];
    let mut c = [0.0_f64; 4];
    assert!(plan.execute(&a, &b, &mut c).is_err());
}
