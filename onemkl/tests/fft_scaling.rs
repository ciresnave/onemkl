#![cfg(feature = "fft")]

//! Verify FFT forward / backward scaling.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::fft::{FftPlan, FftPlanOutOfPlace, RealFftPlan};

#[test]
fn unscaled_round_trip_is_n_times_input() {
    // Default behavior: IFFT(FFT(x)) = N * x.
    let n: usize = 8;
    let mut plan = FftPlan::<f64>::complex_1d(n).unwrap();
    let mut buf: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, 0.0))
        .collect();
    let original = buf.clone();
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after.re, before.re * n as f64, epsilon = 1e-10);
        assert_abs_diff_eq!(after.im, before.im * n as f64, epsilon = 1e-10);
    }
}

#[test]
fn backward_scaled_round_trip_returns_input() {
    let n: usize = 8;
    let mut plan =
        FftPlan::<f64>::complex_nd_with_scales(&[n], 1.0, 1.0 / n as f64).unwrap();
    let mut buf: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, 0.0))
        .collect();
    let original = buf.clone();
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after.re, before.re, epsilon = 1e-10);
        assert_abs_diff_eq!(after.im, before.im, epsilon = 1e-10);
    }
}

#[test]
fn unitary_scaling_preserves_norm() {
    // Forward = backward = 1/sqrt(N) — Parseval-style unitary FFT.
    let n: usize = 8;
    let scale = 1.0 / (n as f64).sqrt();
    let mut plan =
        FftPlan::<f64>::complex_nd_with_scales(&[n], scale, scale).unwrap();
    let mut buf: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, 0.0))
        .collect();
    let original = buf.clone();
    let norm_before: f64 = original.iter().map(|c| c.norm_sqr()).sum();
    plan.forward_in_place(&mut buf).unwrap();
    let norm_after: f64 = buf.iter().map(|c| c.norm_sqr()).sum();
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-10);
}

#[test]
fn out_of_place_scaling() {
    let n: usize = 4;
    let mut plan = FftPlanOutOfPlace::<f64>::complex_nd_with_scales(
        &[n],
        1.0,
        1.0 / n as f64,
    )
    .unwrap();
    let input: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let mut spectrum: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    plan.forward(&input, &mut spectrum).unwrap();
    let mut recovered: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    plan.backward(&spectrum, &mut recovered).unwrap();
    for (a, b) in recovered.iter().zip(&input) {
        assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-10);
        assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-10);
    }
}

#[test]
fn real_fft_with_backward_scale() {
    let n: usize = 8;
    let mut plan = RealFftPlan::<f64>::real_nd_with_scales(
        &[n],
        1.0,
        1.0 / n as f64,
    )
    .unwrap();
    let input: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut spectrum: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); plan.complex_len()];
    plan.forward(&input, &mut spectrum).unwrap();
    let mut recovered: Vec<f64> = vec![0.0; n];
    plan.backward(&spectrum, &mut recovered).unwrap();
    for (a, b) in recovered.iter().zip(&input) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
}

#[test]
fn f32_scaling() {
    // Confirm scales propagate correctly for single precision too.
    let n: usize = 4;
    let mut plan = FftPlan::<f32>::complex_nd_with_scales(
        &[n],
        1.0,
        1.0 / n as f32,
    )
    .unwrap();
    let mut buf: Vec<num_complex::Complex32> = (0..n)
        .map(|i| num_complex::Complex32::new(i as f32, 0.0))
        .collect();
    let original = buf.clone();
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after.re, before.re, epsilon = 1e-5);
    }
}
