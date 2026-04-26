#![cfg(feature = "fft")]

//! Verify FFT wrappers via known DFT values and round-tripping.

use approx::assert_abs_diff_eq;
use num_complex::{Complex32, Complex64};

use onemkl::fft::{cce_complex_len, FftPlan, FftPlanOutOfPlace, RealFftPlan};

#[test]
fn fft_of_delta_is_constant_double() {
    // FFT of [1, 0, 0, 0, 0, 0, 0, 0] is all 1s.
    let mut plan = FftPlan::<f64>::complex_1d(8).unwrap();
    let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); 8];
    buf[0] = Complex64::new(1.0, 0.0);
    plan.forward_in_place(&mut buf).unwrap();
    for x in &buf {
        assert_abs_diff_eq!(x.re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x.im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn fft_of_constant_is_delta_double() {
    // FFT of all 1s is [N, 0, 0, ..., 0].
    let mut plan = FftPlan::<f64>::complex_1d(8).unwrap();
    let mut buf: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); 8];
    plan.forward_in_place(&mut buf).unwrap();
    assert_abs_diff_eq!(buf[0].re, 8.0, epsilon = 1e-12);
    assert_abs_diff_eq!(buf[0].im, 0.0, epsilon = 1e-12);
    for x in &buf[1..] {
        assert_abs_diff_eq!(x.re, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x.im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn round_trip_is_identity_double() {
    // forward then backward gives N * x (un-normalized).
    let n = 16;
    let original: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, (i + 1) as f64 * 0.5))
        .collect();
    let mut plan = FftPlan::<f64>::complex_1d(n).unwrap();
    let mut buf = original.clone();
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for (a, b) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(a.re, b.re * n as f64, epsilon = 1e-9);
        assert_abs_diff_eq!(a.im, b.im * n as f64, epsilon = 1e-9);
    }
}

#[test]
fn round_trip_single_precision() {
    let n = 16;
    let original: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(i as f32, 0.0))
        .collect();
    let mut plan = FftPlan::<f32>::complex_1d(n).unwrap();
    let mut buf = original.clone();
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for (a, b) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(a.re, b.re * n as f32, epsilon = 1e-3);
        assert_abs_diff_eq!(a.im, b.im * n as f32, epsilon = 1e-3);
    }
}

#[test]
fn out_of_place_forward() {
    let n = 8;
    let mut input: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    input[0] = Complex64::new(1.0, 0.0);
    let mut output = vec![Complex64::new(0.0, 0.0); n];

    let mut plan = FftPlanOutOfPlace::<f64>::complex_1d(n).unwrap();
    plan.forward(&input, &mut output).unwrap();

    // Input is preserved.
    assert_abs_diff_eq!(input[0].re, 1.0, epsilon = 1e-12);
    // Output is all 1s.
    for x in &output {
        assert_abs_diff_eq!(x.re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x.im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn out_of_place_round_trip() {
    let n = 8;
    let original: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, 0.0))
        .collect();
    let mut transformed = vec![Complex64::new(0.0, 0.0); n];
    let mut restored = vec![Complex64::new(0.0, 0.0); n];

    let mut plan = FftPlanOutOfPlace::<f64>::complex_1d(n).unwrap();
    plan.forward(&original, &mut transformed).unwrap();
    plan.backward(&transformed, &mut restored).unwrap();

    for (r, o) in restored.iter().zip(&original) {
        assert_abs_diff_eq!(r.re, o.re * n as f64, epsilon = 1e-9);
        assert_abs_diff_eq!(r.im, o.im * n as f64, epsilon = 1e-9);
    }
}

#[test]
fn fft_2d_of_delta_is_constant() {
    // 2-D FFT of [[1,0,0,0],[0,0,0,0]] (a 2×4 array with one nonzero
    // at (0,0)) should give all 1s in the output.
    let mut plan = FftPlan::<f64>::complex_2d(2, 4).unwrap();
    let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); 8];
    buf[0] = Complex64::new(1.0, 0.0);
    plan.forward_in_place(&mut buf).unwrap();
    for x in &buf {
        assert_abs_diff_eq!(x.re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x.im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn fft_2d_round_trip() {
    let n0 = 4;
    let n1 = 6;
    let n = n0 * n1;
    let original: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i + 1) as f64 * 0.5, (i as f64) * 0.25))
        .collect();
    let mut plan = FftPlan::<f64>::complex_2d(n0, n1).unwrap();
    let mut buf = original.clone();
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for (a, b) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(a.re, b.re * n as f64, epsilon = 1e-9);
        assert_abs_diff_eq!(a.im, b.im * n as f64, epsilon = 1e-9);
    }
}

#[test]
fn fft_3d_round_trip() {
    let n = 4 * 4 * 4;
    let original: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, (i as f64) * 0.1))
        .collect();
    let mut plan = FftPlan::<f64>::complex_3d(4, 4, 4).unwrap();
    let mut buf = original.clone();
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for (a, b) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(a.re, b.re * n as f64, epsilon = 1e-8);
        assert_abs_diff_eq!(a.im, b.im * n as f64, epsilon = 1e-8);
    }
}

#[test]
fn cce_complex_len_formula() {
    assert_eq!(cce_complex_len(8), 5);
    assert_eq!(cce_complex_len(7), 4);
    assert_eq!(cce_complex_len(1), 1);
}

#[test]
fn real_fft_1d_of_delta_is_constant() {
    // Forward FFT of real [1, 0, ..., 0] gives complex all-1s
    // (output length = n/2 + 1 in CCE format).
    let n = 8;
    let mut plan = RealFftPlan::<f64>::real_1d(n).unwrap();
    let mut input = vec![0.0_f64; n];
    input[0] = 1.0;
    let mut output = vec![Complex64::new(0.0, 0.0); cce_complex_len(n)];
    plan.forward(&input, &mut output).unwrap();
    for c in &output {
        assert_abs_diff_eq!(c.re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c.im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn real_fft_1d_round_trip() {
    let n = 16;
    let original: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let mut plan = RealFftPlan::<f64>::real_1d(n).unwrap();
    let mut spectrum = vec![Complex64::new(0.0, 0.0); cce_complex_len(n)];
    plan.forward(&original, &mut spectrum).unwrap();
    let mut restored = vec![0.0_f64; n];
    plan.backward(&spectrum, &mut restored).unwrap();
    for (a, b) in restored.iter().zip(&original) {
        assert_abs_diff_eq!(*a, *b * n as f64, epsilon = 1e-9);
    }
}

#[test]
fn real_fft_2d_round_trip() {
    let n0 = 8;
    let n1 = 4;
    let total = n0 * n1;
    let original: Vec<f64> = (0..total).map(|i| (i as f64).cos()).collect();
    let mut plan = RealFftPlan::<f64>::real_2d(n0, n1).unwrap();
    let cce_len = plan.complex_len();
    assert_eq!(cce_len, n0 * cce_complex_len(n1));
    let mut spectrum = vec![Complex64::new(0.0, 0.0); cce_len];
    plan.forward(&original, &mut spectrum).unwrap();
    let mut restored = vec![0.0_f64; total];
    plan.backward(&spectrum, &mut restored).unwrap();
    for (a, b) in restored.iter().zip(&original) {
        assert_abs_diff_eq!(*a, *b * total as f64, epsilon = 1e-8);
    }
}

#[test]
fn fft_of_cosine_2_period() {
    // x[k] = cos(2π * k / 8). FFT bin 1 should have magnitude N/2 = 4.
    let n = 8;
    let mut buf: Vec<Complex64> = (0..n)
        .map(|k| {
            let theta = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            Complex64::new(theta.cos(), 0.0)
        })
        .collect();
    let mut plan = FftPlan::<f64>::complex_1d(n).unwrap();
    plan.forward_in_place(&mut buf).unwrap();
    // Bin 1 and bin N-1 should each have magnitude 4 for a real cosine.
    assert_abs_diff_eq!(buf[1].re, 4.0, epsilon = 1e-9);
    assert_abs_diff_eq!(buf[n - 1].re, 4.0, epsilon = 1e-9);
}
