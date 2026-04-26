//! Verify FFT wrappers via known DFT values and round-tripping.

use approx::assert_abs_diff_eq;
use num_complex::{Complex32, Complex64};

use onemkl::fft::{FftPlan, FftPlanOutOfPlace};

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
