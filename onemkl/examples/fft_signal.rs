//! Generate a sinusoid, run a real-input FFT, find the peak frequency
//! bin. Demonstrates `RealFftPlan` with backward-scaling normalization.
//!
//! Run with `cargo run --example fft_signal`.

use num_complex::Complex64;
use onemkl::fft::RealFftPlan;

fn main() {
    let n: usize = 64;
    let freq_hz = 5.0;       // 5 cycles in n samples
    let sample_rate = n as f64;

    // Generate the signal.
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * freq_hz * t).sin()
        })
        .collect();

    // Build a real FFT plan with 1/N backward scaling so IFFT(FFT(x)) = x.
    let mut plan = RealFftPlan::<f64>::real_nd_with_scales(
        &[n],
        1.0,
        1.0 / n as f64,
    )
    .unwrap();
    let mut spectrum: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); plan.complex_len()];
    plan.forward(&signal, &mut spectrum).unwrap();

    // Find the peak bin (skipping DC).
    let mut peak_bin = 1;
    let mut peak_mag = 0.0;
    for (i, c) in spectrum.iter().enumerate().skip(1) {
        let m = c.norm();
        if m > peak_mag {
            peak_mag = m;
            peak_bin = i;
        }
    }
    println!(
        "Input: {} samples of a {} Hz sinusoid at sample rate {} Hz.",
        n, freq_hz, sample_rate
    );
    println!(
        "Peak FFT bin: {} (corresponds to {} Hz)",
        peak_bin, peak_bin as f64
    );
    assert_eq!(peak_bin, freq_hz as usize);
}
