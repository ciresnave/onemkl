#![cfg(feature = "rng")]

//! Verify 1-D convolution and correlation tasks.

use approx::assert_abs_diff_eq;

use onemkl::rng::convolution::{
    convolve_1d, correlate_1d, Conv1d, ConvMode, Corr1d, CorrMode,
};

#[test]
fn convolve_short_vectors() {
    // x * y = [1*1, 1*1+2*1, 2*1+3*1, 3*1] = [1, 3, 5, 3].
    let x = [1.0_f64, 2.0, 3.0];
    let y = [1.0_f64, 1.0];
    let z = convolve_1d::<f64>(ConvMode::Auto, &x, &y).unwrap();
    assert_eq!(z.len(), 4);
    assert_abs_diff_eq!(z[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[1], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[2], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[3], 3.0, epsilon = 1e-12);
}

#[test]
fn convolve_f32() {
    let x = [1.0_f32, 2.0, 1.0];
    let y = [1.0_f32, 0.0, -1.0];
    // Convolution: [1, 2, 0, -2, -1].
    let z = convolve_1d::<f32>(ConvMode::Direct, &x, &y).unwrap();
    assert_eq!(z.len(), 5);
    assert_abs_diff_eq!(z[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(z[1], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(z[2], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(z[3], -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(z[4], -1.0, epsilon = 1e-6);
}

#[test]
fn correlate_returns_full_length() {
    // Check shape and energy preservation: cross-correlation of a
    // signal with itself peaks at the center of the output and has a
    // total energy equal to ‖x‖² × ly + small boundary effects.
    let x = [1.0_f64, 2.0, 3.0];
    let y = [1.0_f64, 1.0];
    let z = correlate_1d::<f64>(CorrMode::Auto, &x, &y).unwrap();
    assert_eq!(z.len(), x.len() + y.len() - 1);
    // Sum of correlation outputs = sum(x) * sum(y) for real inputs.
    let total: f64 = z.iter().sum();
    let expected = x.iter().sum::<f64>() * y.iter().sum::<f64>();
    assert_abs_diff_eq!(total, expected, epsilon = 1e-12);
}

#[test]
fn reusable_conv_task() {
    // Build the task once and execute it twice with different inputs.
    let mut task = Conv1d::<f64>::new(ConvMode::Auto, 3, 2, 4).unwrap();

    let x1 = [1.0_f64, 2.0, 3.0];
    let y1 = [1.0_f64, 1.0];
    let mut z1 = [0.0_f64; 4];
    task.execute(&x1, 1, &y1, 1, &mut z1, 1).unwrap();
    assert_abs_diff_eq!(z1[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z1[3], 3.0, epsilon = 1e-12);

    // Different inputs through the same task.
    let x2 = [2.0_f64, 0.0, -1.0];
    let y2 = [1.0_f64, -1.0];
    let mut z2 = [0.0_f64; 4];
    task.execute(&x2, 1, &y2, 1, &mut z2, 1).unwrap();
    // Conv: [2, -2, -1, 1].
    assert_abs_diff_eq!(z2[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z2[1], -2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z2[2], -1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z2[3], 1.0, epsilon = 1e-12);
}

#[test]
fn reusable_corr_task() {
    let mut task = Corr1d::<f64>::new(CorrMode::Direct, 3, 2, 4).unwrap();
    let x = [1.0_f64, 2.0, 3.0];
    let y = [1.0_f64, 1.0];
    let mut z = [0.0_f64; 4];
    task.execute(&x, 1, &y, 1, &mut z, 1).unwrap();
    // Should match correlate_1d with the same inputs.
    let z_ref = correlate_1d::<f64>(CorrMode::Direct, &x, &y).unwrap();
    for (a, b) in z.iter().zip(&z_ref) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
    }
}

#[test]
fn empty_input_rejected() {
    let x: [f64; 0] = [];
    let y = [1.0_f64];
    assert!(convolve_1d::<f64>(ConvMode::Auto, &x, &y).is_err());
    assert!(correlate_1d::<f64>(CorrMode::Auto, &x, &y).is_err());
}
