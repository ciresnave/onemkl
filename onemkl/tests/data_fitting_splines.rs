#![cfg(feature = "data-fitting")]

//! Verify Bessel, Akima, and Hermite cubic spline subtypes.

use approx::assert_abs_diff_eq;

use onemkl::data_fitting::CubicSpline1d;

#[test]
fn bessel_passes_through_knots() {
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    let spline = CubicSpline1d::bessel(x.clone(), y.clone()).unwrap();
    let evaluated = spline.interpolate(&x).unwrap();
    for (yi, ei) in y.iter().zip(&evaluated) {
        assert_abs_diff_eq!(*yi, *ei, epsilon = 1e-10);
    }
}

#[test]
fn akima_passes_through_knots() {
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    // Akima needs at least 5 knots.
    let spline = CubicSpline1d::akima(x.clone(), y.clone()).unwrap();
    let evaluated = spline.interpolate(&x).unwrap();
    for (yi, ei) in y.iter().zip(&evaluated) {
        assert_abs_diff_eq!(*yi, *ei, epsilon = 1e-10);
    }
}

#[test]
fn hermite_passes_through_knots() {
    // y = x²; dy/dx = 2x.
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    let dy: Vec<f64> = x.iter().map(|t| 2.0 * t).collect();
    let spline = CubicSpline1d::hermite(x.clone(), y.clone(), dy).unwrap();
    let evaluated = spline.interpolate(&x).unwrap();
    for (yi, ei) in y.iter().zip(&evaluated) {
        assert_abs_diff_eq!(*yi, *ei, epsilon = 1e-10);
    }
}

#[test]
fn hermite_rejects_mismatched_derivatives_length() {
    let x = vec![0.0_f64, 1.0, 2.0];
    let y = vec![0.0_f64, 1.0, 4.0];
    let dy = vec![0.0_f64, 2.0]; // wrong length
    assert!(CubicSpline1d::hermite(x, y, dy).is_err());
}

#[test]
fn bessel_smooth_interpolation() {
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t.sin()).collect();
    let spline = CubicSpline1d::bessel(x, y).unwrap();
    let probe = vec![0.5_f64, 1.5, 2.5, 3.5, 4.5];
    let evaluated = spline.interpolate(&probe).unwrap();
    for (s, t) in probe.iter().zip(&evaluated) {
        // Bessel cubic should be within a few % of true sin value.
        assert!((s.sin() - *t).abs() < 0.05, "site {} → {}", s, t);
    }
}
