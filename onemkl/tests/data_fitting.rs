//! Verify cubic spline interpolation and integration.

use approx::assert_abs_diff_eq;

use onemkl::data_fitting::CubicSpline1d;

#[test]
fn cubic_spline_passes_through_knots() {
    // Knot points should be exactly reproduced by interpolation.
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect(); // y = x²

    let spline = CubicSpline1d::natural(x.clone(), y.clone()).unwrap();
    let evaluated = spline.interpolate(&x).unwrap();
    for (yi, ei) in y.iter().zip(&evaluated) {
        assert_abs_diff_eq!(*yi, *ei, epsilon = 1e-10);
    }
}

#[test]
fn cubic_spline_interpolates_smoothly() {
    // For y = x², the spline should interpolate to within close
    // tolerance everywhere on the knot range.
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();

    let spline = CubicSpline1d::natural(x.clone(), y.clone()).unwrap();
    let probe = vec![0.5_f64, 1.5, 2.5, 3.5, 4.5];
    let evaluated = spline.interpolate(&probe).unwrap();
    // Cubic spline of x² won't be exact except at knots, but should be
    // within a few percent of the true value for this smooth function.
    for (s, t) in probe.iter().zip(&evaluated) {
        assert!((s * s - *t).abs() < 0.1, "site {} → {}", s, t);
    }
}

#[test]
fn integrate_constant_function() {
    // y = 5 over x in [0, 4]; integral over [0, 4] = 20.
    let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let y = vec![5.0_f64; 5];
    let spline = CubicSpline1d::natural(x, y).unwrap();
    let res = spline.integrate(&[0.0_f64], &[4.0_f64]).unwrap();
    assert_abs_diff_eq!(res[0], 20.0, epsilon = 1e-9);
}

#[test]
fn integrate_linear_function() {
    // y = 2x over [0, 4]; integral = x² evaluated at 4 = 16.
    let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|t| 2.0 * t).collect();
    let spline = CubicSpline1d::natural(x, y).unwrap();
    let res = spline.integrate(&[0.0_f64], &[4.0_f64]).unwrap();
    assert_abs_diff_eq!(res[0], 16.0, epsilon = 1e-9);
}

#[test]
fn invalid_lengths_rejected() {
    let r = CubicSpline1d::<f64>::natural(vec![1.0, 2.0], vec![1.0]);
    assert!(r.is_err());
}

#[test]
fn too_few_knots_rejected() {
    let r = CubicSpline1d::<f64>::natural(vec![1.0], vec![1.0]);
    assert!(r.is_err());
}
