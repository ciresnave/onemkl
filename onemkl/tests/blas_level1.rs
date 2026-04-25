//! Verify BLAS Level 1 wrappers against hand-computed values.

use approx::assert_abs_diff_eq;
use num_complex::{Complex32, Complex64};

use onemkl::blas::level1::{
    asum, asum_inc, axpy, copy, dot, dot_inc, dotc, dotu, iamax, iamin, nrm2,
    rot, rotg, scal, scal_real, swap,
};

#[test]
fn asum_real_and_complex() {
    let xf: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
    assert_abs_diff_eq!(asum(&xf), 10.0_f32);

    let xd: Vec<f64> = vec![1.0, -2.0, 3.0, -4.0];
    assert_abs_diff_eq!(asum(&xd), 10.0_f64);

    // Complex `asum` is Σ (|re| + |im|).
    let xc: Vec<Complex64> = vec![
        Complex64::new(1.0, -1.0),
        Complex64::new(-2.0, 3.0),
        Complex64::new(0.5, -0.5),
    ];
    assert_abs_diff_eq!(asum(&xc), 1.0 + 1.0 + 2.0 + 3.0 + 0.5 + 0.5, epsilon = 1e-12);
}

#[test]
fn nrm2_real() {
    let x: Vec<f64> = vec![3.0, 4.0];
    assert_abs_diff_eq!(nrm2(&x), 5.0_f64, epsilon = 1e-12);

    let xf: Vec<f32> = vec![3.0, 4.0];
    assert_abs_diff_eq!(nrm2(&xf), 5.0_f32, epsilon = 1e-6);
}

#[test]
fn nrm2_complex() {
    let x: Vec<Complex64> = vec![
        Complex64::new(3.0, 4.0),
        Complex64::new(0.0, 0.0),
    ];
    // |3+4i| = 5 → norm = 5.
    assert_abs_diff_eq!(nrm2(&x), 5.0_f64, epsilon = 1e-12);
}

#[test]
fn dot_real() {
    let x: Vec<f64> = vec![1.0, 2.0, 3.0];
    let y: Vec<f64> = vec![4.0, -5.0, 6.0];
    // 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12.
    assert_abs_diff_eq!(dot(&x, &y).unwrap(), 12.0_f64, epsilon = 1e-12);
}

#[test]
fn dotc_and_dotu_complex() {
    let x: Vec<Complex64> = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
    ];
    let y: Vec<Complex64> = vec![
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];

    // dotu = Σ x_i * y_i
    //      = (1+i)(3) + (2-i)(i) = (3+3i) + (1+2i) = 4 + 5i.
    let u = dotu(&x, &y).unwrap();
    assert_abs_diff_eq!(u.re, 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(u.im, 5.0, epsilon = 1e-12);

    // dotc = Σ conj(x_i) * y_i
    //      = (1-i)(3) + (2+i)(i) = (3-3i) + (-1+2i) = 2 - i.
    let c = dotc(&x, &y).unwrap();
    assert_abs_diff_eq!(c.re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c.im, -1.0, epsilon = 1e-12);
}

#[test]
fn axpy_real() {
    let x: Vec<f32> = vec![1.0, 2.0, 3.0];
    let mut y: Vec<f32> = vec![10.0, 20.0, 30.0];
    axpy(2.5_f32, &x, &mut y).unwrap();
    assert_abs_diff_eq!(y[0], 12.5);
    assert_abs_diff_eq!(y[1], 25.0);
    assert_abs_diff_eq!(y[2], 37.5);
}

#[test]
fn scal_real_and_complex() {
    let mut x: Vec<f64> = vec![1.0, 2.0, 3.0];
    scal(0.5_f64, &mut x);
    assert_eq!(x, vec![0.5, 1.0, 1.5]);

    let mut z: Vec<Complex32> = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, -1.0)];
    scal(Complex32::new(0.0, 1.0), &mut z); // multiply by i.
    assert_abs_diff_eq!(z[0].re, -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(z[0].im, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(z[1].re, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(z[1].im, 3.0, epsilon = 1e-6);
}

#[test]
fn scal_real_for_complex_uses_csscal() {
    let mut z: Vec<Complex64> = vec![
        Complex64::new(1.0, -2.0),
        Complex64::new(3.0, 4.0),
    ];
    scal_real(2.0_f64, &mut z);
    assert_abs_diff_eq!(z[0].re, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[0].im, -4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[1].re, 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[1].im, 8.0, epsilon = 1e-12);
}

#[test]
fn copy_swap_match_naive() {
    let x: Vec<f64> = vec![1.0, 2.0, 3.0];
    let mut y: Vec<f64> = vec![0.0; 3];
    copy(&x, &mut y).unwrap();
    assert_eq!(x, y);

    let mut a: Vec<f64> = vec![10.0, 20.0, 30.0];
    let mut b: Vec<f64> = vec![1.0, 2.0, 3.0];
    swap(&mut a, &mut b).unwrap();
    assert_eq!(a, vec![1.0, 2.0, 3.0]);
    assert_eq!(b, vec![10.0, 20.0, 30.0]);
}

#[test]
fn iamax_and_iamin() {
    let x: Vec<f64> = vec![1.0, -3.0, 2.0, -5.0, 4.0];
    // |x| = [1, 3, 2, 5, 4] → max at index 3, min at index 0.
    assert_eq!(iamax(&x), Some(3));
    assert_eq!(iamin(&x), Some(0));

    let empty: Vec<f64> = Vec::new();
    assert_eq!(iamax(&empty), None);
    assert_eq!(iamin(&empty), None);
}

#[test]
fn strided_dot() {
    // Dot product over every other element.
    let x: Vec<f64> = vec![1.0, 99.0, 2.0, 99.0, 3.0];
    let y: Vec<f64> = vec![4.0, 99.0, 5.0, 99.0, 6.0];
    let d = dot_inc(&x, 2, &y, 2).unwrap();
    assert_abs_diff_eq!(d, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0, epsilon = 1e-12);
}

#[test]
fn strided_asum() {
    let x: Vec<f64> = vec![1.0, -100.0, -2.0, 100.0, 3.0];
    let s = asum_inc(&x, 2).unwrap();
    assert_abs_diff_eq!(s, 6.0, epsilon = 1e-12);
}

#[test]
fn rot_applies_2d_rotation() {
    // 90° rotation: c = 0, s = 1.
    let c = 0.0_f64;
    let s = 1.0_f64;
    let mut x = vec![1.0, 2.0, 3.0];
    let mut y = vec![4.0, 5.0, 6.0];
    rot(&mut x, &mut y, c, s).unwrap();
    // x' = c*x + s*y = y; y' = -s*x + c*y = -x.
    assert_abs_diff_eq!(x[0], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(x[2], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[0], -1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[1], -2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(y[2], -3.0, epsilon = 1e-12);
}

#[test]
fn rotg_zeros_b() {
    let mut a = 3.0_f64;
    let mut b = 4.0_f64;
    let (c, s) = rotg(&mut a, &mut b);
    // Applying [[c, s], [-s, c]] to [3, 4] should give [r, 0] = [5, 0].
    let r_check = c * 3.0 + s * 4.0;
    let z_check = -s * 3.0 + c * 4.0;
    assert_abs_diff_eq!(r_check, 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z_check, 0.0, epsilon = 1e-12);
}
