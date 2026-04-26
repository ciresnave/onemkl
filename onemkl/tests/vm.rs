//! Verify Vector Math (VM) wrappers against std math reference values.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use onemkl::vm;

#[test]
fn arithmetic_real() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0];
    let y = vec![10.0_f64, 20.0, 30.0, 40.0];
    let mut r = vec![0.0_f64; 4];

    vm::add(&x, &y, &mut r).unwrap();
    assert_eq!(r, vec![11.0, 22.0, 33.0, 44.0]);

    vm::sub(&y, &x, &mut r).unwrap();
    assert_eq!(r, vec![9.0, 18.0, 27.0, 36.0]);

    vm::mul(&x, &y, &mut r).unwrap();
    assert_eq!(r, vec![10.0, 40.0, 90.0, 160.0]);

    vm::div(&y, &x, &mut r).unwrap();
    assert_eq!(r, vec![10.0, 10.0, 10.0, 10.0]);

    vm::sqr(&x, &mut r).unwrap();
    assert_eq!(r, vec![1.0, 4.0, 9.0, 16.0]);

    vm::inv(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(r[2], 1.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[3], 0.25, epsilon = 1e-12);
}

#[test]
fn power_root() {
    let x = vec![4.0_f64, 9.0, 16.0, 25.0];
    let mut r = vec![0.0_f64; 4];

    vm::sqrt(&x, &mut r).unwrap();
    assert_eq!(r, vec![2.0, 3.0, 4.0, 5.0]);

    vm::inv_sqrt(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0 / 3.0, epsilon = 1e-12);

    let cubes = vec![8.0_f64, 27.0, 64.0, 125.0];
    vm::cbrt(&cubes, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[2], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[3], 5.0, epsilon = 1e-12);

    let mut r2 = vec![0.0_f64; 4];
    vm::powx(&x, 2.0, &mut r2).unwrap();
    assert_eq!(r2, vec![16.0, 81.0, 256.0, 625.0]);
}

#[test]
fn exp_log() {
    let x = vec![0.0_f64, 1.0, 2.0];
    let mut r = vec![0.0_f64; 3];
    vm::exp(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0_f64.exp(), epsilon = 1e-12);
    assert_abs_diff_eq!(r[2], 2.0_f64.exp(), epsilon = 1e-12);

    let y = vec![1.0_f64, 10.0, 100.0];
    let mut s = vec![0.0_f64; 3];
    vm::log10(&y, &mut s).unwrap();
    assert_abs_diff_eq!(s[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[2], 2.0, epsilon = 1e-12);

    vm::ln(&y, &mut s).unwrap();
    assert_abs_diff_eq!(s[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[1], 10.0_f64.ln(), epsilon = 1e-12);
}

#[test]
fn trig_real() {
    use std::f64::consts::FRAC_PI_2;
    let x = vec![0.0_f64, FRAC_PI_2, std::f64::consts::PI];
    let mut s = vec![0.0_f64; 3];
    let mut c = vec![0.0_f64; 3];

    vm::sin(&x, &mut s).unwrap();
    assert_abs_diff_eq!(s[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[2], 0.0, epsilon = 1e-12);

    vm::cos(&x, &mut c).unwrap();
    assert_abs_diff_eq!(c[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c[2], -1.0, epsilon = 1e-12);

    let mut sb = vec![0.0_f64; 3];
    let mut cb = vec![0.0_f64; 3];
    vm::sincos(&x, &mut sb, &mut cb).unwrap();
    for i in 0..3 {
        assert_abs_diff_eq!(sb[i], s[i], epsilon = 1e-12);
        assert_abs_diff_eq!(cb[i], c[i], epsilon = 1e-12);
    }
}

#[test]
fn inverse_trig() {
    let x = vec![-1.0_f64, 0.0, 1.0];
    let mut r = vec![0.0_f64; 3];

    vm::asin(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], -std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[2], std::f64::consts::FRAC_PI_2, epsilon = 1e-12);

    let y = vec![1.0_f64, 0.0, -1.0];
    let xv = vec![0.0_f64, 1.0, 0.0];
    vm::atan2(&y, &xv, &mut r).unwrap();
    // atan2(1, 0) = π/2; atan2(0, 1) = 0; atan2(-1, 0) = -π/2
    assert_abs_diff_eq!(r[0], std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[2], -std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
}

#[test]
fn hyperbolic() {
    let x = vec![0.0_f64, 1.0];
    let mut r = vec![0.0_f64; 2];
    vm::sinh(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0_f64.sinh(), epsilon = 1e-12);

    vm::cosh(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0_f64.cosh(), epsilon = 1e-12);

    vm::tanh(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0_f64.tanh(), epsilon = 1e-12);
}

#[test]
fn special_functions() {
    let x = vec![0.0_f64, 1.0, 2.0];
    let mut r = vec![0.0_f64; 3];

    vm::erf(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 0.842_700_792_949_715, epsilon = 1e-10);
    assert_abs_diff_eq!(r[2], 0.995_322_265_018_953, epsilon = 1e-10);

    vm::tgamma(&[1.0_f64, 2.0, 5.0], &mut r).unwrap();
    // Γ(1)=1, Γ(2)=1, Γ(5)=24
    assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[2], 24.0, epsilon = 1e-10);
}

#[test]
fn rounding() {
    let x = vec![1.4_f64, 1.5, 1.6, -1.4, -1.5, -1.6];
    let mut r = vec![0.0_f64; 6];

    vm::floor(&x, &mut r).unwrap();
    assert_eq!(r, vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);

    vm::ceil(&x, &mut r).unwrap();
    assert_eq!(r, vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);

    vm::trunc(&x, &mut r).unwrap();
    assert_eq!(r, vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
}

#[test]
fn complex_arithmetic() {
    let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)];
    let y = vec![Complex64::new(0.0, 1.0), Complex64::new(2.0, 0.0)];
    let mut r = vec![Complex64::new(0.0, 0.0); 2];

    vm::add(&x, &y, &mut r).unwrap();
    assert_abs_diff_eq!(r[0].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[0].im, 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1].re, 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1].im, -1.0, epsilon = 1e-12);

    vm::mul(&x, &y, &mut r).unwrap();
    // (1+2i)(0+i) = -2 + i; (3-i)(2+0i) = 6 - 2i
    assert_abs_diff_eq!(r[0].re, -2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[0].im, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1].re, 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1].im, -2.0, epsilon = 1e-12);
}

#[test]
fn complex_to_real() {
    let x = vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)];
    let mut r = vec![0.0_f64; 2];

    vm::abs_complex(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0], 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 0.0, epsilon = 1e-12);

    vm::arg(&x, &mut r).unwrap();
    // arg(3+4i) = atan2(4, 3); arg(0) = 0 (by convention)
    assert_abs_diff_eq!(r[0], 4.0_f64.atan2(3.0), epsilon = 1e-12);
}

#[test]
fn complex_conj_and_cis() {
    let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    let mut r = vec![Complex64::new(0.0, 0.0); 2];

    vm::conj(&x, &mut r).unwrap();
    assert_abs_diff_eq!(r[0].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[0].im, -2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1].re, 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1].im, 4.0, epsilon = 1e-12);

    let theta = vec![0.0_f64, std::f64::consts::FRAC_PI_2];
    let mut z = vec![Complex64::new(0.0, 0.0); 2];
    vm::cis::<Complex64>(&theta, &mut z).unwrap();
    // exp(0) = 1; exp(iπ/2) = i
    assert_abs_diff_eq!(z[0].re, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[0].im, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[1].re, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(z[1].im, 1.0, epsilon = 1e-12);
}

#[test]
fn linear_frac_works() {
    // r = (2*x + 1) / (1*y + 0) = (2x+1) / y
    let x = vec![1.0_f64, 2.0];
    let y = vec![3.0_f64, 5.0];
    let mut r = vec![0.0_f64; 2];
    vm::linear_frac(&x, &y, 2.0, 1.0, 1.0, 0.0, &mut r).unwrap();
    // (2*1+1)/3 = 1; (2*2+1)/5 = 1
    assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0, epsilon = 1e-12);
}

#[test]
fn modf_split() {
    let x = vec![1.5_f64, -2.75, 3.0];
    let mut int_part = vec![0.0_f64; 3];
    let mut frac_part = vec![0.0_f64; 3];
    vm::modf(&x, &mut int_part, &mut frac_part).unwrap();
    assert_abs_diff_eq!(int_part[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(frac_part[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(int_part[1], -2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(frac_part[1], -0.75, epsilon = 1e-12);
}
