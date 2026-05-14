#![cfg(feature = "rng")]

//! Verify VSL RNG wrappers via statistical sanity checks.

use approx::assert_abs_diff_eq;

use onemkl::rng::{BasicRng, MatrixStorage, Stream};

const N: usize = 100_000;

fn mean_f64(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn var_f64(xs: &[f64]) -> f64 {
    let m = mean_f64(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

#[test]
fn uniform_mean_in_band() {
    let mut s = Stream::new(BasicRng::Mt19937, 42).unwrap();
    let mut buf = vec![0.0_f64; N];
    s.uniform(&mut buf, 0.0, 1.0).unwrap();
    let mean = mean_f64(&buf);
    // Expected mean is 0.5; allow plenty of slack for randomness.
    assert_abs_diff_eq!(mean, 0.5, epsilon = 0.01);
    // All values should be in [0, 1).
    assert!(buf.iter().all(|&x| (0.0..1.0).contains(&x)));
}

#[test]
fn uniform_f32() {
    let mut s = Stream::new(BasicRng::Mt19937, 7).unwrap();
    let mut buf = vec![0.0_f32; N];
    s.uniform(&mut buf, -1.0_f32, 1.0_f32).unwrap();
    let m: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
    assert!((m as f64).abs() < 0.02);
    assert!(buf.iter().all(|&x| (-1.0..1.0).contains(&x)));
}

#[test]
fn gaussian_mean_and_variance() {
    let mut s = Stream::new(BasicRng::Mt19937, 100).unwrap();
    let mut buf = vec![0.0_f64; N];
    s.gaussian(&mut buf, 5.0, 2.0).unwrap();
    let mean = mean_f64(&buf);
    let var = var_f64(&buf);
    // Generous tolerance for sample statistics.
    assert!((mean - 5.0).abs() < 0.05, "mean = {}", mean);
    assert!((var - 4.0).abs() < 0.1, "var = {}", var);
}

#[test]
fn exponential_mean() {
    // mean of Exp(displacement=0, beta=2) is 0 + 2 = 2.
    let mut s = Stream::new(BasicRng::Mt19937, 11).unwrap();
    let mut buf = vec![0.0_f64; N];
    s.exponential(&mut buf, 0.0, 2.0).unwrap();
    let mean = mean_f64(&buf);
    assert!((mean - 2.0).abs() < 0.05, "mean = {}", mean);
    assert!(buf.iter().all(|&x| x >= 0.0));
}

#[test]
fn poisson_mean() {
    let mut s = Stream::new(BasicRng::Mt19937, 17).unwrap();
    let mut buf = vec![0_i32; N];
    s.poisson(&mut buf, 5.0).unwrap();
    let m: f64 = buf.iter().map(|&x| x as f64).sum::<f64>() / buf.len() as f64;
    assert!((m - 5.0).abs() < 0.1);
}

#[test]
fn bernoulli_proportion() {
    let mut s = Stream::new(BasicRng::Mt19937, 23).unwrap();
    let mut buf = vec![0_i32; N];
    s.bernoulli(&mut buf, 0.3).unwrap();
    let count = buf.iter().filter(|&&v| v == 1).count();
    let p = count as f64 / N as f64;
    assert!((p - 0.3).abs() < 0.01);
    assert!(buf.iter().all(|&v| v == 0 || v == 1));
}

#[test]
fn uniform_int_range() {
    let mut s = Stream::new(BasicRng::Mt19937, 31).unwrap();
    let mut buf = vec![0_i32; 1000];
    s.uniform_int(&mut buf, 10, 20).unwrap();
    assert!(buf.iter().all(|&v| (10..20).contains(&v)));
}

#[test]
fn binomial_mean() {
    let mut s = Stream::new(BasicRng::Mt19937, 41).unwrap();
    let mut buf = vec![0_i32; N];
    let p = 0.4;
    let n = 20;
    s.binomial(&mut buf, n, p).unwrap();
    let m: f64 = buf.iter().map(|&x| x as f64).sum::<f64>() / buf.len() as f64;
    // Expected mean is n*p = 8.0.
    assert!((m - 8.0).abs() < 0.05, "mean = {}", m);
}

#[test]
fn deterministic_seed_reproduces() {
    let mut s1 = Stream::new(BasicRng::Mt19937, 12345).unwrap();
    let mut s2 = Stream::new(BasicRng::Mt19937, 12345).unwrap();
    let mut a = vec![0.0_f64; 10];
    let mut b = vec![0.0_f64; 10];
    s1.uniform(&mut a, 0.0, 1.0).unwrap();
    s2.uniform(&mut b, 0.0, 1.0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn skip_ahead_decimates() {
    let mut s_full = Stream::new(BasicRng::Mt19937, 99).unwrap();
    let mut s_skip = s_full.clone();
    let mut full = vec![0.0_f64; 20];
    s_full.uniform(&mut full, 0.0, 1.0).unwrap();
    // Skip the first 10, then draw 10 more.
    s_skip.skip_ahead(10).unwrap();
    let mut skipped = vec![0.0_f64; 10];
    s_skip.uniform(&mut skipped, 0.0, 1.0).unwrap();
    assert_eq!(&full[10..20], &skipped[..]);
}

#[test]
fn clone_independent_state() {
    let mut s1 = Stream::new(BasicRng::Mt19937, 555).unwrap();
    let mut s2 = s1.clone();
    let mut a = vec![0.0_f64; 10];
    let mut b = vec![0.0_f64; 10];
    s1.uniform(&mut a, 0.0, 1.0).unwrap();
    s2.uniform(&mut b, 0.0, 1.0).unwrap();
    // Clone shares state at the moment of cloning, so first draws match.
    assert_eq!(a, b);
}

#[test]
fn gumbel_location() {
    let mut s = Stream::new(BasicRng::Mt19937, 7).unwrap();
    let mut buf = vec![0.0_f64; N];
    s.gumbel(&mut buf, 0.0, 1.0).unwrap();
    // MKL's vsRngGumbel returns the min-Gumbel (Type I extreme value
    // for minima), with mean a − γβ. With a=0, β=1 that's −γ.
    let mean = mean_f64(&buf);
    assert_abs_diff_eq!(mean, -0.5772, epsilon = 5e-2);
}

#[test]
fn laplace_zero_mean() {
    let mut s = Stream::new(BasicRng::Mt19937, 11).unwrap();
    let mut buf = vec![0.0_f64; N];
    s.laplace(&mut buf, 0.0, 1.0).unwrap();
    assert_abs_diff_eq!(mean_f64(&buf), 0.0, epsilon = 5e-2);
}

#[test]
fn chi_square_mean_equals_dof() {
    let mut s = Stream::new(BasicRng::Mt19937, 13).unwrap();
    let mut buf = vec![0.0_f64; N];
    s.chi_square(&mut buf, 5).unwrap();
    // χ²_k has mean k.
    assert_abs_diff_eq!(mean_f64(&buf), 5.0, epsilon = 0.1);
}

#[test]
fn truncated_normal_stays_in_bounds() {
    let mut s = Stream::new(BasicRng::Mt19937, 17).unwrap();
    let mut buf = vec![0.0_f64; N];
    s.truncated_normal(&mut buf, 0.0, 1.0, -2.0, 2.0).unwrap();
    assert!(buf.iter().all(|&x| (-2.0..=2.0).contains(&x)));
}

#[test]
fn gaussian_mv_diagonal_matches_independent() {
    let mut s = Stream::new(BasicRng::Mt19937, 19).unwrap();
    let mut buf = vec![0.0_f64; N * 3];
    // Diagonal storage: stddevs along each of 3 axes.
    let mean = [0.0, 1.0, -1.0];
    let stds = [1.0, 2.0, 0.5];
    s.gaussian_mv(&mut buf, 3, MatrixStorage::Diagonal, &mean, &stds)
        .unwrap();
    // Each axis should match its target mean / variance independently.
    for axis in 0..3 {
        let col: Vec<f64> = buf.iter().skip(axis).step_by(3).copied().collect();
        assert_abs_diff_eq!(mean_f64(&col), mean[axis], epsilon = 0.05);
        let var = var_f64(&col);
        assert_abs_diff_eq!(var.sqrt(), stds[axis], epsilon = 0.05);
    }
}

#[test]
fn multinomial_row_sums_match_ntrial() {
    let mut s = Stream::new(BasicRng::Mt19937, 23).unwrap();
    let p = [0.2, 0.3, 0.5];
    let mut out = vec![0_i32; 100 * p.len()];
    s.multinomial(&mut out, 10, &p).unwrap();
    for row in out.chunks_exact(p.len()) {
        assert_eq!(row.iter().sum::<i32>(), 10);
    }
}

#[test]
fn categorical_indices_in_range() {
    let mut s = Stream::new(BasicRng::Mt19937, 29).unwrap();
    let p = [0.1, 0.4, 0.5];
    let mut out = vec![0_i32; 5_000];
    s.categorical(&mut out, &p).unwrap();
    assert!(out.iter().all(|&i| (0..3).contains(&i)));
    // Empirical frequency of category 2 should be ≈ 0.5.
    let freq2 = out.iter().filter(|&&i| i == 2).count() as f64 / out.len() as f64;
    assert_abs_diff_eq!(freq2, 0.5, epsilon = 0.05);
}

#[test]
fn dirichlet_rows_sum_to_one() {
    let mut s = Stream::new(BasicRng::Mt19937, 31).unwrap();
    let alpha = [1.0, 2.0, 5.0];
    let mut out = vec![0.0_f64; 1_000 * alpha.len()];
    s.dirichlet(&mut out, &alpha).unwrap();
    for row in out.chunks_exact(alpha.len()) {
        let s: f64 = row.iter().sum();
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        assert!(row.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }
}

#[test]
fn geometric_mean_is_one_minus_p_over_p() {
    let mut s = Stream::new(BasicRng::Mt19937, 37).unwrap();
    let mut out = vec![0_i32; N];
    s.geometric(&mut out, 0.25).unwrap();
    let mean = out.iter().map(|&x| x as f64).sum::<f64>() / out.len() as f64;
    // Geometric(p) on {0,1,...} has mean (1-p)/p.
    assert_abs_diff_eq!(mean, 0.75 / 0.25, epsilon = 0.1);
}

#[test]
fn quasi_random_sobol_dimension_2() {
    let mut s = Stream::quasi_random(BasicRng::Sobol, 2).unwrap();
    let mut buf = vec![0.0_f64; 2_048];
    s.uniform(&mut buf, 0.0, 1.0).unwrap();
    assert!(buf.iter().all(|&x| (0.0..1.0).contains(&x)));
    // Sobol on [0,1]^2 should integrate the constant 1 very accurately.
    let est = mean_f64(&buf);
    assert_abs_diff_eq!(est, 0.5, epsilon = 1e-3);
}

#[test]
fn quasi_random_rejects_pseudo_brng() {
    let r = Stream::quasi_random(BasicRng::Mt19937, 1);
    assert!(r.is_err());
}

#[test]
fn quasi_random_rejects_zero_dimension() {
    let r = Stream::quasi_random(BasicRng::Sobol, 0);
    assert!(r.is_err());
}
