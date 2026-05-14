#![cfg(feature = "rng")]

//! Verify summary statistics tasks.

use approx::assert_abs_diff_eq;

use onemkl::rng::summary_stats::SummaryStats;

#[test]
fn mean_of_arithmetic_sequence() {
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 5).unwrap();
    let mean = ss.mean().unwrap();
    assert_eq!(mean.len(), 1);
    assert_abs_diff_eq!(mean[0], 3.0, epsilon = 1e-12);
}

#[test]
fn variance_of_known_sample() {
    // Sample variance of [1, 2, 3, 4, 5] is 10 / (5 - 1) = 2.5.
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 5).unwrap();
    let var = ss.variance().unwrap();
    assert_eq!(var.len(), 1);
    assert_abs_diff_eq!(var[0], 2.5, epsilon = 1e-12);
}

#[test]
fn min_max_sum() {
    let data = [3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 7).unwrap();
    let mn = ss.min().unwrap();
    let mx = ss.max().unwrap();
    let sm = ss.sum().unwrap();
    assert_abs_diff_eq!(mn[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(mx[0], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(sm[0], 25.0, epsilon = 1e-12);
}

#[test]
fn multivariate_means() {
    // 2 variables × 4 observations, row-major:
    //   var 0: [1, 2, 3, 4]  → mean 2.5
    //   var 1: [10, 20, 30, 40] → mean 25.0
    let data = [1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut ss = SummaryStats::<f64>::new(&data, 2, 4).unwrap();
    let mean = ss.mean().unwrap();
    assert_eq!(mean.len(), 2);
    assert_abs_diff_eq!(mean[0], 2.5, epsilon = 1e-12);
    assert_abs_diff_eq!(mean[1], 25.0, epsilon = 1e-12);
}

#[test]
fn f32_variant() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let mut ss = SummaryStats::<f32>::new(&data, 1, 5).unwrap();
    let mean = ss.mean().unwrap();
    assert_abs_diff_eq!(mean[0], 3.0, epsilon = 1e-6);
}

#[test]
fn rejects_undersized_buffer() {
    // Want 2 vars × 3 obs but supply only 5 elements.
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    assert!(SummaryStats::<f64>::new(&data, 2, 3).is_err());
}

#[test]
fn quantiles_at_quartiles() {
    // Sample [1..=11]; expected median = 6 exactly. Q1 / Q3 depend
    // on the interpolation convention; just check they're in band.
    let data: Vec<f64> = (1..=11).map(|x| x as f64).collect();
    let mut ss = SummaryStats::<f64>::new(&data, 1, data.len()).unwrap();
    let q = ss.quantiles(&[0.25, 0.5, 0.75]).unwrap();
    assert_eq!(q.len(), 3);
    assert_abs_diff_eq!(q[1], 6.0, epsilon = 1e-9);
    assert!((q[0] - 3.5).abs() < 0.5);
    assert!((q[2] - 8.5).abs() < 0.5);
}

#[test]
fn quantiles_rejects_empty_orders() {
    let data = [1.0_f64, 2.0, 3.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 3).unwrap();
    assert!(ss.quantiles(&[]).is_err());
}

#[test]
fn robust_covariance_tbs_runs() {
    // TBS (Tukey biweight S-estimator) is the algorithm MKL exposes
    // for VSL_SS_ROBUST_COV. Needs a multivariate, sufficiently large
    // sample.
    const N: usize = 200;
    let var1: Vec<f64> = (0..N).map(|x| x as f64).collect();
    let var2: Vec<f64> = (0..N).map(|x| (x as f64) * 2.0).collect();
    let mut data = Vec::with_capacity(2 * N);
    data.extend_from_slice(&var1);
    data.extend_from_slice(&var2);
    let mut ss = SummaryStats::<f64>::new(&data, 2, N).unwrap();
    let (mean, cov) = ss.robust_covariance_tbs(&[]).unwrap();
    assert_eq!(mean.len(), 2);
    assert_eq!(cov.len(), 4);
    // Robust mean of an evenly-spaced sequence sits in its middle.
    assert!(mean[0] > 50.0 && mean[0] < 150.0);
}

#[test]
fn outliers_bacon_flags_extreme() {
    // 5 normal points + 1 huge outlier.
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 1000.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, data.len()).unwrap();
    let params = SummaryStats::<f64>::bacon_default_params();
    let weights = ss.outliers_bacon(&params).unwrap();
    assert_eq!(weights.len(), data.len());
    assert!(
        weights[5] < 1.0,
        "expected last sample flagged, got weight {}",
        weights[5]
    );
}
