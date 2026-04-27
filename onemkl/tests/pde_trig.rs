#![cfg(feature = "pde")]

//! Verify trigonometric transforms (DCT / DST).

use approx::assert_abs_diff_eq;

use onemkl::pde::{TrigTransform, TrigTransformType};

#[test]
fn cosine_round_trip_recovers_input() {
    // Cosine transform expects buffer length n + 1.
    let n: usize = 8;
    let original: Vec<f64> = (0..=n).map(|i| (i as f64) * 0.5).collect();
    let mut buf = original.clone();

    let mut plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-10);
    }
}

#[test]
fn sine_round_trip_recovers_input() {
    let n: usize = 8;
    // Sine transform requires zero boundary samples. The interior
    // values are arbitrary.
    let original: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let mut buf = original.clone();

    let mut plan = TrigTransform::<f64>::new(TrigTransformType::Sine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-10);
    }
}

#[test]
fn staggered_cosine_round_trip() {
    let n: usize = 8;
    let original: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    let mut buf = original.clone();

    let mut plan =
        TrigTransform::<f64>::new(TrigTransformType::StaggeredCosine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-10);
    }
}

#[test]
fn buffer_len_matches_type() {
    assert_eq!(TrigTransformType::Cosine.buffer_len(8), 9);
    assert_eq!(TrigTransformType::Sine.buffer_len(8), 9);
    assert_eq!(TrigTransformType::StaggeredCosine.buffer_len(8), 8);
    assert_eq!(TrigTransformType::Staggered2Sine.buffer_len(8), 8);
}

#[test]
fn buffer_len_query_matches_plan() {
    let plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, 8).unwrap();
    assert_eq!(plan.buffer_len(), 9);
}

#[test]
fn rejects_undersized_buffer() {
    let mut plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, 8).unwrap();
    let mut too_small = vec![0.0_f64; 4];
    assert!(plan.forward(&mut too_small).is_err());
}

#[test]
fn f32_round_trip() {
    let n: usize = 8;
    let original: Vec<f32> = (0..=n).map(|i| (i as f32) * 0.25).collect();
    let mut buf = original.clone();

    let mut plan = TrigTransform::<f32>::new(TrigTransformType::Cosine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-5);
    }
}
