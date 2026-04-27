//! Run an int8 × uint8 → int32 GEMM as in quantized DL inference.
//! `A` is signed activations, `B` is unsigned weights, `C` is the
//! int32 accumulator. Demonstrates the `gemm_s8u8_s32` API with a
//! per-column zero-point bias.
//!
//! Run with `cargo run --example quantized_inference`.

use onemkl::blas::mixed_precision::{gemm_s8u8_s32, CblasOffset};
use onemkl::{Layout, Transpose};

fn main() {
    // 4×8 activations × 8×4 weights → 4×4 output.
    let m = 4;
    let n = 4;
    let k = 8;

    let a: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
    let b: Vec<u8> = (0..k * n).map(|i| ((i * 3) % 13) as u8).collect();

    // Per-column bias (4 entries) — broadcast across all rows of C.
    let cb = vec![10_i32, 20, 30, 40];

    let mut c = vec![0_i32; m * n];

    gemm_s8u8_s32(
        Layout::RowMajor,
        Transpose::NoTrans, Transpose::NoTrans,
        CblasOffset::Row,
        m, n, k,
        1.0,         // alpha
        &a, k, 0,    // A and zero-point ao = 0 (symmetric int8)
        &b, n, 0,    // B and zero-point bo = 0
        0.0,         // beta
        &mut c, n,
        &cb,
    )
    .unwrap();

    println!("Quantized output C ({}×{}):", m, n);
    for row in c.chunks(n) {
        println!("  {:?}", row);
    }
}
