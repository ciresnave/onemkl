//! Transformer-style projection: 32 input tokens × 64 hidden dims
//! projected to query / key / value matrices via three GEMMs against
//! shared weight matrices. Demonstrates the JIT GEMM API for
//! repeating-shape inference.
//!
//! Run with `cargo run --example transformer_qkv_projection`.

use onemkl::blas::jit::JitGemm;
use onemkl::{Layout, Transpose};

fn main() {
    let batch_seq = 32; // tokens
    let hidden = 64;
    let head_dim = 64;

    // Activations (batch_seq × hidden) and three weight matrices
    // (hidden × head_dim) — random data for illustration.
    let activations: Vec<f32> = (0..batch_seq * hidden)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let wq: Vec<f32> = (0..hidden * head_dim).map(|i| (i as f32) * 0.0005).collect();
    let wk: Vec<f32> = (0..hidden * head_dim).map(|i| (i as f32) * 0.0007).collect();
    let wv: Vec<f32> = (0..hidden * head_dim).map(|i| (i as f32) * 0.0011).collect();

    // Build the JIT'd kernel once — same shape for Q, K, V.
    let plan = JitGemm::<f32>::new(
        Layout::RowMajor,
        Transpose::NoTrans, Transpose::NoTrans,
        batch_seq, head_dim, hidden,
        1.0, hidden, head_dim, 0.0, head_dim,
    )
    .unwrap();
    println!("JIT kernel status: {:?}", plan.status());

    let mut q = vec![0.0_f32; batch_seq * head_dim];
    let mut k = vec![0.0_f32; batch_seq * head_dim];
    let mut v = vec![0.0_f32; batch_seq * head_dim];

    plan.execute(&activations, &wq, &mut q).unwrap();
    plan.execute(&activations, &wk, &mut k).unwrap();
    plan.execute(&activations, &wv, &mut v).unwrap();

    println!("Q[0..4] = {:?}", &q[0..4]);
    println!("K[0..4] = {:?}", &k[0..4]);
    println!("V[0..4] = {:?}", &v[0..4]);
}
