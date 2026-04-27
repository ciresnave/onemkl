# onemkl

Safe, idiomatic Rust wrapper over [Intel oneAPI Math Kernel Library (oneMKL)][onemkl].

Built on top of [`onemkl-sys`](../onemkl-sys), which provides the raw FFI
bindings.

## What's covered

Every major oneMKL domain has a safe Rust wrapper. Some highlights, by
audience:

### For ML inference / training

- **Mixed-precision GEMM** (`blas::mixed_precision`): `gemm_bf16_f32`,
  `gemm_f16_f32`, `gemm_e5m2_f32`, `gemm_e4m3_f32` (FP8 for both
  activations and weights), `gemm_s8u8_s32`, `gemm_s16_s32` (quantized
  inference), `hgemm` (pure fp16).
- **JIT GEMM** (`blas::jit`): runtime-specialized small kernels for
  transformer attention heads, MLP blocks, and any
  shape-stable hot loop.
- **Pack / compute** (`blas::packed`): pre-pack a fixed weight matrix
  once and amortize the data reordering across many input batches.
- **Compact BLAS** (`blas::compact`): SIMD-packed batched kernels for
  many independent small matrices in lockstep.
- **Pointer-array batched GEMM / GEMV / TRSM** (`blas::extensions`):
  the canonical batched dispatch model.

### For scientific computing

- **Direct sparse solvers**: [`pardiso`] (with `pardiso_64`, save /
  restore handle, custom pivot callback, diagonal extraction) and
  [`dss`] (with timing / memory / determinant / inertia statistics).
- **Iterative sparse solvers** (`iss`): CG and FGMRES with closure
  matvec, optional preconditioning, and a step-by-step
  `CgSession` / `FgmresSession` API for callers driving the loop
  themselves.
- **Sparse manipulation** (`sparse`): CSR / COO / CSC / BSR
  construction, conversion between formats, sparse arithmetic
  (`add` / `spmm` / `spmmd`), iterative-solver primitives
  (`dot_mv`, `symgs`, `symgs_mv`).
- **Eigensolvers**: [`lapack`] (general, symmetric, Hermitian, banded,
  packed, generalized, divide-and-conquer, RRR) and [`feast`]
  (contour-integration solver for dense / CSR / banded /
  generalized problems).
- **FFT** (`fft`): 1-D / 2-D / 3-D / N-D complex and real-input
  transforms, configurable forward / backward scaling.
- **PDE building blocks** (`pde`): trigonometric transforms (DCT,
  DST, four staggered variants) for spectral PDE solvers.
- **Data fitting** (`data_fitting`): natural / Bessel / Akima /
  Hermite cubic splines.
- **Nonlinear optimization** (`optim`): trust-region nonlinear least
  squares (TRNLS) with bound constraints, plus numerical Jacobian.
- **Random numbers** (`rng`): Mersenne Twister, Philox4x32x10, MT2203,
  R250, and others; full set of continuous and discrete
  distributions; 1-D convolution / correlation; multivariate
  summary statistics.
- **Vector math** (`vm`): element-wise transcendentals.

See the crate-level rustdoc for the per-domain reference.

## Quick example

```rust,no_run
use onemkl::prelude::*;
use onemkl::blas::level3::gemm;

// Compute C = A * B for two 2×2 matrices.
let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
let b_data = vec![5.0_f64, 6.0, 7.0, 8.0];
let mut c_data = vec![0.0_f64; 4];

let a = MatrixRef::new(&a_data, 2, 2, Layout::RowMajor)?;
let b = MatrixRef::new(&b_data, 2, 2, Layout::RowMajor)?;
let mut c = MatrixMut::new(&mut c_data, 2, 2, Layout::RowMajor)?;

gemm(Transpose::NoTrans, Transpose::NoTrans, 1.0, &a, &b, 0.0, &mut c)?;
// c_data ≈ [19, 22, 43, 50]
# Ok::<(), onemkl::Error>(())
```

## More examples

The `examples/` directory contains short standalone programs for
common patterns:

- `solve_linear_system` — LAPACK `gesv` for `A x = b`.
- `sparse_cg` — Conjugate Gradient on a sparse SPD matrix.
- `fft_signal` — real-input FFT to find a peak frequency.
- `quantized_inference` — int8 × uint8 → int32 GEMM with bias for
  quantized DL.
- `transformer_qkv_projection` — JIT GEMM driving Q / K / V
  projections in a transformer layer.

Run any example with `cargo run --example <name>`.

## Cargo features

Each major oneMKL domain is gated by a feature so projects can opt out
of unused parts. **All domain features are enabled by default.**

| Feature | Module |
| --- | --- |
| `blas` | `blas` |
| `data-fitting` | `data_fitting` |
| `dss` | `dss` |
| `feast` | `feast` |
| `fft` | `fft` |
| `iss` | `iss` |
| `lapack` | `lapack` |
| `optim` | `optim` |
| `pardiso` | `pardiso` |
| `pde` | `pde` |
| `preconditioners` | `preconditioners` |
| `rng` | `rng` |
| `sparse` | `sparse` |
| `vm` | `vm` |

Build configuration (one from each group must be selected; defaults
shown):

- Interface: `lp64` (default) or `ilp64`.
- Threading: `threading-sequential` (default), `threading-intel-openmp`,
  or `threading-tbb`.
- Linkage: `link-dynamic` (default) or `link-static`.

For a minimal build with only BLAS and LAPACK:

```toml
onemkl = { version = "0.1", default-features = false, features = [
    "lp64", "threading-sequential", "link-dynamic",
    "blas", "lapack",
]}
```

## Building against MKL

The build script in `onemkl-sys` looks for MKL in this order:

1. `MKLROOT` environment variable.
2. `ONEMKL_SYS_INCLUDE_DIR` / `ONEMKL_SYS_LIB_DIR` overrides.
3. `ONEAPI_ROOT/mkl/latest`.
4. Platform-standard paths (`C:\Program Files (x86)\Intel\oneAPI` on
   Windows, `/opt/intel/oneapi` on Linux/macOS).

At runtime, MKL's `bin/` (Windows) or `lib/` (Linux/macOS) directory
must be on the loader search path.

## Anything not yet wrapped

Reachable through `onemkl::sys` (the raw `onemkl-sys` re-export).
See [`ROADMAP.md`](../ROADMAP.md) for what's planned per release.

## License

Dual-licensed under MIT or Apache-2.0, at your option.

[onemkl]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
