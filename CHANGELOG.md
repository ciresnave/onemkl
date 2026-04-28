# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Added — every major oneMKL domain has a safe Rust wrapper

#### BLAS and BLAS-like extensions (`blas`)

- Level 1 / 2 / 3 covering all universal, real-only, and complex-only
  routines, including banded and packed variants.
- Strided batched: `gemm_batch_strided`, `trsm_batch_strided`,
  `gemv_batch_strided`, `dgmm_batch_strided`, `axpy_batch_strided`,
  `copy_batch_strided`, `gemm3m_batch_strided`.
- Pointer-array batched: `gemm_batch`, `gemv_batch`, `trsm_batch`
  with matching `*BatchGroup` structs that flatten group descriptors
  into MKL's parallel parameter arrays.
- Mixed-precision GEMMs in `blas::mixed_precision`: `gemm_bf16_f32`,
  `gemm_f16_f32`, `gemm_e5m2_f32`, `gemm_e4m3_f32`, `gemm_s8u8_s32`,
  `gemm_s16_s32`, `hgemm`. Inputs are taken as raw bit-pattern
  slices so callers can integrate with whatever bf16 / fp16 / fp8
  representation they already use.
- JIT GEMM in `blas::jit`: `JitGemm<T>` for f32 / f64 / Complex32 /
  Complex64 with runtime-specialized small kernels and
  `JitStatus::{Compiled, Fallback}` introspection.
- Pack / compute API in `blas::packed`: `PackedMatrix<T>` plus
  `gemm_compute_packed_a` / `gemm_compute_packed_b` for amortized
  packing across many GEMMs.
- Compact BLAS in `blas::compact`: SIMD-packed batched GEMM with
  `gepack` / `geunpack` / `gemm` / `get_size` and three pack widths
  (Sse / Avx / Avx512).
- Out-of-place / in-place transpose, axpby, omatcopy, omatadd.

#### LAPACK (`lapack`)

- Linear solvers (general, symmetric, Hermitian, banded, packed, PD).
- Least squares and QR (`gels`, `gelsd`, `geqrf`, `gelqf`, `orgqr`,
  `ungqr`).
- Eigenvalue and SVD (`syev`, `heev`, `geev_real`, `geev_complex`,
  `gesdd`, `gesvd`).
- Generalized eigenvalue (`sygv`, `hegv`, `ggev_real`,
  `ggev_complex`).
- Divide-and-conquer + RRR (`syevd`, `heevd`, `syevr`, `heevr`).
- Auxiliary: `lacpy`, `lange`, `gecon`, `laswp`, `larfg`.

#### Sparse BLAS (`sparse`)

- `SparseMatrix<T>` with constructors for CSR / COO / CSC / BSR.
- Inspector-Executor: `mv` / `mm` / `trsv` / `optimize`, plus
  Sparse QR via `qr_factor` / `qr_solve`.
- Format conversion (`copy`, `convert_csr`, `convert_csc`,
  `convert_coo`, `convert_bsr`, `order`) and analysis-stage hint
  setters (`set_mv_hint`, `set_mm_hint`, `set_sv_hint`).
- Sparse arithmetic: `add`, `spmm`, `spmmd`.
- Iterative-solver primitives: `dot_mv`, `symgs`, `symgs_mv`.

#### Sparse direct solvers

- PARDISO (`pardiso`): factor + solve, multi-RHS, cached
  factorization, diagonal extraction (`with_diagonal_enabled` /
  `get_diagonal`), save / restore handle (`save_handle` /
  `load_handle` / `delete_handle_files`), low-level `export`,
  user-supplied `perm`, low-level `pardiso_64_raw` for the always-
  64-bit interface, `set_pardiso_pivot_callback` for custom pivot
  logic.
- DSS (`dss`): real and complex factor / solve, multi-RHS, statistics
  readout (timing / memory / determinant / inertia).

#### Iterative sparse solvers (`iss`)

- Closure-driven `solve_cg` and `solve_fgmres` returning `IssResult`
  with iterations, residual norms, and `IssStopReason`.
- User-driven step-by-step API: `CgSession` / `FgmresSession` with
  `step` returning an `RciAction` enum.

#### Preconditioners (`preconditioners`)

- ILU0 and ILUT factorization plus `apply_ilu` for the two-step
  triangular solve.

#### Eigensolvers (`feast`)

- Dense / CSR / banded standard problems and dense generalized
  problem, real and complex.

#### FFT (`fft`)

- 1-D / 2-D / 3-D / N-D complex transforms, in-place and out-of-place.
- Real-input transforms with CCE storage.
- Configurable forward / backward scaling (`*_with_scales`
  constructors).

#### Random numbers and statistics (`rng`)

- VSL `Stream` RAII type with all major BRNGs.
- Continuous and discrete distributions.
- 1-D convolution / correlation tasks (`Conv1d`, `Corr1d`).
- Multivariate summary statistics (`SummaryStats` with `mean`,
  `variance`, `min`, `max`, `sum`).

#### Vector math (`vm`)

- All major function families (transcendental, trigonometric,
  hyperbolic, special, rounding) with elementwise vector operations.

#### Optimization (`optim`)

- Trust-region nonlinear least squares (`solve_trnls`) and bound-
  constrained variant (`solve_trnls_bounded`), each returning
  `TrnlsResult` with iteration count, stopping criterion, and
  initial / final residual norms.
- RCI numerical Jacobian and low-level `djacobi_with_callback` for
  direct-FFI Jacobian computation.

#### Data fitting (`data_fitting`)

- Cubic splines: natural, Bessel, Akima, Hermite — all interpolatable
  and integrable.

#### PDE support (`pde`)

- Trigonometric transforms: Sine, Cosine, four staggered variants
  for spectral PDE solvers.

#### Service (`service`)

- Version, threading, memory, verbose mode, finalize.

### Build infrastructure

- Two-crate workspace: `onemkl-sys` (raw bindgen FFI) and `onemkl`
  (safe wrappers).
- Per-domain Cargo features all enabled by default; downstream
  callers can `default-features = false` and pick exactly what they
  need.
- Build configuration features: `lp64` / `ilp64` (interface),
  `threading-sequential` / `threading-intel-openmp` /
  `threading-tbb`, `link-dynamic` / `link-static`.
- MPI-gated domains as opt-in features: `cluster-sparse-solver`,
  `cdft`, `scalapack`, `blacs`, `pblas`.

### Documentation

- Crate-level rustdoc with a use-case-oriented orientation guide.
- Per-module rustdoc on every public domain.
- `examples/` directory with five runnable programs:
  `solve_linear_system`, `sparse_cg`, `fft_signal`,
  `quantized_inference`, `transformer_qkv_projection`.
- README with feature table and minimal example.
- `ROADMAP.md` tracking coverage tiers per domain.

### Known gaps

These are wrapped at the FFI level via `onemkl::sys` but lack a
high-level safe wrapper:

- FFT alternate real-input storage formats (CCS / PACK / PERM) —
  CCE format is fully wrapped.
- PARDISO Schur complement extraction — the `export` method is
  exposed but the exact MKL call sequence for Schur output is
  finicky enough that it currently crashes inside MKL.
- High-level `Pardiso64<T>` state-managed wrapper — only the
  low-level `pardiso_64_raw` is wrapped.
- Sparse `mkl_sparse_sp2m` (with request flags) and `_syprd`.
