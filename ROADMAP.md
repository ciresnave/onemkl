# Roadmap

This document tracks the current state of the wrappers and what's planned
next. As planned items become reality they collapse into the *Status*
section's table; the *Plans* section then shrinks accordingly.

## Coverage tiers

Each oneMKL domain is rated against three tiers. We use these as the
acceptance criteria for the milestones below.

- **MVP** — the core most-used routines work, with a safe API and real
  + complex coverage where applicable. Enough that someone wanting "the
  obvious thing" from this domain finds it.
- **Common** — every standard routine in the domain is exposed, including
  banded / packed / generalized / expert variants where they exist.
- **Comprehensive** — every entry point oneMKL ships under the domain is
  wrapped, including obscure / specialized variants (compact, JIT,
  mixed-precision, MPI, etc.).

A tier is "done" only when (a) the public API exists, (b) it returns
`Result` and validates inputs at boundaries, (c) every wrapped routine
has at least one test, and (d) the docs name the underlying MKL
function(s).

## Status

| Domain | Tier | Notes |
| --- | --- | --- |
| BLAS L1/L2/L3 | Common | All universal + real-only + complex-only routines, including banded and packed |
| BLAS-like extensions | MVP | `axpby`, `imatcopy`/`omatcopy`/`omatcopy2`/`omatadd`, batched (strided) `gemm`/`trsm`/`gemv`/`dgmm`/`axpy`/`copy`/`gemm3m` |
| LAPACK | MVP | Linear solve, QR, LS, eigenvalue, SVD; banded/packed/generalized still TODO |
| Sparse BLAS (IE) | MVP | CSR `mv`/`mm`/`trsv`/`optimize` + Sparse QR factor / solve |
| PARDISO | MVP | Factor + solve, multi-RHS, cached factorization |
| DSS | MVP | Symmetric SPD solve, multi-RHS |
| ISS (CG, FGMRES) | MVP | Closure-driven mat-vec; preconditioned CG |
| Preconditioners | MVP | ILU0, ILUT |
| FEAST | MVP | Dense symmetric / Hermitian; CSR / banded / generalized / RCI still TODO |
| VM (Vector Math) | Common | All major function families |
| RNG (VSL) | MVP | RAII `Stream` + 8 continuous + 4 discrete distributions |
| FFT (DFTI) | MVP | 1-D complex (in-place + out-of-place) |
| Data fitting | MVP | Natural cubic spline |
| Optimization | MVP | TRNLS, TRNLSPBC, numerical Jacobian |
| Service | Common | Version, threading, memory, verbose, finalize |

170 tests pass workspace-wide; each test maps directly to a wrapped
routine or trait method.

## Plans

### 0.2.0 — All domains at Common

Move every domain currently at MVP up to Common. Specifically:

- **BLAS-like extensions**: pointer-array batched (`?gemm_batch`,
  `?trsm_batch`, `?gemv_batch`), pack/compute API (`?gemm_pack`,
  `?gemm_compute`), JIT (`mkl_jit_create_?gemm`).
- **LAPACK**: banded variants (`?gbsv`, `?gbtrf`/`?gbtrs`,
  `?pbsv`/`?pbtrf`/`?pbtrs`), packed variants (`?spsv`/`?sptrf`,
  `?ppsv`/`?pptrf`), generalized eigenvalue (`?ggev`, `?sygv`,
  `?hegv`), expert / RRR drivers (`?gesvx`, `?syevr`, `?heevr`),
  utility routines (`?lange`, `?gecon`, `?syrfs`).
- **Sparse BLAS**: COO, CSC, BSR formats; `mkl_sparse_copy`,
  `mkl_sparse_convert_*`, `mkl_sparse_order`; the analysis-stage hint
  setters (`mkl_sparse_set_mv_hint`, etc.).
- **PARDISO**: `pardiso_64`, `mkl_pardiso_pivot`, `pardiso_getdiag`,
  `pardiso_export`, save/restore handle pair.
- **DSS**: complex matrices, statistics readout, generalized symmetric
  options.
- **ISS**: full RCI surface for users who want to drive their own
  loop; expose `?cg_get`/`?fgmres_get` reasoning codes.
- **FEAST**: CSR-input drivers (`?feast_scsrev`, `?feast_hcsrev`),
  banded (`?feast_sbev`, `?feast_hbev`), generalized
  (`?feast_sygv`/`?feast_hegv`/`?feast_hcsrgv`).
- **RNG**: convolution and correlation tasks (VSL_*Conv*, VSL_*Corr*),
  summary statistics (mean/variance/covariance/order statistics),
  remaining BRNGs and method codes.
- **FFT**: multi-dimensional, real-input variants, configurable
  scaling and storage formats.
- **Data fitting**: Hermite, Bessel, Akima, lookup / linear /
  step-function spline types; user-supplied derivatives;
  `?SearchCells1D` and the cell-based interpolators.

### 0.3.0 — Comprehensive coverage of three priority domains

After 0.2, push BLAS, LAPACK, and Sparse to Comprehensive — these
account for the vast majority of downstream use:

- **BLAS Comprehensive**: compact BLAS (`mkl_?gemm_compact`,
  `mkl_?trsm_compact`, `mkl_?potrf_compact`, etc.), mixed-precision
  GEMMs (`bf16bf16f32`, `f16f16f32`, `e5m2e5m2f32`, `e4m3e4m3f32`),
  full JIT GEMM lifecycle, packed compute (`cblas_gemm_*_pack`,
  `cblas_gemm_*_compute`).
- **LAPACK Comprehensive**: full auxiliary routine set (`?lacgv`,
  `?lacrm`, `?syconv`, `?larfg`, `?larft`, `?larfb`, `?lacpy`, etc.),
  Netlib-compatibility additions, `?tppack` / `?tpunpack`.
- **Sparse Comprehensive**: matrix manipulation (`mkl_sparse_*_add`,
  `mkl_sparse_*_spmm`, `mkl_sparse_*_spmmd`), inspector-executor
  analysis routines, full export API.

### 0.4.0 — Domain-completion sweep

Push all remaining MVP domains to Common+ and start filling in the
specialized ones:

- **Optimization**: full RCI surface (`?trnlsp_check`, `?trnlsp_get`
  with all return shapes), `?jacobi`/`?jacobix` direct callers.
- **Service**: CBWR (`mkl_cbwr_*`), `mkl_set_progress`,
  `mkl_set_xerbla`, `mkl_set_pardiso_pivot`,
  `mkl_enable_instructions`, single-dynamic-library control.
- **PDE Support**: trigonometric transforms (Cartesian + spherical
  Poisson solver, sine/cosine/staggered-cosine).
- **Compact BLAS / LAPACK**: full set (`mkl_?gepack_compact`,
  `mkl_?geunpack_compact`, `mkl_?get_size_compact`, `*_potrf`,
  `*_getrfnp`, `*_geqrf`, `*_getrinp`).

### 0.5.0 — All domains Comprehensive (single-node)

Every non-MPI domain at Comprehensive:

- **FFT Comprehensive**: every storage format, transform type, and
  config flag DFTI exposes.
- **Data Fitting Comprehensive**: integration call-back form, cell
  search call-backs, all spline editors.
- **VSL Comprehensive**: every BRNG + every method code; summary
  statistics with all weights / outliers / partial-task variants.
- **FEAST Comprehensive**: full RCI interface.

### 0.6.0 — MPI domains

Wrap the MPI-gated domains behind their existing opt-in features:

- **ScaLAPACK**: distributed dense computational + driver routines.
- **PBLAS**: parallel BLAS L1 / L2 / L3.
- **BLACS**: communication primitives.
- **Cluster FFT**: distributed DFTI.
- **Cluster PARDISO**: distributed direct sparse solve.

### 1.0.0

- All domains at Comprehensive.
- API frozen; SemVer commitments documented.
- Every `unsafe` block reviewed; safety invariants documented at the
  trait-method level.
- Tests run against multiple oneMKL versions in CI.
- Bench suite landed for the hot-path routines.

## Out of scope

These are explicit non-goals — we do not intend to wrap them.

- **DPC++ / SYCL APIs** — those are a C++ surface; users wanting GPU
  acceleration through a Rust wrapper should look at vendor-specific
  routes (CUDA, ROCm) instead.
- **OpenMP offload variants** (`*_omp_offload.h`) — same reasoning.
- **Fortran-style `LAPACK_*` wrappers** when LAPACKE provides the
  equivalent — we standardize on the LAPACKE interface throughout.
- **Routines users can build themselves trivially** with the existing
  primitives (e.g., wrappers that just pre-compute argument values).
