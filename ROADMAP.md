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
| BLAS-like extensions | Common | `axpby`, `imatcopy`/`omatcopy`/`omatcopy2`/`omatadd`, batched (strided) `gemm`/`trsm`/`gemv`/`dgmm`/`axpy`/`copy`/`gemm3m`; pointer-array `gemm_batch` / `gemv_batch` / `trsm_batch` |
| LAPACK | Common | Linear solve, QR, LS, eigenvalue (incl. RRR + D&C), SVD, banded, packed, generalized |
| Sparse BLAS (IE) | Common | CSR / COO / CSC / BSR construction; `mv`/`mm`/`trsv`/`optimize` + Sparse QR factor / solve; copy / convert / order / mv-mm-sv hint setters |
| PARDISO | Common | Factor + solve, multi-RHS, cached factorization, diagonal extraction, save/restore handle, low-level `export`, user `perm`, `pardiso_64_raw`, custom pivot callback |
| DSS | Common | Real + complex (single + double precision) factor / solve, multi-RHS, statistics (timing / memory / determinant / inertia) |
| ISS (CG, FGMRES) | Common | Closure-driven mat-vec; preconditioned CG; `IssResult` with iterations / residual norms / stop reason; full RCI surface via `CgSession` / `FgmresSession` |
| Preconditioners | Common | ILU0, ILUT, plus `apply_ilu` for two-step triangular solve. ILU0 / ILUT are MKL's only RCI preconditioners — no SSOR / Jacobi / Chebyshev exposed. |
| FEAST | Common | Dense / CSR / banded standard problems; dense generalized problem; real + complex; RCI still TODO |
| VM (Vector Math) | Common | All major function families |
| RNG (VSL) | Common | RAII `Stream` + 8 continuous + 4 discrete distributions; 1-D convolution / correlation tasks; summary statistics (mean / variance / min / max / sum) |
| FFT (DFTI) | Common | 1-D / 2-D / 3-D / N-D complex; real-input (CCE) variants; configurable forward / backward scaling |
| Data fitting | Common | Natural / Bessel / Akima / Hermite cubic splines, with interpolate / integrate |
| Optimization | Common | TRNLS, TRNLSPBC with initial / final residual norms; RCI numerical Jacobian; low-level `djacobi_with_callback` direct-FFI Jacobian |
| PDE Support | MVP+ | DCT / DST trigonometric transforms (sine, cosine, four staggered variants) for spectral PDE solvers |
| Service | Common | Version, threading, memory, verbose, finalize |

170 tests pass workspace-wide; each test maps directly to a wrapped
routine or trait method.

## Plans

### 0.2.0 — All domains at Common

Move every domain currently at MVP up to Common. Specifically:

- **BLAS-like extensions**: ~~pointer-array `?gemm_batch`,
  `?gemv_batch`, `?trsm_batch`~~ (done — `gemm_batch` /
  `gemv_batch` / `trsm_batch` plus matching `*BatchGroup` structs);
  pack/compute API (`?gemm_pack`, `?gemm_compute`), JIT
  (`mkl_jit_create_?gemm`).
- **LAPACK**: banded variants (`?gbsv`, `?gbtrf`/`?gbtrs`,
  `?pbsv`/`?pbtrf`/`?pbtrs`), packed variants (`?spsv`/`?sptrf`,
  `?ppsv`/`?pptrf`), generalized eigenvalue (`?ggev`, `?sygv`,
  `?hegv`), expert / RRR drivers (`?gesvx`, `?syevr`, `?heevr`),
  utility routines (`?lange`, `?gecon`, `?syrfs`).
- **Sparse BLAS**: ~~COO, CSC, BSR formats~~ (done — `from_coo` /
  `from_csc` / `from_bsr` constructors); ~~`mkl_sparse_copy`,
  `mkl_sparse_convert_*`, `mkl_sparse_order`; the analysis-stage hint
  setters (`mkl_sparse_set_mv_hint`, etc.)~~ (done — `copy`,
  `convert_{csr,csc,coo,bsr}`, `order`, `set_{mv,mm,sv}_hint`).
- **PARDISO**: ~~`pardiso_getdiag`, save/restore handle pair,
  `pardiso_export`, `perm` parameter, `pardiso_64`,
  `mkl_pardiso_pivot`~~ (all done — `with_diagonal_enabled` /
  `get_diagonal`, `save_handle` / `load_handle` /
  `delete_handle_files`, low-level `export`, `set_perm` / `perm`,
  `pardiso_64_raw`, `set_pardiso_pivot_callback`). The Schur
  extraction call sequence via `export` still needs to be nailed
  down (currently crashes inside MKL). High-level `Pardiso64<T>`
  state-managed wrapper and Schur fix remain as polish.
- **DSS**: ~~complex matrices, statistics readout~~ (done — complex
  factor/solve already covered, plus `factor_time` / `peak_memory_kb`
  / `determinant` / `inertia` etc.); generalized symmetric options.
- **ISS**: ~~expose `?cg_get`/`?fgmres_get` reasoning codes; full RCI
  surface for users who want to drive their own loop~~ (done —
  `IssResult` plus `CgSession` / `FgmresSession` with `step` →
  `RciAction { Converged, NeedMatVec, NeedPrecondition,
  StoppingTest, Other }`).
- **FEAST**: ~~CSR-input drivers (`?feast_scsrev`, `?feast_hcsrev`),
  banded (`?feast_sbev`, `?feast_hbev`), dense generalized
  (`?feast_sygv` / `?feast_hegv`)~~ (done — `eigh_real_csr` /
  `eigh_complex_csr` / `eigh_real_banded` / `eigh_complex_banded` /
  `gen_eigh_real_dense` / `gen_eigh_complex_dense`); CSR
  generalized (`?feast_scsrgv` / `?feast_hcsrgv`); banded
  generalized (`?feast_sbgv` / `?feast_hbgv`); RCI variants.
- **RNG**: ~~convolution and correlation tasks (VSL_*Conv*,
  VSL_*Corr*)~~ (done — `Conv1d` / `Corr1d` plus `convolve_1d` /
  `correlate_1d` free functions); ~~summary statistics
  (mean/variance)~~ (done — `SummaryStats::{mean, variance, min, max,
  sum}`); covariance / order statistics / quantiles; remaining BRNGs
  and method codes.
- **FFT**: ~~multi-dimensional, real-input variants, configurable
  scaling~~ (done — `complex_nd` / `real_nd` plus `*_with_scales`
  constructors); alternate real-input storage formats (CCS, PACK,
  PERM); per-axis stride / distance config.
- **Data fitting**: ~~Hermite, Bessel, Akima cubic splines;
  user-supplied derivatives~~ (done — `CubicSpline1d::{bessel,
  akima, hermite}`); lookup / linear / step-function spline types;
  `?SearchCells1D` and the cell-based interpolators.

### 0.3.0 — Comprehensive coverage of three priority domains

After 0.2, push BLAS, LAPACK, and Sparse to Comprehensive — these
account for the vast majority of downstream use:

- **BLAS Comprehensive**: ~~mixed-precision GEMMs (`bf16bf16f32`,
  `f16f16f32`, `e5m2e5m2f32`, `e4m3e4m3f32`)~~ (done — `gemm_bf16_f32`
  / `gemm_f16_f32` / `gemm_e5m2_f32` / `gemm_e4m3_f32` plus
  `gemm_s8u8_s32` / `gemm_s16_s32` / `hgemm` in
  `blas::mixed_precision`); ~~JIT GEMM lifecycle~~ (done — `JitGemm<T>`
  in `blas::jit`); ~~packed compute (`cblas_gemm_*_pack`,
  `cblas_gemm_*_compute`)~~ (done — `PackedMatrix<T>` plus
  `gemm_compute_packed_{a,b}` in `blas::packed`); ~~compact BLAS
  GEMM + pack / unpack lifecycle~~ (done — `blas::compact`); other
  compact kernels (`?trsm_compact`, `?potrf_compact`,
  `?getrfnp_compact`, `?geqrf_compact`).
- **LAPACK Comprehensive**: ~~core auxiliary routines (`?lacpy`,
  `?lange`, `?gecon`, `?larfg`, `?laswp`)~~ (done in
  `lapack::{lacpy, lange, gecon, laswp, larfg}`); remaining
  auxiliary set (`?lacgv`, `?lacrm`, `?syconv`, `?larft`, `?larfb`,
  etc.), Netlib-compatibility additions, `?tppack` / `?tpunpack`.
- **Sparse Comprehensive**: ~~matrix manipulation
  (`mkl_sparse_*_add`, `mkl_sparse_*_spmm`,
  `mkl_sparse_*_spmmd`)~~ (done — `SparseMatrix::{add, spmm,
  spmmd}`); `mkl_sparse_sp2m` (with request flags), `_dotmv`,
  `_symgs` / `_symgs_mv` (Gauss–Seidel sweep), `_syprd`
  (`A · diag · Aᵀ`), inspector-executor analysis routines, full
  export API.

### 0.4.0 — Domain-completion sweep

Push all remaining MVP domains to Common+ and start filling in the
specialized ones:

- **Optimization**: ~~`?trnlsp_get` with full return shape (initial /
  final residual norms), `?jacobi` direct caller~~ (done — residual
  norms in `TrnlsResult`, `djacobi_with_callback`); `?trnlsp_check`
  validation (currently subsumed by `_init`), `?jacobix` (Jacobian
  with central differences), session API for step-by-step driving.
- **Service**: CBWR (`mkl_cbwr_*`), `mkl_set_progress`,
  `mkl_set_xerbla`, `mkl_set_pardiso_pivot`,
  `mkl_enable_instructions`, single-dynamic-library control.
- **PDE Support**: ~~trigonometric transforms (sine / cosine /
  staggered)~~ (done — `TrigTransform<T>` in `pde`); Cartesian and
  spherical Poisson / Helmholtz solvers (`d_helmholtz_2d` /
  `d_helmholtz_3d`).
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
