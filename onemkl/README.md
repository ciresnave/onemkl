# onemkl

Safe, idiomatic Rust wrapper over [Intel oneAPI Math Kernel Library (oneMKL)][onemkl].

Built on top of [`onemkl-sys`](../onemkl-sys), which provides the raw FFI
bindings.

## Status

Coverage is being built out incrementally; this crate is not yet a
complete wrapper of every oneMKL domain. Currently implemented:

- **BLAS Level 1**: `asum`, `nrm2`, `iamax`, `iamin`, `dot`, `dotc`,
  `dotu`, `axpy`, `copy`, `swap`, `scal`, `scal_real`, `rot`, `rotg`,
  plus strided variants.
- **BLAS Level 2**: `gemv`.
- **BLAS Level 3**: `gemm`.

Anything not yet wrapped is reachable through `onemkl::sys` (i.e. the
raw `onemkl-sys` re-export).

Planned: LAPACK, VM (vector math), VSL (RNG / statistics), Sparse BLAS,
Sparse Solvers (PARDISO, DSS, RCI ISS), FFT (DFTI), Extended Eigensolver
(FEAST), Data Fitting, PDE Support, Nonlinear Optimization.

## Example

```rust
use onemkl::blas::level3::gemm;
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::{Layout, Transpose};

// 2x3 * 3x2 = 2x2.
let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let b = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
let mut c = [0.0_f64; 4];

let a = MatrixRef::new(&a, 2, 3, Layout::RowMajor)?;
let b = MatrixRef::new(&b, 3, 2, Layout::RowMajor)?;
let mut c = MatrixMut::new(&mut c, 2, 2, Layout::RowMajor)?;

gemm(Transpose::NoTrans, Transpose::NoTrans, 1.0, &a, &b, 0.0, &mut c)?;
# Ok::<(), onemkl::Error>(())
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.

[onemkl]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
