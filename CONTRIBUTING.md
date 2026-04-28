# Contributing to onemkl-rs

Thanks for your interest in contributing.

## Development setup

1. Install Intel oneAPI Math Kernel Library. The build script
   searches in this order:
   - `MKLROOT` environment variable
   - `ONEMKL_SYS_INCLUDE_DIR` / `ONEMKL_SYS_LIB_DIR` overrides
   - `ONEAPI_ROOT/mkl/latest`
   - Platform-standard install paths (`C:\Program Files (x86)\Intel\oneAPI`
     on Windows, `/opt/intel/oneapi` elsewhere)
2. Install LLVM / `libclang` for `bindgen` (set `LIBCLANG_PATH` to your
   LLVM `bin/` directory if not auto-detected).
3. A recent Rust toolchain (Edition 2024, MSRV 1.95).

## Build and test

```bash
cargo build --workspace
cargo test --workspace
cargo doc --workspace --no-deps
```

For a faster iteration cycle, scope to a single feature:

```bash
cargo test -p onemkl --features sparse --test sparse_arithmetic
```

At runtime, MKL's `bin/` (Windows) or `lib/intel64/` (Linux/macOS)
directory must be on the loader search path. On Linux this usually
means `source /opt/intel/oneapi/setvars.sh` before running tests.

## Adding a wrapper for a new MKL routine

1. Locate the routine in the bindgen-generated bindings under
   `target/debug/build/onemkl-sys-*/out/bindings.rs`.
2. If it's part of a scalar-generic family (real / complex), add a
   trait method to the appropriate `*Scalar` trait and wire it through
   the `impl_*!` macros for f32 / f64 / Complex32 / Complex64.
3. Add a public free function (or a method on the relevant struct)
   that takes safe Rust types, validates input dimensions, and
   forwards to the trait method.
4. Add at least one integration test under
   `onemkl/tests/<domain>_<routine>.rs` that exercises a known input
   and verifies the numerical result.
5. Update `ROADMAP.md` to mark the routine as done within its
   domain's planned work, and `CHANGELOG.md` under "Unreleased".

## Style notes

- Keep `unsafe` blocks small and targeted at the FFI call. The
  surrounding wrapper does the validation; the `unsafe` block does
  the call only.
- Numeric input dimensions (`m`, `n`, `k`, leading dims) are taken as
  `usize` in the public API and converted to `MKL_INT` internally via
  `dim_to_mkl_int`.
- Errors are returned as `Result<T, onemkl::Error>`; never panic on
  user input, only on internal invariant violations.
- Doc-link to the underlying MKL routine name (e.g., "Wraps
  `LAPACKE_dgesv`.") so users can grep the MKL reference.
- Output buffers are allocated by the caller when the size is known
  up front; allocate inside the wrapper only when MKL determines the
  size dynamically.
- For new types that hold MKL handles, implement `Drop` to free the
  handle, and document whether the type is `Send` / `Sync`.

## Submitting changes

- One logical change per PR.
- Run `cargo fmt --all` and `cargo clippy --workspace --all-features
  -- -D warnings` before opening the PR.
- Add a CHANGELOG entry under "Unreleased".
- The CI workflow runs `cargo build`, `cargo test`, `cargo doc`, and
  `cargo clippy` on Linux against the latest Intel oneAPI MKL.

## License

By contributing, you agree that your contributions will be dual-
licensed under MIT or Apache-2.0, matching the rest of the project.
