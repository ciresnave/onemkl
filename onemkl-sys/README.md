# onemkl-sys

Raw, `unsafe` FFI bindings to the C interface of [Intel oneAPI Math Kernel Library (oneMKL)][onemkl].

For a safe, idiomatic Rust API, see the companion crate [`onemkl`](../onemkl).

## Requirements

Intel oneAPI Math Kernel Library, installed locally. The crate's `build.rs`
searches in this order:

1. `MKLROOT` environment variable
2. `ONEMKL_SYS_INCLUDE_DIR` / `ONEMKL_SYS_LIB_DIR` (explicit override)
3. `ONEAPI_ROOT/mkl/latest`
4. Platform-standard install paths

The recommended setup on Windows is to run Intel's `setvars.bat` once per
shell session before invoking `cargo build`, which sets `MKLROOT`,
`ONEAPI_ROOT`, and adds the runtime DLLs to `PATH`.

On Linux/macOS, the equivalent is `source /opt/intel/oneapi/setvars.sh`.

## Cargo features

Exactly one feature must be selected from each of:

| Group         | Features                                                                            |
| ------------- | ----------------------------------------------------------------------------------- |
| Interface     | `lp64` (default), `ilp64`                                                           |
| Threading     | `threading-sequential` (default), `threading-intel-openmp`, `threading-tbb`         |
| Linkage       | `link-dynamic` (default), `link-static`                                             |

Opt-in MPI-dependent domains (off by default): `cluster-sparse-solver`,
`cdft`, `scalapack`, `blacs`, `pblas`.

## License

Dual-licensed under MIT or Apache-2.0, at your option.

Note: this crate links against Intel oneMKL, which has its own license terms
(the [Intel Simplified Software License][isl]). It is your responsibility to
comply with those terms in any product that ships against this binding.

[onemkl]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
[isl]: https://www.intel.com/content/www/us/en/developer/articles/license/onemkl-license-faq.html
