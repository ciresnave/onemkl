//! Smoke-test the `prelude` exports — make sure each common type is
//! reachable through `use onemkl::prelude::*;`.

use onemkl::prelude::*;

#[test]
#[cfg(feature = "blas")]
fn prelude_lets_us_run_a_gemm() {
    use onemkl::blas::level3::gemm;

    let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let b_data = vec![1.0_f64, 0.0, 0.0, 1.0];
    let mut c_data = vec![0.0_f64; 4];
    let a = MatrixRef::new(&a_data, 2, 2, Layout::RowMajor).unwrap();
    let b = MatrixRef::new(&b_data, 2, 2, Layout::RowMajor).unwrap();
    let mut c = MatrixMut::new(&mut c_data, 2, 2, Layout::RowMajor).unwrap();
    gemm(
        Transpose::NoTrans, Transpose::NoTrans,
        1.0, &a, &b, 0.0, &mut c,
    )
    .unwrap();
    // A * I = A.
    assert_eq!(c_data, a_data);
}

#[test]
#[cfg(feature = "sparse")]
fn prelude_brings_sparse_types_in_scope() {
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![1.0_f64, 2.0, 3.0];
    let mat: SparseMatrix<f64> =
        SparseMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
    let x = [1.0_f64; 3];
    let mut y = [0.0_f64; 3];
    mat.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
    assert_eq!(y, [1.0, 2.0, 3.0]);
}

#[test]
fn prelude_exports_error_and_result() {
    fn returns_result() -> Result<()> {
        Err(Error::DimensionOverflow)
    }
    assert!(returns_result().is_err());
}

#[test]
fn prelude_exports_matrix_views() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let view = MatrixRef::new(&data, 2, 2, Layout::RowMajor).unwrap();
    assert_eq!(view.rows(), 2);
    assert_eq!(view.cols(), 2);
}
