//! Inspector-Executor Sparse BLAS — sparse linear algebra with reusable
//! handles.
//!
//! Currently provides [`CsrMatrix`], an owned wrapper over a CSR-format
//! sparse matrix handle (`sparse_matrix_t`). It supports matrix-vector
//! and matrix-matrix multiplication, triangular solves, and the
//! `mkl_sparse_optimize` hint for repeated execution.
//!
//! ```no_run
//! use onemkl::sparse::{CsrMatrix, IndexBase, Operation, MatrixType};
//!
//! // 3x3 sparse matrix in CSR (zero-indexed):
//! //   [[1, 0, 0],
//! //    [0, 2, 0],
//! //    [0, 0, 3]]
//! let row_ptr = vec![0, 1, 2, 3];
//! let col_idx = vec![0, 1, 2];
//! let values  = vec![1.0_f64, 2.0, 3.0];
//! let mat = CsrMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
//!
//! let x = [1.0_f64; 3];
//! let mut y = [0.0_f64; 3];
//! mat.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
//! ```

mod scalar;

pub use scalar::SparseScalar;

use core::marker::PhantomData;
use core::ptr;

use onemkl_sys::{
    self as sys, matrix_descr, sparse_diag_type_t, sparse_fill_mode_t,
    sparse_index_base_t, sparse_layout_t, sparse_matrix_t, sparse_matrix_type_t,
    sparse_operation_t, sparse_status_t,
};

use crate::error::{Error, Result, SparseStatus};

// =====================================================================
// Public enums
// =====================================================================

/// Index base for sparse matrix indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexBase {
    /// 0-based indexing (C-style, default in Rust).
    Zero,
    /// 1-based indexing (Fortran-style).
    One,
}

impl IndexBase {
    #[inline]
    fn as_sys(self) -> sparse_index_base_t::Type {
        match self {
            Self::Zero => sparse_index_base_t::SPARSE_INDEX_BASE_ZERO,
            Self::One => sparse_index_base_t::SPARSE_INDEX_BASE_ONE,
        }
    }
}

/// Operation applied to the sparse matrix in a routine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Operation {
    /// Use the matrix as-is.
    #[default]
    NoTrans,
    /// Use the transpose.
    Trans,
    /// Use the conjugate transpose (only meaningful for complex types).
    ConjTrans,
}

impl Operation {
    #[inline]
    fn as_sys(self) -> sparse_operation_t::Type {
        match self {
            Self::NoTrans => sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE,
            Self::Trans => sparse_operation_t::SPARSE_OPERATION_TRANSPOSE,
            Self::ConjTrans => sparse_operation_t::SPARSE_OPERATION_CONJUGATE_TRANSPOSE,
        }
    }
}

/// Type of sparse matrix used to build a [`MatrixDescr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixType {
    /// General (non-symmetric, non-triangular).
    General,
    /// Symmetric.
    Symmetric,
    /// Hermitian.
    Hermitian,
    /// Triangular (combine with `FillMode` and `DiagType`).
    Triangular,
    /// Diagonal.
    Diagonal,
}

impl MatrixType {
    #[inline]
    fn as_sys(self) -> sparse_matrix_type_t::Type {
        match self {
            Self::General => sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL,
            Self::Symmetric => sparse_matrix_type_t::SPARSE_MATRIX_TYPE_SYMMETRIC,
            Self::Hermitian => sparse_matrix_type_t::SPARSE_MATRIX_TYPE_HERMITIAN,
            Self::Triangular => sparse_matrix_type_t::SPARSE_MATRIX_TYPE_TRIANGULAR,
            Self::Diagonal => sparse_matrix_type_t::SPARSE_MATRIX_TYPE_DIAGONAL,
        }
    }
}

/// Which triangle of a triangular / symmetric / Hermitian sparse matrix
/// is referenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FillMode {
    /// Lower triangle.
    #[default]
    Lower,
    /// Upper triangle.
    Upper,
    /// Full (used for general matrices).
    Full,
}

impl FillMode {
    #[inline]
    fn as_sys(self) -> sparse_fill_mode_t::Type {
        match self {
            Self::Lower => sparse_fill_mode_t::SPARSE_FILL_MODE_LOWER,
            Self::Upper => sparse_fill_mode_t::SPARSE_FILL_MODE_UPPER,
            Self::Full => sparse_fill_mode_t::SPARSE_FILL_MODE_FULL,
        }
    }
}

/// Whether a triangular matrix has implicit unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DiagType {
    /// Diagonal entries are read from the matrix.
    #[default]
    NonUnit,
    /// Diagonal entries are assumed to be 1 and not referenced.
    Unit,
}

impl DiagType {
    #[inline]
    fn as_sys(self) -> sparse_diag_type_t::Type {
        match self {
            Self::NonUnit => sparse_diag_type_t::SPARSE_DIAG_NON_UNIT,
            Self::Unit => sparse_diag_type_t::SPARSE_DIAG_UNIT,
        }
    }
}

/// Layout for dense operands in matrix-matrix routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DenseLayout {
    /// Row-major.
    #[default]
    RowMajor,
    /// Column-major.
    ColMajor,
}

impl DenseLayout {
    #[inline]
    fn as_sys(self) -> sparse_layout_t::Type {
        match self {
            Self::RowMajor => sparse_layout_t::SPARSE_LAYOUT_ROW_MAJOR,
            Self::ColMajor => sparse_layout_t::SPARSE_LAYOUT_COLUMN_MAJOR,
        }
    }
}

/// A descriptor used to specialize sparse routines for the structure
/// of the matrix at execution time.
#[derive(Debug, Clone, Copy)]
pub struct MatrixDescr {
    inner: matrix_descr,
}

impl MatrixDescr {
    /// General matrix descriptor.
    #[inline]
    #[must_use]
    pub fn general() -> Self {
        Self {
            inner: matrix_descr {
                type_: MatrixType::General.as_sys(),
                mode: FillMode::Full.as_sys(),
                diag: DiagType::NonUnit.as_sys(),
            },
        }
    }

    /// Triangular matrix descriptor.
    #[inline]
    #[must_use]
    pub fn triangular(fill: FillMode, diag: DiagType) -> Self {
        Self {
            inner: matrix_descr {
                type_: MatrixType::Triangular.as_sys(),
                mode: fill.as_sys(),
                diag: diag.as_sys(),
            },
        }
    }

    /// Symmetric matrix descriptor.
    #[inline]
    #[must_use]
    pub fn symmetric(fill: FillMode) -> Self {
        Self {
            inner: matrix_descr {
                type_: MatrixType::Symmetric.as_sys(),
                mode: fill.as_sys(),
                diag: DiagType::NonUnit.as_sys(),
            },
        }
    }

    /// Hermitian matrix descriptor.
    #[inline]
    #[must_use]
    pub fn hermitian(fill: FillMode) -> Self {
        Self {
            inner: matrix_descr {
                type_: MatrixType::Hermitian.as_sys(),
                mode: fill.as_sys(),
                diag: DiagType::NonUnit.as_sys(),
            },
        }
    }
}

impl From<MatrixType> for MatrixDescr {
    fn from(t: MatrixType) -> Self {
        Self {
            inner: matrix_descr {
                type_: t.as_sys(),
                mode: FillMode::Full.as_sys(),
                diag: DiagType::NonUnit.as_sys(),
            },
        }
    }
}

// =====================================================================
// CsrMatrix
// =====================================================================

/// Owned sparse matrix in CSR format.
///
/// Stores the underlying `row_ptr`, `col_idx`, and `values` arrays
/// internally; MKL's handle borrows from these for the lifetime of the
/// `CsrMatrix`. Drop releases both the MKL handle and the Rust-owned
/// buffers.
pub struct CsrMatrix<T: SparseScalar> {
    handle: sparse_matrix_t,
    // The buffers below back the MKL handle. They are kept Boxed and
    // owned by this struct, with MKL holding `*mut` pointers into them.
    // Do not access them directly — even reads could race with MKL's
    // executor stage.
    _row_ptr: Box<[i32]>,
    _col_idx: Box<[i32]>,
    _values: Box<[T]>,
    rows: usize,
    cols: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: SparseScalar + Send> Send for CsrMatrix<T> {}

impl<T: SparseScalar> CsrMatrix<T> {
    /// Build a CSR matrix from standard 3-array CSR storage.
    ///
    /// `row_ptr` has length `rows + 1`. `col_idx` and `values` have
    /// length equal to the number of non-zeros (i.e. `row_ptr[rows]`
    /// for zero-indexing).
    pub fn from_csr(
        rows: usize,
        cols: usize,
        indexing: IndexBase,
        row_ptr: Vec<i32>,
        col_idx: Vec<i32>,
        values: Vec<T>,
    ) -> Result<Self> {
        if row_ptr.len() != rows + 1 {
            return Err(Error::InvalidArgument(
                "row_ptr must have length rows + 1",
            ));
        }
        if col_idx.len() != values.len() {
            return Err(Error::InvalidArgument(
                "col_idx and values must have the same length",
            ));
        }
        let m: i32 = rows.try_into().map_err(|_| Error::DimensionOverflow)?;
        let n: i32 = cols.try_into().map_err(|_| Error::DimensionOverflow)?;

        let mut row_ptr_box = row_ptr.into_boxed_slice();
        let mut col_idx_box = col_idx.into_boxed_slice();
        let mut values_box = values.into_boxed_slice();

        let mut handle: sparse_matrix_t = ptr::null_mut();
        // For 3-array CSR, rows_start[i] = row_ptr[i] and rows_end[i] = row_ptr[i+1].
        let rows_start_ptr = row_ptr_box.as_mut_ptr();
        let rows_end_ptr = unsafe { rows_start_ptr.add(1) };

        let status = unsafe {
            T::sparse_create_csr(
                &mut handle,
                indexing.as_sys(),
                m,
                n,
                rows_start_ptr,
                rows_end_ptr,
                col_idx_box.as_mut_ptr(),
                values_box.as_mut_ptr(),
            )
        };
        check_sparse(status)?;

        Ok(Self {
            handle,
            _row_ptr: row_ptr_box,
            _col_idx: col_idx_box,
            _values: values_box,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Number of rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Hint to MKL that the matrix will be used many times for `?_mv`.
    /// Calls `mkl_sparse_optimize`.
    pub fn optimize(&self) -> Result<()> {
        let status = unsafe { sys::mkl_sparse_optimize(self.handle) };
        check_sparse(status)
    }

    /// `y ← alpha * op(A) * x + beta * y`.
    pub fn mv(
        &self,
        op: Operation,
        alpha: T,
        descr: impl Into<MatrixDescr>,
        x: &[T],
        beta: T,
        y: &mut [T],
    ) -> Result<()> {
        let needed_x = match op {
            Operation::NoTrans => self.cols,
            Operation::Trans | Operation::ConjTrans => self.rows,
        };
        let needed_y = match op {
            Operation::NoTrans => self.rows,
            Operation::Trans | Operation::ConjTrans => self.cols,
        };
        if x.len() < needed_x {
            return Err(Error::InvalidArgument("x is too short for op(A)*x"));
        }
        if y.len() < needed_y {
            return Err(Error::InvalidArgument("y is too short for op(A)*x"));
        }
        let descr_inner = descr.into().inner;
        let status = unsafe {
            T::sparse_mv(
                op.as_sys(),
                alpha,
                self.handle,
                descr_inner,
                x.as_ptr(),
                beta,
                y.as_mut_ptr(),
            )
        };
        check_sparse(status)
    }

    /// `Y ← alpha * op(A) * X + beta * Y` for dense `X` and `Y`.
    /// `X` and `Y` are stored according to `layout`; `columns` is the
    /// number of columns of `X` / `Y`.
    #[allow(clippy::too_many_arguments)]
    pub fn mm(
        &self,
        op: Operation,
        alpha: T,
        descr: impl Into<MatrixDescr>,
        layout: DenseLayout,
        x: &[T],
        columns: usize,
        ldx: usize,
        beta: T,
        y: &mut [T],
        ldy: usize,
    ) -> Result<()> {
        let descr_inner = descr.into().inner;
        let status = unsafe {
            T::sparse_mm(
                op.as_sys(),
                alpha,
                self.handle,
                descr_inner,
                layout.as_sys(),
                x.as_ptr(),
                columns.try_into().map_err(|_| Error::DimensionOverflow)?,
                ldx.try_into().map_err(|_| Error::DimensionOverflow)?,
                beta,
                y.as_mut_ptr(),
                ldy.try_into().map_err(|_| Error::DimensionOverflow)?,
            )
        };
        check_sparse(status)
    }

    /// Solve a triangular system `op(A) * y = alpha * x`.
    pub fn trsv(
        &self,
        op: Operation,
        alpha: T,
        descr: impl Into<MatrixDescr>,
        x: &[T],
        y: &mut [T],
    ) -> Result<()> {
        if x.len() < self.rows || y.len() < self.rows {
            return Err(Error::InvalidArgument(
                "x and y must each have at least n entries",
            ));
        }
        let descr_inner = descr.into().inner;
        let status = unsafe {
            T::sparse_trsv(
                op.as_sys(),
                alpha,
                self.handle,
                descr_inner,
                x.as_ptr(),
                y.as_mut_ptr(),
            )
        };
        check_sparse(status)
    }
}

impl<T: SparseScalar> Drop for CsrMatrix<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::mkl_sparse_destroy(self.handle);
            }
        }
    }
}

// =====================================================================
// Helpers
// =====================================================================

#[inline]
fn check_sparse(status: sparse_status_t::Type) -> Result<()> {
    if status == sparse_status_t::SPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        let s = match status {
            sparse_status_t::SPARSE_STATUS_NOT_INITIALIZED => SparseStatus::NotInitialized,
            sparse_status_t::SPARSE_STATUS_ALLOC_FAILED => SparseStatus::AllocFailed,
            sparse_status_t::SPARSE_STATUS_INVALID_VALUE => SparseStatus::InvalidValue,
            sparse_status_t::SPARSE_STATUS_EXECUTION_FAILED => SparseStatus::ExecutionFailed,
            sparse_status_t::SPARSE_STATUS_INTERNAL_ERROR => SparseStatus::InternalError,
            sparse_status_t::SPARSE_STATUS_NOT_SUPPORTED => SparseStatus::NotSupported,
            other => SparseStatus::Unknown(other),
        };
        Err(Error::SparseStatus(s))
    }
}
