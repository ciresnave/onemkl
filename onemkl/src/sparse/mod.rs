//! Inspector-Executor Sparse BLAS — sparse linear algebra with reusable
//! handles.
//!
//! Provides [`SparseMatrix`] (also exported as `CsrMatrix` for backward
//! compatibility), an owned wrapper over a sparse matrix handle
//! (`sparse_matrix_t`). The handle can be built from CSR, COO, CSC, or
//! BSR storage via the corresponding `from_*` constructor and supports
//! matrix-vector / matrix-matrix multiplication, triangular solves, and
//! the `mkl_sparse_optimize` hint for repeated execution.
//!
//! ```no_run
//! use onemkl::sparse::{SparseMatrix, IndexBase, Operation, MatrixType};
//!
//! // 3x3 sparse matrix in CSR (zero-indexed):
//! //   [[1, 0, 0],
//! //    [0, 2, 0],
//! //    [0, 0, 3]]
//! let row_ptr = vec![0, 1, 2, 3];
//! let col_idx = vec![0, 1, 2];
//! let values  = vec![1.0_f64, 2.0, 3.0];
//! let mat = SparseMatrix::from_csr(3, 3, IndexBase::Zero, row_ptr, col_idx, values).unwrap();
//!
//! let x = [1.0_f64; 3];
//! let mut y = [0.0_f64; 3];
//! mat.mv(Operation::NoTrans, 1.0, MatrixType::General, &x, 0.0, &mut y).unwrap();
//! ```

mod scalar;

pub use scalar::SparseScalar;
// `RealSparseScalar` is defined further down in this file.

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

/// Storage layout of the dense blocks within a BSR matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlockLayout {
    /// Each block stored row-major.
    #[default]
    RowMajor,
    /// Each block stored column-major.
    ColMajor,
}

impl BlockLayout {
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
// SparseMatrix
// =====================================================================

/// Owned sparse matrix handle.
///
/// The handle can be built from any of the supported MKL formats — CSR,
/// COO, CSC, or BSR — via the corresponding `from_*` constructor. MKL
/// stores a generic handle internally and supports the same set of
/// executor routines (`mv`, `mm`, `trsv`, …) regardless of the source
/// format. The handle borrows from the index/value buffers passed in,
/// which this struct owns for the lifetime of the matrix.
pub struct SparseMatrix<T: SparseScalar> {
    handle: sparse_matrix_t,
    // The buffers below back the MKL handle. They are kept Boxed and
    // owned by this struct, with MKL holding `*mut` pointers into them.
    // Do not access them directly — even reads could race with MKL's
    // executor stage.
    _idx_a: Box<[i32]>,
    _idx_b: Box<[i32]>,
    _values: Box<[T]>,
    rows: usize,
    cols: usize,
    _marker: PhantomData<T>,
}

/// Compatibility alias for [`SparseMatrix`].
pub type CsrMatrix<T> = SparseMatrix<T>;

unsafe impl<T: SparseScalar + Send> Send for SparseMatrix<T> {}

impl<T: SparseScalar> SparseMatrix<T> {
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
            _idx_a: row_ptr_box,
            _idx_b: col_idx_box,
            _values: values_box,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Build a sparse matrix from coordinate-format storage.
    ///
    /// `row_indx` and `col_indx` give the row/column of each non-zero;
    /// they and `values` all have length `nnz`.
    pub fn from_coo(
        rows: usize,
        cols: usize,
        indexing: IndexBase,
        row_indx: Vec<i32>,
        col_indx: Vec<i32>,
        values: Vec<T>,
    ) -> Result<Self> {
        if row_indx.len() != col_indx.len() || col_indx.len() != values.len() {
            return Err(Error::InvalidArgument(
                "row_indx, col_indx, and values must all have the same length",
            ));
        }
        let m: i32 = rows.try_into().map_err(|_| Error::DimensionOverflow)?;
        let n: i32 = cols.try_into().map_err(|_| Error::DimensionOverflow)?;
        let nnz: i32 = values.len().try_into().map_err(|_| Error::DimensionOverflow)?;

        let mut row_box = row_indx.into_boxed_slice();
        let mut col_box = col_indx.into_boxed_slice();
        let mut values_box = values.into_boxed_slice();

        let mut handle: sparse_matrix_t = ptr::null_mut();
        let status = unsafe {
            T::sparse_create_coo(
                &mut handle,
                indexing.as_sys(),
                m,
                n,
                nnz,
                row_box.as_mut_ptr(),
                col_box.as_mut_ptr(),
                values_box.as_mut_ptr(),
            )
        };
        check_sparse(status)?;

        Ok(Self {
            handle,
            _idx_a: row_box,
            _idx_b: col_box,
            _values: values_box,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Build a sparse matrix from compressed-column (CSC) storage.
    ///
    /// `col_ptr` has length `cols + 1`; `row_indx` and `values` have
    /// length `nnz` (equal to `col_ptr[cols]` for zero-indexing).
    pub fn from_csc(
        rows: usize,
        cols: usize,
        indexing: IndexBase,
        col_ptr: Vec<i32>,
        row_indx: Vec<i32>,
        values: Vec<T>,
    ) -> Result<Self> {
        if col_ptr.len() != cols + 1 {
            return Err(Error::InvalidArgument("col_ptr must have length cols + 1"));
        }
        if row_indx.len() != values.len() {
            return Err(Error::InvalidArgument(
                "row_indx and values must have the same length",
            ));
        }
        let m: i32 = rows.try_into().map_err(|_| Error::DimensionOverflow)?;
        let n: i32 = cols.try_into().map_err(|_| Error::DimensionOverflow)?;

        let mut col_ptr_box = col_ptr.into_boxed_slice();
        let mut row_indx_box = row_indx.into_boxed_slice();
        let mut values_box = values.into_boxed_slice();

        let mut handle: sparse_matrix_t = ptr::null_mut();
        let cols_start_ptr = col_ptr_box.as_mut_ptr();
        let cols_end_ptr = unsafe { cols_start_ptr.add(1) };

        let status = unsafe {
            T::sparse_create_csc(
                &mut handle,
                indexing.as_sys(),
                m,
                n,
                cols_start_ptr,
                cols_end_ptr,
                row_indx_box.as_mut_ptr(),
                values_box.as_mut_ptr(),
            )
        };
        check_sparse(status)?;

        Ok(Self {
            handle,
            _idx_a: col_ptr_box,
            _idx_b: row_indx_box,
            _values: values_box,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Build a sparse matrix from block sparse row (BSR) storage.
    ///
    /// `block_size` is the side length of each square dense block.
    /// `row_ptr` has length `block_rows + 1`, where `block_rows = rows / block_size`.
    /// `col_idx` has length `nnz_blocks`. `values` has length
    /// `nnz_blocks * block_size * block_size`, with each block stored
    /// according to `block_layout`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bsr(
        rows: usize,
        cols: usize,
        block_size: usize,
        block_layout: BlockLayout,
        indexing: IndexBase,
        row_ptr: Vec<i32>,
        col_idx: Vec<i32>,
        values: Vec<T>,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(Error::InvalidArgument("block_size must be > 0"));
        }
        if rows % block_size != 0 || cols % block_size != 0 {
            return Err(Error::InvalidArgument(
                "rows and cols must be multiples of block_size",
            ));
        }
        let block_rows = rows / block_size;
        let block_cols = cols / block_size;
        if row_ptr.len() != block_rows + 1 {
            return Err(Error::InvalidArgument(
                "row_ptr must have length (rows / block_size) + 1",
            ));
        }
        let nnz_blocks = col_idx.len();
        if values.len() != nnz_blocks * block_size * block_size {
            return Err(Error::InvalidArgument(
                "values must have length col_idx.len() * block_size * block_size",
            ));
        }
        // MKL takes rows/cols in block units, not element units.
        let m: i32 = block_rows.try_into().map_err(|_| Error::DimensionOverflow)?;
        let n: i32 = block_cols.try_into().map_err(|_| Error::DimensionOverflow)?;
        let bs: i32 = block_size.try_into().map_err(|_| Error::DimensionOverflow)?;

        let mut row_ptr_box = row_ptr.into_boxed_slice();
        let mut col_idx_box = col_idx.into_boxed_slice();
        let mut values_box = values.into_boxed_slice();

        let mut handle: sparse_matrix_t = ptr::null_mut();
        let rows_start_ptr = row_ptr_box.as_mut_ptr();
        let rows_end_ptr = unsafe { rows_start_ptr.add(1) };

        let status = unsafe {
            T::sparse_create_bsr(
                &mut handle,
                indexing.as_sys(),
                block_layout.as_sys(),
                m,
                n,
                bs,
                rows_start_ptr,
                rows_end_ptr,
                col_idx_box.as_mut_ptr(),
                values_box.as_mut_ptr(),
            )
        };
        check_sparse(status)?;

        Ok(Self {
            handle,
            _idx_a: row_ptr_box,
            _idx_b: col_idx_box,
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

    /// Clone the matrix. The new handle and its internal storage are
    /// owned by the returned `SparseMatrix` and freed on drop. Wraps
    /// `mkl_sparse_copy`.
    pub fn copy(&self, descr: impl Into<MatrixDescr>) -> Result<Self> {
        let descr_inner = descr.into().inner;
        let mut dest: sparse_matrix_t = ptr::null_mut();
        let status = unsafe { sys::mkl_sparse_copy(self.handle, descr_inner, &mut dest) };
        check_sparse(status)?;
        Ok(Self::new_mkl_owned(dest, self.rows, self.cols))
    }

    /// Convert this matrix to CSR storage. If `op` is
    /// `Operation::Trans` or `ConjTrans`, the conversion implicitly
    /// transposes. Wraps `mkl_sparse_convert_csr`.
    pub fn convert_csr(&self, op: Operation) -> Result<Self> {
        let mut dest: sparse_matrix_t = ptr::null_mut();
        let status =
            unsafe { sys::mkl_sparse_convert_csr(self.handle, op.as_sys(), &mut dest) };
        check_sparse(status)?;
        let (rows, cols) = transposed_shape(op, self.rows, self.cols);
        Ok(Self::new_mkl_owned(dest, rows, cols))
    }

    /// Convert this matrix to CSC storage. Wraps
    /// `mkl_sparse_convert_csc`.
    pub fn convert_csc(&self, op: Operation) -> Result<Self> {
        let mut dest: sparse_matrix_t = ptr::null_mut();
        let status =
            unsafe { sys::mkl_sparse_convert_csc(self.handle, op.as_sys(), &mut dest) };
        check_sparse(status)?;
        let (rows, cols) = transposed_shape(op, self.rows, self.cols);
        Ok(Self::new_mkl_owned(dest, rows, cols))
    }

    /// Convert this matrix to COO storage. Wraps
    /// `mkl_sparse_convert_coo`.
    pub fn convert_coo(&self, op: Operation) -> Result<Self> {
        let mut dest: sparse_matrix_t = ptr::null_mut();
        let status =
            unsafe { sys::mkl_sparse_convert_coo(self.handle, op.as_sys(), &mut dest) };
        check_sparse(status)?;
        let (rows, cols) = transposed_shape(op, self.rows, self.cols);
        Ok(Self::new_mkl_owned(dest, rows, cols))
    }

    /// Convert this matrix to BSR storage with the given `block_size`
    /// and per-block storage layout. Wraps `mkl_sparse_convert_bsr`.
    pub fn convert_bsr(
        &self,
        block_size: usize,
        block_layout: BlockLayout,
        op: Operation,
    ) -> Result<Self> {
        let bs: core::ffi::c_int =
            block_size.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut dest: sparse_matrix_t = ptr::null_mut();
        let status = unsafe {
            sys::mkl_sparse_convert_bsr(
                self.handle,
                bs,
                block_layout.as_sys(),
                op.as_sys(),
                &mut dest,
            )
        };
        check_sparse(status)?;
        let (rows, cols) = transposed_shape(op, self.rows, self.cols);
        Ok(Self::new_mkl_owned(dest, rows, cols))
    }

    /// Sort column indices within each row (CSR / BSR) or row indices
    /// within each column (CSC) so that subsequent executor routines
    /// can take faster paths. Wraps `mkl_sparse_order`.
    pub fn order(&self) -> Result<()> {
        let status = unsafe { sys::mkl_sparse_order(self.handle) };
        check_sparse(status)
    }

    /// Hint that `mv` will be called approximately `expected_calls`
    /// times with the given operation and descriptor. Used by
    /// `optimize` to choose internal layouts. Wraps
    /// `mkl_sparse_set_mv_hint`.
    pub fn set_mv_hint(
        &self,
        op: Operation,
        descr: impl Into<MatrixDescr>,
        expected_calls: usize,
    ) -> Result<()> {
        let calls: core::ffi::c_int = expected_calls
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let descr_inner = descr.into().inner;
        let status = unsafe {
            sys::mkl_sparse_set_mv_hint(self.handle, op.as_sys(), descr_inner, calls)
        };
        check_sparse(status)
    }

    /// Hint that `mm` will be called approximately `expected_calls`
    /// times. Wraps `mkl_sparse_set_mm_hint`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_mm_hint(
        &self,
        op: Operation,
        descr: impl Into<MatrixDescr>,
        layout: DenseLayout,
        dense_matrix_size: usize,
        expected_calls: usize,
    ) -> Result<()> {
        let dms: core::ffi::c_int = dense_matrix_size
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let calls: core::ffi::c_int = expected_calls
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let descr_inner = descr.into().inner;
        let status = unsafe {
            sys::mkl_sparse_set_mm_hint(
                self.handle,
                op.as_sys(),
                descr_inner,
                layout.as_sys(),
                dms,
                calls,
            )
        };
        check_sparse(status)
    }

    /// Hint that `trsv` will be called approximately `expected_calls`
    /// times. Wraps `mkl_sparse_set_sv_hint`.
    pub fn set_sv_hint(
        &self,
        op: Operation,
        descr: impl Into<MatrixDescr>,
        expected_calls: usize,
    ) -> Result<()> {
        let calls: core::ffi::c_int = expected_calls
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let descr_inner = descr.into().inner;
        let status = unsafe {
            sys::mkl_sparse_set_sv_hint(self.handle, op.as_sys(), descr_inner, calls)
        };
        check_sparse(status)
    }

    /// `C ← op(self) + alpha · other`, returning a freshly-allocated
    /// sparse matrix `C` whose storage is owned by MKL. Wraps
    /// `mkl_sparse_*_add`.
    pub fn add(
        &self,
        op: Operation,
        alpha: T,
        other: &Self,
    ) -> Result<Self> {
        let mut dest: sparse_matrix_t = ptr::null_mut();
        let status = unsafe {
            T::sparse_add(op.as_sys(), self.handle, alpha, other.handle, &mut dest)
        };
        check_sparse(status)?;
        let (rows, cols) = transposed_shape(op, self.rows, self.cols);
        Ok(Self::new_mkl_owned(dest, rows, cols))
    }

    /// `C ← op(self) · other` returning a sparse `C`. Wraps
    /// `mkl_sparse_spmm` (which dispatches generically by scalar
    /// type).
    pub fn spmm(&self, op: Operation, other: &Self) -> Result<Self> {
        // For A * B, result is (rows of op(A)) × (cols of B);
        // for Aᵀ * B, result is (cols of A) × (cols of B).
        let result_rows = match op {
            Operation::NoTrans => self.rows,
            Operation::Trans | Operation::ConjTrans => self.cols,
        };
        let result_cols = other.cols;
        let mut dest: sparse_matrix_t = ptr::null_mut();
        let status = unsafe {
            sys::mkl_sparse_spmm(op.as_sys(), self.handle, other.handle, &mut dest)
        };
        check_sparse(status)?;
        Ok(Self::new_mkl_owned(dest, result_rows, result_cols))
    }

    /// `C ← op(self) · other` returning a *dense* `C` in the supplied
    /// layout. Useful when the product is dense even though both
    /// operands are sparse (e.g. small sparse blocks producing a
    /// dense head). Wraps `mkl_sparse_*_spmmd`.
    ///
    /// `ldc` is the leading dimension of `C` in the chosen layout
    /// (≥ number of columns for row-major, ≥ number of rows for
    /// column-major).
    pub fn spmmd(
        &self,
        op: Operation,
        other: &Self,
        layout: DenseLayout,
        ldc: usize,
    ) -> Result<Vec<T>>
    where
        T: Default,
    {
        let result_rows = match op {
            Operation::NoTrans => self.rows,
            Operation::Trans | Operation::ConjTrans => self.cols,
        };
        let result_cols = other.cols;
        let needed = match layout {
            DenseLayout::RowMajor => result_rows.saturating_mul(ldc),
            DenseLayout::ColMajor => result_cols.saturating_mul(ldc),
        };
        let ldc_i: core::ffi::c_int =
            ldc.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..needed).map(|_| T::default()).collect();
        let status = unsafe {
            T::sparse_spmmd(
                op.as_sys(),
                self.handle,
                other.handle,
                layout.as_sys(),
                out.as_mut_ptr(),
                ldc_i,
            )
        };
        check_sparse(status)?;
        Ok(out)
    }

    /// Build a [`SparseMatrix`] around an MKL handle whose internal
    /// storage MKL owns (e.g. the result of copy or convert).
    fn new_mkl_owned(handle: sparse_matrix_t, rows: usize, cols: usize) -> Self {
        Self {
            handle,
            _idx_a: Box::new([]),
            _idx_b: Box::new([]),
            _values: Box::new([]),
            rows,
            cols,
            _marker: PhantomData,
        }
    }
}

#[inline]
fn transposed_shape(op: Operation, rows: usize, cols: usize) -> (usize, usize) {
    match op {
        Operation::NoTrans => (rows, cols),
        Operation::Trans | Operation::ConjTrans => (cols, rows),
    }
}

// =====================================================================
// Sparse QR — real-only (oneMKL doesn't ship complex variants)
// =====================================================================

/// Real-only sparse routines, currently the QR family.
#[allow(missing_docs)]
pub trait RealSparseScalar: SparseScalar {
    unsafe fn sparse_qr_factorize(
        a: sparse_matrix_t,
        alt_values: *mut Self,
    ) -> sparse_status_t::Type;

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_qr_solve(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        alt_values: *mut Self,
        layout: sparse_layout_t::Type,
        columns: core::ffi::c_int,
        x: *mut Self,
        ldx: core::ffi::c_int,
        b: *const Self,
        ldb: core::ffi::c_int,
    ) -> sparse_status_t::Type;
}

impl RealSparseScalar for f32 {
    unsafe fn sparse_qr_factorize(
        a: sparse_matrix_t,
        alt_values: *mut f32,
    ) -> sparse_status_t::Type {
        unsafe { sys::mkl_sparse_s_qr_factorize(a, alt_values) }
    }
    unsafe fn sparse_qr_solve(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        alt_values: *mut f32,
        layout: sparse_layout_t::Type,
        columns: core::ffi::c_int,
        x: *mut f32,
        ldx: core::ffi::c_int,
        b: *const f32,
        ldb: core::ffi::c_int,
    ) -> sparse_status_t::Type {
        unsafe {
            sys::mkl_sparse_s_qr_solve(op, a, alt_values, layout, columns, x, ldx, b, ldb)
        }
    }
}
impl RealSparseScalar for f64 {
    unsafe fn sparse_qr_factorize(
        a: sparse_matrix_t,
        alt_values: *mut f64,
    ) -> sparse_status_t::Type {
        unsafe { sys::mkl_sparse_d_qr_factorize(a, alt_values) }
    }
    unsafe fn sparse_qr_solve(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        alt_values: *mut f64,
        layout: sparse_layout_t::Type,
        columns: core::ffi::c_int,
        x: *mut f64,
        ldx: core::ffi::c_int,
        b: *const f64,
        ldb: core::ffi::c_int,
    ) -> sparse_status_t::Type {
        unsafe {
            sys::mkl_sparse_d_qr_solve(op, a, alt_values, layout, columns, x, ldx, b, ldb)
        }
    }
}

impl<T: RealSparseScalar> SparseMatrix<T> {
    /// Symbolic + numerical QR factorization of the matrix. Equivalent
    /// to calling `mkl_sparse_qr_reorder` followed by
    /// `mkl_sparse_?_qr_factorize`. After calling this, [`qr_solve`]
    /// can solve `A * x = b` and `Aᵀ * x = b` repeatedly.
    pub fn qr_factor(&self, descr: impl Into<MatrixDescr>) -> Result<()> {
        let descr_inner = descr.into().inner;
        let status = unsafe { sys::mkl_sparse_qr_reorder(self.handle, descr_inner) };
        check_sparse(status)?;
        let status = unsafe {
            T::sparse_qr_factorize(self.handle, core::ptr::null_mut())
        };
        check_sparse(status)
    }

    /// Solve `op(A) * X = B` using the cached QR factorization.
    /// Call [`qr_factor`] first.
    #[allow(clippy::too_many_arguments)]
    pub fn qr_solve(
        &self,
        op: Operation,
        layout: DenseLayout,
        b: &[T],
        columns: usize,
        ldx: usize,
        x: &mut [T],
        ldb: usize,
    ) -> Result<()> {
        let status = unsafe {
            T::sparse_qr_solve(
                op.as_sys(),
                self.handle,
                core::ptr::null_mut(),
                layout.as_sys(),
                columns.try_into().map_err(|_| Error::DimensionOverflow)?,
                x.as_mut_ptr(),
                ldx.try_into().map_err(|_| Error::DimensionOverflow)?,
                b.as_ptr(),
                ldb.try_into().map_err(|_| Error::DimensionOverflow)?,
            )
        };
        check_sparse(status)
    }
}

impl<T: SparseScalar> Drop for SparseMatrix<T> {
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
