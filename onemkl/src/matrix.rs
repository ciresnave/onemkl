//! Strided matrix views.
//!
//! These types are the primary input / output for Level 2 and Level 3
//! BLAS routines. They borrow from a backing slice, carry shape and
//! layout information, and validate at construction time that the slice
//! is large enough for the declared dimensions.
//!
//! For raw vectors (Level 1 BLAS), use plain `&[T]` / `&mut [T]` instead
//! — there's no need for an extra view type.

use core::marker::PhantomData;

use crate::enums::Layout;
use crate::error::{Error, Result};

/// Read-only view of a 2-D matrix.
///
/// The underlying buffer is borrowed for the lifetime `'a`; the matrix
/// holds no ownership.
#[derive(Debug, Clone, Copy)]
pub struct MatrixRef<'a, T> {
    data: *const T,
    rows: usize,
    cols: usize,
    leading_dim: usize,
    layout: Layout,
    _marker: PhantomData<&'a [T]>,
}

/// Mutable view of a 2-D matrix.
///
/// Borrowed exclusively for the lifetime `'a`.
#[derive(Debug)]
pub struct MatrixMut<'a, T> {
    data: *mut T,
    rows: usize,
    cols: usize,
    leading_dim: usize,
    layout: Layout,
    _marker: PhantomData<&'a mut [T]>,
}

// SAFETY: the views are no more permissive than the underlying borrows;
// `Send`/`Sync` follow from `&[T]` / `&mut [T]` for `T: Send`/`T: Sync`.
unsafe impl<T: Sync> Send for MatrixRef<'_, T> {}
unsafe impl<T: Sync> Sync for MatrixRef<'_, T> {}
unsafe impl<T: Send> Send for MatrixMut<'_, T> {}
unsafe impl<T: Sync> Sync for MatrixMut<'_, T> {}

impl<'a, T> MatrixRef<'a, T> {
    /// Build a view over a slice with the natural (tightly-packed)
    /// leading dimension for `layout`:
    ///
    /// - [`Layout::RowMajor`] → `leading_dim = cols`
    /// - [`Layout::ColMajor`] → `leading_dim = rows`
    pub fn new(data: &'a [T], rows: usize, cols: usize, layout: Layout) -> Result<Self> {
        let lda = match layout {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        };
        Self::with_lda(data, rows, cols, layout, lda)
    }

    /// Build a view with an explicit leading dimension.
    ///
    /// The leading dimension must be at least as large as the contiguous
    /// dimension implied by `layout`:
    ///
    /// - [`Layout::RowMajor`] requires `leading_dim ≥ cols`
    /// - [`Layout::ColMajor`] requires `leading_dim ≥ rows`
    ///
    /// `data` must contain at least `(other - 1) * leading_dim + min` elements,
    /// where `min` is the contiguous dimension and `other` is the strided one.
    pub fn with_lda(
        data: &'a [T],
        rows: usize,
        cols: usize,
        layout: Layout,
        leading_dim: usize,
    ) -> Result<Self> {
        check_lda(rows, cols, layout, leading_dim, data.len())?;
        Ok(Self {
            data: data.as_ptr(),
            rows,
            cols,
            leading_dim,
            layout,
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
    /// Leading dimension (the stride between successive rows in row-major,
    /// or successive columns in column-major).
    #[inline]
    #[must_use]
    pub fn leading_dim(&self) -> usize {
        self.leading_dim
    }
    /// Storage layout.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// Raw pointer to the first element. Valid for `'a`.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data
    }
}

impl<'a, T> MatrixMut<'a, T> {
    /// Build a mutable view over a slice with the natural (tightly-packed)
    /// leading dimension for `layout`.
    pub fn new(data: &'a mut [T], rows: usize, cols: usize, layout: Layout) -> Result<Self> {
        let lda = match layout {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        };
        Self::with_lda(data, rows, cols, layout, lda)
    }

    /// Build a mutable view with an explicit leading dimension.
    pub fn with_lda(
        data: &'a mut [T],
        rows: usize,
        cols: usize,
        layout: Layout,
        leading_dim: usize,
    ) -> Result<Self> {
        check_lda(rows, cols, layout, leading_dim, data.len())?;
        Ok(Self {
            data: data.as_mut_ptr(),
            rows,
            cols,
            leading_dim,
            layout,
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
    /// Leading dimension.
    #[inline]
    #[must_use]
    pub fn leading_dim(&self) -> usize {
        self.leading_dim
    }
    /// Storage layout.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// Raw pointer to the first element. Valid for `'a`.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data
    }
    /// Raw mutable pointer to the first element. Valid for `'a`.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data
    }

    /// Re-borrow as an immutable view, without giving up the underlying
    /// mutable borrow.
    #[must_use]
    pub fn as_ref(&self) -> MatrixRef<'_, T> {
        MatrixRef {
            data: self.data,
            rows: self.rows,
            cols: self.cols,
            leading_dim: self.leading_dim,
            layout: self.layout,
            _marker: PhantomData,
        }
    }
}

fn check_lda(
    rows: usize,
    cols: usize,
    layout: Layout,
    leading_dim: usize,
    buf_len: usize,
) -> Result<()> {
    let (contiguous, strided) = match layout {
        Layout::RowMajor => (cols, rows),
        Layout::ColMajor => (rows, cols),
    };
    if leading_dim < contiguous {
        return Err(Error::InvalidArgument(
            "leading dimension is smaller than the contiguous dimension",
        ));
    }
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let needed = (strided - 1)
        .checked_mul(leading_dim)
        .and_then(|v| v.checked_add(contiguous))
        .ok_or(Error::DimensionOverflow)?;
    if buf_len < needed {
        return Err(Error::InvalidArgument(
            "backing slice is too small for the declared rows × cols × leading_dim",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_leading_dim_row_major() {
        let buf = vec![0.0_f32; 12];
        let m = MatrixRef::new(&buf, 3, 4, Layout::RowMajor).unwrap();
        assert_eq!(m.leading_dim(), 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
    }

    #[test]
    fn natural_leading_dim_col_major() {
        let buf = vec![0.0_f32; 12];
        let m = MatrixRef::new(&buf, 3, 4, Layout::ColMajor).unwrap();
        assert_eq!(m.leading_dim(), 3);
    }

    #[test]
    fn small_buffer_rejected() {
        let buf = vec![0.0_f32; 11];
        assert!(MatrixRef::new(&buf, 3, 4, Layout::RowMajor).is_err());
    }

    #[test]
    fn lda_smaller_than_contiguous_rejected() {
        let buf = vec![0.0_f32; 12];
        assert!(
            MatrixRef::with_lda(&buf, 3, 4, Layout::RowMajor, 3).is_err(),
            "lda < cols in row-major must be rejected"
        );
    }

    #[test]
    fn padded_lda_accepted() {
        let buf = vec![0.0_f32; 30]; // 3 rows × lda 10
        let m = MatrixRef::with_lda(&buf, 3, 4, Layout::RowMajor, 10).unwrap();
        assert_eq!(m.leading_dim(), 10);
    }
}
