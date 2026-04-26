//! Scalar dispatch trait for Inspector-Executor Sparse BLAS.

use core::ffi::c_int;

use num_complex::{Complex32, Complex64};
use onemkl_sys::{
    self as sys, matrix_descr, sparse_index_base_t, sparse_layout_t, sparse_matrix_t,
    sparse_operation_t, sparse_status_t,
};

use crate::scalar::Scalar;

/// Scalar types supported by the IE sparse BLAS.
///
/// Implemented for [`f32`], [`f64`], [`Complex32`], and [`Complex64`].
/// Methods are `unsafe` thunks that forward to the corresponding
/// `mkl_sparse_*` C symbol.
#[allow(missing_docs)]
pub trait SparseScalar: Scalar {
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_csr(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: c_int,
        cols: c_int,
        rows_start: *mut c_int,
        rows_end: *mut c_int,
        col_indx: *mut c_int,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    unsafe fn sparse_mv(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        x: *const Self,
        beta: Self,
        y: *mut Self,
    ) -> sparse_status_t::Type;

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_mm(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        layout: sparse_layout_t::Type,
        x: *const Self,
        columns: c_int,
        ldx: c_int,
        beta: Self,
        y: *mut Self,
        ldy: c_int,
    ) -> sparse_status_t::Type;

    unsafe fn sparse_trsv(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        x: *const Self,
        y: *mut Self,
    ) -> sparse_status_t::Type;
}

macro_rules! impl_sparse_real {
    ($ty:ty,
        create_csr=$create:ident, mv=$mv:ident, mm=$mm:ident, trsv=$trsv:ident,
    ) => {
        impl SparseScalar for $ty {
            unsafe fn sparse_create_csr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int, cols: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create(
                        a, indexing, rows, cols, rows_start, rows_end, col_indx, values,
                    )
                }
            }
            unsafe fn sparse_mv(
                op: sparse_operation_t::Type, alpha: Self,
                a: sparse_matrix_t, descr: matrix_descr,
                x: *const Self, beta: Self, y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$mv(op, alpha, a, descr, x, beta, y) }
            }
            unsafe fn sparse_mm(
                op: sparse_operation_t::Type, alpha: Self,
                a: sparse_matrix_t, descr: matrix_descr,
                layout: sparse_layout_t::Type,
                x: *const Self, columns: c_int, ldx: c_int,
                beta: Self, y: *mut Self, ldy: c_int,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$mm(op, alpha, a, descr, layout, x, columns, ldx, beta, y, ldy)
                }
            }
            unsafe fn sparse_trsv(
                op: sparse_operation_t::Type, alpha: Self,
                a: sparse_matrix_t, descr: matrix_descr,
                x: *const Self, y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$trsv(op, alpha, a, descr, x, y) }
            }
        }
    };
}

impl_sparse_real!(
    f32,
    create_csr=mkl_sparse_s_create_csr, mv=mkl_sparse_s_mv,
    mm=mkl_sparse_s_mm, trsv=mkl_sparse_s_trsv,
);
impl_sparse_real!(
    f64,
    create_csr=mkl_sparse_d_create_csr, mv=mkl_sparse_d_mv,
    mm=mkl_sparse_d_mm, trsv=mkl_sparse_d_trsv,
);

macro_rules! impl_sparse_complex {
    ($ty:ty,
        create_csr=$create:ident, mv=$mv:ident, mm=$mm:ident, trsv=$trsv:ident,
    ) => {
        impl SparseScalar for $ty {
            unsafe fn sparse_create_csr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int, cols: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create(
                        a, indexing, rows, cols, rows_start, rows_end, col_indx,
                        values.cast(),
                    )
                }
            }
            unsafe fn sparse_mv(
                op: sparse_operation_t::Type, alpha: Self,
                a: sparse_matrix_t, descr: matrix_descr,
                x: *const Self, beta: Self, y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$mv(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a, descr, x.cast(),
                        core::mem::transmute_copy(&beta),
                        y.cast(),
                    )
                }
            }
            unsafe fn sparse_mm(
                op: sparse_operation_t::Type, alpha: Self,
                a: sparse_matrix_t, descr: matrix_descr,
                layout: sparse_layout_t::Type,
                x: *const Self, columns: c_int, ldx: c_int,
                beta: Self, y: *mut Self, ldy: c_int,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$mm(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a, descr, layout, x.cast(), columns, ldx,
                        core::mem::transmute_copy(&beta),
                        y.cast(), ldy,
                    )
                }
            }
            unsafe fn sparse_trsv(
                op: sparse_operation_t::Type, alpha: Self,
                a: sparse_matrix_t, descr: matrix_descr,
                x: *const Self, y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$trsv(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a, descr, x.cast(), y.cast(),
                    )
                }
            }
        }
    };
}

impl_sparse_complex!(
    Complex32,
    create_csr=mkl_sparse_c_create_csr, mv=mkl_sparse_c_mv,
    mm=mkl_sparse_c_mm, trsv=mkl_sparse_c_trsv,
);
impl_sparse_complex!(
    Complex64,
    create_csr=mkl_sparse_z_create_csr, mv=mkl_sparse_z_mv,
    mm=mkl_sparse_z_mm, trsv=mkl_sparse_z_trsv,
);
