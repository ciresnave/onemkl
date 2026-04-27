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

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_coo(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: c_int,
        cols: c_int,
        nnz: c_int,
        row_indx: *mut c_int,
        col_indx: *mut c_int,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_csc(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: c_int,
        cols: c_int,
        cols_start: *mut c_int,
        cols_end: *mut c_int,
        row_indx: *mut c_int,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_bsr(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        block_layout: onemkl_sys::sparse_layout_t::Type,
        rows: c_int,
        cols: c_int,
        block_size: c_int,
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

    unsafe fn sparse_add(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        alpha: Self,
        b: sparse_matrix_t,
        c: *mut sparse_matrix_t,
    ) -> sparse_status_t::Type;

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_dotmv(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        x: *const Self,
        beta: Self,
        y: *mut Self,
        d: *mut Self,
    ) -> sparse_status_t::Type;

    unsafe fn sparse_symgs(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        descr: matrix_descr,
        alpha: Self,
        b: *const Self,
        x: *mut Self,
    ) -> sparse_status_t::Type;

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_symgs_mv(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        descr: matrix_descr,
        alpha: Self,
        b: *const Self,
        x: *mut Self,
        y: *mut Self,
    ) -> sparse_status_t::Type;

    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_spmmd(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        b: sparse_matrix_t,
        layout: onemkl_sys::sparse_layout_t::Type,
        c: *mut Self,
        ldc: c_int,
    ) -> sparse_status_t::Type;
}

macro_rules! impl_sparse_real {
    ($ty:ty,
        create_csr=$create:ident, create_coo=$create_coo:ident,
        create_csc=$create_csc:ident, create_bsr=$create_bsr:ident,
        mv=$mv:ident, mm=$mm:ident, trsv=$trsv:ident,
        add=$add:ident, spmmd=$spmmd:ident,
        dotmv=$dotmv:ident, symgs=$symgs:ident, symgs_mv=$symgs_mv:ident,
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
            unsafe fn sparse_create_coo(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int, cols: c_int, nnz: c_int,
                row_indx: *mut c_int, col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_coo(
                        a, indexing, rows, cols, nnz, row_indx, col_indx, values,
                    )
                }
            }
            unsafe fn sparse_create_csc(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int, cols: c_int,
                cols_start: *mut c_int, cols_end: *mut c_int,
                row_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_csc(
                        a, indexing, rows, cols, cols_start, cols_end, row_indx, values,
                    )
                }
            }
            unsafe fn sparse_create_bsr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                block_layout: onemkl_sys::sparse_layout_t::Type,
                rows: c_int, cols: c_int, block_size: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_bsr(
                        a, indexing, block_layout, rows, cols, block_size,
                        rows_start, rows_end, col_indx, values,
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
            unsafe fn sparse_add(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                alpha: Self,
                b: sparse_matrix_t,
                c: *mut sparse_matrix_t,
            ) -> sparse_status_t::Type {
                unsafe { sys::$add(op, a, alpha, b, c) }
            }
            unsafe fn sparse_spmmd(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                b: sparse_matrix_t,
                layout: onemkl_sys::sparse_layout_t::Type,
                c: *mut Self,
                ldc: c_int,
            ) -> sparse_status_t::Type {
                unsafe { sys::$spmmd(op, a, b, layout, c, ldc) }
            }
            unsafe fn sparse_dotmv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                beta: Self,
                y: *mut Self,
                d: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$dotmv(op, alpha, a, descr, x, beta, y, d) }
            }
            unsafe fn sparse_symgs(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$symgs(op, a, descr, alpha, b, x) }
            }
            unsafe fn sparse_symgs_mv(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$symgs_mv(op, a, descr, alpha, b, x, y) }
            }
        }
    };
}

impl_sparse_real!(
    f32,
    create_csr=mkl_sparse_s_create_csr,
    create_coo=mkl_sparse_s_create_coo,
    create_csc=mkl_sparse_s_create_csc,
    create_bsr=mkl_sparse_s_create_bsr,
    mv=mkl_sparse_s_mv, mm=mkl_sparse_s_mm, trsv=mkl_sparse_s_trsv,
    add=mkl_sparse_s_add, spmmd=mkl_sparse_s_spmmd,
    dotmv=mkl_sparse_s_dotmv, symgs=mkl_sparse_s_symgs, symgs_mv=mkl_sparse_s_symgs_mv,
);
impl_sparse_real!(
    f64,
    create_csr=mkl_sparse_d_create_csr,
    create_coo=mkl_sparse_d_create_coo,
    create_csc=mkl_sparse_d_create_csc,
    create_bsr=mkl_sparse_d_create_bsr,
    mv=mkl_sparse_d_mv, mm=mkl_sparse_d_mm, trsv=mkl_sparse_d_trsv,
    add=mkl_sparse_d_add, spmmd=mkl_sparse_d_spmmd,
    dotmv=mkl_sparse_d_dotmv, symgs=mkl_sparse_d_symgs, symgs_mv=mkl_sparse_d_symgs_mv,
);

macro_rules! impl_sparse_complex {
    ($ty:ty,
        create_csr=$create:ident, create_coo=$create_coo:ident,
        create_csc=$create_csc:ident, create_bsr=$create_bsr:ident,
        mv=$mv:ident, mm=$mm:ident, trsv=$trsv:ident,
        add=$add:ident, spmmd=$spmmd:ident,
        dotmv=$dotmv:ident, symgs=$symgs:ident, symgs_mv=$symgs_mv:ident,
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
            unsafe fn sparse_create_coo(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int, cols: c_int, nnz: c_int,
                row_indx: *mut c_int, col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_coo(
                        a, indexing, rows, cols, nnz, row_indx, col_indx, values.cast(),
                    )
                }
            }
            unsafe fn sparse_create_csc(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int, cols: c_int,
                cols_start: *mut c_int, cols_end: *mut c_int,
                row_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_csc(
                        a, indexing, rows, cols, cols_start, cols_end, row_indx,
                        values.cast(),
                    )
                }
            }
            unsafe fn sparse_create_bsr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                block_layout: onemkl_sys::sparse_layout_t::Type,
                rows: c_int, cols: c_int, block_size: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_bsr(
                        a, indexing, block_layout, rows, cols, block_size,
                        rows_start, rows_end, col_indx, values.cast(),
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
            unsafe fn sparse_add(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                alpha: Self,
                b: sparse_matrix_t,
                c: *mut sparse_matrix_t,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$add(op, a, core::mem::transmute_copy(&alpha), b, c)
                }
            }
            unsafe fn sparse_spmmd(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                b: sparse_matrix_t,
                layout: onemkl_sys::sparse_layout_t::Type,
                c: *mut Self,
                ldc: c_int,
            ) -> sparse_status_t::Type {
                unsafe { sys::$spmmd(op, a, b, layout, c.cast(), ldc) }
            }
            unsafe fn sparse_dotmv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                beta: Self,
                y: *mut Self,
                d: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$dotmv(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a, descr, x.cast(),
                        core::mem::transmute_copy(&beta),
                        y.cast(), d.cast(),
                    )
                }
            }
            unsafe fn sparse_symgs(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$symgs(
                        op, a, descr,
                        core::mem::transmute_copy(&alpha),
                        b.cast(), x.cast(),
                    )
                }
            }
            unsafe fn sparse_symgs_mv(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$symgs_mv(
                        op, a, descr,
                        core::mem::transmute_copy(&alpha),
                        b.cast(), x.cast(), y.cast(),
                    )
                }
            }
        }
    };
}

impl_sparse_complex!(
    Complex32,
    create_csr=mkl_sparse_c_create_csr,
    create_coo=mkl_sparse_c_create_coo,
    create_csc=mkl_sparse_c_create_csc,
    create_bsr=mkl_sparse_c_create_bsr,
    mv=mkl_sparse_c_mv, mm=mkl_sparse_c_mm, trsv=mkl_sparse_c_trsv,
    add=mkl_sparse_c_add, spmmd=mkl_sparse_c_spmmd,
    dotmv=mkl_sparse_c_dotmv, symgs=mkl_sparse_c_symgs, symgs_mv=mkl_sparse_c_symgs_mv,
);
impl_sparse_complex!(
    Complex64,
    create_csr=mkl_sparse_z_create_csr,
    create_coo=mkl_sparse_z_create_coo,
    create_csc=mkl_sparse_z_create_csc,
    create_bsr=mkl_sparse_z_create_bsr,
    mv=mkl_sparse_z_mv, mm=mkl_sparse_z_mm, trsv=mkl_sparse_z_trsv,
    add=mkl_sparse_z_add, spmmd=mkl_sparse_z_spmmd,
    dotmv=mkl_sparse_z_dotmv, symgs=mkl_sparse_z_symgs, symgs_mv=mkl_sparse_z_symgs_mv,
);
