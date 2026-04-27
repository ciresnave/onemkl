#![cfg(feature = "pardiso")]

//! Verify PARDISO direct sparse solver wrappers.

use approx::assert_abs_diff_eq;

use onemkl::pardiso::{
    pardiso_64_raw, set_pardiso_pivot_callback, IndexBase, Pardiso, PardisoMatrixType,
};

#[test]
fn solve_real_spd_3x3() {
    // SPD matrix:
    //   [[ 4, -1,  0],
    //    [-1,  4, -1],
    //    [ 0, -1,  4]]
    // PARDISO needs upper triangle for SPD storage.
    // CSR (1-based, upper triangle only): row 0 → cols [1,2], row 1 → cols [2,3], row 2 → col [3]
    //
    // ia (row pointers, 1-based, length n+1=4):
    //   row 0 starts at index 1, row 1 at 3, row 2 at 5, end at 6
    let ia = vec![1_i32, 3, 5, 6];
    // ja (column indices, 1-based, length nnz=5)
    let ja = vec![1_i32, 2, 2, 3, 3];
    // values (matching ja order)
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    // RHS: A * x_true with x_true = [1, 1, 1] = [3, 2, 3]
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
}

#[test]
fn solve_real_unsymmetric_3x3() {
    // Asymmetric matrix:
    //   [[2, 1, 0],
    //    [0, 3, 1],
    //    [1, 0, 4]]
    // Stored as full CSR (1-based).
    // row 0: cols [1, 2], values [2, 1]
    // row 1: cols [2, 3], values [3, 1]
    // row 2: cols [1, 3], values [1, 4]
    let ia = vec![1_i32, 3, 5, 7];
    let ja = vec![1_i32, 2, 2, 3, 1, 3];
    let a = vec![2.0_f64, 1.0, 3.0, 1.0, 1.0, 4.0];
    // x_true = [1, 1, 1]; b = A * x_true = [3, 4, 5]
    let b = vec![3.0_f64, 4.0, 5.0];
    let mut x = vec![0.0_f64; 3];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealUnsym)
        .with_indexing(IndexBase::One);
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
}

#[test]
fn solve_zero_indexed_csr() {
    // Same SPD as the first test but using 0-based CSR.
    let ia = vec![0_i32, 2, 4, 5];
    let ja = vec![0_i32, 1, 1, 2, 2];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::Zero);
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
}

#[test]
fn solve_two_rhs() {
    // Same SPD; two right-hand sides at once.
    //   B = [[3, 6], [2, 4], [3, 6]] (column-major: [3,2,3,6,4,6])
    //   X_true = [[1, 2], [1, 2], [1, 2]]
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    let b = vec![3.0_f64, 2.0, 3.0, 6.0, 4.0, 6.0];
    let mut x = vec![0.0_f64; 6];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.solve_multi(3, 2, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[3], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[4], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[5], 2.0, epsilon = 1e-10);
}

#[test]
fn second_solve_uses_cached_factorization() {
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);

    // First solve: factorizes.
    let b1 = vec![3.0_f64, 2.0, 3.0];
    let mut x1 = vec![0.0_f64; 3];
    solver.solve(3, &a, &ia, &ja, &b1, &mut x1).unwrap();
    assert_abs_diff_eq!(x1[0], 1.0, epsilon = 1e-10);

    // Second solve: should reuse the existing factorization (phase 33 only).
    let b2 = vec![6.0_f64, 4.0, 6.0];
    let mut x2 = vec![0.0_f64; 3];
    solver.solve(3, &a, &ia, &ja, &b2, &mut x2).unwrap();
    assert_abs_diff_eq!(x2[0], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x2[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x2[2], 2.0, epsilon = 1e-10);
}

#[test]
fn analyze_then_solve() {
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.analyze_and_factorize(3, &a, &ia, &ja).unwrap();

    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
}

#[test]
fn get_diagonal_returns_da_match() {
    // SPD diag(1, 2, 3) — the diagonal of A is [1, 2, 3].
    let ia = vec![1_i32, 2, 3, 4];
    let ja = vec![1_i32, 2, 3];
    let a = vec![1.0_f64, 2.0, 3.0];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One)
        .with_diagonal_enabled();
    solver.analyze_and_factorize(3, &a, &ia, &ja).unwrap();

    let (df, da) = solver.get_diagonal(3).unwrap();
    assert_eq!(df.len(), 3);
    assert_eq!(da.len(), 3);
    // Diagonal of A.
    assert_abs_diff_eq!(da[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(da[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(da[2], 3.0, epsilon = 1e-10);
    // For SPD with no permutation, df is the diagonal of D in LDLᵀ —
    // for a diagonal matrix D = A, so df should equal A's diagonal.
    assert_abs_diff_eq!(df[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(df[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(df[2], 3.0, epsilon = 1e-10);
}

#[test]
fn get_diagonal_before_factorize_errors() {
    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd);
    assert!(solver.get_diagonal(3).is_err());
}

unsafe extern "C" fn passthrough_pivot(
    _aii: *mut f64,
    _bii: *mut f64,
    _eps: *mut f64,
) -> core::ffi::c_int {
    // Return 0 so MKL applies its default pivot adjustment.
    0
}

#[test]
fn pardiso_pivot_callback_install_and_solve() {
    // Install a passthrough pivot callback, run a normal SPD solve,
    // and confirm both that the callback installation succeeds and
    // that the solve still produces the right answer. Clear the
    // callback at the end so it doesn't leak into other tests in
    // this binary.
    let prev = unsafe { set_pardiso_pivot_callback(Some(passthrough_pivot)) };
    // Don't make assertions on `prev` — its value depends on test
    // ordering and global state.
    let _ = prev;

    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);

    // Clear the callback to restore default behavior.
    unsafe {
        let _ = set_pardiso_pivot_callback(None);
    }
}

#[test]
fn pardiso_64_raw_solves_3x3_spd() {
    // Same SPD test problem as solve_real_spd_3x3 but driven through
    // the always-64-bit pardiso_64 interface.
    let ia = vec![1_i64, 3, 5, 6];
    let ja = vec![1_i64, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];

    let mut pt: [*mut core::ffi::c_void; 64] = [core::ptr::null_mut(); 64];
    let mut iparm: [i64; 64] = [0; 64];
    // iparm[0] = 0 tells PARDISO to populate the rest with defaults.
    // iparm[34] = 0 selects 1-based indexing.
    let mtype: i64 = 2; // RealSpd
    let no_b: [f64; 0] = [];
    let mut no_x: [f64; 0] = [];

    // Phase 13 = analyze + factorize + solve.
    pardiso_64_raw::<f64>(
        &mut pt, 1, 1, mtype, 13, 3,
        &a, &ia, &ja, None, 1, &mut iparm, 0,
        &b, &mut x,
    )
    .unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);

    // Phase -1 = release all internal memory.
    pardiso_64_raw::<f64>(
        &mut pt, 1, 1, mtype, -1, 3,
        &a, &ia, &ja, None, 0, &mut iparm, 0,
        &no_b, &mut no_x,
    )
    .unwrap();
}

#[test]
fn pardiso_64_raw_rejects_complex_real_mismatch() {
    // mtype = 13 is ComplexUnsym, but we pass T = f64 which is real.
    let ia = vec![1_i64, 2];
    let ja = vec![1_i64];
    let a = vec![1.0_f64];
    let b = vec![1.0_f64];
    let mut x = vec![0.0_f64];
    let mut pt: [*mut core::ffi::c_void; 64] = [core::ptr::null_mut(); 64];
    let mut iparm: [i64; 64] = [0; 64];
    let r = pardiso_64_raw::<f64>(
        &mut pt, 1, 1, 13, 13, 1,
        &a, &ia, &ja, None, 1, &mut iparm, 0,
        &b, &mut x,
    );
    assert!(r.is_err());
}

#[test]
fn perm_records_fill_reducing_permutation() {
    // With iparm[4] = 2, MKL writes the fill-reducing permutation it
    // chose into the user's perm array. We provide a length-n
    // pre-allocated perm and verify after factorization that MKL
    // populated it (i.e. it's not all zeros and contains values
    // within [1, n]).
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];

    let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver.set_perm(Some(vec![0_i32; 3]));
    solver.iparm()[4] = 2; // ask MKL to write its perm choice
    solver.analyze_and_factorize(3, &a, &ia, &ja).unwrap();

    let perm = solver.perm().expect("perm should still be present");
    // Permutation is 1-based and a permutation of [1..=n].
    let mut sorted: Vec<i32> = perm.to_vec();
    sorted.sort();
    assert_eq!(sorted, vec![1, 2, 3]);
}

#[test]
fn save_then_restore_handle_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path();

    // Factorize once and save.
    let ia = vec![1_i32, 3, 5, 6];
    let ja = vec![1_i32, 2, 2, 3, 3];
    let a = vec![4.0_f64, -1.0, 4.0, -1.0, 4.0];
    {
        let mut solver = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
            .with_indexing(IndexBase::One);
        solver.analyze_and_factorize(3, &a, &ia, &ja).unwrap();
        solver.save_handle(dir_path).unwrap();
    }

    // Load into a fresh solver and solve.
    let mut solver2 = Pardiso::<f64>::new(PardisoMatrixType::RealSpd)
        .with_indexing(IndexBase::One);
    solver2.load_handle(dir_path).unwrap();
    let b = vec![3.0_f64, 2.0, 3.0];
    let mut x = vec![0.0_f64; 3];
    solver2.solve(3, &a, &ia, &ja, &b, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);

    // Clean up the disk artifacts.
    Pardiso::<f64>::delete_handle_files(dir_path).unwrap();
}
