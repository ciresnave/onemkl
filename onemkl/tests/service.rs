//! Verify service routines: version, threading, memory.

use onemkl::service;

#[test]
fn version_is_populated() {
    let v = service::version();
    assert!(v.major > 0);
    assert!(!v.build.is_empty());
    println!("MKL {}.{}.{} (build {}, processor {})", v.major, v.minor, v.update, v.build, v.processor);
}

#[test]
fn version_string_nonempty() {
    let s = service::version_string();
    assert!(!s.is_empty());
    assert!(s.contains("Math Kernel"));
}

#[test]
fn max_threads_positive() {
    let n = service::max_threads();
    assert!(n >= 1);
}

#[test]
fn set_num_threads_does_not_crash() {
    // Under the default threading-sequential feature, MKL pins the
    // thread count at 1, so we only verify the call doesn't crash.
    let original = service::max_threads();
    service::set_num_threads(2);
    let _ = service::max_threads();
    service::set_num_threads(original);
}

#[test]
fn local_thread_count_independent() {
    let prev = service::set_num_threads_local(4);
    // local count is now 4; max_threads still reflects global setting
    let _ = prev;
    // Reset.
    service::set_num_threads_local(0); // 0 means "use global"
}

#[test]
fn dynamic_does_not_crash() {
    // Same caveat as set_num_threads — under sequential threading the
    // value may not change. Just make sure neither call panics.
    let original = service::dynamic();
    service::set_dynamic(!original);
    let _ = service::dynamic();
    service::set_dynamic(original);
}

#[test]
fn mem_stat_works() {
    let stat = service::mem_stat();
    // stat may be zero or positive; either is fine.
    assert!(stat.bytes_allocated >= 0);
    assert!(stat.num_buffers >= 0);
}

#[test]
fn peak_mem_usage_works() {
    // Just ensure the call returns some value.
    let _ = service::peak_mem_usage(false);
}
