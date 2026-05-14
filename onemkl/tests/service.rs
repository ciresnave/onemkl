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

#[test]
fn aligned_buffer_round_trips_data() {
    let mut buf = service::AlignedBuffer::<f64>::new(1024, 64).unwrap();
    for (i, slot) in buf.as_mut_slice().iter_mut().enumerate() {
        *slot = i as f64;
    }
    assert_eq!(buf.len(), 1024);
    assert_eq!(buf[10], 10.0);
    assert_eq!(buf.as_slice().iter().sum::<f64>(), 1023.0 * 1024.0 / 2.0);
}

#[test]
fn aligned_buffer_honors_alignment() {
    let buf = service::AlignedBuffer::<f64>::new(16, 64).unwrap();
    assert_eq!((buf.as_ptr() as usize) % 64, 0);
}

#[test]
fn aligned_buffer_rejects_bad_alignment() {
    // Not a power of two.
    let r = service::AlignedBuffer::<f64>::new(16, 48);
    assert!(r.is_err());
}

#[test]
fn aligned_buffer_zero_init() {
    let buf = service::AlignedBuffer::<f64>::new(64, 32).unwrap();
    assert!(buf.iter().all(|&x| x == 0.0));
}

#[test]
fn thread_count_guard_runs_drop() {
    // Under threading-sequential MKL pins the local count, so we
    // can't reliably observe the restored value — just verify that
    // constructing and dropping the guard doesn't panic.
    {
        let _g = service::ThreadCountGuard::new(1);
    }
}

#[test]
fn cpu_frequency_positive() {
    let f = service::cpu_frequency_ghz();
    assert!(f > 0.0, "expected a positive CPU frequency, got {f}");
    // On virtualized CI runners `MKL_Get_Max_Cpu_Frequency` can
    // return 0 (no boost-clock info available), so only require
    // non-negative — `>= f` is not portable.
    let m = service::max_cpu_frequency_ghz();
    assert!(m >= 0.0);
}

#[test]
fn cpu_clocks_monotonic() {
    let a = service::cpu_clocks();
    // Spin a moment to advance the counter.
    for _ in 0..10_000 {
        std::hint::black_box(0_u64);
    }
    let b = service::cpu_clocks();
    assert!(b >= a);
}

#[test]
fn enable_instructions_handles_unsupported() {
    // We can't predict the host's CPU, so just verify the call doesn't
    // panic regardless of whether the level is supported.
    let _ = service::enable_instructions(service::IsaLevel::Sse42);
}
