//! Service routines: version info, threading control, memory tracking,
//! verbose mode, and overall library lifecycle.

use core::ffi::{c_char, c_int};
use std::ffi::{CStr, CString};

use onemkl_sys as sys;

use crate::error::{Error, Result};

// =====================================================================
// Version
// =====================================================================

/// Information about the running oneMKL library version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    /// Major version (e.g. `2025` for MKL 2025.x).
    pub major: i32,
    /// Minor version.
    pub minor: i32,
    /// Update / patch number.
    pub update: i32,
    /// Build identifier (e.g. `"20251007"`).
    pub build: String,
    /// Targeted processor architecture (e.g. `"Intel(R) Architecture processors"`).
    pub processor: String,
    /// Platform string (e.g. `"Intel(R) 64 architecture"`).
    pub platform: String,
}

/// Query the oneMKL version.
pub fn version() -> Version {
    let mut v = sys::MKLVersion::default();
    unsafe { sys::MKL_Get_Version(&mut v) };
    Version {
        major: v.MajorVersion,
        minor: v.MinorVersion,
        update: v.UpdateVersion,
        build: c_str_to_owned(v.Build),
        processor: c_str_to_owned(v.Processor),
        platform: c_str_to_owned(v.Platform),
    }
}

/// Single-line description of the running oneMKL library, as returned
/// by `MKL_Get_Version_String`.
pub fn version_string() -> String {
    let mut buf = vec![0u8; 256];
    unsafe {
        sys::MKL_Get_Version_String(buf.as_mut_ptr().cast::<c_char>(), buf.len() as c_int);
    }
    if let Some(end) = buf.iter().position(|&b| b == 0) {
        buf.truncate(end);
    }
    let trimmed: Vec<u8> = buf.into_iter().rev().skip_while(|&b| b == b' ').collect();
    let mut out: Vec<u8> = trimmed.into_iter().rev().collect();
    out.retain(|&b| b != 0);
    String::from_utf8_lossy(&out).into_owned()
}

// =====================================================================
// Threading
// =====================================================================

/// Set the global thread count for oneMKL routines.
pub fn set_num_threads(n: i32) {
    unsafe { sys::MKL_Set_Num_Threads(n) };
}

/// Set the thread count for the current thread only. Returns the
/// previous local thread count for that thread.
pub fn set_num_threads_local(n: i32) -> i32 {
    unsafe { sys::MKL_Set_Num_Threads_Local(n) }
}

/// Maximum number of threads oneMKL is allowed to use.
pub fn max_threads() -> i32 {
    unsafe { sys::MKL_Get_Max_Threads() }
}

/// Enable or disable dynamic adjustment of the number of threads
/// (`MKL_Set_Dynamic`).
pub fn set_dynamic(flag: bool) {
    unsafe { sys::MKL_Set_Dynamic(if flag { 1 } else { 0 }) };
}

/// Read the current dynamic-threading setting.
pub fn dynamic() -> bool {
    unsafe { sys::MKL_Get_Dynamic() != 0 }
}

// =====================================================================
// Verbose mode
// =====================================================================

/// Enable or disable verbose mode (`MKL_Verbose`). When enabled, oneMKL
/// prints a one-line trace of every supported routine call.
pub fn set_verbose(flag: bool) {
    unsafe { sys::MKL_Verbose(if flag { 1 } else { 0 }) };
}

/// Redirect verbose output to the given file path. Pass `None` to
/// reset to stdout.
pub fn set_verbose_output_file(path: Option<&str>) -> Result<()> {
    let status = match path {
        Some(p) => {
            let cstr = CString::new(p).map_err(|_| {
                Error::InvalidArgument("verbose output path contains a NUL byte")
            })?;
            unsafe { sys::MKL_Verbose_Output_File(cstr.as_ptr()) }
        }
        None => unsafe { sys::MKL_Verbose_Output_File(core::ptr::null()) },
    };
    if status == 0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "MKL_Verbose_Output_File failed (path inaccessible?)",
        ))
    }
}

// =====================================================================
// Memory
// =====================================================================

/// Snapshot of MKL's internal allocator state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemStat {
    /// Total bytes allocated by MKL.
    pub bytes_allocated: i64,
    /// Number of distinct buffers currently held.
    pub num_buffers: i32,
}

/// Read the current allocation statistics.
pub fn mem_stat() -> MemStat {
    let mut nbuffers: c_int = 0;
    let bytes = unsafe { sys::MKL_Mem_Stat(&mut nbuffers) };
    MemStat {
        bytes_allocated: bytes,
        num_buffers: nbuffers,
    }
}

/// Peak memory usage in bytes since the last reset. Pass `reset=true`
/// to clear the high-water mark for subsequent calls.
pub fn peak_mem_usage(reset: bool) -> i64 {
    let r: c_int = if reset { 1 } else { 0 };
    unsafe { sys::MKL_Peak_Mem_Usage(r) }
}

/// Free internal MKL buffers (best-effort cleanup).
pub fn free_buffers() {
    unsafe { sys::MKL_Free_Buffers() };
}

/// Free per-thread MKL buffers for the current thread.
pub fn thread_free_buffers() {
    unsafe { sys::MKL_Thread_Free_Buffers() };
}

/// Set a soft memory limit for MKL allocations. `mem_type` is reserved
/// (currently 0 for total).
pub fn set_memory_limit(mem_type: i32, limit_bytes: usize) -> Result<()> {
    let status = unsafe { sys::MKL_Set_Memory_Limit(mem_type, limit_bytes) };
    if status == 0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "MKL_Set_Memory_Limit rejected the requested limit",
        ))
    }
}

// =====================================================================
// Library lifecycle
// =====================================================================

/// Tear down all internal oneMKL state. Call once at program exit if
/// running in an environment that needs deterministic cleanup.
pub fn finalize() {
    unsafe { sys::MKL_Finalize() };
}

// =====================================================================
// Helpers
// =====================================================================

#[inline]
fn c_str_to_owned(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: MKL guarantees these strings are NUL-terminated and live
    // for the program's lifetime.
    unsafe { CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}
