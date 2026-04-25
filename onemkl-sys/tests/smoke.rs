//! End-to-end check: bindings compile, link, and a trivial MKL call works.

use std::ffi::CStr;
use std::os::raw::c_char;

use onemkl_sys::{MKLVersion, MKL_Get_Version, MKL_Get_Version_String};

#[test]
fn version_struct_is_populated() {
    let mut v = MKLVersion::default();
    unsafe {
        MKL_Get_Version(&mut v);
    }
    assert!(v.MajorVersion > 0, "MKL major version should be > 0");
    println!(
        "MKL {}.{}.{} (build {}, processor {})",
        v.MajorVersion,
        v.MinorVersion,
        v.UpdateVersion,
        unsafe { CStr::from_ptr(v.Build) }.to_string_lossy(),
        unsafe { CStr::from_ptr(v.Processor) }.to_string_lossy(),
    );
}

#[test]
fn version_string_is_nonempty() {
    let mut buf = [0u8; 256];
    unsafe {
        MKL_Get_Version_String(buf.as_mut_ptr().cast::<c_char>(), buf.len() as i32);
    }
    let s = CStr::from_bytes_until_nul(&buf)
        .expect("version string should be NUL-terminated");
    let text = s.to_string_lossy();
    assert!(!text.is_empty());
    println!("Version string: {text}");
}
