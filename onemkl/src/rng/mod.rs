//! Random number generation via the Vector Statistical Library (VSL).
//!
//! oneMKL's VSL exposes a family of basic pseudo-random number
//! generators (BRNGs) plus distribution wrappers (`vsRngGaussian`,
//! `viRngPoisson`, etc.). This module wraps them in an idiomatic
//! [`Stream`] type that owns a `VSLStreamStatePtr` and releases it on
//! drop.
//!
//! ```no_run
//! use onemkl::rng::{Stream, BasicRng};
//!
//! let mut s = Stream::new(BasicRng::Mt19937, 42).unwrap();
//! let mut buf = [0.0_f64; 1000];
//! s.uniform(&mut buf, 0.0, 1.0).unwrap();
//! ```

use core::ffi::c_int;
use core::ptr;

use onemkl_sys::{self as sys, VSLStreamStatePtr};

use crate::error::{Error, Result};
use crate::util::dim_to_mkl_int;

/// Basic random number generators supported by oneMKL VSL.
///
/// Pick one for [`Stream::new`]. Most users want
/// [`Mt19937`](Self::Mt19937) (Mersenne Twister) for general work or
/// [`Philox4x32x10`](Self::Philox4x32x10) for highly parallel use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasicRng {
    /// 31-bit multiplicative congruential generator.
    Mcg31,
    /// 59-bit multiplicative congruential generator.
    Mcg59,
    /// L'Ecuyer's combined multiple recursive generator.
    Mrg32k3a,
    /// Mersenne Twister (period 2^19937 − 1).
    Mt19937,
    /// MT2203 — a set of 6024 independent Mersenne-Twister-like
    /// generators useful for parallel work.
    Mt2203,
    /// Counter-based Philox 4×32-10.
    Philox4x32x10,
    /// R250 generator.
    R250,
    /// SIMD-oriented Fast Mersenne Twister.
    Sfmt19937,
    /// Wichmann-Hill generator.
    Wh,
    /// Non-deterministic generator backed by hardware entropy
    /// (`RDRAND`-style). Re-tries up to
    /// [`VSL_BRNG_NONDETERM_NRETRIES`](onemkl_sys::VSL_BRNG_NONDETERM_NRETRIES)
    /// times on failure.
    NonDeterm,
}

impl BasicRng {
    #[inline]
    fn as_brng(self) -> c_int {
        let v = match self {
            Self::Mcg31 => sys::VSL_BRNG_MCG31,
            Self::Mcg59 => sys::VSL_BRNG_MCG59,
            Self::Mrg32k3a => sys::VSL_BRNG_MRG32K3A,
            Self::Mt19937 => sys::VSL_BRNG_MT19937,
            Self::Mt2203 => sys::VSL_BRNG_MT2203,
            Self::Philox4x32x10 => sys::VSL_BRNG_PHILOX4X32X10,
            Self::R250 => sys::VSL_BRNG_R250,
            Self::Sfmt19937 => sys::VSL_BRNG_SFMT19937,
            Self::Wh => sys::VSL_BRNG_WH,
            Self::NonDeterm => sys::VSL_BRNG_NONDETERM,
        };
        v as c_int
    }
}

/// An owned VSL stream. Release the underlying generator state on
/// drop.
///
/// `Send` but not `Sync` — random number generation is stateful and
/// non-reentrant. To use the same stream across threads, wrap in a
/// `Mutex` or use [`leapfrog`](Self::leapfrog) /
/// [`skip_ahead`](Self::skip_ahead) to derive independent streams.
pub struct Stream {
    handle: VSLStreamStatePtr,
}

// SAFETY: VSL stream state is heap-allocated and not tied to any
// particular thread; the C library is happy to be moved between
// threads as long as no two access it concurrently.
unsafe impl Send for Stream {}

impl Stream {
    /// Create a new stream backed by the given BRNG and seeded with
    /// `seed`.
    pub fn new(brng: BasicRng, seed: u32) -> Result<Self> {
        let mut handle: VSLStreamStatePtr = ptr::null_mut();
        let status = unsafe {
            sys::vslNewStream(&mut handle, brng.as_brng(), seed)
        };
        check_vsl(status)?;
        Ok(Self { handle })
    }

    /// Skip the first `nskip` numbers in the stream. Useful for
    /// parallel decomposition.
    pub fn skip_ahead(&mut self, nskip: i64) -> Result<()> {
        let status = unsafe { sys::vslSkipAheadStream(self.handle, nskip) };
        check_vsl(status)
    }

    /// Re-cast as the leapfrog'th generator out of `nstreams`. Each
    /// resulting stream draws every `nstreams`-th element of the
    /// original sequence.
    pub fn leapfrog(&mut self, leap_index: c_int, nstreams: c_int) -> Result<()> {
        let status = unsafe {
            sys::vslLeapfrogStream(self.handle, leap_index, nstreams)
        };
        check_vsl(status)
    }

    /// Raw pointer to the underlying state — for advanced use only.
    /// Borrowed for the lifetime of `&mut self`.
    #[inline]
    #[must_use]
    pub fn as_handle(&mut self) -> VSLStreamStatePtr {
        self.handle
    }

    // -----------------------------------------------------------------
    // Continuous distributions
    // -----------------------------------------------------------------

    /// Fill `out` with samples uniformly distributed on `[a, b)`.
    pub fn uniform<T: RngFloat>(&mut self, out: &mut [T], a: T, b: T) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_uniform(0, self.handle, n, out.as_mut_ptr(), a, b)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Normal(mean, sigma)`.
    pub fn gaussian<T: RngFloat>(
        &mut self,
        out: &mut [T],
        mean: T,
        sigma: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_gaussian(0, self.handle, n, out.as_mut_ptr(), mean, sigma)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from the exponential distribution with
    /// displacement `displacement` and scale `beta`
    /// (mean = displacement + beta).
    pub fn exponential<T: RngFloat>(
        &mut self,
        out: &mut [T],
        displacement: T,
        beta: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_exponential(0, self.handle, n, out.as_mut_ptr(), displacement, beta)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Lognormal(mean, sigma, displacement, scale)`.
    /// `mean` and `sigma` describe the underlying normal; `displacement`
    /// shifts and `scale` scales the exponentiated value.
    pub fn lognormal<T: RngFloat>(
        &mut self,
        out: &mut [T],
        mean: T,
        sigma: T,
        displacement: T,
        scale: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_lognormal(
                0, self.handle, n, out.as_mut_ptr(), mean, sigma, displacement, scale,
            )
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Cauchy(displacement, scale)`.
    pub fn cauchy<T: RngFloat>(
        &mut self,
        out: &mut [T],
        displacement: T,
        scale: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_cauchy(0, self.handle, n, out.as_mut_ptr(), displacement, scale)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Weibull(alpha, displacement, beta)`.
    pub fn weibull<T: RngFloat>(
        &mut self,
        out: &mut [T],
        alpha: T,
        displacement: T,
        beta: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_weibull(0, self.handle, n, out.as_mut_ptr(), alpha, displacement, beta)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Gamma(alpha, displacement, beta)`.
    pub fn gamma<T: RngFloat>(
        &mut self,
        out: &mut [T],
        alpha: T,
        displacement: T,
        beta: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_gamma(0, self.handle, n, out.as_mut_ptr(), alpha, displacement, beta)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Beta(p, q, displacement, scale)`.
    pub fn beta<T: RngFloat>(
        &mut self,
        out: &mut [T],
        p: T,
        q: T,
        displacement: T,
        scale: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_beta(0, self.handle, n, out.as_mut_ptr(), p, q, displacement, scale)
        };
        check_vsl(status)
    }

    // -----------------------------------------------------------------
    // Discrete distributions (i32 outputs)
    // -----------------------------------------------------------------

    /// Fill `out` with integers uniformly distributed on `[a, b)`.
    pub fn uniform_int(&mut self, out: &mut [i32], a: i32, b: i32) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            sys::viRngUniform(0, self.handle, n, out.as_mut_ptr(), a, b)
        };
        check_vsl(status)
    }

    /// Fill `out` with `Bernoulli(p)` samples (0 or 1).
    pub fn bernoulli(&mut self, out: &mut [i32], p: f64) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            sys::viRngBernoulli(0, self.handle, n, out.as_mut_ptr(), p)
        };
        check_vsl(status)
    }

    /// Fill `out` with `Binomial(ntrial, p)` samples.
    pub fn binomial(&mut self, out: &mut [i32], ntrial: i32, p: f64) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            sys::viRngBinomial(0, self.handle, n, out.as_mut_ptr(), ntrial, p)
        };
        check_vsl(status)
    }

    /// Fill `out` with `Poisson(lambda)` samples.
    pub fn poisson(&mut self, out: &mut [i32], lambda: f64) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            sys::viRngPoisson(0, self.handle, n, out.as_mut_ptr(), lambda)
        };
        check_vsl(status)
    }
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        let mut handle: VSLStreamStatePtr = ptr::null_mut();
        let status = unsafe { sys::vslCopyStream(&mut handle, self.handle) };
        if status != 0 {
            panic!("vslCopyStream failed (status {status})");
        }
        Self { handle }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle was obtained from vslNewStream (or
            // vslCopyStream) and is owned by this Stream.
            unsafe {
                let _ = sys::vslDeleteStream(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// Trait wiring
// =====================================================================

/// Floating-point scalar types supported by the VSL distribution
/// generators ([`f32`] and [`f64`]).
#[allow(missing_docs)]
pub trait RngFloat: Copy + 'static {
    unsafe fn rng_uniform(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, a: Self, b: Self,
    ) -> c_int;
    unsafe fn rng_gaussian(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, mean: Self, sigma: Self,
    ) -> c_int;
    unsafe fn rng_exponential(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, displacement: Self, beta: Self,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    unsafe fn rng_lognormal(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self,
        mean: Self, sigma: Self, displacement: Self, scale: Self,
    ) -> c_int;
    unsafe fn rng_cauchy(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, displacement: Self, scale: Self,
    ) -> c_int;
    unsafe fn rng_weibull(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, alpha: Self, displacement: Self, beta: Self,
    ) -> c_int;
    unsafe fn rng_gamma(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, alpha: Self, displacement: Self, beta: Self,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    unsafe fn rng_beta(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, p: Self, q: Self, displacement: Self, scale: Self,
    ) -> c_int;
}

macro_rules! impl_rng_float {
    ($ty:ty,
        uniform=$uniform:ident, gaussian=$gaussian:ident,
        exponential=$exp:ident, lognormal=$logn:ident,
        cauchy=$cauchy:ident, weibull=$weibull:ident,
        gamma=$gamma:ident, beta=$beta:ident,
    ) => {
        impl RngFloat for $ty {
            unsafe fn rng_uniform(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, a: Self, b: Self,
            ) -> c_int {
                unsafe { sys::$uniform(method, stream, n, r, a, b) }
            }
            unsafe fn rng_gaussian(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, mean: Self, sigma: Self,
            ) -> c_int {
                unsafe { sys::$gaussian(method, stream, n, r, mean, sigma) }
            }
            unsafe fn rng_exponential(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, displacement: Self, beta: Self,
            ) -> c_int {
                unsafe { sys::$exp(method, stream, n, r, displacement, beta) }
            }
            unsafe fn rng_lognormal(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self,
                mean: Self, sigma: Self, displacement: Self, scale: Self,
            ) -> c_int {
                unsafe {
                    sys::$logn(method, stream, n, r, mean, sigma, displacement, scale)
                }
            }
            unsafe fn rng_cauchy(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, displacement: Self, scale: Self,
            ) -> c_int {
                unsafe { sys::$cauchy(method, stream, n, r, displacement, scale) }
            }
            unsafe fn rng_weibull(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, alpha: Self, displacement: Self, beta: Self,
            ) -> c_int {
                unsafe { sys::$weibull(method, stream, n, r, alpha, displacement, beta) }
            }
            unsafe fn rng_gamma(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, alpha: Self, displacement: Self, beta: Self,
            ) -> c_int {
                unsafe { sys::$gamma(method, stream, n, r, alpha, displacement, beta) }
            }
            unsafe fn rng_beta(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self,
                p: Self, q: Self, displacement: Self, scale: Self,
            ) -> c_int {
                unsafe {
                    sys::$beta(method, stream, n, r, p, q, displacement, scale)
                }
            }
        }
    };
}

impl_rng_float!(
    f32,
    uniform=vsRngUniform, gaussian=vsRngGaussian,
    exponential=vsRngExponential, lognormal=vsRngLognormal,
    cauchy=vsRngCauchy, weibull=vsRngWeibull,
    gamma=vsRngGamma, beta=vsRngBeta,
);

impl_rng_float!(
    f64,
    uniform=vdRngUniform, gaussian=vdRngGaussian,
    exponential=vdRngExponential, lognormal=vdRngLognormal,
    cauchy=vdRngCauchy, weibull=vdRngWeibull,
    gamma=vdRngGamma, beta=vdRngBeta,
);

#[inline]
fn check_vsl(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::VslStatus(status))
    }
}
