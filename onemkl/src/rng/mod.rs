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

pub mod convolution;
pub mod summary_stats;

/// Basic random number generators supported by oneMKL VSL.
///
/// Pick one for [`Stream::new`]. Most users want
/// [`Mt19937`](Self::Mt19937) (Mersenne Twister) for general work or
/// [`Philox4x32x10`](Self::Philox4x32x10) for highly parallel use.
///
/// [`Sobol`](Self::Sobol) and [`Niederreiter`](Self::Niederreiter) are
/// quasi-random (low-discrepancy) sequences rather than pseudo-random
/// generators. They require a dimension and must be created with
/// [`Stream::quasi_random`].
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
    /// Counter-based ARS-5 (AES-based, requires AES-NI).
    Ars5,
    /// R250 generator.
    R250,
    /// SIMD-oriented Fast Mersenne Twister.
    Sfmt19937,
    /// Wichmann-Hill generator.
    Wh,
    /// Sobol low-discrepancy quasi-random sequence. Use with
    /// [`Stream::quasi_random`] to set the dimension.
    Sobol,
    /// Niederreiter low-discrepancy quasi-random sequence. Use with
    /// [`Stream::quasi_random`] to set the dimension.
    Niederreiter,
    /// Hardware `RDRAND` instruction.
    Rdrand,
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
            Self::Ars5 => sys::VSL_BRNG_ARS5,
            Self::R250 => sys::VSL_BRNG_R250,
            Self::Sfmt19937 => sys::VSL_BRNG_SFMT19937,
            Self::Wh => sys::VSL_BRNG_WH,
            Self::Sobol => sys::VSL_BRNG_SOBOL,
            Self::Niederreiter => sys::VSL_BRNG_NIEDERR,
            Self::Rdrand => sys::VSL_BRNG_RDRAND,
            Self::NonDeterm => sys::VSL_BRNG_NONDETERM,
        };
        v as c_int
    }

    /// True if this BRNG is a quasi-random sequence rather than a
    /// pseudo-random generator. Quasi-random streams must be created
    /// via [`Stream::quasi_random`] with an explicit dimension.
    #[inline]
    #[must_use]
    pub fn is_quasi_random(self) -> bool {
        matches!(self, Self::Sobol | Self::Niederreiter)
    }
}

/// Storage layout for the covariance Cholesky factor passed to
/// [`Stream::gaussian_mv`].
///
/// All variants expect a *Cholesky factor* `T` such that `T·Tᵀ`
/// equals the desired covariance — not the covariance matrix itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixStorage {
    /// Full `dimen × dimen` Cholesky factor in column-major layout
    /// (`dimen²` elements).
    Full,
    /// Lower-triangular Cholesky factor in packed column-major layout
    /// (`dimen·(dimen + 1) / 2` elements).
    Packed,
    /// Diagonal entries only — equivalent to independent univariate
    /// Gaussians with the given per-axis standard deviations
    /// (`dimen` elements).
    Diagonal,
}

impl MatrixStorage {
    #[inline]
    fn as_int(self) -> c_int {
        let v = match self {
            Self::Full => sys::VSL_MATRIX_STORAGE_FULL,
            Self::Packed => sys::VSL_MATRIX_STORAGE_PACKED,
            Self::Diagonal => sys::VSL_MATRIX_STORAGE_DIAGONAL,
        };
        v as c_int
    }

    fn factor_len(self, dimen: usize) -> usize {
        match self {
            Self::Full => dimen * dimen,
            Self::Packed => dimen * (dimen + 1) / 2,
            Self::Diagonal => dimen,
        }
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
    ///
    /// For quasi-random sequences (`Sobol`, `Niederreiter`) use
    /// [`Self::quasi_random`] instead — they need a dimension, not a
    /// scalar seed.
    pub fn new(brng: BasicRng, seed: u32) -> Result<Self> {
        if brng.is_quasi_random() {
            return Err(Error::InvalidArgument(
                "use Stream::quasi_random for Sobol / Niederreiter",
            ));
        }
        let mut handle: VSLStreamStatePtr = ptr::null_mut();
        let status = unsafe {
            sys::vslNewStream(&mut handle, brng.as_brng(), seed)
        };
        check_vsl(status)?;
        Ok(Self { handle })
    }

    /// Create a quasi-random stream backed by Sobol or Niederreiter
    /// with the given dimension.
    ///
    /// Quasi-random sequences fill space more uniformly than
    /// pseudo-random samples (low discrepancy) at the cost of being
    /// deterministic. They're standard for Quasi-Monte Carlo
    /// integration, Bayesian optimization, and MC dropout uncertainty
    /// estimates.
    ///
    /// `dimension` must be `>= 1`. Each draw from the underlying
    /// uniform distribution produces `dimension` correlated values in
    /// `[0, 1)`, so distribution calls should target a buffer length
    /// that's a multiple of `dimension`.
    pub fn quasi_random(brng: BasicRng, dimension: u32) -> Result<Self> {
        if !brng.is_quasi_random() {
            return Err(Error::InvalidArgument(
                "Stream::quasi_random requires Sobol or Niederreiter",
            ));
        }
        if dimension == 0 {
            return Err(Error::InvalidArgument(
                "quasi-random dimension must be >= 1",
            ));
        }
        let mut handle: VSLStreamStatePtr = ptr::null_mut();
        let params = [dimension];
        let status = unsafe {
            sys::vslNewStreamEx(
                &mut handle,
                brng.as_brng(),
                params.len() as c_int,
                params.as_ptr(),
            )
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

    /// Fill `out` with samples from `Gumbel(displacement, scale)` —
    /// specifically the **min-stable** (Type I extreme value for
    /// minima) parameterization MKL exposes, with PDF
    /// `(1/β)·exp((x−a)/β)·exp(−exp((x−a)/β))` and mean `a − γβ`.
    ///
    /// For the **max-Gumbel** (mean `a + γβ`) used in the
    /// Gumbel-softmax trick for differentiable categorical sampling,
    /// negate the samples: draw `min_gumbel(0, 1)` then take `-x` to
    /// get `max_gumbel(0, 1)`.
    pub fn gumbel<T: RngFloat>(
        &mut self,
        out: &mut [T],
        displacement: T,
        scale: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_gumbel(0, self.handle, n, out.as_mut_ptr(), displacement, scale)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Laplace(displacement, scale)`
    /// (double-exponential). Common as a heavy-tailed prior and in
    /// L1-regularized models.
    pub fn laplace<T: RngFloat>(
        &mut self,
        out: &mut [T],
        displacement: T,
        scale: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_laplace(0, self.handle, n, out.as_mut_ptr(), displacement, scale)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Rayleigh(displacement, scale)`.
    pub fn rayleigh<T: RngFloat>(
        &mut self,
        out: &mut [T],
        displacement: T,
        scale: T,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_rayleigh(0, self.handle, n, out.as_mut_ptr(), displacement, scale)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from a chi-square distribution with
    /// `ndf` degrees of freedom.
    pub fn chi_square<T: RngFloat>(&mut self, out: &mut [T], ndf: i32) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            T::rng_chi_square(0, self.handle, n, out.as_mut_ptr(), ndf)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from a `dimen`-dimensional multivariate
    /// normal distribution with mean `mean` and covariance
    /// `cov_factor · cov_factorᵀ`.
    ///
    /// `out` is filled with `out.len() / dimen` samples, each
    /// `dimen` values long, packed contiguously. `mean` must have
    /// length `dimen`. `cov_factor` is the Cholesky factor of the
    /// covariance matrix; its expected length depends on `storage`:
    ///
    /// - [`MatrixStorage::Full`] → `dimen² ` (column-major).
    /// - [`MatrixStorage::Packed`] → `dimen·(dimen + 1) / 2`.
    /// - [`MatrixStorage::Diagonal`] → `dimen` (independent axes).
    ///
    /// To draw from a covariance `C` directly, factor it first with
    /// LAPACK `?potrf` and pass the resulting lower-triangular factor.
    pub fn gaussian_mv<T: RngFloat>(
        &mut self,
        out: &mut [T],
        dimen: usize,
        storage: MatrixStorage,
        mean: &[T],
        cov_factor: &[T],
    ) -> Result<()> {
        if dimen == 0 {
            return Err(Error::InvalidArgument("dimen must be >= 1"));
        }
        if out.len() % dimen != 0 {
            return Err(Error::InvalidArgument(
                "out.len() must be a multiple of dimen",
            ));
        }
        if mean.len() != dimen {
            return Err(Error::InvalidArgument(
                "mean must have length dimen",
            ));
        }
        if cov_factor.len() != storage.factor_len(dimen) {
            return Err(Error::InvalidArgument(
                "cov_factor length must match storage layout for dimen",
            ));
        }
        let n_samples = dim_to_mkl_int(out.len() / dimen)?;
        let dimen_i = dim_to_mkl_int(dimen)?;
        let status = unsafe {
            T::rng_gaussian_mv(
                0,
                self.handle,
                n_samples,
                out.as_mut_ptr(),
                dimen_i,
                storage.as_int(),
                mean.as_ptr(),
                cov_factor.as_ptr(),
            )
        };
        check_vsl(status)
    }

    // -----------------------------------------------------------------
    // Derived distributions (rejection sampling / composition)
    // -----------------------------------------------------------------

    /// Fill `out` with samples from a normal distribution truncated to
    /// `[lo, hi]`. Common for weight initialization (e.g. Glorot/He
    /// with a ±2σ cutoff).
    ///
    /// Implemented via rejection sampling on top of [`Self::gaussian`].
    /// Efficient when the bounds are loose; for very tight bounds
    /// consider an inverse-CDF formulation instead.
    pub fn truncated_normal(
        &mut self,
        out: &mut [f64],
        mean: f64,
        sigma: f64,
        lo: f64,
        hi: f64,
    ) -> Result<()> {
        if !(lo < hi) {
            return Err(Error::InvalidArgument("require lo < hi"));
        }
        if !(sigma > 0.0) {
            return Err(Error::InvalidArgument("sigma must be > 0"));
        }
        rejection_truncated_normal_f64(self, out, mean, sigma, lo, hi)
    }

    /// `f32` counterpart of [`Self::truncated_normal`].
    pub fn truncated_normal_f32(
        &mut self,
        out: &mut [f32],
        mean: f32,
        sigma: f32,
        lo: f32,
        hi: f32,
    ) -> Result<()> {
        if !(lo < hi) {
            return Err(Error::InvalidArgument("require lo < hi"));
        }
        if !(sigma > 0.0) {
            return Err(Error::InvalidArgument("sigma must be > 0"));
        }
        rejection_truncated_normal_f32(self, out, mean, sigma, lo, hi)
    }

    /// Fill `out` with samples from a `Dirichlet(alpha)` distribution.
    ///
    /// `alpha.len()` is the simplex dimension `k`. `out` is filled
    /// with `out.len() / k` samples, each a `k`-vector summing to 1.
    /// `out.len()` must be a multiple of `k`.
    ///
    /// Implemented via the standard recipe: draw `k` independent
    /// `Gamma(αᵢ, 1)` and normalize. Used in Bayesian mixture models,
    /// LDA, RLHF reward modeling, and stick-breaking constructions.
    pub fn dirichlet(&mut self, out: &mut [f64], alpha: &[f64]) -> Result<()> {
        let k = alpha.len();
        if k == 0 {
            return Err(Error::InvalidArgument("alpha must be non-empty"));
        }
        if alpha.iter().any(|&a| !(a > 0.0)) {
            return Err(Error::InvalidArgument(
                "every alpha entry must be > 0",
            ));
        }
        if out.len() % k != 0 {
            return Err(Error::InvalidArgument(
                "out.len() must be a multiple of alpha.len()",
            ));
        }
        for row in out.chunks_exact_mut(k) {
            for (slot, &a) in row.iter_mut().zip(alpha) {
                let mut one = [0.0_f64; 1];
                self.gamma(&mut one, a, 0.0, 1.0)?;
                *slot = one[0];
            }
            let sum: f64 = row.iter().sum();
            if !(sum > 0.0) {
                // Re-draw with a tiny perturbation to avoid div-by-zero
                // from underflow when alpha is very small.
                for slot in row.iter_mut() {
                    *slot += f64::MIN_POSITIVE;
                }
            }
            let s: f64 = row.iter().sum();
            for slot in row.iter_mut() {
                *slot /= s;
            }
        }
        Ok(())
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

    /// Fill `out` with `Geometric(p)` samples — the number of failures
    /// before the first success when each trial succeeds with
    /// probability `p`.
    pub fn geometric(&mut self, out: &mut [i32], p: f64) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            sys::viRngGeometric(0, self.handle, n, out.as_mut_ptr(), p)
        };
        check_vsl(status)
    }

    /// Fill `out` with `Hypergeometric(lot_size, marked, sample_size)`
    /// samples — the number of marked items observed when drawing
    /// `sample_size` items without replacement from a population of
    /// `lot_size` containing `marked` marked items.
    pub fn hypergeometric(
        &mut self,
        out: &mut [i32],
        lot_size: i32,
        marked: i32,
        sample_size: i32,
    ) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            sys::viRngHypergeometric(
                0,
                self.handle,
                n,
                out.as_mut_ptr(),
                lot_size,
                marked,
                sample_size,
            )
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from the negative binomial distribution
    /// with shape parameter `a` (the number of successful trials in
    /// the Polya / Pascal parameterization) and success probability
    /// `p`. Useful as an overdispersed Poisson replacement and in
    /// count-data regression.
    pub fn neg_binomial(&mut self, out: &mut [i32], a: f64, p: f64) -> Result<()> {
        let n = dim_to_mkl_int(out.len())?;
        let status = unsafe {
            sys::viRngNegBinomial(0, self.handle, n, out.as_mut_ptr(), a, p)
        };
        check_vsl(status)
    }

    /// Fill `out` with samples from `Multinomial(ntrial, p)`.
    ///
    /// `p` is the probability vector over `k = p.len()` categories
    /// and must sum to 1 (numerically). Each call draws
    /// `out.len() / k` independent samples; each sample is a `k`-long
    /// count vector summing to `ntrial`. `out.len()` must therefore
    /// be a multiple of `k`.
    ///
    /// For LLM-style token sampling (one draw, position of the
    /// non-zero entry), use `ntrial = 1` and post-process — or build
    /// on top of this with a thin "categorical" helper.
    pub fn multinomial(
        &mut self,
        out: &mut [i32],
        ntrial: i32,
        p: &[f64],
    ) -> Result<()> {
        let k = p.len();
        if k == 0 {
            return Err(Error::InvalidArgument("p must be non-empty"));
        }
        if out.len() % k != 0 {
            return Err(Error::InvalidArgument(
                "out.len() must be a multiple of p.len()",
            ));
        }
        if ntrial < 0 {
            return Err(Error::InvalidArgument("ntrial must be >= 0"));
        }
        let n_samples = dim_to_mkl_int(out.len() / k)?;
        let k_i = dim_to_mkl_int(k)?;
        let status = unsafe {
            sys::viRngMultinomial(
                0,
                self.handle,
                n_samples,
                out.as_mut_ptr(),
                ntrial,
                k_i,
                p.as_ptr(),
            )
        };
        check_vsl(status)
    }

    /// Sample categorical indices in `0..p.len()` according to the
    /// probability vector `p`. `out[i]` is the chosen category index
    /// for sample `i`. Built on top of [`Self::multinomial`] with
    /// `ntrial = 1`.
    ///
    /// This is the typical "draw the next token" primitive for LLM
    /// inference.
    pub fn categorical(&mut self, out: &mut [i32], p: &[f64]) -> Result<()> {
        let k = p.len();
        if k == 0 {
            return Err(Error::InvalidArgument("p must be non-empty"));
        }
        let n = out.len();
        let mut counts = vec![0_i32; n * k];
        self.multinomial(&mut counts, 1, p)?;
        for (i, slot) in out.iter_mut().enumerate() {
            let row = &counts[i * k..(i + 1) * k];
            *slot = row
                .iter()
                .position(|&c| c == 1)
                .map(|j| j as i32)
                .unwrap_or(0);
        }
        Ok(())
    }
}

#[inline]
fn rejection_truncated_normal_f64(
    stream: &mut Stream,
    out: &mut [f64],
    mean: f64,
    sigma: f64,
    lo: f64,
    hi: f64,
) -> Result<()> {
    let mut filled = 0_usize;
    let mut scratch = vec![0.0_f64; out.len()];
    // Cap the number of refill passes so a tight truncation can't loop
    // forever. 64 passes accept anything down to ~10⁻²⁰ acceptance ratio.
    for _ in 0..64 {
        if filled == out.len() {
            return Ok(());
        }
        let need = out.len() - filled;
        let buf = &mut scratch[..need];
        stream.gaussian(buf, mean, sigma)?;
        for &x in buf.iter() {
            if x >= lo && x <= hi {
                out[filled] = x;
                filled += 1;
                if filled == out.len() {
                    return Ok(());
                }
            }
        }
    }
    Err(Error::InvalidArgument(
        "truncated_normal rejection rate too high — tighten sigma or widen [lo, hi]",
    ))
}

#[inline]
fn rejection_truncated_normal_f32(
    stream: &mut Stream,
    out: &mut [f32],
    mean: f32,
    sigma: f32,
    lo: f32,
    hi: f32,
) -> Result<()> {
    let mut filled = 0_usize;
    let mut scratch = vec![0.0_f32; out.len()];
    for _ in 0..64 {
        if filled == out.len() {
            return Ok(());
        }
        let need = out.len() - filled;
        let buf = &mut scratch[..need];
        stream.gaussian(buf, mean, sigma)?;
        for &x in buf.iter() {
            if x >= lo && x <= hi {
                out[filled] = x;
                filled += 1;
                if filled == out.len() {
                    return Ok(());
                }
            }
        }
    }
    Err(Error::InvalidArgument(
        "truncated_normal rejection rate too high — tighten sigma or widen [lo, hi]",
    ))
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
    unsafe fn rng_gumbel(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, displacement: Self, scale: Self,
    ) -> c_int;
    unsafe fn rng_laplace(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, displacement: Self, scale: Self,
    ) -> c_int;
    unsafe fn rng_rayleigh(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, displacement: Self, scale: Self,
    ) -> c_int;
    unsafe fn rng_chi_square(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, ndf: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    unsafe fn rng_gaussian_mv(
        method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
        r: *mut Self, dimen: c_int, mstorage: c_int,
        mean: *const Self, cov_factor: *const Self,
    ) -> c_int;
}

macro_rules! impl_rng_float {
    ($ty:ty,
        uniform=$uniform:ident, gaussian=$gaussian:ident,
        exponential=$exp:ident, lognormal=$logn:ident,
        cauchy=$cauchy:ident, weibull=$weibull:ident,
        gamma=$gamma:ident, beta=$beta:ident,
        gumbel=$gumbel:ident, laplace=$laplace:ident,
        rayleigh=$rayleigh:ident, chi_square=$chi_square:ident,
        gaussian_mv=$gmv:ident,
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
            unsafe fn rng_gumbel(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, displacement: Self, scale: Self,
            ) -> c_int {
                unsafe { sys::$gumbel(method, stream, n, r, displacement, scale) }
            }
            unsafe fn rng_laplace(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, displacement: Self, scale: Self,
            ) -> c_int {
                unsafe { sys::$laplace(method, stream, n, r, displacement, scale) }
            }
            unsafe fn rng_rayleigh(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, displacement: Self, scale: Self,
            ) -> c_int {
                unsafe { sys::$rayleigh(method, stream, n, r, displacement, scale) }
            }
            unsafe fn rng_chi_square(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, ndf: c_int,
            ) -> c_int {
                unsafe { sys::$chi_square(method, stream, n, r, ndf) }
            }
            unsafe fn rng_gaussian_mv(
                method: c_int, stream: VSLStreamStatePtr, n: onemkl_sys::MKL_INT,
                r: *mut Self, dimen: c_int, mstorage: c_int,
                mean: *const Self, cov_factor: *const Self,
            ) -> c_int {
                unsafe {
                    sys::$gmv(method, stream, n, r, dimen, mstorage, mean, cov_factor)
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
    gumbel=vsRngGumbel, laplace=vsRngLaplace,
    rayleigh=vsRngRayleigh, chi_square=vsRngChiSquare,
    gaussian_mv=vsRngGaussianMV,
);

impl_rng_float!(
    f64,
    uniform=vdRngUniform, gaussian=vdRngGaussian,
    exponential=vdRngExponential, lognormal=vdRngLognormal,
    cauchy=vdRngCauchy, weibull=vdRngWeibull,
    gamma=vdRngGamma, beta=vdRngBeta,
    gumbel=vdRngGumbel, laplace=vdRngLaplace,
    rayleigh=vdRngRayleigh, chi_square=vdRngChiSquare,
    gaussian_mv=vdRngGaussianMV,
);

#[inline]
fn check_vsl(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::VslStatus(status))
    }
}
