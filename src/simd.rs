//! SIMD vectors of real numbers, and the precision trait.
//!
//! The state vector is stored in blocks of `LANES` amplitudes: `LANES` real
//! parts followed by `LANES` imaginary parts (64 bytes, one cache line, in
//! both precisions). With split real and imaginary registers, a complex
//! multiply-accumulate is four fused multiply-adds and no shuffles.
//!
//! With AVX2 and FMA enabled at compile time (`-C target-cpu=native` on a
//! recent x86-64 CPU) the vectors are `__m256`/`__m256d`; otherwise a
//! portable array implementation with the same layout is used.

use std::fmt::Debug;

/// A floating-point precision for amplitudes: `f32` or `f64`.
pub trait Real: Copy + Send + Sync + Default + Debug + PartialEq + 'static {
    /// Amplitudes per block (and SIMD lanes per vector).
    const LANES: usize;
    /// `log2(LANES)`: the number of lowest qubits held inside one vector.
    const LANE_BITS: u32;
    type V: Vector<Scalar = Self>;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

/// Operations on a SIMD vector of `Real`s.
pub trait Vector: Copy {
    type Scalar: Real;
    /// A precomputed lane permutation.
    type Perm: Copy + Send + Sync;

    /// Load `LANES` values. The pointer needs no alignment.
    ///
    /// # Safety
    /// `ptr` must be valid for reading `LANES` values.
    unsafe fn load(ptr: *const Self::Scalar) -> Self;
    /// Store `LANES` values.
    ///
    /// # Safety
    /// `ptr` must be valid for writing `LANES` values.
    unsafe fn store(self, ptr: *mut Self::Scalar);
    fn splat(x: Self::Scalar) -> Self;
    fn zero() -> Self;
    /// `a * b + c`
    fn mul_add(a: Self, b: Self, c: Self) -> Self;
    /// `c - a * b`
    fn neg_mul_add(a: Self, b: Self, c: Self) -> Self;
    fn mul(a: Self, b: Self) -> Self;
    fn add(a: Self, b: Self) -> Self;
    fn sub(a: Self, b: Self) -> Self;
    /// The permutation where lane `l` takes the value of lane `l ^ xor`.
    fn xor_perm(xor: usize) -> Self::Perm;
    fn permute(self, perm: Self::Perm) -> Self;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod imp {
    #![allow(unused_unsafe)]
    use std::arch::x86_64::*;

    use super::{Real, Vector};

    #[derive(Clone, Copy)]
    pub struct F32x8(__m256);
    #[derive(Clone, Copy)]
    pub struct F64x4(__m256d);

    impl Real for f32 {
        const LANES: usize = 8;
        const LANE_BITS: u32 = 3;
        type V = F32x8;
        #[inline(always)]
        fn from_f64(x: f64) -> f32 {
            x as f32
        }
        #[inline(always)]
        fn to_f64(self) -> f64 {
            self as f64
        }
    }

    impl Real for f64 {
        const LANES: usize = 4;
        const LANE_BITS: u32 = 2;
        type V = F64x4;
        #[inline(always)]
        fn from_f64(x: f64) -> f64 {
            x
        }
        #[inline(always)]
        fn to_f64(self) -> f64 {
            self
        }
    }

    impl Vector for F32x8 {
        type Scalar = f32;
        type Perm = __m256i;
        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self {
            F32x8(unsafe { _mm256_loadu_ps(ptr) })
        }
        #[inline(always)]
        unsafe fn store(self, ptr: *mut f32) {
            unsafe { _mm256_storeu_ps(ptr, self.0) }
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            F32x8(unsafe { _mm256_set1_ps(x) })
        }
        #[inline(always)]
        fn zero() -> Self {
            F32x8(unsafe { _mm256_setzero_ps() })
        }
        #[inline(always)]
        fn mul_add(a: Self, b: Self, c: Self) -> Self {
            F32x8(unsafe { _mm256_fmadd_ps(a.0, b.0, c.0) })
        }
        #[inline(always)]
        fn neg_mul_add(a: Self, b: Self, c: Self) -> Self {
            F32x8(unsafe { _mm256_fnmadd_ps(a.0, b.0, c.0) })
        }
        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            F32x8(unsafe { _mm256_mul_ps(a.0, b.0) })
        }
        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            F32x8(unsafe { _mm256_add_ps(a.0, b.0) })
        }
        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            F32x8(unsafe { _mm256_sub_ps(a.0, b.0) })
        }
        #[inline(always)]
        fn xor_perm(xor: usize) -> __m256i {
            let x = xor as i32;
            unsafe { _mm256_setr_epi32(x, 1 ^ x, 2 ^ x, 3 ^ x, 4 ^ x, 5 ^ x, 6 ^ x, 7 ^ x) }
        }
        #[inline(always)]
        fn permute(self, perm: __m256i) -> Self {
            F32x8(unsafe { _mm256_permutevar8x32_ps(self.0, perm) })
        }
    }

    impl Vector for F64x4 {
        type Scalar = f64;
        type Perm = __m256i;
        #[inline(always)]
        unsafe fn load(ptr: *const f64) -> Self {
            F64x4(unsafe { _mm256_loadu_pd(ptr) })
        }
        #[inline(always)]
        unsafe fn store(self, ptr: *mut f64) {
            unsafe { _mm256_storeu_pd(ptr, self.0) }
        }
        #[inline(always)]
        fn splat(x: f64) -> Self {
            F64x4(unsafe { _mm256_set1_pd(x) })
        }
        #[inline(always)]
        fn zero() -> Self {
            F64x4(unsafe { _mm256_setzero_pd() })
        }
        #[inline(always)]
        fn mul_add(a: Self, b: Self, c: Self) -> Self {
            F64x4(unsafe { _mm256_fmadd_pd(a.0, b.0, c.0) })
        }
        #[inline(always)]
        fn neg_mul_add(a: Self, b: Self, c: Self) -> Self {
            F64x4(unsafe { _mm256_fnmadd_pd(a.0, b.0, c.0) })
        }
        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            F64x4(unsafe { _mm256_mul_pd(a.0, b.0) })
        }
        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            F64x4(unsafe { _mm256_add_pd(a.0, b.0) })
        }
        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            F64x4(unsafe { _mm256_sub_pd(a.0, b.0) })
        }
        #[inline(always)]
        fn xor_perm(xor: usize) -> __m256i {
            // Move 64-bit lanes as pairs of 32-bit lanes.
            let s = |lane: usize| ((lane ^ xor) * 2) as i32;
            unsafe {
                _mm256_setr_epi32(
                    s(0),
                    s(0) + 1,
                    s(1),
                    s(1) + 1,
                    s(2),
                    s(2) + 1,
                    s(3),
                    s(3) + 1,
                )
            }
        }
        #[inline(always)]
        fn permute(self, perm: __m256i) -> Self {
            F64x4(unsafe {
                _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(self.0), perm))
            })
        }
    }
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
mod imp {
    use super::{Real, Vector};

    macro_rules! portable {
        ($name:ident, $t:ty, $lanes:expr, $bits:expr) => {
            #[derive(Clone, Copy)]
            pub struct $name([$t; $lanes]);

            impl Real for $t {
                const LANES: usize = $lanes;
                const LANE_BITS: u32 = $bits;
                type V = $name;
                #[inline(always)]
                fn from_f64(x: f64) -> $t {
                    x as $t
                }
                #[inline(always)]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }

            impl Vector for $name {
                type Scalar = $t;
                type Perm = [u8; $lanes];
                #[inline(always)]
                unsafe fn load(ptr: *const $t) -> Self {
                    $name(unsafe { std::ptr::read_unaligned(ptr as *const [$t; $lanes]) })
                }
                #[inline(always)]
                unsafe fn store(self, ptr: *mut $t) {
                    unsafe { std::ptr::write_unaligned(ptr as *mut [$t; $lanes], self.0) }
                }
                #[inline(always)]
                fn splat(x: $t) -> Self {
                    $name([x; $lanes])
                }
                #[inline(always)]
                fn zero() -> Self {
                    $name([0.0; $lanes])
                }
                #[inline(always)]
                fn mul_add(a: Self, b: Self, c: Self) -> Self {
                    $name(std::array::from_fn(|i| a.0[i].mul_add(b.0[i], c.0[i])))
                }
                #[inline(always)]
                fn neg_mul_add(a: Self, b: Self, c: Self) -> Self {
                    $name(std::array::from_fn(|i| (-a.0[i]).mul_add(b.0[i], c.0[i])))
                }
                #[inline(always)]
                fn mul(a: Self, b: Self) -> Self {
                    $name(std::array::from_fn(|i| a.0[i] * b.0[i]))
                }
                #[inline(always)]
                fn add(a: Self, b: Self) -> Self {
                    $name(std::array::from_fn(|i| a.0[i] + b.0[i]))
                }
                #[inline(always)]
                fn sub(a: Self, b: Self) -> Self {
                    $name(std::array::from_fn(|i| a.0[i] - b.0[i]))
                }
                #[inline(always)]
                fn xor_perm(xor: usize) -> [u8; $lanes] {
                    std::array::from_fn(|i| (i ^ xor) as u8)
                }
                #[inline(always)]
                fn permute(self, perm: [u8; $lanes]) -> Self {
                    $name(std::array::from_fn(|i| self.0[perm[i] as usize]))
                }
            }
        };
    }

    portable!(F32x8, f32, 8, 3);
    portable!(F64x4, f64, 4, 2);
}

pub use imp::{F32x8, F64x4};

/// Whether the AVX2 + FMA kernels were compiled in.
pub const fn simd_backend() -> &'static str {
    if cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )) {
        "avx2+fma"
    } else {
        "portable"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T: Real>(values: &[f64]) -> Vec<f64> {
        let input: Vec<T> = values.iter().map(|&x| T::from_f64(x)).collect();
        let mut out = vec![T::default(); T::LANES];
        unsafe {
            T::V::load(input.as_ptr())
                .permute(T::V::xor_perm(1))
                .store(out.as_mut_ptr())
        };
        out.into_iter().map(T::to_f64).collect()
    }

    #[test]
    fn xor_permutation_swaps_adjacent_lanes() {
        assert_eq!(
            roundtrip::<f32>(&[0., 1., 2., 3., 4., 5., 6., 7.]),
            vec![1., 0., 3., 2., 5., 4., 7., 6.]
        );
        assert_eq!(roundtrip::<f64>(&[0., 1., 2., 3.]), vec![1., 0., 3., 2.]);
    }

    #[test]
    fn fused_multiply_add() {
        let a = <f64 as Real>::V::splat(2.0);
        let b = <f64 as Real>::V::splat(3.0);
        let c = <f64 as Real>::V::splat(1.0);
        let mut out = [0.0f64; 4];
        unsafe { <f64 as Real>::V::neg_mul_add(a, b, c).store(out.as_mut_ptr()) };
        assert_eq!(out, [-5.0; 4]);
    }
}
