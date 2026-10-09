//! The state vector in the blocked layout (see [`crate::simd`]).

use rayon::prelude::*;

use crate::kernels::Plan;
use crate::matrix::{C64, Matrix};
use crate::memory::Buffer;
use crate::simd::Real;

/// A pure state of `num_qubits` qubits, with amplitudes stored as `T`.
///
/// Amplitude `i` has qubit `q` equal to bit `q` of `i` (qubit 0 is the least
/// significant bit, as in Qiskit). Internally, logical qubits may sit at
/// other *physical* bit positions: the cache-blocked scheduler relabels
/// qubits instead of moving data back. Every public method takes and
/// returns logical qubits and indices.
pub struct StateVector<T: Real> {
    num_qubits: usize,
    buffer: Buffer<T>,
    /// `layout[logical] = physical` bit position.
    layout: Vec<usize>,
}

/// A raw pointer to blocks that threads of one pass may share; every thread
/// touches a disjoint set of blocks.
#[derive(Clone, Copy)]
struct Blocks<T>(*mut T);
unsafe impl<T> Send for Blocks<T> {}
unsafe impl<T> Sync for Blocks<T> {}
impl<T> Blocks<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

impl<T: Real> StateVector<T> {
    /// The state |0...0>. Fails if the memory cannot be mapped.
    pub fn zero(num_qubits: usize) -> std::io::Result<Self> {
        assert!(num_qubits < usize::BITS as usize - 8, "too many qubits");
        let amplitudes = (1usize << num_qubits).max(T::LANES);
        let mut state = StateVector {
            num_qubits,
            buffer: Buffer::zeroed(2 * amplitudes)?,
            layout: (0..num_qubits).collect(),
        };
        state.buffer.as_mut_slice()[0] = T::from_f64(1.0);
        Ok(state)
    }

    /// A state with the given amplitudes (`2^n` of them).
    pub fn from_amplitudes(amplitudes: &[C64]) -> std::io::Result<Self> {
        assert!(
            amplitudes.len().is_power_of_two(),
            "the number of amplitudes must be a power of two"
        );
        let mut state = Self::zero(amplitudes.len().trailing_zeros() as usize)?;
        state.buffer.as_mut_slice()[0] = T::from_f64(0.0);
        for (i, &a) in amplitudes.iter().enumerate() {
            state.set(i, a);
        }
        Ok(state)
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of amplitudes, `2^num_qubits`.
    pub fn len(&self) -> usize {
        1 << self.num_qubits
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Bytes of memory holding amplitudes.
    pub fn memory_bytes(&self) -> usize {
        self.buffer.len() * size_of::<T>()
    }

    /// Number of blocks of `T::LANES` amplitudes.
    pub fn blocks(&self) -> usize {
        self.buffer.len() / (2 * T::LANES)
    }

    /// `log2(blocks)`.
    pub fn block_bits(&self) -> u32 {
        self.blocks().trailing_zeros()
    }

    /// `layout()[q]` is the physical bit position of logical qubit `q`.
    pub fn layout(&self) -> &[usize] {
        &self.layout
    }

    /// The physical index of logical amplitude `index`.
    pub fn physical_index(&self, index: usize) -> usize {
        self.layout
            .iter()
            .enumerate()
            .map(|(q, &p)| ((index >> q) & 1) << p)
            .sum()
    }

    /// The logical index of physical amplitude `index`.
    pub fn logical_index(&self, index: usize) -> usize {
        self.layout
            .iter()
            .enumerate()
            .map(|(q, &p)| ((index >> p) & 1) << q)
            .sum()
    }

    fn is_identity_layout(&self) -> bool {
        self.layout.iter().enumerate().all(|(q, &p)| q == p)
    }

    #[inline]
    fn offsets(physical: usize) -> (usize, usize) {
        let re = (physical >> T::LANE_BITS) * 2 * T::LANES + (physical & (T::LANES - 1));
        (re, re + T::LANES)
    }

    fn get_physical(&self, physical: usize) -> C64 {
        let (re, im) = Self::offsets(physical);
        let data = self.buffer.as_slice();
        C64::new(data[re].to_f64(), data[im].to_f64())
    }

    pub fn get(&self, index: usize) -> C64 {
        assert!(index < self.len());
        self.get_physical(self.physical_index(index))
    }

    pub fn set(&mut self, index: usize, value: C64) {
        assert!(index < self.len());
        let (re, im) = Self::offsets(self.physical_index(index));
        let data = self.buffer.as_mut_slice();
        data[re] = T::from_f64(value.re);
        data[im] = T::from_f64(value.im);
    }

    /// All amplitudes, in (logical) index order.
    pub fn to_vec(&self) -> Vec<C64> {
        if self.is_identity_layout() {
            return (0..self.len())
                .into_par_iter()
                .with_min_len(1 << 14)
                .map(|i| self.get_physical(i))
                .collect();
        }
        (0..self.len())
            .into_par_iter()
            .with_min_len(1 << 14)
            .map(|i| self.get(i))
            .collect()
    }

    /// The raw blocked data in physical order: per block, `LANES` real then
    /// `LANES` imaginary parts.
    pub fn raw(&self) -> &[T] {
        self.buffer.as_slice()
    }

    pub fn raw_mut(&mut self) -> &mut [T] {
        self.buffer.as_mut_slice()
    }

    /// `<psi|psi>`, accumulated in double precision.
    pub fn norm_sqr(&self) -> f64 {
        self.raw()
            .par_chunks(1 << 14)
            .map(|chunk| chunk.iter().map(|&x| x.to_f64() * x.to_f64()).sum::<f64>())
            .sum()
    }

    /// Apply a unitary `matrix` (index bit j = `qubits[j]`) when all
    /// `controls` `(qubit, value)` hold. Qubits are logical.
    pub fn apply(&mut self, matrix: &Matrix, qubits: &[usize], controls: &[(usize, bool)]) {
        for &q in qubits.iter().chain(controls.iter().map(|(q, _)| q)) {
            assert!(
                q < self.num_qubits,
                "qubit {q} out of range for {} qubits",
                self.num_qubits
            );
        }
        if qubits.is_empty() {
            return;
        }
        let targets: Vec<usize> = qubits.iter().map(|&q| self.layout[q]).collect();
        let controls: Vec<(usize, bool)> =
            controls.iter().map(|&(q, v)| (self.layout[q], v)).collect();
        let plan = Plan::<T>::new(matrix, &targets, &controls, self.block_bits());
        // SAFETY: the plan was built for this state's full block range.
        unsafe { plan.apply(self.buffer.as_mut_ptr(), true) };
    }

    /// Multiply every amplitude by `factor` (used to renormalise).
    pub fn scale(&mut self, factor: f64) {
        self.raw_mut().par_chunks_mut(1 << 14).for_each(|chunk| {
            for x in chunk {
                *x = T::from_f64(x.to_f64() * factor);
            }
        });
    }

    /// Exchange physical bit positions `a <-> b` for each pair, in one pass
    /// over memory, and relabel the logical qubits that lived there. Every
    /// position must be a block bit (`>= LANE_BITS`) and appear only once.
    pub fn swap_physical(&mut self, pairs: &[(usize, usize)]) {
        if pairs.is_empty() {
            return;
        }
        let lane_bits = T::LANE_BITS as usize;
        let mut seen = 0usize;
        for &(a, b) in pairs {
            assert!(
                a >= lane_bits && b >= lane_bits && a != b,
                "swap of a lane qubit"
            );
            let (ba, bb) = (1usize << (a - lane_bits), 1usize << (b - lane_bits));
            assert!(seen & (ba | bb) == 0, "qubit swapped twice in one pass");
            seen |= ba | bb;
        }
        let shifts: Vec<(usize, usize)> = pairs
            .iter()
            .map(|&(a, b)| (a - lane_bits, b - lane_bits))
            .collect();
        let permute = move |i: usize| -> usize {
            let mut j = i & !seen;
            for &(a, b) in &shifts {
                j |= ((i >> a) & 1) << b | ((i >> b) & 1) << a;
            }
            j
        };
        let block_len = 2 * T::LANES;
        let blocks = Blocks(self.buffer.as_mut_ptr());
        // Every block i whose a-bits and b-bits differ pairs with
        // permute(i); visiting only i < permute(i) swaps each pair once.
        (0..self.blocks())
            .into_par_iter()
            .with_min_len(1 << 12)
            .for_each(|i| {
                let j = permute(i);
                if i < j {
                    // SAFETY: i and j are distinct blocks and each pair is
                    // visited by exactly one iteration.
                    unsafe {
                        std::ptr::swap_nonoverlapping(
                            blocks.get().add(i * block_len),
                            blocks.get().add(j * block_len),
                            block_len,
                        )
                    };
                }
            });
        for q in self.layout.iter_mut() {
            for &(a, b) in pairs {
                if *q == a {
                    *q = b;
                } else if *q == b {
                    *q = a;
                }
            }
        }
    }
}
