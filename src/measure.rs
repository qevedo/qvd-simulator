//! Measurement: probabilities, sampling and collapse. Every sum is
//! accumulated in double precision, whatever the amplitude precision.

use rand::{Rng, RngExt};
use rayon::prelude::*;

use crate::kernels::DiagonalPlan;
use crate::matrix::C64;
use crate::simd::Real;
use crate::state::StateVector;

/// Blocks per parallel chunk when summing probabilities.
const CHUNK_BLOCKS: usize = 1 << 12;

impl<T: Real> StateVector<T> {
    fn block_probabilities(block: &[T]) -> impl Iterator<Item = f64> + '_ {
        let lanes = T::LANES;
        (0..lanes).map(move |l| {
            let (re, im) = (block[l].to_f64(), block[l + lanes].to_f64());
            re * re + im * im
        })
    }

    /// Probability of each basis state, in (logical) index order.
    pub fn probabilities(&self) -> Vec<f64> {
        let lanes = T::LANES;
        let mut physical: Vec<f64> = self
            .raw()
            .par_chunks(2 * lanes)
            .flat_map_iter(Self::block_probabilities)
            .collect();
        physical.truncate(self.len());
        if self.layout().iter().enumerate().all(|(q, &p)| q == p) {
            return physical;
        }
        (0..self.len())
            .into_par_iter()
            .map(|i| physical[self.physical_index(i)])
            .collect()
    }

    /// Probability that `qubit` measures 1.
    pub fn probability_one(&self, qubit: usize) -> f64 {
        assert!(qubit < self.num_qubits());
        let qubit = self.layout()[qubit];
        let lanes = T::LANES;
        let lane_bits = T::LANE_BITS as usize;
        self.raw()
            .par_chunks(2 * lanes * CHUNK_BLOCKS)
            .enumerate()
            .map(|(chunk, data)| {
                data.chunks_exact(2 * lanes)
                    .enumerate()
                    .map(|(b, block)| {
                        let index = (chunk * CHUNK_BLOCKS + b) << lane_bits;
                        Self::block_probabilities(block)
                            .enumerate()
                            .filter(|&(lane, _)| ((index | lane) >> qubit) & 1 == 1)
                            .map(|(_, p)| p)
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    /// Measure `qubit`, collapse the state and return the outcome.
    pub fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> bool {
        let p1 = self.probability_one(qubit).clamp(0.0, 1.0);
        let outcome = rng.random::<f64>() < p1;
        let p = if outcome { p1 } else { 1.0 - p1 };
        let keep = C64::new(1.0 / p.sqrt(), 0.0);
        let zero = C64::new(0.0, 0.0);
        let entries = if outcome { [zero, keep] } else { [keep, zero] };
        let plan =
            DiagonalPlan::<T>::new(&entries, &[self.layout()[qubit]], &[], self.block_bits());
        // SAFETY: the plan covers this state's full block range.
        unsafe { plan.apply(self.raw_mut().as_mut_ptr(), true) };
        outcome
    }

    /// Reset `qubit` to |0> (measure, then flip if the outcome was 1).
    pub fn reset(&mut self, qubit: usize, rng: &mut impl Rng) {
        if self.measure(qubit, rng) {
            self.apply(&crate::circuit::Gate::X.matrix(), &[qubit], &[]);
        }
    }

    /// Draw `shots` basis-state indices from |amplitude|^2, without
    /// modifying the state. Two passes over memory whatever the shot count:
    /// one for per-chunk sums, one to walk sorted uniforms through them.
    pub fn sample(&self, shots: usize, rng: &mut impl Rng) -> Vec<usize> {
        let mut outcomes = self.sample_unordered(shots, rng);
        // The walk produces outcomes in index order; shuffle into shot order.
        for i in (1..outcomes.len()).rev() {
            let j = rng.random_range(0..=i);
            outcomes.swap(i, j);
        }
        outcomes
    }

    /// Like [`sample`](Self::sample), but the outcomes come in no particular
    /// order (cheaper when only counts are needed).
    pub fn sample_unordered(&self, shots: usize, rng: &mut impl Rng) -> Vec<usize> {
        if shots == 0 {
            return Vec::new();
        }
        let lanes = T::LANES;
        let chunk_len = 2 * lanes * CHUNK_BLOCKS;
        let sums: Vec<f64> = self
            .raw()
            .par_chunks(chunk_len)
            .map(|data| {
                data.chunks_exact(2 * lanes)
                    .flat_map(Self::block_probabilities)
                    .sum()
            })
            .collect();
        let mut prefix = Vec::with_capacity(sums.len() + 1);
        prefix.push(0.0);
        for s in &sums {
            prefix.push(prefix.last().unwrap() + s);
        }
        let total = *prefix.last().unwrap();
        let mut targets: Vec<f64> = (0..shots).map(|_| rng.random::<f64>() * total).collect();
        targets.par_sort_unstable_by(f64::total_cmp);
        // Shots falling into each chunk.
        let bounds: Vec<usize> = prefix
            .iter()
            .map(|&p| targets.partition_point(|&t| t < p))
            .collect();
        let lane_bits = T::LANE_BITS as usize;
        let len = self.len();
        let mut outcomes: Vec<usize> = self
            .raw()
            .par_chunks(chunk_len)
            .enumerate()
            .flat_map_iter(|(chunk, data)| {
                let (lo, hi) = (bounds[chunk], bounds[chunk + 1].min(shots));
                let mut found = Vec::with_capacity(hi.saturating_sub(lo));
                if lo < hi {
                    let mut cumulative = prefix[chunk];
                    let mut shot = lo;
                    let mut last_nonzero = (chunk * CHUNK_BLOCKS) << lane_bits;
                    'walk: for (b, block) in data.chunks_exact(2 * lanes).enumerate() {
                        for (lane, p) in Self::block_probabilities(block).enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let index = ((chunk * CHUNK_BLOCKS + b) << lane_bits) | lane;
                            last_nonzero = index;
                            cumulative += p;
                            while shot < hi && targets[shot] < cumulative {
                                found.push(index);
                                shot += 1;
                            }
                            if shot == hi {
                                break 'walk;
                            }
                        }
                    }
                    // Rounding can leave the last few targets just past the end.
                    found.resize(hi - lo, last_nonzero);
                }
                found.into_iter().map(move |i| i.min(len - 1))
            })
            .collect();
        if self.layout().iter().enumerate().any(|(q, &p)| q != p) {
            for outcome in outcomes.iter_mut() {
                *outcome = self.logical_index(*outcome);
            }
        }
        outcomes
    }
}
