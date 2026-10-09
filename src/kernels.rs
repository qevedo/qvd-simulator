//! Gate kernels for the blocked state-vector layout.
//!
//! A gate's targets split into *high* qubits (bit position >= `LANE_BITS`,
//! i.e. they select whole blocks) and *low* qubits (inside a block, i.e.
//! inside one SIMD register). For each group of blocks that differ only in
//! the high target bits, the kernel loads the `NH = 2^high` blocks, makes
//! `NT = 2^low` lane-permuted copies of each (lane `l` reads lane `l ^ s`
//! for every pattern `s` of the low target bits), and computes every output
//! block as a sum of `NH * NT` complex products with coefficient vectors
//! precomputed per gate. Without low targets, coefficients are broadcast
//! scalars. This is the scheme of qsim's AVX kernels.
//!
//! Kernels operate on a *region*: `2^region_bits` consecutive blocks. The
//! whole state is one region; the cache-blocked scheduler applies gates to
//! cache-sized regions one at a time.

use rayon::prelude::*;

use crate::matrix::{C64, Matrix};
use crate::simd::{Real, Vector};

/// Deposit the low bits of `value` at the set positions of `mask` (BMI2 pdep).
#[inline(always)]
pub fn deposit(value: usize, mask: usize) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // SAFETY: the bmi2 target feature is enabled at compile time.
        unsafe { std::arch::x86_64::_pdep_u64(value as u64, mask as u64) as usize }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let (mut out, mut mask, mut bit) = (0usize, mask, 1usize);
        while mask != 0 {
            let lowest = mask & mask.wrapping_neg();
            if value & bit != 0 {
                out |= lowest;
            }
            mask &= mask - 1;
            bit <<= 1;
        }
        out
    }
}

/// A raw pointer that may be shared between the threads of one kernel call;
/// every thread writes a disjoint set of blocks.
#[derive(Clone, Copy)]
struct SharedPtr<T>(*mut T);
unsafe impl<T> Send for SharedPtr<T> {}
unsafe impl<T> Sync for SharedPtr<T> {}

impl<T> SharedPtr<T> {
    /// Accessing the pointer through a method makes closures capture the
    /// whole wrapper (which is `Sync`) rather than the raw field.
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// Iterations (groups of blocks) per parallel task. Large enough to amortise
/// scheduling, small enough to balance across threads.
const TASK_ITERATIONS: usize = 1 << 11;

/// Run `body(start, end)` over `0..count`, in parallel when worthwhile.
fn for_ranges(count: usize, parallel: bool, body: impl Fn(usize, usize) + Sync) {
    if !parallel || count <= TASK_ITERATIONS || rayon::current_num_threads() == 1 {
        body(0, count);
        return;
    }
    let tasks = count.div_ceil(TASK_ITERATIONS);
    (0..tasks).into_par_iter().for_each(|task| {
        let start = task * TASK_ITERATIONS;
        body(start, (start + TASK_ITERATIONS).min(count));
    });
}

/// Targets and controls of a gate, split by position relative to a block.
struct Layout {
    /// `(matrix bit, block bit)` of each high target.
    high: Vec<(usize, u32)>,
    /// `(matrix bit, lane bit)` of each low target.
    low: Vec<(usize, u32)>,
    /// Block offset of each combination of high target bits.
    offsets: Vec<usize>,
    /// Region block bits not used by targets or controls: the loop index.
    free_mask: usize,
    /// Block bits fixed by controls, and their required values.
    control_value: usize,
}

impl Layout {
    fn new<T: Real>(targets: &[usize], controls: &[(usize, bool)], region_bits: u32) -> Layout {
        let lane_bits = T::LANE_BITS as usize;
        let mut high = Vec::new();
        let mut low = Vec::new();
        for (j, &q) in targets.iter().enumerate() {
            if q < lane_bits {
                low.push((j, q as u32));
            } else {
                high.push((j, (q - lane_bits) as u32));
            }
        }
        // `deposit` numbers the high combinations by ascending bit position,
        // so the matrix-bit lookup must use the same order.
        high.sort_by_key(|&(_, bit)| bit);
        let high_mask: usize = high.iter().map(|&(_, b)| 1 << b).sum();
        let offsets = (0..1usize << high.len())
            .map(|h| deposit(h, high_mask))
            .collect();
        let mut control_mask = 0;
        let mut control_value = 0;
        for &(q, value) in controls {
            assert!(
                q >= lane_bits,
                "low control qubits must be folded into the matrix"
            );
            control_mask |= 1 << (q - lane_bits);
            if value {
                control_value |= 1 << (q - lane_bits);
            }
        }
        let region_mask = (1usize << region_bits) - 1;
        assert!(
            (high_mask | control_mask) & !region_mask == 0,
            "gate qubit outside the region"
        );
        Layout {
            high,
            low,
            offsets,
            free_mask: region_mask & !high_mask & !control_mask,
            control_value,
        }
    }

    /// The matrix index of the amplitude in high combination `h`, lane `lane`.
    fn matrix_index(&self, h: usize, lane: usize) -> usize {
        let mut index = 0;
        for (m, &(j, _)) in self.high.iter().enumerate() {
            index |= ((h >> m) & 1) << j;
        }
        for &(j, bit) in &self.low {
            index |= ((lane >> bit) & 1) << j;
        }
        index
    }

    fn low_lane_mask(&self) -> usize {
        self.low.iter().map(|&(_, bit)| 1 << bit).sum()
    }

    fn iterations(&self) -> usize {
        1 << self.free_mask.count_ones()
    }
}

/// A dense gate prepared for one state layout and region size.
pub struct DensePlan<T: Real> {
    layout: Layout,
    nh: usize,
    nt: usize,
    perms: Vec<<T::V as Vector>::Perm>,
    /// For each `(row, col, t)`, `LANES` real then `LANES` imaginary
    /// coefficients (one per lane; without low targets all lanes are equal,
    /// which still beats broadcasting a scalar on every use).
    coefficients: Vec<T>,
}

impl<T: Real> DensePlan<T> {
    /// Prepare `matrix` on `targets` (index bit j = `targets[j]`), applied when
    /// every `(qubit, value)` control holds. Controls must be high qubits.
    pub fn new(
        matrix: &Matrix,
        targets: &[usize],
        controls: &[(usize, bool)],
        region_bits: u32,
    ) -> Self {
        assert_eq!(matrix.qubits(), targets.len());
        let layout = Layout::new::<T>(targets, controls, region_bits);
        let nh = 1 << layout.high.len();
        let nt = 1 << layout.low.len();
        let low_mask = layout.low_lane_mask();
        let spreads: Vec<usize> = (0..nt).map(|t| deposit(t, low_mask)).collect();
        let perms = spreads.iter().map(|&s| T::V::xor_perm(s)).collect();
        let lanes = T::LANES;
        let mut coefficients = Vec::new();
        for row in 0..nh {
            for col in 0..nh {
                for &s in &spreads {
                    let entry = |lane: usize| {
                        matrix[(
                            layout.matrix_index(row, lane),
                            layout.matrix_index(col, lane ^ s),
                        )]
                    };
                    coefficients.extend((0..lanes).map(|lane| T::from_f64(entry(lane).re)));
                    coefficients.extend((0..lanes).map(|lane| T::from_f64(entry(lane).im)));
                }
            }
        }
        DensePlan {
            layout,
            nh,
            nt,
            perms,
            coefficients,
        }
    }

    /// Apply the gate to the region of `2^region_bits` blocks at `region`.
    ///
    /// # Safety
    /// `region` must point to the start of a region of the size the plan was
    /// built for, valid for reads and writes and not accessed concurrently
    /// except by this call.
    pub unsafe fn apply(&self, region: *mut T, parallel: bool) {
        macro_rules! dispatch {
            ($(($nh:literal, $nt:literal)),*) => {
                match (self.nh, self.nt) {
                    $(($nh, $nt) => unsafe { dense_sweep::<T, $nh, $nt>(self, region, parallel) },)*
                    (nh, nt) => panic!("unsupported dense gate shape: {nh} high x {nt} low combinations"),
                }
            };
        }
        dispatch!(
            (1, 2),
            (1, 4),
            (1, 8),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 8),
            (4, 1),
            (4, 2),
            (4, 4),
            (4, 8),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 8),
            (16, 1),
            (16, 2),
            (16, 4),
            (32, 1),
            (32, 2),
            (64, 1)
        )
    }
}

unsafe fn dense_sweep<T: Real, const NH: usize, const NT: usize>(
    plan: &DensePlan<T>,
    region: *mut T,
    parallel: bool,
) {
    let lanes = T::LANES;
    let region = SharedPtr(region);
    let free = plan.layout.free_mask;
    let control_value = plan.layout.control_value;
    let offsets: [usize; NH] = std::array::from_fn(|h| plan.layout.offsets[h] * 2 * lanes);
    let perms: [<T::V as Vector>::Perm; NT] = std::array::from_fn(|t| plan.perms[t]);
    let coefficients = plan.coefficients.as_ptr();
    let coefficients = SharedPtr(coefficients as *mut T);
    for_ranges(plan.layout.iterations(), parallel, |start, end| {
        let coefficients = coefficients.get() as *const T;
        let mut base = deposit(start, free) | control_value;
        for _ in start..end {
            // SAFETY: `base | offsets[h]` enumerates distinct blocks inside the
            // region; each iteration owns its NH blocks exclusively.
            unsafe {
                let block = region.get().add(base * 2 * lanes);
                // Inputs and their lane-permuted copies, built in place.
                let re: [[T::V; NT]; NH] = std::array::from_fn(|h| {
                    let x = T::V::load(block.add(offsets[h]));
                    std::array::from_fn(|t| if t == 0 { x } else { x.permute(perms[t]) })
                });
                let im: [[T::V; NT]; NH] = std::array::from_fn(|h| {
                    let x = T::V::load(block.add(offsets[h] + lanes));
                    std::array::from_fn(|t| if t == 0 { x } else { x.permute(perms[t]) })
                });
                // Two output rows at a time, each with four independent
                // accumulators, so enough FMAs are in flight to hide their
                // latency (a single accumulator chain runs ~4x slower).
                if NH == 1 {
                    dense_row::<T, NH, NT>(0, &re, &im, coefficients, block, &offsets);
                } else {
                    for pair in 0..NH / 2 {
                        dense_row_pair::<T, NH, NT>(
                            2 * pair,
                            &re,
                            &im,
                            coefficients,
                            block,
                            &offsets,
                        );
                    }
                }
            }
            base = (((base | !free).wrapping_add(1)) & free) | control_value;
        }
    });
}

/// Compute output rows `row` and `row + 1` of a dense gate and store them.
///
/// The eight accumulators are separate variables (not an array) so they stay
/// in registers; each takes one FMA per input term, so eight independent
/// FMA chains hide the FMA latency.
#[inline(always)]
unsafe fn dense_row_pair<T: Real, const NH: usize, const NT: usize>(
    row: usize,
    re: &[[T::V; NT]; NH],
    im: &[[T::V; NT]; NH],
    coefficients: *const T,
    block: *mut T,
    offsets: &[usize; NH],
) {
    let lanes = T::LANES;
    let stride = NT * 2 * lanes; // coefficients per (row, col)
    let (mut rr0, mut ir0, mut ii0, mut ri0) =
        (T::V::zero(), T::V::zero(), T::V::zero(), T::V::zero());
    let (mut rr1, mut ir1, mut ii1, mut ri1) =
        (T::V::zero(), T::V::zero(), T::V::zero(), T::V::zero());
    // SAFETY: indices stay within the plan's coefficient table and the
    // iteration's own output blocks.
    unsafe {
        let c0 = coefficients.add(row * NH * stride);
        let c1 = c0.add(NH * stride);
        for col in 0..NH {
            for t in 0..NT {
                let (x_re, x_im) = (re[col][t], im[col][t]);
                let o = col * stride + t * 2 * lanes;
                let (w_re, w_im) = (T::V::load(c0.add(o)), T::V::load(c0.add(o + lanes)));
                rr0 = T::V::mul_add(w_re, x_re, rr0);
                ir0 = T::V::mul_add(w_re, x_im, ir0);
                ii0 = T::V::mul_add(w_im, x_im, ii0);
                ri0 = T::V::mul_add(w_im, x_re, ri0);
                let (w_re, w_im) = (T::V::load(c1.add(o)), T::V::load(c1.add(o + lanes)));
                rr1 = T::V::mul_add(w_re, x_re, rr1);
                ir1 = T::V::mul_add(w_re, x_im, ir1);
                ii1 = T::V::mul_add(w_im, x_im, ii1);
                ri1 = T::V::mul_add(w_im, x_re, ri1);
            }
        }
        let p = block.add(offsets[row]);
        T::V::sub(rr0, ii0).store(p);
        T::V::add(ir0, ri0).store(p.add(lanes));
        let p = block.add(offsets[row + 1]);
        T::V::sub(rr1, ii1).store(p);
        T::V::add(ir1, ri1).store(p.add(lanes));
    }
}

/// Compute a single output row (gates with only in-register targets).
#[inline(always)]
unsafe fn dense_row<T: Real, const NH: usize, const NT: usize>(
    row: usize,
    re: &[[T::V; NT]; NH],
    im: &[[T::V; NT]; NH],
    coefficients: *const T,
    block: *mut T,
    offsets: &[usize; NH],
) {
    let lanes = T::LANES;
    let stride = NT * 2 * lanes;
    let (mut rr, mut ir, mut ii, mut ri) = (T::V::zero(), T::V::zero(), T::V::zero(), T::V::zero());
    // SAFETY: as in `dense_row_pair`.
    unsafe {
        let c0 = coefficients.add(row * NH * stride);
        for col in 0..NH {
            for t in 0..NT {
                let (x_re, x_im) = (re[col][t], im[col][t]);
                let o = col * stride + t * 2 * lanes;
                let (w_re, w_im) = (T::V::load(c0.add(o)), T::V::load(c0.add(o + lanes)));
                rr = T::V::mul_add(w_re, x_re, rr);
                ir = T::V::mul_add(w_re, x_im, ir);
                ii = T::V::mul_add(w_im, x_im, ii);
                ri = T::V::mul_add(w_im, x_re, ri);
            }
        }
        let p = block.add(offsets[row]);
        T::V::sub(rr, ii).store(p);
        T::V::add(ir, ri).store(p.add(lanes));
    }
}

/// A diagonal gate prepared for one state layout and region size.
pub struct DiagonalPlan<T: Real> {
    layout: Layout,
    nh: usize,
    /// For each high combination, `LANES` real then `LANES` imaginary factors.
    factors: Vec<T>,
}

impl<T: Real> DiagonalPlan<T> {
    /// Prepare the diagonal `entries` (index bit j = `targets[j]`).
    pub fn new(
        entries: &[C64],
        targets: &[usize],
        controls: &[(usize, bool)],
        region_bits: u32,
    ) -> Self {
        assert_eq!(entries.len(), 1 << targets.len());
        let layout = Layout::new::<T>(targets, controls, region_bits);
        let nh = 1 << layout.high.len();
        let mut factors = Vec::with_capacity(nh * 2 * T::LANES);
        for h in 0..nh {
            factors.extend(
                (0..T::LANES).map(|lane| T::from_f64(entries[layout.matrix_index(h, lane)].re)),
            );
            factors.extend(
                (0..T::LANES).map(|lane| T::from_f64(entries[layout.matrix_index(h, lane)].im)),
            );
        }
        DiagonalPlan {
            layout,
            nh,
            factors,
        }
    }

    /// Apply the gate; see [`DensePlan::apply`].
    ///
    /// # Safety
    /// As for [`DensePlan::apply`].
    pub unsafe fn apply(&self, region: *mut T, parallel: bool) {
        match self.nh {
            1 => unsafe { diagonal_sweep::<T, 1>(self, region, parallel) },
            2 => unsafe { diagonal_sweep::<T, 2>(self, region, parallel) },
            4 => unsafe { diagonal_sweep::<T, 4>(self, region, parallel) },
            8 => unsafe { diagonal_sweep::<T, 8>(self, region, parallel) },
            16 => unsafe { diagonal_sweep::<T, 16>(self, region, parallel) },
            32 => unsafe { diagonal_sweep::<T, 32>(self, region, parallel) },
            64 => unsafe { diagonal_sweep::<T, 64>(self, region, parallel) },
            nh => panic!("unsupported diagonal gate with {nh} high combinations"),
        }
    }
}

unsafe fn diagonal_sweep<T: Real, const NH: usize>(
    plan: &DiagonalPlan<T>,
    region: *mut T,
    parallel: bool,
) {
    let lanes = T::LANES;
    let region = SharedPtr(region);
    let free = plan.layout.free_mask;
    let control_value = plan.layout.control_value;
    let offsets: [usize; NH] = std::array::from_fn(|h| plan.layout.offsets[h] * 2 * lanes);
    let factors = SharedPtr(plan.factors.as_ptr() as *mut T);
    for_ranges(plan.layout.iterations(), parallel, |start, end| {
        let factors = factors.get() as *const T;
        let mut base = deposit(start, free) | control_value;
        for _ in start..end {
            // SAFETY: as in `dense_sweep`.
            unsafe {
                let block = region.get().add(base * 2 * lanes);
                for (h, &offset) in offsets.iter().enumerate() {
                    let f = factors.add(h * 2 * lanes);
                    let (f_re, f_im) = (T::V::load(f), T::V::load(f.add(lanes)));
                    let p = block.add(offset);
                    let (re, im) = (T::V::load(p), T::V::load(p.add(lanes)));
                    let out_re = T::V::neg_mul_add(f_im, im, T::V::mul(f_re, re));
                    let out_im = T::V::mul_add(f_im, re, T::V::mul(f_re, im));
                    out_re.store(p);
                    out_im.store(p.add(lanes));
                }
            }
            base = (((base | !free).wrapping_add(1)) & free) | control_value;
        }
    });
}

/// Widest gate (targets plus folded low controls) the kernels apply.
pub const MAX_GATE_QUBITS: usize = 6;

/// A gate prepared for a state layout: the dense or diagonal kernel.
pub enum Plan<T: Real> {
    Dense(DensePlan<T>),
    Diagonal(DiagonalPlan<T>),
}

impl<T: Real> Plan<T> {
    /// Prepare `matrix` on physical `qubits`, with `controls` as
    /// `(physical qubit, required value)`. Controls on low qubits are folded
    /// into the matrix, since they cannot select whole blocks.
    pub fn new(
        matrix: &Matrix,
        qubits: &[usize],
        controls: &[(usize, bool)],
        region_bits: u32,
    ) -> Plan<T> {
        let lane_bits = T::LANE_BITS as usize;
        let (low, high): (Vec<_>, Vec<_>) = controls.iter().partition(|&&(q, _)| q < lane_bits);
        let (matrix, targets) = if low.is_empty() {
            (matrix.clone(), qubits.to_vec())
        } else {
            let state = low
                .iter()
                .enumerate()
                .map(|(i, &(_, v))| (v as usize) << i)
                .sum();
            let targets: Vec<usize> = low
                .iter()
                .map(|&(q, _)| q)
                .chain(qubits.iter().copied())
                .collect();
            (matrix.controlled(low.len(), state), targets)
        };
        assert!(
            targets.len() <= MAX_GATE_QUBITS,
            "gates on more than {MAX_GATE_QUBITS} qubits (including controls on qubits below {lane_bits}) are not supported"
        );
        if matrix.is_diagonal() {
            Plan::Diagonal(DiagonalPlan::new(
                &matrix.diagonal_entries(),
                &targets,
                &high,
                region_bits,
            ))
        } else {
            Plan::Dense(DensePlan::new(&matrix, &targets, &high, region_bits))
        }
    }

    /// # Safety
    /// As for [`DensePlan::apply`].
    pub unsafe fn apply(&self, region: *mut T, parallel: bool) {
        match self {
            Plan::Dense(plan) => unsafe { plan.apply(region, parallel) },
            Plan::Diagonal(plan) => unsafe { plan.apply(region, parallel) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deposit_spreads_bits() {
        assert_eq!(deposit(0b101, 0b1101_0000), 0b1000_0000 | 0b0001_0000);
        assert_eq!(deposit(0b11, 0b1010), 0b1010);
        assert_eq!(deposit(0, 0xff), 0);
    }
}
