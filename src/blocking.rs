//! Cache-blocked execution: many gates per pass over memory.
//!
//! The state is split into *regions* of `2^region_bits` consecutive blocks
//! (1 MiB by default, which stays in a core's L2 cache). Physical qubits
//! inside a region are *local*; the others are *global*. A *stage* is a run
//! of gates that only need local qubits as (non-diagonal) targets. Each
//! thread takes a region, applies every gate of the stage to it while it
//! sits in cache, and moves on: the whole stage costs one pass over memory
//! instead of one pass per gate.
//!
//! Diagonal gates and controls never force a qubit to be local ("insular"
//! qubits in the Atlas paper): within one region, global qubits have fixed
//! values, so a global control either holds or not, and a diagonal gate on
//! global qubits is a diagonal gate on its local qubits only.
//!
//! When a gate needs a global target, the scheduler looks ahead, picks the
//! set of qubits the coming gates need, and swaps them with local qubits
//! that are not needed, in a single pass (the state's layout records where
//! each logical qubit now lives, so nothing is moved back). This follows
//! Häner & Steiger (SC17), who needed only a few such swaps for a deep
//! 42-qubit circuit.

use rayon::prelude::*;

use crate::fusion::{Op, fuse, fuse_where};
use crate::kernels::Plan;
use crate::matrix::{C64, Matrix};
use crate::simd::Real;
use crate::state::StateVector;

/// Block bits per region: 2^14 blocks of 64 bytes = 1 MiB.
pub const DEFAULT_REGION_BITS: u32 = 14;

/// Counts of what a blocked run did.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockingStats {
    /// Stages executed (each one pass over memory).
    pub stages: usize,
    /// Relabeling passes (each one pass over memory).
    pub swaps: usize,
    /// Fused gates applied.
    pub kernels: usize,
}

#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// A gate prepared for one region of a stage.
struct RegionOp<T: Real> {
    /// Region-index bits that global controls fix, and their values.
    control_mask: usize,
    control_value: usize,
    /// Region-index bits of global diagonal targets (in matrix-bit order).
    diagonal_global: Vec<usize>,
    /// One plan, or one per assignment of the global diagonal targets.
    plans: Vec<Plan<T>>,
}

impl<T: Real> RegionOp<T> {
    fn new(op: &Op, layout: &[usize], local_limit: usize, region_bits: u32) -> RegionOp<T> {
        let mut control_mask = 0;
        let mut control_value = 0;
        let mut local_controls = Vec::new();
        for &(q, value) in &op.controls {
            let p = layout[q];
            if p >= local_limit {
                control_mask |= 1 << (p - local_limit);
                control_value |= (value as usize) << (p - local_limit);
            } else {
                local_controls.push((p, value));
            }
        }
        let physical: Vec<usize> = op.targets.iter().map(|&q| layout[q]).collect();
        if physical.iter().all(|&p| p < local_limit) {
            let plan = Plan::new(&op.matrix, &physical, &local_controls, region_bits);
            return RegionOp {
                control_mask,
                control_value,
                diagonal_global: Vec::new(),
                plans: vec![plan],
            };
        }
        // A diagonal gate with global targets: one local diagonal per
        // assignment of the global target bits.
        let diagonal = op.matrix.diagonal_entries();
        let (global, local): (Vec<_>, Vec<_>) =
            (0..physical.len()).partition(|&j| physical[j] >= local_limit);
        let local_targets: Vec<usize> = local.iter().map(|&j| physical[j]).collect();
        let diagonal_global = global.iter().map(|&j| physical[j] - local_limit).collect();
        let plans = (0..1usize << global.len())
            .map(|assignment| {
                let entries: Vec<C64> = (0..1usize << local.len())
                    .map(|sub| {
                        let mut index = 0;
                        for (m, &j) in global.iter().enumerate() {
                            index |= ((assignment >> m) & 1) << j;
                        }
                        for (m, &j) in local.iter().enumerate() {
                            index |= ((sub >> m) & 1) << j;
                        }
                        diagonal[index]
                    })
                    .collect();
                // Plan::new folds controls on lane qubits into the matrix.
                Plan::new(
                    &Matrix::diagonal(&entries),
                    &local_targets,
                    &local_controls,
                    region_bits,
                )
            })
            .collect();
        RegionOp {
            control_mask,
            control_value,
            diagonal_global,
            plans,
        }
    }

    /// # Safety
    /// `ptr` must point to region `region` of the state the op was built for.
    unsafe fn apply(&self, region: usize, ptr: *mut T) {
        if region & self.control_mask != self.control_value {
            return;
        }
        let plan = if self.diagonal_global.is_empty() {
            &self.plans[0]
        } else {
            let assignment: usize = self
                .diagonal_global
                .iter()
                .enumerate()
                .map(|(m, &bit)| ((region >> bit) & 1) << m)
                .sum();
            &self.plans[assignment]
        };
        unsafe { plan.apply(ptr, false) };
    }
}

/// Whether `op` can run inside a stage with the current layout: its
/// targets are local, or it is diagonal (global diagonal targets and
/// global controls are fixed within a region).
fn runs_locally(op: &Op, layout: &[usize], local_limit: usize) -> bool {
    op.matrix.is_diagonal() || op.targets.iter().all(|&q| layout[q] < local_limit)
}

/// Walk the pending ops in program order and take every op that can run
/// now: it runs locally (per `can_run`) and no earlier pending op on one of
/// its qubits is left behind. Returns the indices taken.
fn extract(ops: &[Op], done: &[bool], mut can_run: impl FnMut(&Op) -> bool) -> Vec<usize> {
    let mut taken = Vec::new();
    let mut blocked: Vec<usize> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        if done[i] {
            continue;
        }
        let touches_blocked = op.qubits().any(|q| blocked.contains(&q));
        if !touches_blocked && can_run(op) {
            taken.push(i);
        } else {
            for q in op.qubits() {
                if !blocked.contains(&q) {
                    blocked.push(q);
                }
            }
        }
    }
    taken
}

/// Choose swaps that make the next stage as long as possible.
///
/// Simulates stage extraction with a growing set of wanted qubits: an op
/// joins if its non-diagonal targets fit in the local block positions
/// together with the qubits already wanted. The wanted set becomes local;
/// local qubits that are not wanted make room. Lane qubits are always local.
fn plan_swaps(
    ops: &[Op],
    done: &[bool],
    layout: &[usize],
    lane_bits: usize,
    local_limit: usize,
) -> Vec<(usize, usize)> {
    let capacity = local_limit - lane_bits;
    let mut wanted: Vec<usize> = Vec::new();
    extract(ops, done, |op| {
        if op.matrix.is_diagonal() {
            return true;
        }
        let new: Vec<usize> = op
            .targets
            .iter()
            .copied()
            .filter(|&q| layout[q] >= lane_bits && !wanted.contains(&q))
            .collect();
        if wanted.len() + new.len() > capacity {
            return false;
        }
        wanted.extend(new);
        true
    });
    if wanted.is_empty() {
        return Vec::new();
    }
    let incoming: Vec<usize> = wanted
        .iter()
        .map(|&q| layout[q])
        .filter(|&p| p >= local_limit)
        .collect();
    let wanted_positions: Vec<usize> = wanted.iter().map(|&q| layout[q]).collect();
    // Evict from the top of the local range first; any unwanted slot works.
    let victims = (lane_bits..local_limit)
        .rev()
        .filter(|p| !wanted_positions.contains(p));
    incoming.iter().zip(victims).map(|(&g, v)| (v, g)).collect()
}

/// Apply `ops` (unfused, on logical qubits) with cache blocking, fusing up
/// to `max_fused_qubits` within each stage.
pub fn apply_blocked<T: Real>(
    state: &mut StateVector<T>,
    ops: &[Op],
    region_bits: u32,
    max_fused_qubits: usize,
) -> BlockingStats {
    let lane_bits = T::LANE_BITS as usize;
    let block_bits = state.block_bits();
    let mut stats = BlockingStats::default();
    if block_bits <= region_bits + 1 {
        // The whole state fits in a couple of regions: plain fused kernels.
        for op in fuse(ops, max_fused_qubits) {
            state.apply(&op.matrix, &op.targets, &op.controls);
            stats.stages += 1;
            stats.kernels += 1;
        }
        return stats;
    }
    let local_limit = lane_bits + region_bits as usize;
    let regions = 1usize << (block_bits - region_bits);
    let region_len = (1usize << region_bits) * 2 * T::LANES;
    let mut done = vec![false; ops.len()];
    let mut remaining = ops.len();
    while remaining > 0 {
        let layout = state.layout().to_vec();
        let taken = extract(ops, &done, |op| runs_locally(op, &layout, local_limit));
        if !taken.is_empty() {
            for &i in &taken {
                done[i] = true;
            }
            remaining -= taken.len();
            let stage_ops: Vec<Op> = taken.iter().map(|&i| ops[i].clone()).collect();
            // Fuse only ops whose qubits are all local: merging a global
            // control or diagonal target into a dense matrix would make it
            // a global target.
            let fused = fuse_where(&stage_ops, max_fused_qubits, |op| {
                op.qubits().all(|q| layout[q] < local_limit)
            });
            stats.kernels += fused.len();
            let stage: Vec<RegionOp<T>> = fused
                .iter()
                .map(|op| RegionOp::new(op, &layout, local_limit, region_bits))
                .collect();
            let base = Shared(state.raw_mut().as_mut_ptr());
            (0..regions).into_par_iter().for_each(|region| {
                // SAFETY: each region is a disjoint range of the state.
                let ptr = unsafe { base.get().add(region * region_len) };
                for op in &stage {
                    unsafe { op.apply(region, ptr) };
                }
            });
            stats.stages += 1;
        }
        if remaining == 0 {
            break;
        }
        let pairs = plan_swaps(ops, &done, &layout, lane_bits, local_limit);
        if pairs.is_empty() {
            // The next op has more block-qubit targets than a region holds
            // (only possible with tiny regions): one full-state pass.
            let i = done.iter().position(|d| !d).expect("an op remains");
            done[i] = true;
            remaining -= 1;
            state.apply(&ops[i].matrix, &ops[i].targets, &ops[i].controls);
            stats.stages += 1;
            stats.kernels += 1;
        } else {
            state.swap_physical(&pairs);
            stats.swaps += 1;
        }
    }
    stats
}
