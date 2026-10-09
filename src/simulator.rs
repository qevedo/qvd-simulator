//! Running circuits: lowering to unitary ops, fusion, measurement handling.

use std::collections::BTreeMap;
use std::time::Instant;

use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::blocking::{DEFAULT_REGION_BITS, apply_blocked};
use crate::circuit::{Circuit, Gate, Instruction};
use crate::fusion::{Op, fuse};
use crate::simd::Real;
use crate::state::StateVector;

/// Simulator settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// Largest fused gate, in qubits. 1 disables fusion. On AVX2 machines a
    /// fused gate on up to 4 qubits costs about one memory sweep.
    pub max_fused_qubits: usize,
    /// Seed for measurement randomness; `None` draws one from the OS.
    pub seed: Option<u64>,
    /// Cache blocking: apply runs of gates to cache-sized regions of
    /// `2^region_bits` blocks before moving on. `None` disables it.
    pub region_bits: Option<u32>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            max_fused_qubits: 4,
            seed: None,
            region_bits: Some(DEFAULT_REGION_BITS),
        }
    }
}

/// What a run did, for benchmarking.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Gates in the circuit.
    pub gates: usize,
    /// Gates after fusion.
    pub kernels: usize,
    /// Passes over the whole state in memory: one per fused gate without
    /// cache blocking; one per stage and per relabeling with it.
    pub passes: usize,
    /// Seconds spent applying gates.
    pub gate_seconds: f64,
    /// Seconds spent sampling and measuring.
    pub measure_seconds: f64,
}

/// The result of running a circuit with shots.
#[derive(Clone, Debug, Default)]
pub struct RunResult {
    /// Counts per classical bitstring, written with classical bit 0 on the
    /// right (Qiskit's convention).
    pub counts: BTreeMap<String, usize>,
    pub stats: Stats,
}

/// Lower a gate to a unitary op, keeping controls separate so the kernel can
/// skip the uncontrolled half of the state.
pub fn gate_op(gate: &Gate, qubits: &[usize]) -> Op {
    match gate.controlled_form() {
        Some((controls, target)) => Op {
            targets: qubits[controls..].to_vec(),
            controls: qubits[..controls].iter().map(|&q| (q, true)).collect(),
            matrix: target.matrix(),
        },
        None => Op {
            targets: qubits.to_vec(),
            controls: Vec::new(),
            matrix: gate.matrix(),
        },
    }
}

/// Apply ops to a state, fusing them first.
pub fn apply_ops<T: Real>(
    state: &mut StateVector<T>,
    ops: &[Op],
    options: &Options,
    stats: &mut Stats,
) {
    let start = Instant::now();
    let max_fused = options.max_fused_qubits.max(1);
    match options.region_bits {
        Some(region_bits) => {
            let blocked = apply_blocked(state, ops, region_bits, max_fused);
            stats.passes += blocked.stages + blocked.swaps;
            stats.kernels += blocked.kernels;
        }
        None => {
            let fused = fuse(ops, max_fused);
            for op in &fused {
                state.apply(&op.matrix, &op.targets, &op.controls);
            }
            stats.passes += fused.len();
            stats.kernels += fused.len();
        }
    }
    stats.gates += ops.len();
    stats.gate_seconds += start.elapsed().as_secs_f64();
}

/// The final state of a circuit with no measurements or resets.
pub fn statevector<T: Real>(
    circuit: &Circuit,
    options: &Options,
) -> std::io::Result<(StateVector<T>, Stats)> {
    let mut ops = Vec::new();
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => ops.push(gate_op(gate, qubits)),
            Instruction::Barrier { .. } => {}
            other => panic!("statevector() needs a unitary circuit, found {other:?}"),
        }
    }
    let mut state = StateVector::<T>::zero(circuit.num_qubits)?;
    let mut stats = Stats::default();
    apply_ops(&mut state, &ops, options, &mut stats);
    Ok((state, stats))
}

/// Index of the first instruction after which only measurements (and
/// barriers) follow, if every measured qubit is untouched afterwards.
fn terminal_measurements(circuit: &Circuit) -> Option<usize> {
    let mut split = circuit.instructions.len();
    for (i, instruction) in circuit.instructions.iter().enumerate().rev() {
        match instruction {
            Instruction::Measure { .. } | Instruction::Barrier { .. } => split = i,
            _ => break,
        }
    }
    let unitary_tail_free = circuit.instructions[..split]
        .iter()
        .all(|i| matches!(i, Instruction::Gate { .. } | Instruction::Barrier { .. }));
    unitary_tail_free.then_some(split)
}

/// Count sampled basis states as classical bitstrings: map each outcome to
/// an integer key, sort and count runs in parallel, and format only the
/// distinct keys.
fn count_outcomes(
    outcomes: &[usize],
    measures: &[(usize, usize)],
    num_clbits: usize,
) -> BTreeMap<String, usize> {
    use rayon::prelude::*;
    let mut counts = BTreeMap::new();
    if num_clbits > 128 {
        for &index in outcomes {
            let mut clbits = vec![false; num_clbits];
            for &(qubit, clbit) in measures {
                clbits[clbit] = (index >> qubit) & 1 == 1;
            }
            *counts.entry(bitstring(&clbits)).or_default() += 1;
        }
        return counts;
    }
    let mut keys: Vec<u128> = outcomes
        .par_iter()
        .map(|&index| {
            measures.iter().fold(0u128, |key, &(q, c)| {
                key | (((index >> q) & 1) as u128) << c
            })
        })
        .collect();
    keys.par_sort_unstable();
    let mut start = 0;
    while start < keys.len() {
        let key = keys[start];
        let end = start + keys[start..].partition_point(|&k| k == key);
        let clbits: Vec<bool> = (0..num_clbits).map(|c| (key >> c) & 1 == 1).collect();
        counts.insert(bitstring(&clbits), end - start);
        start = end;
    }
    counts
}

fn bitstring(clbits: &[bool]) -> String {
    clbits
        .iter()
        .rev()
        .map(|&b| if b { '1' } else { '0' })
        .collect()
}

/// Run `circuit` for `shots` shots and count the classical outcomes.
///
/// Circuits whose measurements all come at the end are simulated once and
/// sampled. Mid-circuit measurements and resets are simulated shot by shot.
pub fn run<T: Real>(
    circuit: &Circuit,
    shots: usize,
    options: &Options,
) -> std::io::Result<RunResult> {
    let mut rng = match options.seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_rng(&mut rand::rng()),
    };
    let mut result = RunResult::default();
    if let Some(split) = terminal_measurements(circuit) {
        let ops: Vec<Op> = circuit.instructions[..split]
            .iter()
            .filter_map(|i| match i {
                Instruction::Gate { gate, qubits } => Some(gate_op(gate, qubits)),
                _ => None,
            })
            .collect();
        let mut state = StateVector::<T>::zero(circuit.num_qubits)?;
        apply_ops(&mut state, &ops, options, &mut result.stats);
        let start = Instant::now();
        let measures: Vec<(usize, usize)> = circuit.instructions[split..]
            .iter()
            .filter_map(|i| match i {
                Instruction::Measure { qubit, clbit } => Some((*qubit, *clbit)),
                _ => None,
            })
            .collect();
        let outcomes = state.sample_unordered(shots, &mut rng);
        result.counts = count_outcomes(&outcomes, &measures, circuit.num_clbits);
        result.stats.measure_seconds = start.elapsed().as_secs_f64();
        return Ok(result);
    }
    for _ in 0..shots {
        let mut state = StateVector::<T>::zero(circuit.num_qubits)?;
        let mut clbits = vec![false; circuit.num_clbits];
        let mut pending: Vec<Op> = Vec::new();
        for instruction in &circuit.instructions {
            match instruction {
                Instruction::Gate { gate, qubits } => pending.push(gate_op(gate, qubits)),
                Instruction::Barrier { .. } => {}
                Instruction::Measure { qubit, clbit } => {
                    apply_ops(
                        &mut state,
                        &std::mem::take(&mut pending),
                        options,
                        &mut result.stats,
                    );
                    let start = Instant::now();
                    clbits[*clbit] = state.measure(*qubit, &mut rng);
                    result.stats.measure_seconds += start.elapsed().as_secs_f64();
                }
                Instruction::Reset { qubit } => {
                    apply_ops(
                        &mut state,
                        &std::mem::take(&mut pending),
                        options,
                        &mut result.stats,
                    );
                    state.reset(*qubit, &mut rng);
                }
            }
        }
        apply_ops(&mut state, &pending, options, &mut result.stats);
        *result.counts.entry(bitstring(&clbits)).or_default() += 1;
    }
    Ok(result)
}
