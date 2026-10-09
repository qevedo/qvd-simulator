//! Matrix product state (MPS) simulation for circuits with limited
//! entanglement, past the state vector's memory limit.
//!
//! The state is a chain of site tensors `A[l, p, r]`, one qubit per site,
//! joined by bonds of dimension χ. A one-qubit gate updates one site. A gate
//! on several qubits first moves them next to each other with adjacent
//! SWAPs (and leaves them there: lazy permutation), contracts their sites,
//! applies the matrix and splits the block back with SVDs (TEBD). Each SVD
//! keeps at most χ_max singular values and drops those whose squared sum is
//! below a threshold; the state stays in mixed canonical form, so each cut
//! discards exactly the smallest Schmidt coefficients, and the product of
//! kept weights estimates the fidelity.
//!
//! The qubits are placed on sites in the order that best keeps interacting
//! qubits close (see [`qubit_order`]), which matters when the circuit's
//! labels do not follow its geometry.
//!
//! Costs: memory `O(n χ²)`, `O(χ³)` per multi-qubit gate (plus one SWAP per
//! site of distance between its qubits), `O(n χ²)` per sampled shot.

mod layout;
mod state;

use std::time::Instant;

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::circuit::{Circuit, Instruction};
use crate::matrix::Matrix;
use crate::simulator::{Options, RunResult, bitstring, counts_from_keys, terminal_measurements};

pub use layout::{cut_cost, qubit_order};
pub use state::{Mps, Truncation};

/// Shots per independently seeded batch: results depend on the seed only,
/// not on the number of threads.
const SHOT_BATCH: usize = 256;

/// Apply the gates of `circuit` (measurements and resets are not allowed).
pub fn simulate(circuit: &Circuit, truncation: Truncation) -> Mps {
    let mut mps = Mps::with_order(&qubit_order(circuit), truncation);
    if let Some(other) = circuit
        .instructions
        .iter()
        .find(|i| matches!(i, Instruction::Measure { .. } | Instruction::Reset { .. }))
    {
        panic!("simulate covers unitary circuits only, found {other:?}");
    }
    apply_gates(&mut mps, &circuit.instructions);
    mps
}

/// Apply the gates among `instructions` as one batch, so that gates on
/// disjoint qubits run in parallel.
fn apply_gates(mps: &mut Mps, instructions: &[Instruction]) {
    let matrices: Vec<(Matrix, &[usize])> = instructions
        .iter()
        .filter_map(|i| match i {
            Instruction::Gate { gate, qubits } => Some((gate.matrix(), qubits.as_slice())),
            _ => None,
        })
        .collect();
    let gates: Vec<(&Matrix, &[usize])> = matrices.iter().map(|(m, q)| (m, *q)).collect();
    mps.apply_gates(&gates);
}

/// Run `circuit` for `shots` shots on an MPS with the truncation in
/// `options`, and count the classical outcomes. Circuits whose measurements
/// all come at the end are simulated once and sampled; otherwise the gates
/// up to the first measurement or reset run once and each shot continues
/// from a copy.
pub fn run(circuit: &Circuit, shots: usize, options: &Options) -> RunResult {
    let truncation = Truncation {
        max_bond: options.max_bond_dimension,
        threshold: options.truncation_threshold,
    };
    let mut rng = match options.seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_rng(&mut rand::rng()),
    };
    let mut result = RunResult::default();
    result.stats.gates = circuit.gate_count();
    let start = Instant::now();
    let instructions = &circuit.instructions;
    let first = match terminal_measurements(circuit) {
        Some(split) => split,
        None => instructions
            .iter()
            .position(|i| matches!(i, Instruction::Measure { .. } | Instruction::Reset { .. }))
            .unwrap_or(instructions.len()),
    };
    let mut mps = Mps::with_order(&qubit_order(circuit), truncation);
    apply_gates(&mut mps, &instructions[..first]);
    result.stats.gate_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let seeds: Vec<u64> = (0..shots.div_ceil(SHOT_BATCH))
        .map(|_| rng.random())
        .collect();
    let num_clbits = circuit.num_clbits;
    // One classical record per shot, from a batch's own random stream.
    let shots_of = |batch: usize| SHOT_BATCH.min(shots - batch * SHOT_BATCH);
    let (records, fidelity, max_bond): (Vec<Vec<bool>>, f64, usize) =
        if terminal_measurements(circuit).is_some() {
            let measures: Vec<(usize, usize)> = instructions[first..]
                .iter()
                .filter_map(|i| match i {
                    Instruction::Measure { qubit, clbit } => Some((*qubit, *clbit)),
                    _ => None,
                })
                .collect();
            mps.prepare_sampling();
            let n = circuit.num_qubits;
            let records = seeds
                .par_iter()
                .enumerate()
                .flat_map_iter(|(batch, &seed)| {
                    let mut rng = Pcg64::seed_from_u64(seed);
                    let count = shots_of(batch);
                    let qubits = mps.sample(count, &mut rng);
                    let measures = &measures;
                    (0..count)
                        .map(move |i| {
                            let mut clbits = vec![false; num_clbits];
                            for &(q, c) in measures {
                                clbits[c] = qubits[i * n + q];
                            }
                            clbits
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            (records, mps.fidelity(), mps.max_bond_seen())
        } else {
            let tail = &instructions[first..];
            let runs: Vec<(Vec<bool>, f64, usize)> = seeds
                .par_iter()
                .enumerate()
                .flat_map_iter(|(batch, &seed)| {
                    let mut rng = Pcg64::seed_from_u64(seed);
                    let base = &mps;
                    (0..shots_of(batch))
                        .map(move |_| {
                            let mut mps = base.clone();
                            let mut clbits = vec![false; num_clbits];
                            let mut gates_from = 0;
                            for (i, instruction) in tail.iter().enumerate() {
                                let (qubit, clbit) = match instruction {
                                    Instruction::Measure { qubit, clbit } => (*qubit, Some(*clbit)),
                                    Instruction::Reset { qubit } => (*qubit, None),
                                    _ => continue,
                                };
                                apply_gates(&mut mps, &tail[gates_from..i]);
                                gates_from = i + 1;
                                match clbit {
                                    Some(c) => clbits[c] = mps.measure(qubit, &mut rng),
                                    None => mps.reset(qubit, &mut rng),
                                }
                            }
                            apply_gates(&mut mps, &tail[gates_from..]);
                            (clbits, mps.fidelity(), mps.max_bond_seen())
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let fidelity = runs.iter().map(|r| r.1).fold(mps.fidelity(), f64::min);
            let max_bond = runs
                .iter()
                .map(|r| r.2)
                .fold(mps.max_bond_seen(), usize::max);
            (runs.into_iter().map(|r| r.0).collect(), fidelity, max_bond)
        };
    result.counts = if num_clbits <= 128 {
        let keys = records
            .iter()
            .map(|clbits| {
                clbits
                    .iter()
                    .enumerate()
                    .fold(0u128, |key, (c, &b)| key | (b as u128) << c)
            })
            .collect();
        counts_from_keys(keys, num_clbits)
    } else {
        let mut counts = std::collections::BTreeMap::new();
        for clbits in &records {
            *counts.entry(bitstring(clbits)).or_default() += 1;
        }
        counts
    };
    result.stats.measure_seconds = start.elapsed().as_secs_f64();
    result.stats.fidelity = Some(fidelity);
    result.stats.max_bond_dimension = Some(max_bond);
    result
}
