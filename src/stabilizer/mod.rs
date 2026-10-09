//! Stabilizer simulation of Clifford circuits: thousands of qubits,
//! millions of shots.
//!
//! A circuit made only of Clifford gates (H, S, CX, CZ, Paulis, rotations
//! by multiples of π/2, ...) plus measurements and resets keeps the state a
//! stabilizer state, which [`Tableau`] stores in `O(n^2)` bits instead of
//! `2^n` amplitudes. Shots are drawn with Pauli frames, as in Stim: one
//! reference run on the tableau, then every other shot is the reference
//! plus a Pauli "frame" propagated through the circuit, 64 shots per
//! machine word.

mod tableau;

use std::f64::consts::FRAC_PI_2;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::circuit::{Circuit, Gate, Instruction};
use crate::simulator::{RunResult, counts_from_keys};

pub use tableau::{Layout, PauliString, Tableau};

/// A primitive Clifford operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clifford {
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    CX(usize, usize),
    CZ(usize, usize),
    Swap(usize, usize),
}

/// `angle / (π/2)` if it is (within 1e-9) an integer, reduced mod 4.
fn quarter_turns(angle: f64) -> Option<u8> {
    let turns = angle / FRAC_PI_2;
    let rounded = turns.round();
    ((turns - rounded).abs() < 1e-9).then(|| rounded.rem_euclid(4.0) as u8)
}

/// `RZ` by `k` quarter turns, up to global phase.
fn rz(k: u8, q: usize, out: &mut Vec<Clifford>) {
    match k {
        1 => out.push(Clifford::S(q)),
        2 => out.push(Clifford::Z(q)),
        3 => out.push(Clifford::Sdg(q)),
        _ => {}
    }
}

/// Lower a gate to primitive Clifford operations, in application order, or
/// `None` if the gate is not Clifford. Global phases are dropped.
pub fn lower(gate: &Gate, qubits: &[usize]) -> Option<Vec<Clifford>> {
    use Clifford as C;
    let q = qubits;
    let mut out = Vec::new();
    match gate {
        Gate::I => {}
        Gate::H => out.push(C::H(q[0])),
        Gate::X => out.push(C::X(q[0])),
        Gate::Y => out.push(C::Y(q[0])),
        Gate::Z => out.push(C::Z(q[0])),
        Gate::S => out.push(C::S(q[0])),
        Gate::Sdg => out.push(C::Sdg(q[0])),
        Gate::SX => out.extend([C::H(q[0]), C::S(q[0]), C::H(q[0])]),
        Gate::SXdg => out.extend([C::H(q[0]), C::Sdg(q[0]), C::H(q[0])]),
        Gate::RZ(t) | Gate::P(t) => rz(quarter_turns(*t)?, q[0], &mut out),
        Gate::RX(t) => {
            // RX = H RZ H.
            let k = quarter_turns(*t)?;
            out.push(C::H(q[0]));
            rz(k, q[0], &mut out);
            out.push(C::H(q[0]));
        }
        Gate::RY(t) => {
            // RY = S RX S†.
            let k = quarter_turns(*t)?;
            out.extend([C::Sdg(q[0]), C::H(q[0])]);
            rz(k, q[0], &mut out);
            out.extend([C::H(q[0]), C::S(q[0])]);
        }
        Gate::U(theta, phi, lambda) => {
            // U(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ) up to global phase.
            let (kt, kp, kl) = (
                quarter_turns(*theta)?,
                quarter_turns(*phi)?,
                quarter_turns(*lambda)?,
            );
            rz(kl, q[0], &mut out);
            out.extend([C::Sdg(q[0]), C::H(q[0])]);
            rz(kt, q[0], &mut out);
            out.extend([C::H(q[0]), C::S(q[0])]);
            rz(kp, q[0], &mut out);
        }
        Gate::CX => out.push(C::CX(q[0], q[1])),
        Gate::CZ => out.push(C::CZ(q[0], q[1])),
        Gate::CY => out.extend([C::Sdg(q[1]), C::CX(q[0], q[1]), C::S(q[1])]),
        Gate::SWAP => out.push(C::Swap(q[0], q[1])),
        Gate::CP(t) => match quarter_turns(*t)? {
            0 => {}
            2 => out.push(C::CZ(q[0], q[1])),
            _ => return None,
        },
        Gate::RZZ(t) => {
            // RZZ(θ) = CX RZ_b(θ) CX.
            let k = quarter_turns(*t)?;
            out.push(C::CX(q[0], q[1]));
            rz(k, q[1], &mut out);
            out.push(C::CX(q[0], q[1]));
        }
        Gate::RXX(t) => {
            let k = quarter_turns(*t)?;
            out.extend([C::H(q[0]), C::H(q[1]), C::CX(q[0], q[1])]);
            rz(k, q[1], &mut out);
            out.extend([C::CX(q[0], q[1]), C::H(q[0]), C::H(q[1])]);
        }
        Gate::T
        | Gate::Tdg
        | Gate::CH
        | Gate::CRX(_)
        | Gate::CRY(_)
        | Gate::CRZ(_)
        | Gate::CCX
        | Gate::CSWAP
        | Gate::Unitary(_) => return None,
    }
    Some(out)
}

/// Whether every gate of `circuit` is Clifford (so the stabilizer backend
/// can run it).
pub fn is_clifford(circuit: &Circuit) -> bool {
    circuit.instructions.iter().all(|i| match i {
        Instruction::Gate { gate, qubits } => lower(gate, qubits).is_some(),
        _ => true,
    })
}

impl Tableau {
    pub fn apply(&mut self, op: Clifford) {
        match op {
            Clifford::H(q) => self.h(q),
            Clifford::S(q) => self.s(q),
            Clifford::Sdg(q) => self.sdg(q),
            Clifford::X(q) => self.x(q),
            Clifford::Y(q) => self.y(q),
            Clifford::Z(q) => self.z(q),
            Clifford::CX(c, t) => self.cx(c, t),
            Clifford::CZ(a, b) => self.cz(a, b),
            Clifford::Swap(a, b) => self.swap(a, b),
        }
    }

    /// The tableau after running the gates of a circuit (measurements and
    /// resets are applied too, with outcomes drawn from `rng`).
    pub fn from_circuit(circuit: &Circuit, rng: &mut impl Rng) -> Result<Tableau, String> {
        let mut tableau = Tableau::new(circuit.num_qubits);
        for instruction in &circuit.instructions {
            match instruction {
                Instruction::Gate { gate, qubits } => {
                    for op in lower(gate, qubits)
                        .ok_or_else(|| format!("{gate:?} is not a Clifford gate"))?
                    {
                        tableau.apply(op);
                    }
                }
                Instruction::Measure { qubit, .. } => {
                    tableau.measure(*qubit, rng);
                }
                Instruction::Reset { qubit } => tableau.reset(*qubit, rng),
                Instruction::Barrier { .. } => {}
            }
        }
        Ok(tableau)
    }
}

/// Pauli frames for a batch of shots: bit `s` of `x[q]` (`z[q]`) is the X
/// (Z) component of shot `s`'s frame on qubit `q`.
struct Frames {
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    /// One random stream per word of 64 shots, so results depend on the
    /// seed but not on how the shots are split between threads.
    rngs: Vec<Pcg64>,
}

impl Frames {
    fn new(qubits: usize, rngs: Vec<Pcg64>) -> Frames {
        let words = rngs.len();
        let mut frames = Frames {
            words,
            x: vec![0; qubits * words],
            z: vec![0; qubits * words],
            rngs,
        };
        // Z errors on |0> change nothing; random ones make the measurements
        // that should be random come out random once gates turn them into
        // X components (Stim's trick).
        for q in 0..qubits {
            frames.randomize_z(q);
        }
        frames
    }

    fn qubit(&self, q: usize) -> std::ops::Range<usize> {
        q * self.words..(q + 1) * self.words
    }

    fn randomize_z(&mut self, q: usize) {
        let r = self.qubit(q);
        for (z, rng) in self.z[r].iter_mut().zip(&mut self.rngs) {
            *z = rng.next_u64();
        }
    }

    /// `a ^= b` between rows of the same table.
    fn xor(table: &mut [u64], words: usize, dst: usize, src: usize) {
        let (d, s) = (dst * words, src * words);
        if d < s {
            let (lo, hi) = table.split_at_mut(s);
            for (a, b) in lo[d..d + words].iter_mut().zip(&hi[..words]) {
                *a ^= b;
            }
        } else {
            let (lo, hi) = table.split_at_mut(d);
            for (a, b) in hi[..words].iter_mut().zip(&lo[s..s + words]) {
                *a ^= b;
            }
        }
    }

    fn swap_tables(&mut self, q: usize) {
        let r = self.qubit(q);
        self.x[r.clone()].swap_with_slice(&mut self.z[r]);
    }

    fn apply(&mut self, op: Clifford) {
        let w = self.words;
        match op {
            Clifford::H(q) => self.swap_tables(q),
            Clifford::S(q) | Clifford::Sdg(q) => {
                let r = self.qubit(q);
                for i in r {
                    self.z[i] ^= self.x[i];
                }
            }
            Clifford::X(_) | Clifford::Y(_) | Clifford::Z(_) => {}
            Clifford::CX(c, t) => {
                Self::xor(&mut self.x, w, t, c);
                Self::xor(&mut self.z, w, c, t);
            }
            Clifford::CZ(a, b) => {
                // z_a ^= x_b; z_b ^= x_a (x is unchanged).
                for (i, j) in self.qubit(a).zip(self.qubit(b)) {
                    self.z[i] ^= self.x[j];
                    self.z[j] ^= self.x[i];
                }
            }
            Clifford::Swap(a, b) => {
                for (i, j) in self.qubit(a).zip(self.qubit(b)) {
                    self.x.swap(i, j);
                    self.z.swap(i, j);
                }
            }
        }
    }
}

/// Measurement results of many shots, bit-packed: one bit per shot per
/// classical bit, 64 shots per word.
#[derive(Clone, Debug)]
pub struct Samples {
    pub shots: usize,
    pub num_clbits: usize,
    words: usize,
    /// Classical bit `c` of shot `s` is bit `s % 64` of `bits[c * words + s / 64]`.
    bits: Vec<u64>,
}

impl Samples {
    /// Classical bit `clbit` of shot `shot`.
    pub fn get(&self, shot: usize, clbit: usize) -> bool {
        assert!(shot < self.shots && clbit < self.num_clbits);
        (self.bits[clbit * self.words + shot / 64] >> (shot % 64)) & 1 == 1
    }

    /// The shots of classical bit `clbit`, 64 per word (bits past `shots`
    /// in the last word are unspecified).
    pub fn clbit_words(&self, clbit: usize) -> &[u64] {
        &self.bits[clbit * self.words..(clbit + 1) * self.words]
    }

    /// Count the outcomes as bitstrings (classical bit 0 on the right).
    pub fn counts(&self) -> std::collections::BTreeMap<String, usize> {
        if self.num_clbits <= 128 {
            let mut keys = vec![0u128; self.shots];
            for c in 0..self.num_clbits {
                let words = self.clbit_words(c);
                for (s, key) in keys.iter_mut().enumerate() {
                    *key |= (((words[s / 64] >> (s % 64)) & 1) as u128) << c;
                }
            }
            return counts_from_keys(keys, self.num_clbits);
        }
        let mut counts = std::collections::BTreeMap::new();
        for s in 0..self.shots {
            let bits: String = (0..self.num_clbits)
                .rev()
                .map(|c| if self.get(s, c) { '1' } else { '0' })
                .collect();
            *counts.entry(bits).or_default() += 1;
        }
        counts
    }
}

/// A circuit step with the reference outcome of measurements filled in.
#[derive(Clone, Copy)]
enum Step {
    Gate(Clifford),
    Measure {
        qubit: usize,
        clbit: usize,
        reference: bool,
    },
    Reset {
        qubit: usize,
    },
}

/// A pointer that may be shared between threads writing disjoint ranges.
#[derive(Clone, Copy)]
struct Shared(*mut u64);
// SAFETY: only used to write disjoint words from different threads.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    fn get(self) -> *mut u64 {
        self.0
    }
}

/// Sample `shots` shots of a Clifford circuit.
///
/// One reference shot runs on the tableau; the others are Pauli frames
/// relative to it, 64 shots per word. The frames are split into blocks of
/// a few words that run through the whole circuit independently, in
/// parallel and in cache. Fails if the circuit contains a non-Clifford
/// gate.
pub fn sample(circuit: &Circuit, shots: usize, seed: Option<u64>) -> Result<Samples, String> {
    let mut rng = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_rng(&mut rand::rng()),
    };
    let n = circuit.num_qubits;
    // Reference shot.
    let mut tableau = Tableau::new(n);
    let mut steps = Vec::with_capacity(circuit.instructions.len());
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => {
                for op in
                    lower(gate, qubits).ok_or_else(|| format!("{gate:?} is not a Clifford gate"))?
                {
                    tableau.apply(op);
                    steps.push(Step::Gate(op));
                }
            }
            Instruction::Measure { qubit, clbit } => {
                let reference = tableau.measure(*qubit, &mut rng);
                steps.push(Step::Measure {
                    qubit: *qubit,
                    clbit: *clbit,
                    reference,
                });
            }
            Instruction::Reset { qubit } => {
                tableau.reset(*qubit, &mut rng);
                steps.push(Step::Reset { qubit: *qubit });
            }
            Instruction::Barrier { .. } => {}
        }
    }
    drop(tableau);
    // Frames, one block per thread, at most 2048 shots each so a block's
    // frames stay in cache. Run inside a pool of performance cores: the
    // blocks synchronise after every segment of steps, so a slow core holds
    // up all the others.
    let words = shots.div_ceil(64);
    let block = words.div_ceil(rayon::current_num_threads()).clamp(1, 32);
    let mut rngs: Vec<Pcg64> = (0..words)
        .map(|_| Pcg64::seed_from_u64(rng.next_u64()))
        .collect();
    // Latest measurement record of each classical bit, one bit per shot.
    let mut bits = vec![0u64; circuit.num_clbits * words];
    let out = Shared(bits.as_mut_ptr());
    let mut blocks: Vec<(usize, Frames)> = Vec::new();
    while !rngs.is_empty() {
        let rest = rngs.split_off(block.min(rngs.len()));
        blocks.push((
            words - rngs.len() - rest.len(),
            Frames::new(n, std::mem::replace(&mut rngs, rest)),
        ));
    }
    // All blocks walk the same cache-sized segment of steps at a time.
    for segment in steps.chunks(1 << 15) {
        blocks.par_iter_mut().for_each(|(first, frames)| {
            for step in segment {
                match *step {
                    Step::Gate(op) => frames.apply(op),
                    Step::Measure {
                        qubit,
                        clbit,
                        reference,
                    } => {
                        let fill = if reference { u64::MAX } else { 0 };
                        for (i, &x) in frames.x[frames.qubit(qubit)].iter().enumerate() {
                            // SAFETY: this block alone writes words
                            // `first..first + width` of each classical bit.
                            unsafe { *out.get().add(clbit * words + *first + i) = x ^ fill };
                        }
                        frames.randomize_z(qubit);
                    }
                    Step::Reset { qubit } => {
                        let r = frames.qubit(qubit);
                        frames.x[r].fill(0);
                        frames.randomize_z(qubit);
                    }
                }
            }
        });
    }
    Ok(Samples {
        shots,
        num_clbits: circuit.num_clbits,
        words,
        bits,
    })
}

/// Run a Clifford circuit for `shots` shots and count the outcomes.
///
/// Fails if the circuit contains a non-Clifford gate. For circuits with
/// many classical bits, [`sample`] avoids building one string per shot.
pub fn run(circuit: &Circuit, shots: usize, seed: Option<u64>) -> Result<RunResult, String> {
    let start = Instant::now();
    let samples = sample(circuit, shots, seed)?;
    let gate_seconds = start.elapsed().as_secs_f64();
    let count_start = Instant::now();
    let mut result = RunResult {
        counts: samples.counts(),
        ..RunResult::default()
    };
    result.stats.gates = circuit.gate_count();
    result.stats.gate_seconds = gate_seconds;
    result.stats.measure_seconds = count_start.elapsed().as_secs_f64();
    Ok(result)
}
