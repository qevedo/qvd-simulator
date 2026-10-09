#![allow(dead_code)]

use qvd::{C64, Circuit, Gate, Instruction, Matrix, reference};
use rand::{Rng, RngExt, SeedableRng};
use rand_pcg::Pcg64;

pub fn rng(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}

/// A Haar-ish random unitary: Gram-Schmidt on a random complex matrix.
pub fn random_unitary(rng: &mut impl Rng, qubits: usize) -> Matrix {
    let dim = 1 << qubits;
    let mut columns: Vec<Vec<C64>> = Vec::with_capacity(dim);
    while columns.len() < dim {
        let mut v: Vec<C64> = (0..dim)
            .map(|_| C64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
            .collect();
        for u in &columns {
            let dot: C64 = u.iter().zip(&v).map(|(a, b)| a.conj() * b).sum();
            for (x, y) in v.iter_mut().zip(u) {
                *x -= dot * y;
            }
        }
        let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-6 {
            columns.push(v.into_iter().map(|x| x / norm).collect());
        }
    }
    let mut entries = vec![C64::new(0.0, 0.0); dim * dim];
    for (c, column) in columns.iter().enumerate() {
        for (r, &x) in column.iter().enumerate() {
            entries[r * dim + c] = x;
        }
    }
    Matrix::new(entries)
}

pub fn random_diagonal(rng: &mut impl Rng, qubits: usize) -> Matrix {
    let entries: Vec<C64> = (0..1 << qubits)
        .map(|_| C64::from_polar(1.0, rng.random::<f64>() * std::f64::consts::TAU))
        .collect();
    Matrix::diagonal(&entries)
}

pub fn random_state(rng: &mut impl Rng, qubits: usize) -> Vec<C64> {
    let v: Vec<C64> = (0..1 << qubits)
        .map(|_| C64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

/// `k` distinct random qubits out of `n`, in random order.
pub fn random_qubits(rng: &mut impl Rng, n: usize, k: usize) -> Vec<usize> {
    let mut all: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.random_range(i..n);
        all.swap(i, j);
    }
    all.truncate(k);
    all
}

pub fn max_distance(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

/// Project `state` onto `qubit = outcome` and renormalise; returns the
/// probability of the outcome.
pub fn project(state: &mut [C64], qubit: usize, outcome: bool) -> f64 {
    let p: f64 = state
        .iter()
        .enumerate()
        .filter(|(i, _)| ((i >> qubit) & 1 == 1) == outcome)
        .map(|(_, a)| a.norm_sqr())
        .sum();
    for (i, a) in state.iter_mut().enumerate() {
        *a = if ((i >> qubit) & 1 == 1) == outcome {
            *a / p.sqrt()
        } else {
            C64::new(0.0, 0.0)
        };
    }
    p
}

/// The exact distribution of the classical bits, branching on every
/// measurement and reset of a dense simulation.
pub fn exact_distribution(circuit: &Circuit) -> std::collections::BTreeMap<String, f64> {
    fn walk(
        circuit: &Circuit,
        from: usize,
        mut state: Vec<C64>,
        clbits: Vec<bool>,
        weight: f64,
        out: &mut std::collections::BTreeMap<String, f64>,
    ) {
        for (i, instruction) in circuit.instructions.iter().enumerate().skip(from) {
            let (qubit, clbit) = match instruction {
                Instruction::Gate { gate, qubits } => {
                    reference::apply(&mut state, &gate.matrix(), qubits);
                    continue;
                }
                Instruction::Barrier { .. } => continue,
                Instruction::Measure { qubit, clbit } => (*qubit, Some(*clbit)),
                Instruction::Reset { qubit } => (*qubit, None),
            };
            // Branch on the outcome.
            for outcome in [false, true] {
                let mut branch = state.clone();
                let p = project(&mut branch, qubit, outcome);
                if p < 1e-12 {
                    continue;
                }
                let mut bits = clbits.clone();
                match clbit {
                    Some(c) => bits[c] = outcome,
                    None if outcome => reference::apply(&mut branch, &Gate::X.matrix(), &[qubit]),
                    None => {}
                }
                walk(circuit, i + 1, branch, bits, weight * p, out);
            }
            return;
        }
        let key: String = clbits
            .iter()
            .rev()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        *out.entry(key).or_default() += weight;
    }
    let mut state = vec![C64::new(0.0, 0.0); 1 << circuit.num_qubits];
    state[0] = C64::new(1.0, 0.0);
    let mut out = std::collections::BTreeMap::new();
    walk(
        circuit,
        0,
        state,
        vec![false; circuit.num_clbits],
        1.0,
        &mut out,
    );
    out
}
