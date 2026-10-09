//! Standard circuits for tests and benchmarks.

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::circuit::{Circuit, Gate};

/// GHZ state preparation: H on qubit 0, then a CX chain.
pub fn ghz(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    c
}

/// Quantum Fourier transform (without the final qubit reversal).
pub fn qft(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for target in (0..n).rev() {
        c.h(target);
        for control in (0..target).rev() {
            let angle = std::f64::consts::PI / (1u64 << (target - control)) as f64;
            c.cp(angle, control, target);
        }
    }
    c
}

/// A random circuit in the style of Google's supremacy benchmarks (and
/// qsim's): `depth` layers, each a random single-qubit gate from
/// {sqrt(X), sqrt(Y), sqrt(W)} on every qubit followed by CZ gates on a
/// brick pattern of neighbouring pairs (on a line of qubits).
pub fn random_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    let half = std::f64::consts::FRAC_PI_2;
    for layer in 0..depth {
        for q in 0..n {
            let gate = match rng.random_range(0..3) {
                0 => Gate::RX(half),
                1 => Gate::RY(half),
                // sqrt(W), W = (X + Y) / sqrt(2)
                _ => Gate::U(
                    half,
                    -std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_4,
                ),
            };
            c.gate(gate, &[q]);
        }
        let mut q = layer % 2;
        while q + 1 < n {
            c.cz(q, q + 1);
            q += 2;
        }
    }
    c
}

/// A random circuit over the full standard gate set (rotations with random
/// angles, CX/CZ/CP/SWAP/CCX on random qubits). Used for correctness tests.
pub fn random_gate_circuit(n: usize, gates: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let mut qubits: Vec<usize> = (0..n).collect();
        for i in 0..n.min(3) {
            let j = rng.random_range(i..n);
            qubits.swap(i, j);
        }
        let angle = rng.random::<f64>() * std::f64::consts::TAU;
        let choice = rng.random_range(
            0..if n >= 3 {
                14
            } else if n == 2 {
                12
            } else {
                8
            },
        );
        let (gate, arity) = match choice {
            0 => (Gate::H, 1),
            1 => (Gate::X, 1),
            2 => (Gate::RX(angle), 1),
            3 => (Gate::RY(angle), 1),
            4 => (Gate::RZ(angle), 1),
            5 => (Gate::U(angle, angle * 0.5, angle * 0.25), 1),
            6 => (Gate::T, 1),
            7 => (Gate::SX, 1),
            8 => (Gate::CX, 2),
            9 => (Gate::CZ, 2),
            10 => (Gate::CP(angle), 2),
            11 => (Gate::SWAP, 2),
            12 => (Gate::CCX, 3),
            _ => (Gate::CSWAP, 3),
        };
        c.gate(gate, &qubits[..arity]);
    }
    c
}
