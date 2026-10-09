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

/// A rotated surface-code memory experiment of distance `d`: `rounds`
/// rounds of stabilizer measurement (reset ancillas, CNOTs to the
/// neighbouring data qubits, measure ancillas), then every data qubit is
/// measured. Data qubits are `0..d*d`; the `d*d - 1` ancillas follow.
///
/// This is the kind of circuit stabilizer simulators are built for: about
/// `2 d^2` qubits, mostly deterministic measurements.
pub fn surface_code(d: usize, rounds: usize) -> Circuit {
    assert!(d >= 2);
    let data = |r: usize, c: usize| r * d + c;
    // Plaquette (i, j), i, j in 0..=d, touches data (i-1|i, j-1|j). Interior
    // plaquettes alternate X/Z in a checkerboard; on the top and bottom
    // boundaries only X plaquettes are kept, on the left and right only Z.
    let mut plaquettes: Vec<(bool, Vec<Option<usize>>)> = Vec::new();
    for i in 0..=d {
        for j in 0..=d {
            let is_x = (i + j) % 2 == 0;
            let corner = |di: usize, dj: usize| -> Option<usize> {
                let (r, c) = ((i + di).checked_sub(1)?, (j + dj).checked_sub(1)?);
                (r < d && c < d).then(|| data(r, c))
            };
            // Neighbours in hook-safe order: NW, NE, SW, SE for X; NW, SW, NE, SE for Z.
            let order = if is_x {
                [(0, 0), (0, 1), (1, 0), (1, 1)]
            } else {
                [(0, 0), (1, 0), (0, 1), (1, 1)]
            };
            let neighbours: Vec<Option<usize>> = order.iter().map(|&(a, b)| corner(a, b)).collect();
            let weight = neighbours.iter().flatten().count();
            let on_row_boundary = i == 0 || i == d;
            let on_col_boundary = j == 0 || j == d;
            let keep = match weight {
                4 => true,
                2 => (on_row_boundary && is_x) || (on_col_boundary && !is_x),
                _ => false,
            };
            if keep {
                plaquettes.push((is_x, neighbours));
            }
        }
    }
    let ancilla0 = d * d;
    let n = ancilla0 + plaquettes.len();
    let mut c = Circuit::new(n);
    let mut clbit = 0;
    for _ in 0..rounds {
        for a in 0..plaquettes.len() {
            c.reset(ancilla0 + a);
        }
        for (a, (is_x, _)) in plaquettes.iter().enumerate() {
            if *is_x {
                c.h(ancilla0 + a);
            }
        }
        for step in 0..4 {
            for (a, (is_x, neighbours)) in plaquettes.iter().enumerate() {
                if let Some(q) = neighbours[step] {
                    if *is_x {
                        c.cx(ancilla0 + a, q);
                    } else {
                        c.cx(q, ancilla0 + a);
                    }
                }
            }
        }
        for (a, (is_x, _)) in plaquettes.iter().enumerate() {
            if *is_x {
                c.h(ancilla0 + a);
            }
        }
        for a in 0..plaquettes.len() {
            c.measure(ancilla0 + a, clbit);
            clbit += 1;
        }
    }
    for q in 0..d * d {
        c.measure(q, clbit);
        clbit += 1;
    }
    c
}

/// A random Clifford circuit: `depth` layers of random single-qubit
/// Cliffords on every qubit and CX on random disjoint pairs.
pub fn random_clifford(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        for q in 0..n {
            match rng.random_range(0..3) {
                0 => c.h(q),
                1 => c.s(q),
                _ => c.sx(q),
            };
        }
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        for pair in order.chunks_exact(2) {
            c.cx(pair[0], pair[1]);
        }
    }
    c
}
