//! The stabilizer backend against the dense state-vector simulator.

mod common;

use std::f64::consts::FRAC_PI_2;

use qvd::reference;
use qvd::stabilizer::{self, Layout, PauliString, Tableau};
use qvd::{Backend, C64, Circuit, Gate, Instruction, Matrix, Options};
use rand::{Rng, RngExt};

/// A random circuit over every gate form the stabilizer backend accepts.
fn random_clifford(rng: &mut impl Rng, n: usize, gates: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let qubits = common::random_qubits(rng, n, n.min(2));
        let quarter = rng.random_range(0..4) as f64 * FRAC_PI_2;
        let two = n >= 2;
        let gate = match rng.random_range(0..if two { 20 } else { 12 }) {
            0 => Gate::H,
            1 => Gate::S,
            2 => Gate::Sdg,
            3 => Gate::X,
            4 => Gate::Y,
            5 => Gate::Z,
            6 => Gate::SX,
            7 => Gate::SXdg,
            8 => Gate::RX(quarter),
            9 => Gate::RY(quarter),
            10 => Gate::RZ(quarter),
            11 => Gate::U(quarter, rng.random_range(0..4) as f64 * FRAC_PI_2, -quarter),
            12 => Gate::CX,
            13 => Gate::CY,
            14 => Gate::CZ,
            15 => Gate::SWAP,
            16 => Gate::RZZ(quarter),
            17 => Gate::RXX(quarter),
            18 => Gate::CP(2.0 * quarter),
            _ => Gate::P(quarter),
        };
        let arity = gate.arity();
        // CP is Clifford only for multiples of π.
        let gate = match gate {
            Gate::CP(t) if (t / std::f64::consts::PI).fract().abs() > 1e-9 => Gate::CZ,
            g => g,
        };
        c.gate(gate, &qubits[..arity]);
    }
    c
}

fn random_pauli(rng: &mut impl Rng, n: usize) -> PauliString {
    let mut text = String::from(if rng.random_bool(0.5) { "-" } else { "+" });
    for _ in 0..n {
        text.push(['I', 'X', 'Y', 'Z'][rng.random_range(0..4)]);
    }
    PauliString::parse(&text).unwrap()
}

/// `<psi|P|psi>` on a dense state.
fn dense_expectation(state: &[C64], pauli: &PauliString) -> f64 {
    let mut out = state.to_vec();
    for j in 0..pauli.xs.len() {
        let gate = match (pauli.xs[j], pauli.zs[j]) {
            (true, false) => Gate::X,
            (true, true) => Gate::Y,
            (false, true) => Gate::Z,
            (false, false) => continue,
        };
        reference::apply(&mut out, &gate.matrix(), &[j]);
    }
    let value: C64 = state.iter().zip(&out).map(|(a, b)| a.conj() * b).sum();
    assert!(value.im.abs() < 1e-9);
    if pauli.negative { -value.re } else { value.re }
}

fn assert_same_state(tableau: &Tableau, state: &[C64], rng: &mut impl Rng, context: &str) {
    let n = tableau.num_qubits();
    let mut paulis: Vec<PauliString> = (0..60).map(|_| random_pauli(rng, n)).collect();
    // Every single-qubit Z and X too.
    for q in 0..n {
        for p in ['X', 'Z'] {
            let text: String = (0..n).map(|j| if j == q { p } else { 'I' }).collect();
            paulis.push(PauliString::parse(&text).unwrap());
        }
    }
    for pauli in &paulis {
        let expected = dense_expectation(state, pauli);
        let got = tableau.expectation(pauli) as f64;
        assert!(
            (expected - got).abs() < 1e-9,
            "{context}: <{pauli:?}> tableau {got}, dense {expected}"
        );
    }
}

#[test]
fn expectations_match_dense_state() {
    let mut rng = common::rng(21);
    for case in 0..300 {
        let n = 1 + case % 10;
        let circuit = random_clifford(&mut rng, n, 40);
        let tableau = Tableau::from_circuit(&circuit, &mut rng).unwrap();
        let state = reference::statevector(&circuit);
        assert_same_state(&tableau, &state, &mut rng, &format!("case {case}"));
    }
}

#[test]
fn deterministic_measurements_match_dense_probabilities() {
    let mut rng = common::rng(22);
    for case in 0..200 {
        let n = 1 + case % 8;
        let circuit = random_clifford(&mut rng, n, 30);
        let tableau = Tableau::from_circuit(&circuit, &mut rng).unwrap();
        let state = reference::statevector(&circuit);
        for q in 0..n {
            let p1: f64 = state
                .iter()
                .enumerate()
                .filter(|(i, _)| (i >> q) & 1 == 1)
                .map(|(_, a)| a.norm_sqr())
                .sum();
            match tableau.peek_z(q) {
                Some(outcome) => assert!(
                    (p1 - outcome as u8 as f64).abs() < 1e-9,
                    "case {case} q {q}: p1 {p1}"
                ),
                None => assert!(
                    (p1 - 0.5).abs() < 1e-9,
                    "case {case} q {q}: random outcome but p1 {p1}"
                ),
            }
        }
    }
}

/// Project a dense state on `qubit = outcome` and renormalise.
fn project(state: &mut [C64], qubit: usize, outcome: bool) -> f64 {
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

#[test]
fn collapse_matches_dense_projection() {
    let mut rng = common::rng(23);
    for case in 0..600 {
        let n = 1 + case % 9;
        let mut tableau = Tableau::new(n);
        // Every layout policy; a tiny switch cost makes Adaptive switch
        // back and forth.
        tableau.set_layout([Layout::Rows, Layout::Columns, Layout::Adaptive][(case / 9) % 3]);
        tableau.set_switch_cost(2);
        let mut state = vec![C64::new(0.0, 0.0); 1 << n];
        state[0] = C64::new(1.0, 0.0);
        for round in 0..4 {
            let circuit = random_clifford(&mut rng, n, 25);
            for instruction in &circuit.instructions {
                if let Instruction::Gate { gate, qubits } = instruction {
                    for op in stabilizer::lower(gate, qubits).unwrap() {
                        tableau.apply(op);
                    }
                    reference::apply(&mut state, &gate.matrix(), qubits);
                }
            }
            // Measure a few qubits, forcing the random outcomes.
            for _ in 0..rng.random_range(1..=n) {
                let q = rng.random_range(0..n);
                let coin = rng.random_bool(0.5);
                let outcome = tableau.measure_with(q, || coin);
                let p = project(&mut state, q, outcome);
                assert!(
                    p > 0.49,
                    "case {case} round {round}: outcome {outcome} of q{q} has probability {p}"
                );
            }
            assert_same_state(
                &tableau,
                &state,
                &mut rng,
                &format!("case {case} round {round}"),
            );
        }
    }
}

/// The exact distribution of the classical bits, branching on every
/// measurement and reset of a dense simulation.
fn exact_distribution(circuit: &Circuit) -> std::collections::BTreeMap<String, f64> {
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

#[test]
fn sampling_matches_exact_distribution() {
    let mut rng = common::rng(24);
    for case in 0..40 {
        let n = 2 + case % 6;
        let mut circuit = random_clifford(&mut rng, n, 30);
        // Mid-circuit measurements and resets on half the cases.
        if case % 2 == 1 {
            circuit.measure(0, 0);
            circuit.reset(1);
            let more = random_clifford(&mut rng, n, 10);
            circuit.instructions.extend(more.instructions);
        }
        for q in 0..n {
            circuit.measure(q, q + 1);
        }
        let shots = 50_000;
        let options = Options {
            backend: Backend::Stabilizer,
            seed: Some(case as u64),
            ..Options::default()
        };
        let result = qvd::run::<f64>(&circuit, shots, &options).unwrap();
        let exact = exact_distribution(&circuit);
        for key in result.counts.keys() {
            assert!(
                exact.contains_key(key),
                "case {case}: impossible outcome {key}"
            );
        }
        for (key, &p) in &exact {
            let observed = *result.counts.get(key).unwrap_or(&0) as f64 / shots as f64;
            let sigma = (p * (1.0 - p) / shots as f64).sqrt();
            assert!(
                (observed - p).abs() <= 5.0 * sigma + 1e-9,
                "case {case} {key}: observed {observed}, exact {p}"
            );
        }
    }
}

#[test]
fn large_ghz_sampling() {
    let n = 1000;
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    for q in 0..100 {
        c.measure(q * 10, q);
    }
    let result = stabilizer::run(&c, 10_000, Some(7)).unwrap();
    assert_eq!(result.counts.len(), 2);
    let zeros = result.counts["0".repeat(100).as_str()] as f64;
    assert!((zeros - 5000.0).abs() < 300.0, "{zeros}");
}

#[test]
fn auto_backend_and_non_clifford_detection() {
    let mut clifford = Circuit::new(3);
    clifford.h(0).cx(0, 1).s(2).measure_all();
    assert!(stabilizer::is_clifford(&clifford));
    let mut t = Circuit::new(1);
    t.t(0);
    assert!(!stabilizer::is_clifford(&t));
    assert!(stabilizer::run(&t, 1, None).is_err());
    let mut rz = Circuit::new(1);
    rz.rz(0.3, 0);
    assert!(!stabilizer::is_clifford(&rz));
    rz.instructions.clear();
    rz.rz(3.0 * FRAC_PI_2, 0);
    assert!(stabilizer::is_clifford(&rz));
    // A 2000-qubit Clifford circuit only runs on the stabilizer backend.
    let mut big = Circuit::new(2000);
    big.h(0).cx(0, 1999).measure(0, 0).measure(1999, 1);
    let result = qvd::run::<f32>(&big, 1000, &Options::default()).unwrap();
    assert!(result.counts.keys().all(|k| k == "00" || k == "11"));
    let _ = Matrix::identity(1);
}

#[test]
fn surface_code_stabilizers_repeat() {
    // Noiseless memory experiment: Z stabilizers always read 0; X
    // stabilizers are random in round 1 and repeat that value afterwards.
    for d in [3, 5, 7] {
        let rounds = 4;
        let circuit = qvd::library::surface_code(d, rounds);
        let ancillas = d * d - 1;
        let shots = 2000;
        let samples = stabilizer::sample(&circuit, shots, Some(d as u64)).unwrap();
        // Ancilla `a` of round `r` is classical bit `r * ancillas + a`; X
        // plaquettes are the ones Hadamard-conjugated in the circuit.
        let x_ancillas: Vec<usize> = circuit
            .instructions
            .iter()
            .filter_map(|i| match i {
                Instruction::Gate {
                    gate: Gate::H,
                    qubits,
                } => Some(qubits[0] - d * d),
                _ => None,
            })
            .take(2 * ancillas)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(x_ancillas.len(), ancillas / 2, "d={d}");
        let mut random_first_round = 0;
        for a in 0..ancillas {
            let is_x = x_ancillas.contains(&a);
            for shot in 0..shots {
                let first = samples.get(shot, a);
                if is_x {
                    random_first_round += first as usize;
                } else {
                    assert!(!first, "d={d}: Z stabilizer {a} read 1 in round 1");
                }
                for r in 1..rounds {
                    assert_eq!(
                        samples.get(shot, r * ancillas + a),
                        first,
                        "d={d} shot {shot}: stabilizer {a} changed in round {r}"
                    );
                }
            }
        }
        // X stabilizers in round 1 are fair coins.
        let total = (x_ancillas.len() * shots) as f64;
        assert!(
            (random_first_round as f64 / total - 0.5).abs() < 0.02,
            "d={d}: {random_first_round}/{total}"
        );
    }
}

#[test]
fn sampling_is_reproducible_across_thread_counts() {
    let circuit = qvd::library::surface_code(5, 3);
    let sample = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| stabilizer::sample(&circuit, 5000, Some(7)))
            .unwrap()
    };
    let (one, three) = (sample(1), sample(3));
    for c in 0..circuit.num_clbits {
        assert_eq!(
            one.clbit_words(c),
            three.clbit_words(c),
            "classical bit {c}"
        );
    }
}
