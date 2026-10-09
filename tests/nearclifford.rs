//! The near-Clifford backend against the dense reference simulator.

mod common;

use common::{exact_distribution, project};
use qvd::library::{ghz, random_gate_circuit};
use qvd::nearclifford::{self, NearClifford};
use qvd::{Backend, C64, Circuit, Gate, Instruction, Options, reference};
use rand::{Rng, RngExt};

/// A random Pauli string as `(qubit, 'X' | 'Y' | 'Z')` factors.
fn random_pauli(rng: &mut impl Rng, n: usize) -> Vec<(usize, char)> {
    (0..n)
        .filter_map(|q| match rng.random_range(0..4) {
            0 => None,
            1 => Some((q, 'X')),
            2 => Some((q, 'Y')),
            _ => Some((q, 'Z')),
        })
        .collect()
}

/// `<ψ|P|ψ>` on a dense state.
fn dense_expectation(state: &[C64], pauli: &[(usize, char)]) -> f64 {
    let mut out = state.to_vec();
    for &(q, p) in pauli {
        let gate = match p {
            'X' => Gate::X,
            'Y' => Gate::Y,
            _ => Gate::Z,
        };
        reference::apply(&mut out, &gate.matrix(), &[q]);
    }
    state.iter().zip(&out).map(|(a, b)| (a.conj() * b).re).sum()
}

fn simulate(circuit: &Circuit) -> NearClifford {
    let mut state = NearClifford::new(circuit.num_qubits);
    for instruction in &circuit.instructions {
        if let Instruction::Gate { gate, qubits } = instruction {
            state.apply(gate, qubits).unwrap();
        }
    }
    state
}

#[test]
fn expectations_match_dense_state() {
    let mut rng = common::rng(3);
    for seed in 0..80u64 {
        let n = 1 + (seed as usize % 8);
        let circuit = random_gate_circuit(n, 40, seed);
        let state = simulate(&circuit);
        let dense = reference::statevector(&circuit);
        for _ in 0..20 {
            let pauli = random_pauli(&mut rng, n);
            let (got, expected) = (state.expectation(&pauli), dense_expectation(&dense, &pauli));
            assert!(
                (got - expected).abs() < 1e-9,
                "seed {seed}, {pauli:?}: {got} vs {expected}"
            );
        }
        assert!(state.peak_active_qubits() <= n);
    }
}

#[test]
fn single_qubit_unitaries_decompose() {
    let mut rng = common::rng(4);
    for seed in 0..20 {
        let mut circuit = Circuit::new(2);
        circuit.gate(Gate::H, &[0]).cx(0, 1);
        circuit.unitary(common::random_unitary(&mut rng, 1), &[seed % 2]);
        circuit.cx(1, 0);
        let state = simulate(&circuit);
        let dense = reference::statevector(&circuit);
        for _ in 0..10 {
            let pauli = random_pauli(&mut rng, 2);
            assert!((state.expectation(&pauli) - dense_expectation(&dense, &pauli)).abs() < 1e-9);
        }
    }
}

#[test]
fn measurement_matches_dense_projection() {
    let mut rng = common::rng(5);
    for seed in 0..60u64 {
        let n = 2 + (seed as usize % 7);
        let circuit = random_gate_circuit(n, 40, seed);
        let mut state = simulate(&circuit);
        let mut dense = reference::statevector(&circuit);
        for round in 0..n {
            let q = rng.random_range(0..n);
            let outcome = state.measure(q, &mut rng);
            let p = project(&mut dense, q, outcome);
            assert!(
                p > 1e-9,
                "seed {seed} round {round}: outcome {outcome} of q{q} has probability {p}"
            );
            for _ in 0..10 {
                let pauli = random_pauli(&mut rng, n);
                let (got, expected) =
                    (state.expectation(&pauli), dense_expectation(&dense, &pauli));
                assert!(
                    (got - expected).abs() < 1e-9,
                    "seed {seed} round {round}, {pauli:?}: {got} vs {expected}"
                );
            }
        }
        // Measuring every qubit leaves nothing active.
        for q in 0..n {
            state.measure(q, &mut rng);
        }
        assert_eq!(state.active_qubits(), 0, "seed {seed}");
    }
}

#[test]
fn sampling_matches_exact_distribution() {
    for case in 0..30u64 {
        let n = 2 + (case as usize % 5);
        let mut circuit = random_gate_circuit(n, 25, case);
        if case % 2 == 1 {
            circuit.measure(0, 0);
            circuit.reset(1);
            circuit
                .instructions
                .extend(random_gate_circuit(n, 10, case + 100).instructions);
        }
        for q in 0..n {
            circuit.measure(q, q + 1);
        }
        let shots = 40_000;
        let options = Options {
            backend: Backend::NearClifford,
            seed: Some(case),
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
            // A few counts of slack: an outcome of probability 1e-6 seen once
            // in 40,000 shots is 6 sigma off but not evidence of an error.
            let slack = 3.0 / shots as f64;
            assert!(
                (observed - p).abs() <= 5.0 * sigma + slack,
                "case {case} {key}: observed {observed}, exact {p}"
            );
        }
    }
}

#[test]
fn clifford_circuits_stay_inactive() {
    let mut circuit = ghz(50);
    circuit
        .gate(Gate::S, &[3])
        .gate(Gate::SX, &[7])
        .gate(Gate::RZ(std::f64::consts::FRAC_PI_2), &[9]);
    let state = simulate(&circuit);
    assert_eq!(state.peak_active_qubits(), 0);
    assert_eq!(nearclifford::rotation_count(&circuit), 0);
}

#[test]
fn large_ghz_with_a_t_gate() {
    // (|0...0> + e^{iπ/4} |1...1>)/√2 on 200 qubits: <X...X> = cos(π/4),
    // <Y X...X> = sin(π/4), and only one qubit is ever active.
    let n = 200;
    let mut circuit = ghz(n);
    circuit.gate(Gate::T, &[0]);
    let state = simulate(&circuit);
    assert_eq!(state.peak_active_qubits(), 1);
    let xs: Vec<(usize, char)> = (0..n).map(|q| (q, 'X')).collect();
    let mut yxs = xs.clone();
    yxs[0] = (0, 'Y');
    let r = std::f64::consts::FRAC_1_SQRT_2;
    assert!((state.expectation(&xs) - r).abs() < 1e-12);
    assert!((state.expectation(&yxs) - r).abs() < 1e-12);
    // Sampling: all zeros or all ones, half and half.
    circuit.measure_all();
    let options = Options {
        backend: Backend::NearClifford,
        seed: Some(1),
        ..Options::default()
    };
    let result = qvd::run::<f64>(&circuit, 2000, &options).unwrap();
    assert_eq!(result.counts.len(), 2);
    let zeros = result.counts.get(&"0".repeat(n)).copied().unwrap_or(0);
    assert!((850..1150).contains(&zeros), "{zeros}");
    assert_eq!(result.stats.peak_active_qubits, Some(1));
}

#[test]
fn sampling_is_reproducible_across_thread_counts() {
    let mut circuit = random_gate_circuit(10, 60, 9);
    circuit.measure_all();
    let options = Options {
        backend: Backend::NearClifford,
        seed: Some(2),
        ..Options::default()
    };
    let sample = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| qvd::run::<f64>(&circuit, 2000, &options))
            .unwrap()
            .counts
    };
    assert_eq!(sample(1), sample(4));
}

#[test]
fn symbolic_peak_matches_and_auto_selects() {
    for seed in 0..10 {
        let mut circuit = qvd::library::random_clifford_t(12, 8, 6, seed);
        circuit.measure(3, 0);
        circuit
            .instructions
            .extend(qvd::library::random_clifford_t(12, 4, 3, seed + 50).instructions);
        let predicted = nearclifford::peak_active_qubits(&circuit).unwrap();
        let mut state = NearClifford::new(12);
        let mut rng = common::rng(seed);
        for instruction in &circuit.instructions {
            match instruction {
                Instruction::Gate { gate, qubits } => state.apply(gate, qubits).unwrap(),
                Instruction::Measure { qubit, .. } => {
                    state.measure(*qubit, &mut rng);
                }
                _ => {}
            }
        }
        assert_eq!(state.peak_active_qubits(), predicted, "seed {seed}");
        assert!(predicted <= 9);
    }
    // 80 qubits do not fit a state vector; 10 T gates keep at most 10 active.
    let mut circuit = qvd::library::random_clifford_t(80, 10, 10, 1);
    circuit.measure_all();
    let result = qvd::run::<f64>(
        &circuit,
        100,
        &Options {
            seed: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    let peak = result
        .stats
        .peak_active_qubits
        .expect("Auto should choose the near-Clifford backend");
    assert!(peak <= 10);
}

#[test]
fn auto_prefers_near_clifford_when_much_cheaper() {
    // 24 qubits fit a state vector, but with 6 T gates at most 6 qubits are
    // ever active: far less work.
    let mut circuit = qvd::library::random_clifford_t(24, 10, 6, 3);
    circuit.measure_all();
    let result = qvd::run::<f64>(
        &circuit,
        200,
        &Options {
            seed: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    assert!(
        result.stats.peak_active_qubits.is_some(),
        "Auto should choose the near-Clifford backend"
    );
    // A dense random circuit stays on the state vector.
    let mut dense = random_gate_circuit(10, 80, 4);
    dense.measure_all();
    let result = qvd::run::<f64>(
        &dense,
        200,
        &Options {
            seed: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(result.stats.peak_active_qubits, None);
}
