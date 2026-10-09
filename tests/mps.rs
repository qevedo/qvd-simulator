//! The MPS backend against the dense reference simulator.

mod common;

use common::{exact_distribution, max_distance, project};
use qvd::library::{ghz, qft, random_circuit, random_gate_circuit};
use qvd::mps::{self, Mps, Truncation};
use qvd::{Backend, Circuit, Options, reference};
use rand::RngExt;

fn mps_options(seed: u64) -> Options {
    Options {
        backend: Backend::Mps,
        seed: Some(seed),
        max_bond_dimension: usize::MAX,
        ..Options::default()
    }
}

#[test]
fn exact_states_match_reference() {
    // The full gate set, including non-adjacent and three-qubit gates, so
    // the SWAP routing and every block size are exercised.
    for seed in 0..60u64 {
        let n = 1 + (seed as usize % 9);
        let circuit = random_gate_circuit(n, 50, seed);
        let mps = mps::simulate(&circuit, Truncation::exact());
        let error = max_distance(&mps.to_statevector(), &reference::statevector(&circuit));
        assert!(error < 1e-10, "seed {seed}, n {n}: error {error}");
        assert_eq!(
            mps.fidelity(),
            1.0,
            "seed {seed}: nothing should be truncated"
        );
    }
    for circuit in [ghz(10), qft(9), random_circuit(10, 12, 3)] {
        let mps = mps::simulate(&circuit, Truncation::exact());
        let error = max_distance(&mps.to_statevector(), &reference::statevector(&circuit));
        assert!(error < 1e-10, "error {error}");
    }
}

#[test]
fn measurement_matches_dense_projection() {
    let mut rng = common::rng(5);
    for seed in 0..40u64 {
        let n = 2 + (seed as usize % 7);
        let circuit = random_gate_circuit(n, 40, seed);
        let mut mps = mps::simulate(&circuit, Truncation::exact());
        let mut state = reference::statevector(&circuit);
        for _ in 0..n {
            let q = rng.random_range(0..n);
            let outcome = mps.measure(q, &mut rng);
            let p = project(&mut state, q, outcome);
            assert!(
                p > 1e-12,
                "seed {seed}: outcome {outcome} of q{q} has probability {p}"
            );
            let error = max_distance(&mps.to_statevector(), &state);
            assert!(
                error < 1e-10,
                "seed {seed}: error {error} after measuring q{q}"
            );
        }
    }
}

#[test]
fn sampling_matches_exact_distribution() {
    for case in 0..30u64 {
        let n = 2 + (case as usize % 5);
        let mut circuit = random_gate_circuit(n, 25, case);
        // Mid-circuit measurements and resets on half the cases.
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
        let result = qvd::run::<f64>(&circuit, shots, &mps_options(case)).unwrap();
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
fn truncation_caps_the_bond_and_tracks_fidelity() {
    let circuit = random_circuit(12, 10, 9);
    let exact = reference::statevector(&circuit);
    let mut previous = 0.0;
    for max_bond in [2, 4, 8, 16, 64] {
        let mps = mps::simulate(
            &circuit,
            Truncation {
                max_bond,
                threshold: 1e-16,
            },
        );
        assert!(mps.bond_dimensions().iter().all(|&b| b <= max_bond));
        let state = mps.to_statevector();
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10, "χ {max_bond}: norm {norm}");
        // The fidelity estimate tracks the true overlap with the exact state.
        let overlap: f64 = state
            .iter()
            .zip(&exact)
            .map(|(a, b)| a.conj() * b)
            .sum::<qvd::C64>()
            .norm_sqr();
        let estimate = mps.fidelity();
        assert!(estimate >= previous - 1e-12, "fidelity should grow with χ");
        assert!(
            (overlap - estimate).abs() < 0.1,
            "χ {max_bond}: overlap {overlap}, estimate {estimate}"
        );
        previous = estimate;
    }
    assert!(
        (previous - 1.0).abs() < 1e-12,
        "χ 64 is exact for 12 qubits"
    );
}

#[test]
fn large_ghz_and_automatic_selection() {
    // 300 qubits: far past the state vector, trivial for an MPS (χ = 2).
    let mut circuit = ghz(300);
    // Not Clifford, so Auto cannot use the stabilizer backend.
    circuit.gate(qvd::Gate::T, &[0]).gate(qvd::Gate::Tdg, &[0]);
    circuit.measure_all();
    let options = Options {
        seed: Some(3),
        ..Options::default()
    };
    let result = qvd::run::<f32>(&circuit, 2000, &options).unwrap();
    assert_eq!(result.counts.len(), 2, "{:?}", result.counts.keys());
    let zeros = result.counts.get(&"0".repeat(300)).copied().unwrap_or(0);
    assert!(
        (800..1200).contains(&zeros),
        "{zeros} all-zero shots of 2000"
    );
    assert_eq!(result.stats.max_bond_dimension, Some(2));
    assert_eq!(result.stats.fidelity, Some(1.0));
}

#[test]
fn long_range_gates_and_amplitudes() {
    // CX between the two ends of a chain, applied repeatedly: the routing
    // must leave the state consistent wherever the qubits end up.
    let n = 40;
    let mut circuit = Circuit::new(n);
    circuit.h(0);
    for q in [n - 1, 5, 30, 17] {
        circuit.cx(0, q);
    }
    let mps = mps::simulate(&circuit, Truncation::default());
    let mut bits = vec![false; n];
    let a = mps.amplitude(&bits);
    for q in [0, n - 1, 5, 30, 17] {
        bits[q] = true;
    }
    let b = mps.amplitude(&bits);
    let half = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (a.re - half).abs() < 1e-12 && (b.re - half).abs() < 1e-12,
        "{a} {b}"
    );
    bits[5] = false;
    assert!(mps.amplitude(&bits).norm() < 1e-12);
}

#[test]
fn sampling_is_reproducible_across_thread_counts() {
    let mut circuit = random_circuit(20, 8, 4);
    circuit.measure_all();
    let sample = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| qvd::run::<f64>(&circuit, 3000, &mps_options(11)))
            .unwrap()
            .counts
    };
    assert_eq!(sample(1), sample(4));
}

#[test]
fn reset_returns_qubits_to_zero() {
    let mut mps = Mps::new(3, Truncation::exact());
    let h = qvd::Gate::H.matrix();
    let cx = qvd::Gate::CX.matrix();
    mps.apply(&h, &[0]);
    mps.apply(&cx, &[0, 2]);
    let mut rng = common::rng(1);
    mps.reset(2, &mut rng);
    let state = mps.to_statevector();
    // Qubit 2 is |0>; qubit 0 collapsed with it.
    for (index, a) in state.iter().enumerate() {
        if index & 0b100 != 0 {
            assert!(a.norm() < 1e-12);
        }
    }
    let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-12);
}
