mod common;

use common::max_distance;
use qvd::library::{ghz, qft, random_circuit, random_gate_circuit};
use qvd::{Circuit, Options, reference, run, statevector};

#[test]
fn circuits_match_reference_for_every_fusion_width() {
    for seed in 0..40u64 {
        let n = 1 + (seed as usize % 10);
        let circuit = random_gate_circuit(n, 60, seed);
        let expected = reference::statevector(&circuit);
        for max_fused_qubits in 1..=6 {
            let options = Options {
                backend: qvd::Backend::StateVector,
                max_fused_qubits,
                seed: None,
                region_bits: None,
            };
            let (state, stats) = statevector::<f64>(&circuit, &options).unwrap();
            let error = max_distance(&state.to_vec(), &expected);
            assert!(
                error < 1e-11,
                "seed {seed}, n {n}, fusion {max_fused_qubits}: error {error}"
            );
            assert!(stats.kernels <= stats.gates);
            let (state32, _) = statevector::<f32>(&circuit, &options).unwrap();
            let error32 = max_distance(&state32.to_vec(), &expected);
            assert!(error32 < 1e-5, "f32 seed {seed}: error {error32}");
        }
    }
}

#[test]
fn library_circuits_match_reference() {
    for circuit in [ghz(9), qft(9), random_circuit(10, 12, 7)] {
        let expected = reference::statevector(&circuit);
        let (state, stats) = statevector::<f64>(&circuit, &Options::default()).unwrap();
        assert!(max_distance(&state.to_vec(), &expected) < 1e-12);
        assert!(stats.kernels < stats.gates, "fusion should merge gates");
    }
}

#[test]
fn fusion_cuts_kernel_calls() {
    let circuit = random_circuit(20, 20, 1);
    let (_, stats) = statevector::<f32>(&circuit, &Options::default()).unwrap();
    // 20 layers x (20 single-qubit gates + ~10 CZ) = ~590 gates.
    assert!(stats.gates > 500);
    assert!(
        stats.kernels * 4 < stats.gates,
        "{} kernels for {} gates",
        stats.kernels,
        stats.gates
    );
}

#[test]
fn bell_state_sampling() {
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1).measure_all();
    let result = run::<f32>(
        &c,
        100_000,
        &Options {
            seed: Some(5),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(result.counts.len(), 2, "{:?}", result.counts);
    let zeros = result.counts["00"] as f64;
    let ones = result.counts["11"] as f64;
    assert_eq!(zeros + ones, 100_000.0);
    // Binomial standard deviation is ~158; allow 5 sigma.
    assert!((zeros - 50_000.0).abs() < 800.0, "{zeros}");
}

#[test]
fn sampling_matches_probabilities() {
    let mut c = Circuit::new(3);
    c.ry(1.0, 0).ry(2.0, 1).cx(1, 2).measure_all();
    let shots = 200_000;
    let result = run::<f64>(
        &c,
        shots,
        &Options {
            seed: Some(9),
            ..Options::default()
        },
    )
    .unwrap();
    let mut unitary = c.clone();
    unitary
        .instructions
        .retain(|i| !matches!(i, qvd::Instruction::Measure { .. }));
    let probabilities: Vec<f64> = reference::statevector(&unitary)
        .iter()
        .map(|a| a.norm_sqr())
        .collect();
    for (index, p) in probabilities.iter().enumerate() {
        let key = format!("{index:03b}");
        let observed = *result.counts.get(&key).unwrap_or(&0) as f64 / shots as f64;
        let sigma = (p * (1.0 - p) / shots as f64).sqrt();
        assert!(
            (observed - p).abs() <= 5.0 * sigma + 1e-9,
            "{key}: observed {observed}, expected {p}"
        );
    }
}

#[test]
fn mid_circuit_measurement_and_reset() {
    // x; measure -> c0 (always 1); reset; measure -> c1 (always 0); h; measure -> c2 (random).
    let mut c = Circuit::new(1);
    c.x(0)
        .measure(0, 0)
        .reset(0)
        .measure(0, 1)
        .h(0)
        .measure(0, 2);
    let result = run::<f64>(
        &c,
        2000,
        &Options {
            seed: Some(3),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(
        result.counts.keys().cloned().collect::<Vec<_>>(),
        vec!["001".to_string(), "101".to_string()]
    );
    let ones = result.counts["101"] as f64;
    assert!((ones - 1000.0).abs() < 200.0);
}

#[test]
fn measurement_collapse_is_consistent() {
    // GHZ measured qubit by qubit mid-circuit: all outcomes agree.
    let mut c = ghz(5);
    for q in 0..5 {
        c.measure(q, q);
        c.x(q);
    }
    let result = run::<f32>(
        &c,
        500,
        &Options {
            seed: Some(11),
            ..Options::default()
        },
    )
    .unwrap();
    assert!(
        result.counts.keys().all(|k| k == "00000" || k == "11111"),
        "{:?}",
        result.counts
    );
}

#[test]
fn cache_blocking_matches_reference() {
    // Tiny regions force many stages, global controls, global diagonals and
    // relabeling passes on states small enough for the reference.
    for seed in 0..30u64 {
        let n = 8 + (seed as usize % 7);
        let circuit = random_gate_circuit(n, 120, 100 + seed);
        let expected = reference::statevector(&circuit);
        for region_bits in 1..=4 {
            for max_fused_qubits in [1, 3, 4] {
                let options = Options {
                    backend: qvd::Backend::StateVector,
                    max_fused_qubits,
                    seed: None,
                    region_bits: Some(region_bits),
                };
                let (state, stats) = statevector::<f64>(&circuit, &options).unwrap();
                let error = max_distance(&state.to_vec(), &expected);
                assert!(
                    error < 1e-11,
                    "seed {seed} n {n} region {region_bits} fusion {max_fused_qubits}: {error}"
                );
                assert!(stats.passes > 0);
                let (state32, _) = statevector::<f32>(&circuit, &options).unwrap();
                assert!(max_distance(&state32.to_vec(), &expected) < 1e-5);
            }
        }
    }
}

#[test]
fn sampling_after_relabeling() {
    // After cache blocking the physical layout is permuted; samples and
    // probabilities must still be reported in logical order.
    let mut c = Circuit::new(10);
    c.x(9).x(7).h(0).cx(0, 8).measure_all();
    let options = Options {
        seed: Some(1),
        region_bits: Some(1),
        ..Options::default()
    };
    let result = run::<f32>(&c, 1000, &options).unwrap();
    let keys: Vec<&String> = result.counts.keys().collect();
    assert_eq!(
        keys,
        vec!["1010000000", "1110000001"],
        "{:?}",
        result.counts
    );
}
