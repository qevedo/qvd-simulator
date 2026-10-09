//! The SIMD kernels against the reference simulator, for every combination
//! of in-register ("low") and block ("high") target qubits.

mod common;

use common::*;
use qvd::StateVector;
use qvd::reference;
use qvd::simd::Real;
use rand::RngExt;

fn check<T: Real>(seed: u64, tolerance: f64) {
    let mut rng = rng(seed);
    for case in 0..400 {
        let n = rng.random_range(1..=11usize);
        let k = rng.random_range(1..=n.min(6));
        let qubits = random_qubits(&mut rng, n, k);
        let mut rest: Vec<usize> = (0..n).filter(|q| !qubits.contains(q)).collect();
        // Low controls are folded into the matrix, so keep targets plus
        // controls within the 6-qubit kernel limit.
        let controls: Vec<(usize, bool)> = if rest.is_empty() || k >= 6 || rng.random_bool(0.5) {
            vec![]
        } else {
            let c = rng.random_range(1..=rest.len().min(2).min(6 - k));
            rest.truncate(c);
            rest.iter().map(|&q| (q, rng.random_bool(0.7))).collect()
        };
        let diagonal = rng.random_bool(0.3);
        let matrix = if diagonal {
            random_diagonal(&mut rng, k)
        } else {
            random_unitary(&mut rng, k)
        };

        let initial = random_state(&mut rng, n);
        let mut expected = initial.clone();
        let control_qubits: Vec<usize> = controls.iter().map(|&(q, _)| q).collect();
        let state_bits = controls
            .iter()
            .enumerate()
            .map(|(i, &(_, v))| (v as usize) << i)
            .sum();
        let all: Vec<usize> = control_qubits
            .iter()
            .copied()
            .chain(qubits.iter().copied())
            .collect();
        reference::apply(
            &mut expected,
            &matrix.controlled(controls.len(), state_bits),
            &all,
        );

        let mut state = StateVector::<T>::from_amplitudes(&initial).unwrap();
        state.apply(&matrix, &qubits, &controls);
        let distance = max_distance(&state.to_vec(), &expected);
        assert!(
            distance < tolerance,
            "case {case}: n={n} qubits={qubits:?} controls={controls:?} diagonal={diagonal}: error {distance}"
        );
    }
}

#[test]
fn kernels_match_reference_f64() {
    check::<f64>(1, 1e-12);
}

#[test]
fn kernels_match_reference_f32() {
    check::<f32>(2, 2e-6);
}

#[test]
fn large_state_uses_parallel_path() {
    // 2^18 amplitudes: enough iterations for the multithreaded path.
    let mut rng = rng(3);
    let n = 18;
    let initial = random_state(&mut rng, n);
    for k in 1..=5 {
        let qubits = random_qubits(&mut rng, n, k);
        let matrix = random_unitary(&mut rng, k);
        let mut expected = initial.clone();
        reference::apply(&mut expected, &matrix, &qubits);
        let mut state = StateVector::<f64>::from_amplitudes(&initial).unwrap();
        state.apply(&matrix, &qubits, &[]);
        assert!(
            max_distance(&state.to_vec(), &expected) < 1e-12,
            "k={k} qubits={qubits:?}"
        );
    }
}
