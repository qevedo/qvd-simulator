//! A deliberately simple, obviously-correct simulator used to test the fast
//! engine. Double precision, interleaved complex, one gate at a time.

use crate::circuit::{Circuit, Instruction};
use crate::matrix::{C64, Matrix};

/// Apply `matrix` (index bit j = `qubits[j]`) to `state` in place.
pub fn apply(state: &mut [C64], matrix: &Matrix, qubits: &[usize]) {
    let k = qubits.len();
    let mask: usize = qubits.iter().map(|q| 1 << q).sum();
    let mut input = vec![C64::new(0.0, 0.0); 1 << k];
    let offset = |sub: usize| -> usize { (0..k).map(|j| ((sub >> j) & 1) << qubits[j]).sum() };
    for base in 0..state.len() {
        if base & mask != 0 {
            continue;
        }
        for (sub, value) in input.iter_mut().enumerate() {
            *value = state[base | offset(sub)];
        }
        for row in 0..1 << k {
            let mut acc = C64::new(0.0, 0.0);
            for (col, value) in input.iter().enumerate() {
                acc += matrix[(row, col)] * value;
            }
            state[base | offset(row)] = acc;
        }
    }
}

/// The state after running the unitary part of `circuit` from |0...0>.
///
/// Panics on measurements and resets: this reference only covers unitaries.
pub fn statevector(circuit: &Circuit) -> Vec<C64> {
    let mut state = vec![C64::new(0.0, 0.0); 1 << circuit.num_qubits];
    state[0] = C64::new(1.0, 0.0);
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => apply(&mut state, &gate.matrix(), qubits),
            Instruction::Barrier { .. } => {}
            other => panic!("the reference simulator does not support {other:?}"),
        }
    }
    state
}
