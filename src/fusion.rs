//! Gate fusion: merge runs of gates into dense gates on at most
//! `max_qubits` qubits, so each memory sweep does more work.
//!
//! On the target machine a fused gate on up to 4 qubits costs the same as a
//! single-qubit gate (both are bound by memory bandwidth), so fusing 20
//! gates into one cuts the time for them twentyfold.
//!
//! The fuser respects only the true dependencies of the circuit (gates that
//! share a qubit stay in order), so a gate can join a group even when gates
//! on unrelated qubits come between them in the program, as qsim's fuser
//! does.

use crate::matrix::Matrix;

/// A unitary operation on physical qubits.
#[derive(Clone, Debug)]
pub struct Op {
    /// Qubits the matrix acts on (index bit j = `targets[j]`).
    pub targets: Vec<usize>,
    /// `(qubit, value)` controls the matrix is conditioned on.
    pub controls: Vec<(usize, bool)>,
    pub matrix: Matrix,
}

impl Op {
    pub fn qubits(&self) -> impl Iterator<Item = usize> + '_ {
        self.targets
            .iter()
            .copied()
            .chain(self.controls.iter().map(|&(q, _)| q))
    }

    pub fn width(&self) -> usize {
        self.targets.len() + self.controls.len()
    }

    /// The op as a plain matrix on its controls followed by its targets.
    fn full(&self) -> (Vec<usize>, Matrix) {
        if self.controls.is_empty() {
            return (self.targets.clone(), self.matrix.clone());
        }
        let state = self
            .controls
            .iter()
            .enumerate()
            .map(|(i, &(_, v))| (v as usize) << i)
            .sum();
        let qubits = self
            .controls
            .iter()
            .map(|&(q, _)| q)
            .chain(self.targets.iter().copied())
            .collect();
        (qubits, self.matrix.controlled(self.controls.len(), state))
    }
}

/// Fuse `ops` into groups acting on at most `max_qubits` qubits.
///
/// Every op wider than `max_qubits` stays on its own. A group of one op is
/// returned unchanged, so controlled gates keep their cheaper controlled
/// kernel.
pub fn fuse(ops: &[Op], max_qubits: usize) -> Vec<Op> {
    fuse_where(ops, max_qubits, |_| true)
}

/// Like [`fuse`], but ops for which `fusable` is false are never merged
/// (they stay alone and act as barriers on their qubits).
pub fn fuse_where(ops: &[Op], max_qubits: usize, fusable: impl Fn(&Op) -> bool) -> Vec<Op> {
    let width = |op: &Op| if fusable(op) { op.width() } else { usize::MAX };
    let mut done = vec![false; ops.len()];
    let mut fused = Vec::new();
    let mut seed = 0;
    while seed < ops.len() {
        if done[seed] {
            seed += 1;
            continue;
        }
        done[seed] = true;
        let mut group = Group {
            seed,
            members: vec![seed],
            qubits: ops[seed].qubits().collect(),
        };
        if width(&ops[seed]) <= max_qubits {
            // First grow along gates that share a qubit with the group, then
            // fill the remaining room with unrelated gates. Filling first
            // would spend the room on qubits the chain never needed.
            for connected_only in [true, false] {
                group.grow(ops, &mut done, max_qubits, connected_only, &width);
            }
        }
        let Group {
            members,
            mut qubits,
            ..
        } = group;
        if members.len() == 1 {
            fused.push(ops[seed].clone());
            continue;
        }
        qubits.sort_unstable();
        let mut matrix = Matrix::identity(qubits.len());
        for &i in &members {
            let (op_qubits, op_matrix) = ops[i].full();
            matrix = op_matrix.embed(&op_qubits, &qubits).mul(&matrix);
        }
        fused.push(Op {
            targets: qubits,
            controls: Vec::new(),
            matrix,
        });
    }
    fused
}

/// A fused group being built: the op it started from, its member ops (in
/// program order) and the qubits they touch.
struct Group {
    seed: usize,
    members: Vec<usize>,
    qubits: Vec<usize>,
}

impl Group {
    /// Add every later op that can legally move up to the group.
    ///
    /// An op may join if none of its qubits is *blocked* (has an earlier op
    /// that is not in the group and not yet fused) and the group stays
    /// within `max_qubits`. With `connected_only`, ops sharing no qubit with
    /// the group are skipped (and block their qubits).
    fn grow(
        &mut self,
        ops: &[Op],
        done: &mut [bool],
        max_qubits: usize,
        connected_only: bool,
        width: &impl Fn(&Op) -> usize,
    ) {
        let mut blocked: Vec<usize> = Vec::new();
        for (i, op) in ops.iter().enumerate().skip(self.seed + 1) {
            if done[i] {
                continue;
            }
            if connected_only && self.qubits.iter().all(|q| blocked.contains(q)) {
                break;
            }
            let op_qubits: Vec<usize> = op.qubits().collect();
            let touches_blocked = op_qubits.iter().any(|q| blocked.contains(q));
            let new: Vec<usize> = op_qubits
                .iter()
                .copied()
                .filter(|q| !self.qubits.contains(q))
                .collect();
            let connected = new.len() < op_qubits.len();
            let fits = width(op) <= max_qubits && self.qubits.len() + new.len() <= max_qubits;
            if !touches_blocked && fits && (connected || !connected_only) {
                done[i] = true;
                self.members.push(i);
                self.qubits.extend(new);
            } else {
                for q in op_qubits {
                    if !blocked.contains(&q) {
                        blocked.push(q);
                    }
                }
            }
        }
        self.members.sort_unstable();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Gate;

    fn op(gate: Gate, qubits: &[usize]) -> Op {
        Op {
            targets: qubits.to_vec(),
            controls: vec![],
            matrix: gate.matrix(),
        }
    }

    #[test]
    fn fuses_across_unrelated_gates() {
        // h(0) h(5) cx(0,1) h(6) cx(1,2): the chain on 0..2 fuses into one
        // 3-qubit gate first; the unrelated h(5) and h(6) form another.
        let ops = vec![
            op(Gate::H, &[0]),
            op(Gate::H, &[5]),
            op(Gate::CX, &[0, 1]),
            op(Gate::H, &[6]),
            op(Gate::CX, &[1, 2]),
        ];
        let fused = fuse(&ops, 3);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].targets, vec![0, 1, 2]);
        assert_eq!(fused[1].targets, vec![5, 6]);
    }

    #[test]
    fn respects_dependencies() {
        // cx(0,1) h(1) cx(1,2) with max 2: the h(1) must stay between the CXs.
        let ops = vec![
            op(Gate::CX, &[0, 1]),
            op(Gate::H, &[1]),
            op(Gate::CX, &[1, 2]),
        ];
        let fused = fuse(&ops, 2);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].targets, vec![0, 1]);
        assert_eq!(fused[1].targets, vec![1, 2]);
    }
}
