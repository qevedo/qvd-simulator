//! Near-Clifford simulation: a Clifford frame around a small dense state.
//!
//! The state is `|ψ> = C (|φ>_A ⊗ |0...0>_D)`, after the design of Clifft
//! (Chase & Labib, arXiv:2604.27058). `C` is a Clifford on `n` qubits, kept
//! as an inverse stabilizer tableau, and the virtual qubits are split into
//! a few *active* ones, `A`, whose joint state `φ` is a dense vector, and
//! *dormant* ones, `D`, all in `|0>`.
//!
//! - A Clifford gate `G` only updates the frame: `C -> G C`.
//! - Every other gate is lowered to Cliffords and Pauli rotations
//!   `exp(-iθ/2 P)`. Pulled back through the frame, `P` becomes a virtual
//!   Pauli `P'`. On dormant qubits its Z parts act as `+1`. If it has no X
//!   part there, the rotation acts on `φ` alone. Otherwise the dormant X
//!   parts are gathered onto one dormant qubit by CX gates prepended to the
//!   frame (`C -> C W`, and `W` fixes `|0...0>`, so `φ` does not change),
//!   and that qubit becomes active: the dense state doubles.
//! - A Z measurement pulls `Z_q` back the same way. With an X part on a
//!   dormant qubit `d`, the outcome is uniformly random and the collapsed
//!   state is `C V (φ ⊗ |0>)` for a Clifford `V` (a Pauli on `A` controlled
//!   by `d`, then H and maybe X on `d`): only the frame changes. Otherwise
//!   the measurement acts on `φ`. After it, a Clifford on the active qubits
//!   turns the measured Pauli into `Z` on one of them, which then factors
//!   out as dormant: the dense state halves.
//!
//! The cost is exponential only in the number of active qubits, which
//! grows by at most one per non-Clifford rotation and shrinks at
//! measurements. With no non-Clifford gates nothing is ever active and
//! this is a (slower) stabilizer simulator; with many it is a state vector.

mod dense;

use std::f64::consts::FRAC_PI_4;
use std::sync::Arc;
use std::time::Instant;

use rand::{Rng, RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::circuit::{Circuit, Gate, Instruction};
use crate::matrix::{C64, Matrix};
use crate::simulator::{RunResult, bitstring};
use crate::stabilizer::{Clifford, Tableau, lower};

use dense::{Dense, Gate as DenseGate, Pauli};

/// A step of a lowered gate.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Clifford(Clifford),
    /// `exp(-i θ/2 P)` for the Pauli with these `(qubit, 'X' | 'Y' | 'Z')`
    /// factors.
    Rotation(Vec<(usize, char)>, f64),
}

fn rotation(factors: &[(usize, char)], theta: f64) -> Step {
    Step::Rotation(factors.to_vec(), theta)
}

/// The ZYZ Euler angles `(θ, φ, λ)` of a one-qubit unitary, up to global
/// phase: `U ∝ RZ(φ) RY(θ) RZ(λ)`.
fn euler_angles(m: &Matrix) -> (f64, f64, f64) {
    let (a, b, c, d) = (m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
    let theta = 2.0 * c.norm().atan2(a.norm());
    // Remove the global phase so that the determinant is 1.
    let det = a * d - b * c;
    let phase = det.sqrt();
    let (a, c) = (a / phase, c / phase);
    // a = cos(θ/2) e^{-i(φ+λ)/2}, c = sin(θ/2) e^{i(φ-λ)/2}.
    let sum = -2.0 * a.arg();
    let difference = 2.0 * c.arg();
    if c.norm() < 1e-12 {
        (theta, sum, 0.0)
    } else if a.norm() < 1e-12 {
        (theta, difference, 0.0)
    } else {
        (theta, (sum + difference) / 2.0, (sum - difference) / 2.0)
    }
}

/// Lower a gate to Clifford steps and Pauli rotations (exactly, up to
/// global phase), or `None` for gates this backend does not support
/// (arbitrary unitaries on two or more qubits).
pub fn decompose(gate: &Gate, qubits: &[usize]) -> Option<Vec<Step>> {
    if let Some(ops) = lower(gate, qubits) {
        return Some(ops.into_iter().map(Step::Clifford).collect());
    }
    let q = qubits[0];
    let steps = match *gate {
        Gate::T => vec![rotation(&[(q, 'Z')], FRAC_PI_4)],
        Gate::Tdg => vec![rotation(&[(q, 'Z')], -FRAC_PI_4)],
        Gate::RZ(t) | Gate::P(t) => vec![rotation(&[(q, 'Z')], t)],
        Gate::RX(t) => vec![rotation(&[(q, 'X')], t)],
        Gate::RY(t) => vec![rotation(&[(q, 'Y')], t)],
        Gate::U(theta, phi, lambda) => vec![
            rotation(&[(q, 'Z')], lambda),
            rotation(&[(q, 'Y')], theta),
            rotation(&[(q, 'Z')], phi),
        ],
        Gate::Unitary(ref m) if qubits.len() == 1 => {
            let (theta, phi, lambda) = euler_angles(m);
            vec![
                rotation(&[(q, 'Z')], lambda),
                rotation(&[(q, 'Y')], theta),
                rotation(&[(q, 'Z')], phi),
            ]
        }
        // CP(λ) = exp(iλ|11><11|) ∝ RZ_c(λ/2) RZ_t(λ/2) R_ZZ(-λ/2).
        Gate::CP(l) => {
            let (c, t) = (qubits[0], qubits[1]);
            vec![
                rotation(&[(c, 'Z')], l / 2.0),
                rotation(&[(t, 'Z')], l / 2.0),
                rotation(&[(c, 'Z'), (t, 'Z')], -l / 2.0),
            ]
        }
        // CR_P(θ) = exp(-iθ/2 P_t |1><1|_c) = R_{P_t}(θ/2) R_{Z_c P_t}(-θ/2).
        Gate::CRX(t) | Gate::CRY(t) | Gate::CRZ(t) => {
            let p = match gate {
                Gate::CRX(_) => 'X',
                Gate::CRY(_) => 'Y',
                _ => 'Z',
            };
            let (c, target) = (qubits[0], qubits[1]);
            vec![
                rotation(&[(target, p)], t / 2.0),
                rotation(&[(c, 'Z'), (target, p)], -t / 2.0),
            ]
        }
        Gate::RZZ(t) => vec![rotation(&[(qubits[0], 'Z'), (qubits[1], 'Z')], t)],
        Gate::RXX(t) => vec![rotation(&[(qubits[0], 'X'), (qubits[1], 'X')], t)],
        // Qiskit's definition: S H T CX T† H S† on the target.
        Gate::CH => {
            let (c, t) = (qubits[0], qubits[1]);
            vec![
                Step::Clifford(Clifford::S(t)),
                Step::Clifford(Clifford::H(t)),
                rotation(&[(t, 'Z')], FRAC_PI_4),
                Step::Clifford(Clifford::CX(c, t)),
                rotation(&[(t, 'Z')], -FRAC_PI_4),
                Step::Clifford(Clifford::H(t)),
                Step::Clifford(Clifford::Sdg(t)),
            ]
        }
        Gate::CCX => toffoli(qubits[0], qubits[1], qubits[2]),
        Gate::CSWAP => {
            let (c, a, b) = (qubits[0], qubits[1], qubits[2]);
            let mut steps = vec![Step::Clifford(Clifford::CX(b, a))];
            steps.extend(toffoli(c, a, b));
            steps.push(Step::Clifford(Clifford::CX(b, a)));
            steps
        }
        _ => return None,
    };
    Some(steps)
}

/// The standard seven-T Toffoli.
fn toffoli(a: usize, b: usize, t: usize) -> Vec<Step> {
    let (h, cx) = (Clifford::H, Clifford::CX);
    let tg = |q| rotation(&[(q, 'Z')], FRAC_PI_4);
    let tdg = |q| rotation(&[(q, 'Z')], -FRAC_PI_4);
    vec![
        Step::Clifford(h(t)),
        Step::Clifford(cx(b, t)),
        tdg(t),
        Step::Clifford(cx(a, t)),
        tg(t),
        Step::Clifford(cx(b, t)),
        tdg(t),
        Step::Clifford(cx(a, t)),
        tg(b),
        tg(t),
        Step::Clifford(h(t)),
        Step::Clifford(cx(a, b)),
        tg(a),
        tdg(b),
        Step::Clifford(cx(a, b)),
    ]
}

/// Whether every gate of the circuit can be lowered.
pub fn is_supported(circuit: &Circuit) -> bool {
    circuit.instructions.iter().all(|i| match i {
        Instruction::Gate { gate, qubits } => decompose(gate, qubits).is_some(),
        _ => true,
    })
}

/// How many non-Clifford rotations the circuit has: an upper bound on the
/// number of active qubits, since each activates at most one.
pub fn rotation_count(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter_map(|i| match i {
            Instruction::Gate { gate, qubits } => decompose(gate, qubits),
            _ => None,
        })
        .flatten()
        .filter(|s| matches!(s, Step::Rotation(..)))
        .count()
}

/// The largest number of active qubits the circuit reaches, found by
/// running only its Clifford frame (activations do not depend on
/// measurement outcomes), or `None` if it has unsupported gates. The dense
/// part needs `16 * 2^peak` bytes.
pub fn peak_active_qubits(circuit: &Circuit) -> Option<usize> {
    if !is_supported(circuit) {
        return None;
    }
    let mut state = NearClifford::new(circuit.num_qubits);
    state.symbolic = true;
    let mut rng = Pcg64::seed_from_u64(0);
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => state.apply(gate, qubits).ok()?,
            Instruction::Measure { qubit, .. } => {
                state.measure(*qubit, &mut rng);
            }
            Instruction::Reset { qubit } => state.reset(*qubit, &mut rng),
            Instruction::Barrier { .. } => {}
        }
    }
    Some(state.peak_active_qubits())
}

/// A near-Clifford state; see the module documentation.
#[derive(Clone, Debug)]
pub struct NearClifford {
    n: usize,
    /// The inverse tableau of `C`; its columns are the virtual qubits.
    frame: Tableau,
    /// Shared copy-on-write between branches of a measurement tree.
    dense: Arc<Dense>,
    /// The probability last computed by [`NearClifford::probability_one`].
    cached: Option<(usize, f64)>,
    /// The virtual qubit at each dense position.
    active: Vec<usize>,
    peak: usize,
    /// Track only which qubits are active, without amplitudes (see
    /// [`peak_active_qubits`]).
    symbolic: bool,
    /// When set, every Clifford prepended to the frame is also recorded
    /// here, for Pauli frames that must follow the frame (see
    /// [`sample_terminal`]).
    log: Option<Vec<Clifford>>,
}

/// A virtual Pauli: sign and x/z bit words over the virtual qubits.
type Virtual = (bool, Vec<u64>, Vec<u64>);

fn bit(words: &[u64], q: usize) -> bool {
    words[q / 64] >> (q % 64) & 1 == 1
}

fn set_bits(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(w, &word)| {
        std::iter::successors((word != 0).then_some(word), |&rest| {
            Some(rest & (rest - 1)).filter(|&r| r != 0)
        })
        .map(move |rest| w * 64 + rest.trailing_zeros() as usize)
    })
}

impl NearClifford {
    /// `|0...0>` on `n` qubits: nothing active.
    pub fn new(n: usize) -> NearClifford {
        NearClifford {
            n,
            frame: Tableau::new(n),
            dense: Arc::new(Dense::new()),
            cached: None,
            active: Vec::new(),
            peak: 0,
            symbolic: false,
            log: None,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// `C -> C V` for a Clifford `V` on virtual qubits that the caller
    /// compensates for (it fixes the state, or `φ` was changed by `V†`).
    fn prepend(&mut self, op: Clifford) {
        self.frame.prepend(op);
        if let Some(log) = &mut self.log {
            log.push(op);
        }
    }

    /// The number of active qubits now.
    pub fn active_qubits(&self) -> usize {
        self.active.len()
    }

    /// The largest number of active qubits so far.
    pub fn peak_active_qubits(&self) -> usize {
        self.peak
    }

    /// Apply a gate; fails for unsupported gates.
    pub fn apply(&mut self, gate: &Gate, qubits: &[usize]) -> Result<(), String> {
        let steps = decompose(gate, qubits)
            .ok_or_else(|| format!("{gate:?} is not supported by the near-Clifford backend"))?;
        for step in steps {
            self.step(step);
        }
        Ok(())
    }

    pub fn step(&mut self, step: Step) {
        self.cached = None;
        match step {
            Step::Clifford(op) => self.frame.apply(op),
            Step::Rotation(factors, theta) => self.rotate(&factors, theta),
        }
    }

    fn position(&self, virtual_qubit: usize) -> Option<usize> {
        self.active.iter().position(|&v| v == virtual_qubit)
    }

    /// The active part of a virtual Pauli, as a Pauli on positions (its Z
    /// parts on dormant qubits act as `+1` and are dropped).
    fn on_positions(&self, (negative, x, z): &Virtual) -> Pauli {
        let mut p = Pauli {
            x: 0,
            z: 0,
            negative: *negative,
        };
        for (position, &v) in self.active.iter().enumerate() {
            p.x |= (bit(x, v) as u64) << position;
            p.z |= (bit(z, v) as u64) << position;
        }
        p
    }

    /// Dormant qubits where the virtual Pauli has an X or Y.
    fn dormant_x(&self, (_, x, _): &Virtual) -> Vec<usize> {
        set_bits(x)
            .filter(|&v| self.position(v).is_none())
            .collect()
    }

    /// Gather the dormant X parts of the Pauli pulled back from `factors`
    /// onto one dormant qubit with CX gates prepended to the frame, and
    /// return that qubit with the new pulled-back Pauli.
    fn gather_dormant(&mut self, factors: &[(usize, char)]) -> (Option<usize>, Virtual) {
        let pulled = self.frame.pull_back(factors);
        let dormant = self.dormant_x(&pulled);
        let Some((&first, rest)) = dormant.split_first() else {
            return (None, pulled);
        };
        // CX(first, d) maps X_first X_d to X_first; with `first` in |0> it
        // leaves the state alone.
        for &d in rest {
            self.prepend(Clifford::CX(first, d));
        }
        (Some(first), self.frame.pull_back(factors))
    }

    /// `exp(-i θ/2 P)` for a physical Pauli.
    pub fn rotate(&mut self, factors: &[(usize, char)], theta: f64) {
        self.cached = None;
        let (dormant, pulled) = self.gather_dormant(factors);
        if let Some(d) = dormant {
            self.active.push(d);
            self.peak = self.peak.max(self.active.len());
            if self.symbolic {
                return;
            }
            self.dense_mut().grow();
        }
        let p = self.on_positions(&pulled);
        if !p.is_identity() && !self.symbolic {
            self.dense_mut().rotate(p, theta);
        }
        debug_assert!(self.symbolic || self.dense.qubits() == self.active.len());
    }

    /// Measure qubit `q` in the Z basis.
    pub fn measure(&mut self, q: usize, rng: &mut impl Rng) -> bool {
        let outcome = match self.probability_one(q) {
            p if p == 0.0 || p == 1.0 => p == 1.0,
            0.5 => rng.random::<bool>(),
            p => rng.random::<f64>() < p,
        };
        self.collapse(q, outcome);
        outcome
    }

    /// The probability that measuring qubit `q` gives 1: exactly 0 or 1 when
    /// determined, exactly 1/2 when the measured Pauli has an X part on a
    /// dormant qubit. May rewrite the frame without changing the state.
    pub fn probability_one(&mut self, q: usize) -> f64 {
        let (dormant, pulled) = self.gather_dormant(&[(q, 'Z')]);
        if dormant.is_some() {
            return 0.5;
        }
        let p = self.on_positions(&pulled);
        if p.is_identity() {
            return if p.negative { 1.0 } else { 0.0 };
        }
        if self.symbolic {
            return 0.5;
        }
        let one = ((1.0 - self.dense.expectation(p)) / 2.0).clamp(0.0, 1.0);
        self.cached = Some((q, one));
        one
    }

    /// Collapse the state as if measuring qubit `q` gave `outcome`, which
    /// must have nonzero probability.
    pub fn collapse(&mut self, q: usize, outcome: bool) {
        let factors = [(q, 'Z')];
        let (dormant, pulled) = self.gather_dormant(&factors);
        if let Some(d) = dormant {
            self.cached = None;
            return self.collapse_dormant(&factors, d, pulled, outcome);
        }
        let p = self.on_positions(&pulled);
        if p.is_identity() {
            debug_assert_eq!(p.negative, outcome, "collapsing onto an impossible outcome");
            return;
        }
        let eigenvalue = if outcome { -1.0 } else { 1.0 };
        if !self.symbolic {
            let one = match self.cached.take() {
                Some((cached, one)) if cached == q => one,
                _ => ((1.0 - self.dense.expectation(p)) / 2.0).clamp(0.0, 1.0),
            };
            let probability = if outcome { one } else { 1.0 - one };
            assert!(probability > 0.0, "collapsing onto an impossible outcome");
            self.dense_mut().project(p, eigenvalue, probability);
        }
        self.retire(p, eigenvalue);
    }

    /// The dense part, copied first if another state shares it.
    fn dense_mut(&mut self) -> &mut Dense {
        Arc::make_mut(&mut self.dense)
    }

    /// The measured Pauli has an X or Y on dormant qubit `d` (and on no
    /// other dormant qubit): each outcome has probability 1/2, and the
    /// collapse is a Clifford on the frame.
    fn collapse_dormant(
        &mut self,
        factors: &[(usize, char)],
        d: usize,
        mut pulled: Virtual,
        outcome: bool,
    ) {
        if bit(&pulled.2, d) {
            // Y on d: S† Y S = X, and S fixes |0>.
            self.prepend(Clifford::S(d));
            pulled = self.frame.pull_back(factors);
        }
        // P' = ± P_A ⊗ X_d (⊗ Z on dormant qubits), so the collapsed state is
        // (φ ⊗ |0> + c P_A φ ⊗ |1>)/√2 with c = ±1: controlled-P_A (control
        // d) after H on d, and X before the H if c = -1.
        for v in self.active.clone() {
            match (bit(&pulled.1, v), bit(&pulled.2, v)) {
                (true, false) => self.prepend(Clifford::CX(d, v)),
                (false, true) => self.prepend(Clifford::CZ(d, v)),
                (true, true) => {
                    // CY = S CX S† on the target.
                    self.prepend(Clifford::S(v));
                    self.prepend(Clifford::CX(d, v));
                    self.prepend(Clifford::Sdg(v));
                }
                (false, false) => {}
            }
        }
        self.prepend(Clifford::H(d));
        if outcome ^ pulled.0 {
            self.prepend(Clifford::X(d));
        }
    }

    /// After `φ` became an eigenstate of `p` (with `eigenvalue`), turn `p`
    /// into Z on one position with Cliffords on the active qubits (applied
    /// to `φ` and prepended to the frame), and retire that qubit.
    fn retire(&mut self, mut p: Pauli, eigenvalue: f64) {
        let mut gates = Vec::new();
        let target = if p.x != 0 {
            let j = p.x.trailing_zeros() as usize;
            if p.z >> j & 1 == 1 {
                gates.push(Clifford::S(j));
            }
            for i in set_positions(p.x & !(1 << j)) {
                gates.push(Clifford::CX(j, i));
            }
            j
        } else {
            let j = p.z.trailing_zeros() as usize;
            for i in set_positions(p.z & !(1 << j)) {
                gates.push(Clifford::CX(i, j));
            }
            j
        };
        for &g in &gates {
            conjugate(&mut p, g);
        }
        if p.x != 0 {
            // Clear the Z parts left next to the X on `target`.
            let mut more = Vec::new();
            for i in set_positions(p.z & !(1 << target)) {
                more.push(Clifford::CZ(target, i));
            }
            if p.z >> target & 1 == 1 {
                more.push(Clifford::S(target));
            }
            more.push(Clifford::H(target));
            for &g in &more {
                conjugate(&mut p, g);
            }
            gates.extend(more);
        }
        debug_assert_eq!(
            (p.x, p.z),
            (0, 1 << target),
            "the measured Pauli should be Z on one position"
        );
        for &g in &gates {
            self.apply_virtual(g);
        }
        // φ is now an eigenstate of ±Z_target.
        let one = (eigenvalue < 0.0) ^ p.negative;
        if one {
            self.apply_virtual(Clifford::X(target));
        }
        if !self.symbolic {
            self.dense_mut().shrink(target);
        }
        self.active.remove(target);
        debug_assert!(self.symbolic || self.dense.qubits() == self.active.len());
    }

    /// Apply a Clifford `G` on positions as `φ -> G† φ`, `C -> C G`: the
    /// state is unchanged.
    fn apply_virtual(&mut self, g: Clifford) {
        let virt = |p: usize| self.active[p];
        let (dense, frame) = match g {
            Clifford::H(p) => (DenseGate::H(p), Clifford::H(virt(p))),
            Clifford::S(p) => (
                DenseGate::Phase(p, C64::new(0.0, -1.0)),
                Clifford::S(virt(p)),
            ),
            Clifford::Sdg(p) => (
                DenseGate::Phase(p, C64::new(0.0, 1.0)),
                Clifford::Sdg(virt(p)),
            ),
            Clifford::X(p) => (DenseGate::X(p), Clifford::X(virt(p))),
            Clifford::CX(c, t) => (DenseGate::CX(c, t), Clifford::CX(virt(c), virt(t))),
            Clifford::CZ(a, b) => (DenseGate::CZ(a, b), Clifford::CZ(virt(a), virt(b))),
            other => unreachable!("{other:?} is not used on positions"),
        };
        if !self.symbolic {
            self.dense_mut().apply(dense);
        }
        self.prepend(frame);
    }

    /// Reset qubit `q` to |0>.
    pub fn reset(&mut self, q: usize, rng: &mut impl Rng) {
        if self.measure(q, rng) {
            self.frame.apply(Clifford::X(q));
        }
    }

    /// `<ψ|P|ψ>` for a physical Pauli.
    pub fn expectation(&self, factors: &[(usize, char)]) -> f64 {
        let pulled = self.frame.pull_back(factors);
        if !self.dormant_x(&pulled).is_empty() {
            return 0.0;
        }
        let p = self.on_positions(&pulled);
        if p.is_identity() {
            return if p.negative { -1.0 } else { 1.0 };
        }
        self.dense.expectation(p)
    }
}

fn set_positions(mask: u64) -> impl Iterator<Item = usize> {
    (0..64).filter(move |&i| mask >> i & 1 == 1)
}

/// `P -> G† P G` on positions.
fn conjugate(p: &mut Pauli, g: Clifford) {
    let get = |m: u64, i: usize| m >> i & 1 == 1;
    let flip = |m: &mut u64, i: usize, on: bool| *m ^= (on as u64) << i;
    match g {
        Clifford::H(j) => {
            let (x, z) = (get(p.x, j), get(p.z, j));
            p.negative ^= x && z;
            flip(&mut p.x, j, x != z);
            flip(&mut p.z, j, x != z);
        }
        Clifford::S(j) => {
            let (x, z) = (get(p.x, j), get(p.z, j));
            p.negative ^= x && !z;
            flip(&mut p.z, j, x);
        }
        Clifford::CX(c, t) => {
            let (xc, zc, xt, zt) = (get(p.x, c), get(p.z, c), get(p.x, t), get(p.z, t));
            p.negative ^= xc && zt && (xt == zc);
            flip(&mut p.x, t, xc);
            flip(&mut p.z, c, zt);
        }
        Clifford::CZ(a, b) => {
            let (xa, za, xb, zb) = (get(p.x, a), get(p.z, a), get(p.x, b), get(p.z, b));
            p.negative ^= xa && xb && (za != zb);
            flip(&mut p.z, a, xb);
            flip(&mut p.z, b, xa);
        }
        other => unreachable!("{other:?} is not used on positions"),
    }
}

/// Draws from `Binomial(n, p)`, as `n` Bernoulli trials.
fn binomial(rng: &mut impl Rng, n: usize, p: f64) -> usize {
    (0..n).filter(|_| rng.random::<f64>() < p).count()
}

/// The end of one branch of a measurement tree: a classical record, how many
/// shots ended with it, and the peak number of active qubits on the way.
type Leaf = (Vec<bool>, usize, usize);

/// What a Z measurement does to a state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// The outcome is fixed.
    Fixed,
    /// 50/50, changing only the frame.
    Frame,
    /// Acts on the dense part, which then loses a qubit.
    Dense,
}

impl NearClifford {
    fn kind(&self, q: usize) -> Kind {
        let pulled = self.frame.pull_back(&[(q, 'Z')]);
        if !self.dormant_x(&pulled).is_empty() {
            Kind::Frame
        } else if self.on_positions(&pulled).is_identity() {
            Kind::Fixed
        } else {
            Kind::Dense
        }
    }
}

/// Run `shots` shots of `pending` measurements (`(qubit, clbit)`, distinct
/// qubits, so in any order) and then `tail` from `state`, as one
/// measurement tree. Each measurement is done once per branch: its outcome
/// probability splits the branch's shots binomially, and each outcome with
/// shots continues as its own branch (sharing the dense part until one of
/// them changes it). Measurements that act on the dense part go first, while
/// the shots are still together: each halves the dense part for everything
/// after it. Branches with many shots run in parallel; each draws its own
/// seed, so the result depends only on `seed`.
fn branch(
    mut state: NearClifford,
    mut pending: Vec<(usize, usize)>,
    mut tail: &[Instruction],
    shots: usize,
    seed: u64,
    mut clbits: Vec<bool>,
) -> Vec<Leaf> {
    let mut rng = Pcg64::seed_from_u64(seed);
    loop {
        if !pending.is_empty()
            && tail
                .iter()
                .all(|i| matches!(i, Instruction::Barrier { .. }))
        {
            // The final measurements: Pauli frames instead of branches.
            let records = clbits
                .iter()
                .map(|&b| {
                    if b {
                        valid(shots)
                    } else {
                        vec![0; shots.div_ceil(64)]
                    }
                })
                .collect();
            let frames = ShotFrames::new(state.n, shots);
            return sample_terminal(state, pending, frames, records, rng.random());
        }
        // The next measurement or reset: (qubit, clbit or None for reset).
        let (qubit, clbit) = if !pending.is_empty() {
            // Dense measurements first (each halves the dense part), then
            // fixed ones (no branching), then 50/50 ones.
            let index = if state.active_qubits() > 0 {
                let kinds: Vec<Kind> = pending.iter().map(|&(q, _)| state.kind(q)).collect();
                kinds
                    .iter()
                    .position(|&k| k == Kind::Dense)
                    .or_else(|| kinds.iter().position(|&k| k == Kind::Fixed))
                    .unwrap_or(0)
            } else {
                0
            };
            let (q, c) = pending.swap_remove(index);
            (q, Some(c))
        } else {
            let Some((instruction, rest)) = tail.split_first() else {
                break;
            };
            tail = rest;
            match instruction {
                Instruction::Gate { gate, qubits } => {
                    state.apply(gate, qubits).expect("checked by run");
                    continue;
                }
                Instruction::Barrier { .. } => continue,
                Instruction::Reset { qubit } => (*qubit, None),
                Instruction::Measure { qubit, clbit } => {
                    // Gather the run of measurements on distinct qubits.
                    pending.push((*qubit, *clbit));
                    while let Some((Instruction::Measure { qubit, clbit }, rest)) =
                        tail.split_first()
                    {
                        if pending.iter().any(|&(q, _)| q == *qubit) {
                            break;
                        }
                        pending.push((*qubit, *clbit));
                        tail = rest;
                    }
                    continue;
                }
            }
        };
        let finish = |state: &mut NearClifford, clbits: &mut Vec<bool>, outcome: bool| {
            state.collapse(qubit, outcome);
            match clbit {
                Some(c) => clbits[c] = outcome,
                None if outcome => state.frame.apply(Clifford::X(qubit)),
                None => {}
            }
        };
        let p = state.probability_one(qubit);
        let ones = match p {
            0.0 => 0,
            1.0 => shots,
            p => binomial(&mut rng, shots, p),
        };
        if ones == 0 || ones == shots {
            finish(&mut state, &mut clbits, ones == shots);
            continue;
        }
        let (seed0, seed1) = (rng.random::<u64>(), rng.random::<u64>());
        let (mut other, mut other_clbits) = (state.clone(), clbits.clone());
        finish(&mut state, &mut clbits, false);
        finish(&mut other, &mut other_clbits, true);
        let other_pending = pending.clone();
        let (mut zeros, ones_leaves) = if shots >= 64 {
            rayon::join(
                || branch(state, pending, tail, shots - ones, seed0, clbits),
                || branch(other, other_pending, tail, ones, seed1, other_clbits),
            )
        } else {
            (
                branch(state, pending, tail, shots - ones, seed0, clbits),
                branch(other, other_pending, tail, ones, seed1, other_clbits),
            )
        };
        zeros.extend(ones_leaves);
        return zeros;
    }
    let peak = state.peak_active_qubits();
    vec![(clbits, shots, peak)]
}

/// Pauli frames of a group of shots on the virtual qubits, 64 shots a word.
/// Shot `s` is in the state `C P_s (φ ⊗ |0...0>)`; signs of `P_s` are
/// dropped, as they only change a shot's global phase.
#[derive(Clone, Debug)]
struct ShotFrames {
    shots: usize,
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
}

impl ShotFrames {
    fn new(qubits: usize, shots: usize) -> ShotFrames {
        let words = shots.div_ceil(64);
        ShotFrames {
            shots,
            words,
            x: vec![0; qubits * words],
            z: vec![0; qubits * words],
        }
    }

    fn column(&self, q: usize) -> std::ops::Range<usize> {
        q * self.words..(q + 1) * self.words
    }

    fn xor(table: &mut [u64], words: usize, dst: usize, src: usize) {
        for w in 0..words {
            table[dst * words + w] ^= table[src * words + w];
        }
    }

    /// `P_s -> V† P_s V` after `V` was prepended to the shared frame (`C P_s
    /// = (C V)(V† P_s V)`).
    fn conjugate(&mut self, op: Clifford) {
        let w = self.words;
        match op {
            Clifford::H(q) => {
                let r = self.column(q);
                self.x[r.clone()].swap_with_slice(&mut self.z[r]);
            }
            Clifford::S(q) | Clifford::Sdg(q) => {
                for i in self.column(q) {
                    self.z[i] ^= self.x[i];
                }
            }
            Clifford::X(_) | Clifford::Y(_) | Clifford::Z(_) => {}
            Clifford::CX(c, t) => {
                Self::xor(&mut self.x, w, t, c);
                Self::xor(&mut self.z, w, c, t);
            }
            Clifford::CZ(a, b) => {
                for i in 0..w {
                    let (xa, xb) = (self.x[a * w + i], self.x[b * w + i]);
                    self.z[a * w + i] ^= xb;
                    self.z[b * w + i] ^= xa;
                }
            }
            Clifford::Swap(a, b) => {
                for i in 0..w {
                    self.x.swap(a * w + i, b * w + i);
                    self.z.swap(a * w + i, b * w + i);
                }
            }
        }
    }

    /// Per shot, whether `P_s` anticommutes with the virtual Pauli `(x, z)`.
    fn anticommutes(&self, x: &[u64], z: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; self.words];
        for v in set_bits(x) {
            for (o, &f) in out.iter_mut().zip(&self.z[self.column(v)]) {
                *o ^= f;
            }
        }
        for v in set_bits(z) {
            for (o, &f) in out.iter_mut().zip(&self.x[self.column(v)]) {
                *o ^= f;
            }
        }
        out
    }

    /// The frames of the shots whose bit is set in `mask`, in order.
    fn select(&self, mask: &[u64], qubits: usize) -> ShotFrames {
        let chosen: Vec<usize> = (0..self.shots)
            .filter(|&s| mask[s / 64] >> (s % 64) & 1 == 1)
            .collect();
        let mut out = ShotFrames::new(qubits, chosen.len());
        for q in 0..qubits {
            for (i, &s) in chosen.iter().enumerate() {
                let (w, b) = (q * self.words + s / 64, s % 64);
                out.x[q * out.words + i / 64] |= (self.x[w] >> b & 1) << (i % 64);
                out.z[q * out.words + i / 64] |= (self.z[w] >> b & 1) << (i % 64);
            }
        }
        out
    }
}

/// Bits over `shots`: word-wise mask of the valid shots.
fn valid(shots: usize) -> Vec<u64> {
    (0..shots.div_ceil(64))
        .map(|w| {
            if (w + 1) * 64 <= shots {
                u64::MAX
            } else {
                (1u64 << (shots % 64)) - 1
            }
        })
        .collect()
}

/// Sample the final measurements `pending` (on distinct qubits, nothing after
/// them) of `shots` shots, each with its own Pauli frame (see
/// [`ShotFrames`]). Measurements with a fixed outcome, or a 50/50 one that
/// only changes the frame, are done once for all shots: per shot the outcome
/// flips where its frame anticommutes with the measured Pauli, and a random
/// 50/50 outcome goes into its frame. Only measurements on the dense part
/// branch, and only on the outcome they have in the shared state, so the
/// dense work does not grow with the number of shots. `records` holds the
/// classical bits so far, a column of shots per bit.
fn sample_terminal(
    mut state: NearClifford,
    mut pending: Vec<(usize, usize)>,
    mut frames: ShotFrames,
    mut records: Vec<Vec<u64>>,
    seed: u64,
) -> Vec<Leaf> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let n = state.n;
    let sync = |state: &mut NearClifford, frames: &mut ShotFrames| {
        for op in state.log.as_mut().expect("logging").drain(..) {
            frames.conjugate(op);
        }
    };
    state.log = Some(Vec::new());
    let shots = frames.shots;
    let mask = valid(shots);
    while !pending.is_empty() {
        let kinds: Vec<Kind> = pending.iter().map(|&(q, _)| state.kind(q)).collect();
        let index = kinds
            .iter()
            .position(|&k| k == Kind::Dense)
            .or_else(|| kinds.iter().position(|&k| k == Kind::Fixed))
            .unwrap_or(0);
        let kind = kinds[index];
        let (q, c) = pending.swap_remove(index);
        let (dormant, pulled) = state.gather_dormant(&[(q, 'Z')]);
        sync(&mut state, &mut frames);
        let flips = frames.anticommutes(&pulled.1, &pulled.2);
        match kind {
            Kind::Fixed => {
                let fixed = if state.on_positions(&pulled).negative {
                    u64::MAX
                } else {
                    0
                };
                records[c] = flips
                    .iter()
                    .zip(&mask)
                    .map(|(f, m)| (f ^ fixed) & m)
                    .collect();
            }
            Kind::Frame => {
                let d = dormant.expect("a dormant X part");
                // Collapse the shared state onto effective outcome 0; a shot
                // with effective outcome 1 carries X_d in its frame.
                state.collapse(q, false);
                sync(&mut state, &mut frames);
                let effective: Vec<u64> = mask.iter().map(|m| rng.random::<u64>() & m).collect();
                let column = frames.column(d);
                for (x, e) in frames.x[column].iter_mut().zip(&effective) {
                    *x ^= e;
                }
                records[c] = effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
            }
            Kind::Dense => {
                let one = state.probability_one(q);
                let effective: Vec<u64> = (0..frames.words)
                    .map(|w| {
                        (0..64).fold(0u64, |acc, b| {
                            acc | ((rng.random::<f64>() < one) as u64) << b
                        }) & mask[w]
                    })
                    .collect();
                records[c] = effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
                let ones: usize = effective.iter().map(|w| w.count_ones() as usize).sum();
                if ones == 0 || ones == shots {
                    state.collapse(q, ones == shots);
                    sync(&mut state, &mut frames);
                    continue;
                }
                // Branch on the shared outcome.
                let zeros_mask: Vec<u64> =
                    effective.iter().zip(&mask).map(|(e, m)| !e & m).collect();
                let split = |records: &[Vec<u64>], which: &[u64]| -> Vec<Vec<u64>> {
                    let chosen: Vec<usize> = (0..shots)
                        .filter(|&s| which[s / 64] >> (s % 64) & 1 == 1)
                        .collect();
                    records
                        .iter()
                        .map(|column| {
                            let mut out = vec![0u64; chosen.len().div_ceil(64)];
                            for (i, &s) in chosen.iter().enumerate() {
                                out[i / 64] |= (column[s / 64] >> (s % 64) & 1) << (i % 64);
                            }
                            out
                        })
                        .collect()
                };
                let (mut frames0, mut frames1) =
                    (frames.select(&zeros_mask, n), frames.select(&effective, n));
                let (records0, records1) =
                    (split(&records, &zeros_mask), split(&records, &effective));
                let mut other = state.clone();
                state.collapse(q, false);
                sync(&mut state, &mut frames0);
                other.collapse(q, true);
                sync(&mut other, &mut frames1);
                let (seed0, seed1) = (rng.random::<u64>(), rng.random::<u64>());
                let other_pending = pending.clone();
                let (mut leaves, more) = rayon::join(
                    || sample_terminal(state, pending, frames0, records0, seed0),
                    || sample_terminal(other, other_pending, frames1, records1, seed1),
                );
                leaves.extend(more);
                return leaves;
            }
        }
    }
    let peak = state.peak_active_qubits();
    (0..shots)
        .map(|s| {
            let clbits = records
                .iter()
                .map(|column| column[s / 64] >> (s % 64) & 1 == 1)
                .collect();
            (clbits, 1, peak)
        })
        .collect()
}

/// Run `circuit` for `shots` shots and count the classical outcomes. The
/// gates before the first measurement or reset run once; the rest runs as a
/// measurement tree, so each measurement is done once per distinct history
/// rather than once per shot.
pub fn run(circuit: &Circuit, shots: usize, seed: Option<u64>) -> Result<RunResult, String> {
    let mut rng = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_rng(&mut rand::rng()),
    };
    if !is_supported(circuit) {
        return Err("the circuit has gates the near-Clifford backend does not support".into());
    }
    let start = Instant::now();
    let instructions = &circuit.instructions;
    let first = instructions
        .iter()
        .position(|i| matches!(i, Instruction::Measure { .. } | Instruction::Reset { .. }))
        .unwrap_or(instructions.len());
    let mut base = NearClifford::new(circuit.num_qubits);
    for instruction in &instructions[..first] {
        if let Instruction::Gate { gate, qubits } = instruction {
            base.apply(gate, qubits)?;
        }
    }
    let mut result = RunResult::default();
    result.stats.gates = circuit.gate_count();
    result.stats.gate_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let leaves = if shots == 0 {
        Vec::new()
    } else {
        branch(
            base,
            Vec::new(),
            &instructions[first..],
            shots,
            rng.random(),
            vec![false; circuit.num_clbits],
        )
    };
    let mut counts = std::collections::BTreeMap::new();
    let mut peak = 0;
    for (clbits, count, leaf_peak) in leaves {
        *counts.entry(bitstring(&clbits)).or_default() += count;
        peak = peak.max(leaf_peak);
    }
    result.counts = counts;
    result.stats.measure_seconds = start.elapsed().as_secs_f64();
    result.stats.peak_active_qubits = Some(peak);
    Ok(result)
}
