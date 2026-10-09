//! The dense state of the active virtual qubits.
//!
//! Bit `p` of an amplitude's index is the value of the qubit at position
//! `p`. Paulis on positions are `(x, z, negative)` bit masks meaning
//! `(-1)^negative i^{|x & z|} X^x Z^z` (so `(1, 1)` on a qubit is Y).

use rayon::prelude::*;

use crate::matrix::C64;

/// Indices at or above which loops run in parallel.
const PARALLEL: usize = 1 << 14;

/// A Pauli on positions; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pauli {
    pub x: u64,
    pub z: u64,
    pub negative: bool,
}

impl Pauli {
    /// `P|i> = c |i ^ x>`: the coefficient `c`.
    fn coefficient(&self, i: usize) -> C64 {
        let mut c = match (self.x & self.z).count_ones() % 4 {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            _ => C64::new(0.0, -1.0),
        };
        if self.negative ^ ((self.z & i as u64).count_ones() % 2 == 1) {
            c = -c;
        }
        c
    }

    pub fn is_identity(&self) -> bool {
        self.x == 0 && self.z == 0
    }
}

/// A raw pointer that disjoint index pairs may write through in parallel.
#[derive(Clone, Copy)]
struct Shared(*mut C64);
// SAFETY: only used for writes to disjoint indices.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    fn get(self) -> *mut C64 {
        self.0
    }
}

/// The amplitudes of the active qubits.
#[derive(Clone, Debug)]
pub(super) struct Dense {
    pub amps: Vec<C64>,
}

impl Dense {
    /// The one-amplitude state of no qubits.
    pub fn new() -> Dense {
        Dense {
            amps: vec![C64::new(1.0, 0.0)],
        }
    }

    pub fn qubits(&self) -> usize {
        self.amps.len().trailing_zeros() as usize
    }

    /// Add a qubit in |0> at the next position.
    pub fn grow(&mut self) {
        let len = self.amps.len();
        self.amps.resize(2 * len, C64::new(0.0, 0.0));
    }

    /// Remove the qubit at `position`, which must be |0>.
    pub fn shrink(&mut self, position: usize) {
        let half = self.amps.len() / 2;
        let low = (1usize << position) - 1;
        let old = std::mem::take(&mut self.amps);
        self.amps = (0..half)
            .into_par_iter()
            .with_min_len(PARALLEL)
            .map(|i| old[(i & low) | ((i & !low) << 1)])
            .collect();
    }

    /// Run `f(i, j)` for every pair `i < j = i ^ x` (every index once when
    /// `x == 0`, with `j == i`), in parallel for large states.
    fn for_pairs(&mut self, x: u64, f: impl Fn(usize, usize, *mut C64) + Sync) {
        let len = self.amps.len();
        let ptr = Shared(self.amps.as_mut_ptr());
        let low = if x == 0 {
            0
        } else {
            1usize << x.trailing_zeros()
        };
        let run = |i: usize| {
            if i & low == 0 {
                f(i, i ^ x as usize, ptr.get());
            }
        };
        if len >= PARALLEL {
            (0..len)
                .into_par_iter()
                .with_min_len(PARALLEL / 4)
                .for_each(run);
        } else {
            (0..len).for_each(run);
        }
    }

    /// `exp(-i θ/2 P)`.
    pub fn rotate(&mut self, p: Pauli, theta: f64) {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let minus_i_sin = C64::new(0.0, -s);
        if p.x == 0 {
            self.for_pairs(0, |i, _, amps| {
                // SAFETY: each index is visited once.
                unsafe { *amps.add(i) *= C64::new(c, 0.0) + minus_i_sin * p.coefficient(i) };
            });
            return;
        }
        self.for_pairs(p.x, |i, j, amps| {
            // SAFETY: the pairs (i, j) are disjoint.
            unsafe {
                let (a, b) = (*amps.add(i), *amps.add(j));
                *amps.add(i) = a * c + minus_i_sin * p.coefficient(j) * b;
                *amps.add(j) = b * c + minus_i_sin * p.coefficient(i) * a;
            }
        });
    }

    /// `<φ|P|φ>` (real for a Hermitian `P`).
    pub fn expectation(&self, p: Pauli) -> f64 {
        let term =
            |i: usize| (self.amps[i ^ p.x as usize].conj() * p.coefficient(i) * self.amps[i]).re;
        if self.amps.len() >= PARALLEL {
            (0..self.amps.len())
                .into_par_iter()
                .with_min_len(PARALLEL / 4)
                .map(term)
                .sum()
        } else {
            (0..self.amps.len()).map(term).sum()
        }
    }

    /// Project onto the `eigenvalue` (`±1`) eigenspace of `P` and
    /// renormalise, given that eigenspace's probability.
    pub fn project(&mut self, p: Pauli, eigenvalue: f64, probability: f64) {
        let scale = 0.5 / probability.sqrt();
        self.for_pairs(p.x, |i, j, amps| {
            // SAFETY: the pairs (i, j) are disjoint.
            unsafe {
                if i == j {
                    let a = *amps.add(i);
                    *amps.add(i) = (a + eigenvalue * p.coefficient(i) * a) * scale;
                } else {
                    let (a, b) = (*amps.add(i), *amps.add(j));
                    *amps.add(i) = (a + eigenvalue * p.coefficient(j) * b) * scale;
                    *amps.add(j) = (b + eigenvalue * p.coefficient(i) * a) * scale;
                }
            }
        });
    }

    /// Apply a one- or two-position Clifford.
    pub fn apply(&mut self, gate: Gate) {
        let bit = |p: usize| 1usize << p;
        match gate {
            Gate::H(p) => {
                let r = std::f64::consts::FRAC_1_SQRT_2;
                self.for_pairs(bit(p) as u64, |i, j, amps| unsafe {
                    // SAFETY: the pairs (i, j) are disjoint.
                    let (a, b) = (*amps.add(i), *amps.add(j));
                    *amps.add(i) = (a + b) * r;
                    *amps.add(j) = (a - b) * r;
                });
            }
            Gate::Phase(p, factor) => self.for_pairs(0, |i, _, amps| {
                if i & bit(p) != 0 {
                    // SAFETY: each index is visited once.
                    unsafe { *amps.add(i) *= factor };
                }
            }),
            Gate::X(p) => self.for_pairs(bit(p) as u64, |i, j, amps| unsafe {
                // SAFETY: the pairs (i, j) are disjoint.
                std::ptr::swap(amps.add(i), amps.add(j));
            }),
            Gate::CX(c, t) => self.for_pairs(bit(t) as u64, |i, j, amps| {
                if i & bit(c) != 0 {
                    // SAFETY: the pairs (i, j) are disjoint.
                    unsafe { std::ptr::swap(amps.add(i), amps.add(j)) };
                }
            }),
            Gate::CZ(a, b) => self.for_pairs(0, |i, _, amps| {
                if i & bit(a) != 0 && i & bit(b) != 0 {
                    // SAFETY: each index is visited once.
                    unsafe { *amps.add(i) = -*amps.add(i) };
                }
            }),
        }
    }
}

/// Cliffords applied to the dense state, on positions.
#[derive(Clone, Copy, Debug)]
pub(super) enum Gate {
    H(usize),
    /// Multiply the amplitudes where the position is 1.
    Phase(usize, C64),
    X(usize),
    CX(usize, usize),
    CZ(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_state(qubits: usize, seed: u64) -> Dense {
        let mut x = seed;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let amps: Vec<C64> = (0..1 << qubits).map(|_| C64::new(next(), next())).collect();
        let norm = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        Dense {
            amps: amps.into_iter().map(|a| a / norm).collect(),
        }
    }

    #[test]
    fn rotation_by_pi_is_minus_i_times_the_pauli() {
        let p = Pauli {
            x: 0b101,
            z: 0b110,
            negative: true,
        };
        let state = random_state(3, 1);
        let mut rotated = state.clone();
        rotated.rotate(p, std::f64::consts::PI);
        for i in 0..8 {
            // exp(-i π/2 P) = -i P, and P|i> = c(i) |i ^ x>.
            let expected = C64::new(0.0, -1.0) * p.coefficient(i) * state.amps[i];
            assert!((rotated.amps[i ^ p.x as usize] - expected).norm() < 1e-12);
        }
    }

    #[test]
    fn projection_makes_the_expectation_the_eigenvalue() {
        let p = Pauli {
            x: 0b011,
            z: 0b001,
            negative: false,
        };
        let mut state = random_state(3, 2);
        let e = state.expectation(p);
        assert!(e.abs() < 1.0);
        state.project(p, -1.0, (1.0 - e) / 2.0);
        assert!((state.expectation(p) + 1.0).abs() < 1e-12);
        let norm: f64 = state.amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn shrink_removes_a_zero_qubit() {
        let mut state = Dense {
            amps: (0..8)
                .map(|i| C64::new(if i & 2 == 0 { i as f64 } else { 0.0 }, 0.0))
                .collect(),
        };
        state.shrink(1);
        assert_eq!(state.amps, [0.0, 1.0, 4.0, 5.0].map(|r| C64::new(r, 0.0)));
    }
}
