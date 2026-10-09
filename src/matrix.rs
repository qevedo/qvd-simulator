//! Small dense complex matrices for gates.
//!
//! Index convention (the same as Qiskit): for a gate acting on qubits
//! `[q0, q1, ..., q(k-1)]`, bit `j` of a row or column index is the value of
//! qubit `qj`.

use std::fmt;

use num_complex::Complex64;

pub type C64 = Complex64;

/// A square `2^k x 2^k` complex matrix, stored row-major.
#[derive(Clone, PartialEq)]
pub struct Matrix {
    dim: usize,
    data: Vec<C64>,
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix {}x{} [", self.dim, self.dim)?;
        for row in 0..self.dim {
            let cells: Vec<String> = (0..self.dim)
                .map(|col| format!("{:.4}", self[(row, col)]))
                .collect();
            writeln!(f, "  {}", cells.join(", "))?;
        }
        write!(f, "]")
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = C64;
    fn index(&self, (row, col): (usize, usize)) -> &C64 {
        &self.data[row * self.dim + col]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut C64 {
        &mut self.data[row * self.dim + col]
    }
}

impl Matrix {
    /// A matrix from row-major entries; `entries.len()` must be a power of 4.
    pub fn new(entries: Vec<C64>) -> Matrix {
        let dim = (entries.len() as f64).sqrt() as usize;
        assert!(
            dim * dim == entries.len() && dim.is_power_of_two(),
            "not a 2^k x 2^k matrix"
        );
        Matrix { dim, data: entries }
    }

    /// Build from real/imaginary pairs, row-major.
    pub fn from_pairs(entries: &[(f64, f64)]) -> Matrix {
        Matrix::new(entries.iter().map(|&(re, im)| C64::new(re, im)).collect())
    }

    pub fn identity(qubits: usize) -> Matrix {
        let dim = 1 << qubits;
        let mut m = Matrix {
            dim,
            data: vec![C64::new(0.0, 0.0); dim * dim],
        };
        for i in 0..dim {
            m[(i, i)] = C64::new(1.0, 0.0);
        }
        m
    }

    pub fn diagonal(entries: &[C64]) -> Matrix {
        let dim = entries.len();
        let mut m = Matrix {
            dim,
            data: vec![C64::new(0.0, 0.0); dim * dim],
        };
        for (i, &d) in entries.iter().enumerate() {
            m[(i, i)] = d;
        }
        m
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of qubits the matrix acts on.
    pub fn qubits(&self) -> usize {
        self.dim.trailing_zeros() as usize
    }

    pub fn entries(&self) -> &[C64] {
        &self.data
    }

    pub fn is_diagonal(&self) -> bool {
        (0..self.dim).all(|r| (0..self.dim).all(|c| r == c || self[(r, c)].norm_sqr() == 0.0))
    }

    pub fn diagonal_entries(&self) -> Vec<C64> {
        (0..self.dim).map(|i| self[(i, i)]).collect()
    }

    /// `self * other`.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.dim, other.dim);
        let n = self.dim;
        let mut out = vec![C64::new(0.0, 0.0); n * n];
        for r in 0..n {
            for k in 0..n {
                let a = self.data[r * n + k];
                if a.norm_sqr() == 0.0 {
                    continue;
                }
                for c in 0..n {
                    out[r * n + c] += a * other.data[k * n + c];
                }
            }
        }
        Matrix { dim: n, data: out }
    }

    pub fn adjoint(&self) -> Matrix {
        let n = self.dim;
        let mut out = vec![C64::new(0.0, 0.0); n * n];
        for r in 0..n {
            for c in 0..n {
                out[c * n + r] = self.data[r * n + c].conj();
            }
        }
        Matrix { dim: n, data: out }
    }

    /// Embed a gate on `gate_qubits` into the larger qubit list `qubits`
    /// (which must contain every gate qubit), acting as identity on the rest.
    pub fn embed(&self, gate_qubits: &[usize], qubits: &[usize]) -> Matrix {
        assert_eq!(
            1 << gate_qubits.len(),
            self.dim,
            "gate matrix does not match its qubits"
        );
        let position: Vec<usize> = gate_qubits
            .iter()
            .map(|q| {
                qubits
                    .iter()
                    .position(|x| x == q)
                    .expect("qubit missing from embedding")
            })
            .collect();
        let gate_mask: usize = position.iter().map(|p| 1 << p).sum();
        let dim = 1 << qubits.len();
        let mut out = Matrix {
            dim,
            data: vec![C64::new(0.0, 0.0); dim * dim],
        };
        let local = |index: usize| -> usize {
            position
                .iter()
                .enumerate()
                .map(|(j, &p)| ((index >> p) & 1) << j)
                .sum()
        };
        for r in 0..dim {
            for c in 0..dim {
                if r & !gate_mask == c & !gate_mask {
                    out[(r, c)] = self[(local(r), local(c))];
                }
            }
        }
        out
    }

    /// The matrix of this gate with extra control qubits prepended: the
    /// result acts on `[controls..., targets...]` and applies `self` when
    /// every control qubit is in `control_state` (bit `i` for control `i`).
    pub fn controlled(&self, controls: usize, control_state: usize) -> Matrix {
        let k = self.qubits();
        let dim = 1 << (controls + k);
        let mut out = Matrix::identity(controls + k);
        let mask = (1 << controls) - 1;
        for r in 0..dim {
            for c in 0..dim {
                if r & mask == control_state && c & mask == control_state {
                    out[(r, c)] = self[(r >> controls, c >> controls)];
                } else if r & mask == control_state || c & mask == control_state {
                    out[(r, c)] = C64::new(0.0, 0.0);
                }
            }
        }
        out
    }

    /// Largest absolute entry-wise difference.
    pub fn distance(&self, other: &Matrix) -> f64 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_matches_kronecker_order() {
        // X on qubit 1 of [0, 1]: index bit 1 flips.
        let x = Matrix::from_pairs(&[(0., 0.), (1., 0.), (1., 0.), (0., 0.)]);
        let big = x.embed(&[1], &[0, 1]);
        for r in 0..4 {
            for c in 0..4 {
                let expected = if r == c ^ 2 { 1.0 } else { 0.0 };
                assert_eq!(big[(r, c)].re, expected);
            }
        }
    }

    #[test]
    fn controlled_x_is_cnot() {
        let x = Matrix::from_pairs(&[(0., 0.), (1., 0.), (1., 0.), (0., 0.)]);
        let cx = x.controlled(1, 1);
        // Qiskit's CX on [control, target]: |01> <-> |11> (control is bit 0).
        assert_eq!(cx[(1, 3)].re, 1.0);
        assert_eq!(cx[(3, 1)].re, 1.0);
        assert_eq!(cx[(0, 0)].re, 1.0);
        assert_eq!(cx[(2, 2)].re, 1.0);
        assert_eq!(cx[(1, 1)].re, 0.0);
    }
}
