//! The matrix product state and its operations.

use faer::{Mat, MatRef};
use rand::{Rng, RngExt};

use crate::matrix::{C64, Matrix};

/// How much an [`Mps`] may truncate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truncation {
    /// Largest bond dimension χ kept after a gate.
    pub max_bond: usize,
    /// Discard the smallest singular values while their squares add up to
    /// at most this fraction of the total.
    pub threshold: f64,
}

impl Default for Truncation {
    fn default() -> Truncation {
        Truncation {
            max_bond: 256,
            threshold: 1e-16,
        }
    }
}

impl Truncation {
    /// No truncation beyond numerical zeros: exact, but χ can grow to
    /// `2^(n/2)`.
    pub fn exact() -> Truncation {
        Truncation {
            max_bond: usize::MAX,
            threshold: 1e-16,
        }
    }
}

/// One site tensor `A[l, p, r]` (left bond, physical index, right bond),
/// stored row-major.
#[derive(Clone, Debug)]
struct Site {
    left: usize,
    right: usize,
    data: Vec<C64>,
}

impl Site {
    fn product(bit: bool) -> Site {
        let mut data = vec![C64::new(0.0, 0.0); 2];
        data[bit as usize] = C64::new(1.0, 0.0);
        Site {
            left: 1,
            right: 1,
            data,
        }
    }

    fn norm_sqr(&self) -> f64 {
        self.data.iter().map(|a| a.norm_sqr()).sum()
    }
}

/// A row-major copy of a faer matrix (or of its first `cols` columns).
fn row_major(m: MatRef<'_, C64>, cols: usize) -> Vec<C64> {
    let mut out = Vec::with_capacity(m.nrows() * cols);
    for i in 0..m.nrows() {
        for j in 0..cols {
            out.push(m[(i, j)]);
        }
    }
    out
}

/// `a (m x k) * b (k x n)`, all row-major.
fn matmul(a: &[C64], m: usize, k: usize, b: &[C64], n: usize) -> Vec<C64> {
    let c: Mat<C64> = MatRef::from_row_major_slice(a, m, k) * MatRef::from_row_major_slice(b, k, n);
    row_major(c.as_ref(), n)
}

/// A matrix product state over `n` qubits.
///
/// Site `s` holds one qubit (`qubit_at[s]`); two- and more-qubit gates first
/// move their qubits next to each other with SWAPs and leave them there
/// (lazy permutation). The state is kept in mixed canonical form around an
/// orthogonality center, so that truncating a bond discards exactly the
/// smallest Schmidt coefficients.
#[derive(Clone, Debug)]
pub struct Mps {
    n: usize,
    sites: Vec<Site>,
    qubit_at: Vec<usize>,
    site_of: Vec<usize>,
    /// Sites left of the center are left-canonical, sites right of it
    /// right-canonical.
    center: usize,
    truncation: Truncation,
    /// Product over truncations of the kept fraction of the norm.
    fidelity: f64,
    max_bond_seen: usize,
}

impl Mps {
    /// The state `|0...0>` on `n >= 1` qubits.
    pub fn new(n: usize, truncation: Truncation) -> Mps {
        assert!(n >= 1, "an MPS needs at least one qubit");
        assert!(truncation.max_bond >= 1);
        Mps {
            n,
            sites: (0..n).map(|_| Site::product(false)).collect(),
            qubit_at: (0..n).collect(),
            site_of: (0..n).collect(),
            center: 0,
            truncation,
            fidelity: 1.0,
            max_bond_seen: 1,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Bond dimensions between consecutive sites (`n - 1` of them).
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.sites[..self.n - 1].iter().map(|s| s.right).collect()
    }

    /// The largest bond dimension reached so far.
    pub fn max_bond_seen(&self) -> usize {
        self.max_bond_seen
    }

    /// An estimate of the fidelity with the exact state: the product of the
    /// weight kept at every truncation (1 when nothing was discarded).
    pub fn fidelity(&self) -> f64 {
        self.fidelity
    }

    /// Apply a `2^k x 2^k` unitary to `qubits` (bit `j` of the matrix
    /// index is `qubits[j]`, as everywhere in qvd).
    pub fn apply(&mut self, matrix: &Matrix, qubits: &[usize]) {
        assert_eq!(
            matrix.qubits(),
            qubits.len(),
            "matrix size does not match the qubits"
        );
        if qubits.len() == 1 {
            let s = self.site_of[qubits[0]];
            self.apply_one(s, matrix);
            return;
        }
        let start = self.gather(qubits);
        let k = qubits.len();
        // Gate index bits in block order: block site `s` (most significant
        // first) holds gate qubit `bit_of[s]`.
        let bit_of: Vec<usize> = (0..k)
            .map(|s| {
                let q = self.qubit_at[start + s];
                qubits.iter().position(|&g| g == q).expect("gathered qubit")
            })
            .collect();
        let to_gate =
            |b: usize| -> usize { (0..k).map(|s| ((b >> (k - 1 - s)) & 1) << bit_of[s]).sum() };
        let d = 1 << k;
        let permuted: Vec<C64> = (0..d * d)
            .map(|i| matrix[(to_gate(i / d), to_gate(i % d))])
            .collect();
        self.move_center(self.center.clamp(start, start + k - 1));
        let (left, mut theta, right) = self.contract(start, k);
        apply_physical(&mut theta, &permuted, d, left, right);
        self.split(start, k, left, theta, right, true);
    }

    /// Apply a one-qubit gate to site `s`. A unitary on the physical index
    /// keeps the site's canonical form, so the center does not move.
    fn apply_one(&mut self, s: usize, g: &Matrix) {
        let site = &mut self.sites[s];
        let (g00, g01, g10, g11) = (g[(0, 0)], g[(0, 1)], g[(1, 0)], g[(1, 1)]);
        let r = site.right;
        for l in 0..site.left {
            let (zero, one) = site.data[l * 2 * r..(l + 1) * 2 * r].split_at_mut(r);
            for (a0, a1) in zero.iter_mut().zip(one.iter_mut()) {
                let (x0, x1) = (*a0, *a1);
                *a0 = g00 * x0 + g01 * x1;
                *a1 = g10 * x0 + g11 * x1;
            }
        }
    }

    /// Move qubits next to each other with adjacent SWAPs, keeping their
    /// relative order, and return the first site of the block. The qubit at
    /// the median site stays where it is.
    fn gather(&mut self, qubits: &[usize]) -> usize {
        let mut by_site: Vec<usize> = qubits.to_vec();
        by_site.sort_by_key(|&q| self.site_of[q]);
        let k = by_site.len();
        let mid = k / 2;
        let start = self.site_of[by_site[mid]] - mid;
        for j in (0..mid).rev() {
            while self.site_of[by_site[j]] < start + j {
                let s = self.site_of[by_site[j]];
                self.swap_sites(s, true);
            }
        }
        for (j, &q) in by_site.iter().enumerate().skip(mid + 1) {
            while self.site_of[q] > start + j {
                let s = self.site_of[q] - 1;
                self.swap_sites(s, false);
            }
        }
        start
    }

    /// Swap the qubits at sites `s` and `s + 1`. `moving_right` leaves the
    /// center at `s + 1` (ready for the next swap to the right), otherwise
    /// at `s`.
    fn swap_sites(&mut self, s: usize, moving_right: bool) {
        self.move_center(self.center.clamp(s, s + 1));
        let (left, mut theta, right) = self.contract(s, 2);
        // theta[l, p1, p2, r] -> theta[l, p2, p1, r]: swap the middle two
        // of the four physical blocks.
        for l in 0..left {
            let base = l * 4 * right;
            for r in 0..right {
                theta.swap(base + right + r, base + 2 * right + r);
            }
        }
        self.split(s, 2, left, theta, right, moving_right);
        let (a, b) = (self.qubit_at[s], self.qubit_at[s + 1]);
        self.qubit_at.swap(s, s + 1);
        self.site_of[a] = s + 1;
        self.site_of[b] = s;
    }

    /// Contract sites `start..start + k` into one tensor
    /// `theta[l, p, r]` with `p` of `k` bits (first site most significant).
    fn contract(&self, start: usize, k: usize) -> (usize, Vec<C64>, usize) {
        let first = &self.sites[start];
        let left = first.left;
        let mut theta = first.data.clone();
        let mut rows = left * 2;
        let mut bond = first.right;
        for site in &self.sites[start + 1..start + k] {
            theta = matmul(&theta, rows, bond, &site.data, 2 * site.right);
            rows *= 2;
            bond = site.right;
        }
        (left, theta, bond)
    }

    /// Split `theta[l, p, r]` back into sites `start..start + k` by
    /// successive SVDs, truncating each bond. The center ends at the last
    /// site if `center_right`, otherwise at the first.
    fn split(
        &mut self,
        start: usize,
        k: usize,
        left: usize,
        mut theta: Vec<C64>,
        right: usize,
        center_right: bool,
    ) {
        if center_right {
            let mut l = left;
            for s in start..start + k - 1 {
                let rows = l * 2;
                let cols = theta.len() / rows;
                let (u, m, rest) = self.svd_split(&theta, rows, cols, true);
                self.sites[s] = Site {
                    left: l,
                    right: m,
                    data: u,
                };
                theta = rest;
                l = m;
            }
            self.sites[start + k - 1] = Site {
                left: l,
                right,
                data: theta,
            };
            self.center = start + k - 1;
        } else {
            let mut r = right;
            for s in (start + 1..start + k).rev() {
                let cols = 2 * r;
                let rows = theta.len() / cols;
                let (v, m, rest) = self.svd_split(&theta, rows, cols, false);
                self.sites[s] = Site {
                    left: m,
                    right: r,
                    data: v,
                };
                theta = rest;
                r = m;
            }
            self.sites[start] = Site {
                left,
                right: r,
                data: theta,
            };
            self.center = start;
        }
    }

    /// SVD of a `rows x cols` matrix, truncated to `m` singular values and
    /// renormalised. With `keep_left`, returns `(U, m, S V†)`: the left
    /// factor becomes a left-canonical site. Otherwise returns
    /// `(V†, m, U S)`: the right factor becomes a right-canonical site.
    fn svd_split(
        &mut self,
        theta: &[C64],
        rows: usize,
        cols: usize,
        keep_left: bool,
    ) -> (Vec<C64>, usize, Vec<C64>) {
        let (u, s, v) = svd(MatRef::from_row_major_slice(theta, rows, cols));
        let (u, v) = (u.as_ref(), v.as_ref());
        let m = self.keep(&s);
        let total: f64 = s.iter().map(|x| x * x).sum();
        let kept: f64 = s[..m].iter().map(|x| x * x).sum();
        if kept < total {
            self.fidelity *= kept / total;
        }
        let scale = if kept > 0.0 {
            (total / kept).sqrt()
        } else {
            1.0
        };
        self.max_bond_seen = self.max_bond_seen.max(m);
        if keep_left {
            let rest = (0..m)
                .flat_map(|a| (0..cols).map(move |c| (a, c)))
                .map(|(a, c)| v[(c, a)].conj() * (s[a] * scale))
                .collect();
            (row_major(u, m), m, rest)
        } else {
            let site = (0..m)
                .flat_map(|a| (0..cols).map(move |c| (a, c)))
                .map(|(a, c)| v[(c, a)].conj())
                .collect();
            let rest = (0..rows)
                .flat_map(|i| (0..m).map(move |a| (i, a)))
                .map(|(i, a)| u[(i, a)] * (s[a] * scale))
                .collect();
            (site, m, rest)
        }
    }

    /// How many singular values (sorted, descending) to keep.
    fn keep(&self, s: &[f64]) -> usize {
        let total: f64 = s.iter().map(|x| x * x).sum();
        let mut m = s.len().min(self.truncation.max_bond).max(1);
        let mut discarded: f64 = s[m..].iter().map(|x| x * x).sum();
        while m > 1 && discarded + s[m - 1] * s[m - 1] <= self.truncation.threshold * total {
            m -= 1;
            discarded += s[m] * s[m];
        }
        m
    }

    /// Move the orthogonality center to site `to` with QR decompositions.
    fn move_center(&mut self, to: usize) {
        while self.center < to {
            let c = self.center;
            let site = &self.sites[c];
            let (rows, cols) = (site.left * 2, site.right);
            let qr = MatRef::from_row_major_slice(&site.data, rows, cols).qr();
            let q = qr.compute_thin_Q();
            let r = qr.thin_R();
            let m = q.ncols();
            let left = site.left;
            let r = row_major(r, cols);
            let next = &self.sites[c + 1];
            let data = matmul(&r, m, cols, &next.data, 2 * next.right);
            let right = next.right;
            self.sites[c] = Site {
                left,
                right: m,
                data: row_major(q.as_ref(), m),
            };
            self.sites[c + 1] = Site {
                left: m,
                right,
                data,
            };
            self.center += 1;
        }
        while self.center > to {
            let c = self.center;
            let site = &self.sites[c];
            let (rows, cols) = (site.left, site.right * 2);
            // site = R† Q† from the QR of its adjoint.
            let adjoint = MatRef::from_row_major_slice(&site.data, rows, cols)
                .adjoint()
                .to_owned();
            let qr = adjoint.qr();
            let q = qr.compute_thin_Q();
            let r = qr.thin_R();
            let m = q.ncols();
            let right = site.right;
            let q_adj = row_major(q.adjoint().to_owned().as_ref(), cols);
            let r_adj = row_major(r.adjoint().to_owned().as_ref(), m);
            let prev = &self.sites[c - 1];
            let data = matmul(&prev.data, prev.left * 2, rows, &r_adj, m);
            let prev_left = prev.left;
            self.sites[c] = Site {
                left: m,
                right,
                data: q_adj,
            };
            self.sites[c - 1] = Site {
                left: prev_left,
                right: m,
                data,
            };
            self.center -= 1;
        }
    }

    /// Measure `qubit` in the Z basis and collapse the state.
    pub fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> bool {
        let s = self.site_of[qubit];
        self.move_center(s);
        let site = &mut self.sites[s];
        let r = site.right;
        let total = site.norm_sqr();
        let one: f64 = (0..site.left)
            .flat_map(|l| site.data[(l * 2 + 1) * r..(l * 2 + 2) * r].iter())
            .map(|a| a.norm_sqr())
            .sum();
        let outcome = rng.random::<f64>() * total < one;
        let kept = if outcome { one } else { total - one };
        let scale = 1.0 / kept.sqrt();
        for l in 0..site.left {
            for p in 0..2 {
                for a in &mut site.data[(l * 2 + p) * r..(l * 2 + p + 1) * r] {
                    *a = if (p == 1) == outcome {
                        *a * scale
                    } else {
                        C64::new(0.0, 0.0)
                    };
                }
            }
        }
        outcome
    }

    /// Reset `qubit` to |0>.
    pub fn reset(&mut self, qubit: usize, rng: &mut impl Rng) {
        if self.measure(qubit, rng) {
            let x = Matrix::from_pairs(&[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]);
            self.apply(&x, &[qubit]);
        }
    }

    /// Prepare for [`Mps::sample`]: every site but the first becomes
    /// right-canonical.
    pub fn prepare_sampling(&mut self) {
        self.move_center(0);
    }

    /// Draw `shots` shots of all qubits; shot `i` is
    /// `out[i * n..(i + 1) * n]`, indexed by qubit. Sites are sampled left
    /// to right: with the rest right-canonical, each shot only needs its
    /// running left vector, and the vectors of all shots form a matrix, so
    /// each site costs two matrix products. Call [`Mps::prepare_sampling`]
    /// first.
    pub fn sample(&self, shots: usize, rng: &mut impl Rng) -> Vec<bool> {
        assert_eq!(self.center, 0, "call prepare_sampling first");
        let n = self.n;
        let mut out = vec![false; shots * n];
        let mut v = Mat::<C64>::from_fn(shots, 1, |_, _| C64::new(1.0, 0.0));
        for (s, site) in self.sites.iter().enumerate() {
            let (l, r) = (site.left, site.right);
            let slice = |b: usize| {
                MatRef::from_row_major_slice_with_stride(&site.data[b * r..], l, r, 2 * r)
            };
            let w = [&v * slice(0), &v * slice(1)];
            let mut p = [vec![0.0; shots], vec![0.0; shots]];
            for (wb, pb) in w.iter().zip(&mut p) {
                for j in 0..r {
                    for (acc, a) in pb.iter_mut().zip(wb.col_as_slice(j)) {
                        *acc += a.norm_sqr();
                    }
                }
            }
            let bits: Vec<bool> = (0..shots)
                .map(|i| rng.random::<f64>() * (p[0][i] + p[1][i]) < p[1][i])
                .collect();
            let scale: Vec<f64> = (0..shots)
                .map(|i| 1.0 / p[bits[i] as usize][i].sqrt())
                .collect();
            v = Mat::from_fn(shots, r, |i, j| w[bits[i] as usize][(i, j)] * scale[i]);
            let q = self.qubit_at[s];
            for (i, &bit) in bits.iter().enumerate() {
                out[i * n + q] = bit;
            }
        }
        out
    }

    /// `<bits|psi>` for a basis state (`bits[q]` = qubit `q`).
    pub fn amplitude(&self, bits: &[bool]) -> C64 {
        assert_eq!(bits.len(), self.n);
        let mut v = vec![C64::new(1.0, 0.0)];
        for (s, site) in self.sites.iter().enumerate() {
            let b = bits[self.qubit_at[s]] as usize;
            let r = site.right;
            let mut next = vec![C64::new(0.0, 0.0); r];
            for (l, &vl) in v.iter().enumerate() {
                for (acc, &a) in next
                    .iter_mut()
                    .zip(&site.data[(l * 2 + b) * r..(l * 2 + b + 1) * r])
                {
                    *acc += vl * a;
                }
            }
            v = next;
        }
        v[0]
    }

    /// The full state vector (index bit `q` = qubit `q`); for small `n`.
    pub fn to_statevector(&self) -> Vec<C64> {
        assert!(self.n <= 26, "to_statevector is for small states");
        (0..1usize << self.n)
            .map(|index| {
                let bits: Vec<bool> = (0..self.n).map(|q| (index >> q) & 1 == 1).collect();
                self.amplitude(&bits)
            })
            .collect()
    }
}

/// The thin SVD `a = U diag(s) V†`, singular values descending.
///
/// faer 0.24's SVD can return NaN factors for finite, well-scaled input
/// (seen on a 128x128 matrix whose leading singular values were nearly
/// degenerate), so the result is checked; the SVD of the adjoint, then a
/// Hermitian eigendecomposition of the Gram matrix, are the fallbacks.
fn svd(a: MatRef<'_, C64>) -> (Mat<C64>, Vec<f64>, Mat<C64>) {
    svd_with(a, Method::Direct)
        .or_else(|| svd_with(a, Method::Adjoint))
        .or_else(|| svd_with(a, Method::Gram))
        .expect("SVD failed on every method")
}

#[derive(Clone, Copy, Debug)]
enum Method {
    Direct,
    Adjoint,
    Gram,
}

fn finite(m: MatRef<'_, C64>) -> bool {
    (0..m.ncols())
        .all(|j| (0..m.nrows()).all(|i| m[(i, j)].re.is_finite() && m[(i, j)].im.is_finite()))
}

/// The SVD by one method, or `None` if it failed or is not finite.
fn svd_with(a: MatRef<'_, C64>, method: Method) -> Option<(Mat<C64>, Vec<f64>, Mat<C64>)> {
    let (u, s, v) = match method {
        Method::Direct | Method::Adjoint => {
            let adjoint;
            let input = if matches!(method, Method::Adjoint) {
                adjoint = a.adjoint().to_owned();
                adjoint.as_ref()
            } else {
                a
            };
            let svd = input.thin_svd().ok()?;
            let diag = svd.S().column_vector();
            let s: Vec<f64> = (0..diag.nrows()).map(|i| diag[i].re).collect();
            let (u, v) = (svd.U().to_owned(), svd.V().to_owned());
            // a† = V S U†.
            if matches!(method, Method::Adjoint) {
                (v, s, u)
            } else {
                (u, s, v)
            }
        }
        Method::Gram => {
            // Eigenvectors of the smaller Gram matrix give one factor; the
            // other is a times it, divided by the singular values. Squaring
            // loses precision below about 1e-8 of the largest singular
            // value, and directions with zero singular value are dropped.
            let wide = a.nrows() <= a.ncols();
            let gram: Mat<C64> = if wide {
                a * a.adjoint()
            } else {
                a.adjoint() * a
            };
            let eigen = gram.self_adjoint_eigen(faer::Side::Lower).ok()?;
            let values = eigen.S().column_vector();
            let mut order: Vec<usize> = (0..values.nrows()).collect();
            order.sort_by(|&i, &j| values[j].re.total_cmp(&values[i].re));
            let largest = values[order[0]].re.max(0.0);
            order.retain(|&i| values[i].re > largest * 1e-30 && values[i].re > 0.0);
            let s: Vec<f64> = order.iter().map(|&i| values[i].re.sqrt()).collect();
            let vectors = eigen.U();
            let x =
                Mat::<C64>::from_fn(vectors.nrows(), order.len(), |r, c| vectors[(r, order[c])]);
            let y: Mat<C64> = if wide { a.adjoint() * &x } else { a * &x };
            let y = Mat::<C64>::from_fn(y.nrows(), y.ncols(), |r, c| y[(r, c)] / s[c]);
            if wide { (x, s, y) } else { (y, s, x) }
        }
    };
    (s.iter().all(|x| x.is_finite()) && finite(u.as_ref()) && finite(v.as_ref()))
        .then_some((u, s, v))
}

/// `theta[l, p', r] = sum_p g[p', p] theta[l, p, r]` for a `d x d` matrix
/// `g` (row-major).
fn apply_physical(theta: &mut [C64], g: &[C64], d: usize, left: usize, right: usize) {
    let mut block = vec![C64::new(0.0, 0.0); d * right];
    for l in 0..left {
        let slab = &mut theta[l * d * right..(l + 1) * d * right];
        block.fill(C64::new(0.0, 0.0));
        for po in 0..d {
            let out = &mut block[po * right..(po + 1) * right];
            for pi in 0..d {
                let coeff = g[po * d + pi];
                if coeff == C64::new(0.0, 0.0) {
                    continue;
                }
                for (o, &x) in out.iter_mut().zip(&slab[pi * right..(pi + 1) * right]) {
                    *o += coeff * x;
                }
            }
        }
        slab.copy_from_slice(&block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn every_svd_method_decomposes() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(3);
        for (rows, cols) in [(1, 1), (4, 9), (9, 4), (32, 32), (64, 20)] {
            let data: Vec<C64> = (0..rows * cols)
                .map(|_| C64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
                .collect();
            let a = MatRef::from_row_major_slice(&data, rows, cols);
            for method in [Method::Direct, Method::Adjoint, Method::Gram] {
                let (u, s, v) = svd_with(a, method).expect("finite SVD");
                assert!(
                    s.windows(2).all(|w| w[0] >= w[1]),
                    "{method:?}: not descending"
                );
                let mut error = 0.0f64;
                for i in 0..rows {
                    for j in 0..cols {
                        let x: C64 = (0..s.len())
                            .map(|k| u[(i, k)] * s[k] * v[(j, k)].conj())
                            .sum();
                        error = error.max((x - a[(i, j)]).norm());
                    }
                }
                assert!(error < 1e-10, "{method:?} {rows}x{cols}: error {error}");
                let gram = u.adjoint() * &u;
                for i in 0..s.len() {
                    for j in 0..s.len() {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (gram[(i, j)] - C64::new(expected, 0.0)).norm() < 1e-8,
                            "{method:?}: U not orthonormal"
                        );
                    }
                }
            }
        }
    }
}
