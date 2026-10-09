//! The matrix product state and its operations.

use std::sync::{Arc, LazyLock};

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

/// `a * b`, sequential for small products: faer otherwise dispatches every
/// product to the thread pool, which costs far more than a product of
/// χ = 1 or 2 tensors (a 100-qubit QFT spent most of its time there).
fn product(a: MatRef<'_, C64>, b: MatRef<'_, C64>) -> Mat<C64> {
    let work = a.nrows() * a.ncols() * b.ncols();
    let par = if work < 1 << 18 {
        faer::Par::Seq
    } else {
        faer::get_global_parallelism()
    };
    let mut c = Mat::<C64>::zeros(a.nrows(), b.ncols());
    faer::linalg::matmul::matmul(
        c.as_mut(),
        faer::Accum::Replace,
        a,
        b,
        C64::new(1.0, 0.0),
        par,
    );
    c
}

/// `a (m x k) * b (k x n)`, all row-major.
fn matmul(a: &[C64], m: usize, k: usize, b: &[C64], n: usize) -> Vec<C64> {
    if m * k * n <= 512 {
        let mut out = vec![C64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for p in 0..k {
                let x = a[i * k + p];
                for (o, &y) in out[i * n..(i + 1) * n]
                    .iter_mut()
                    .zip(&b[p * n..(p + 1) * n])
                {
                    *o += x * y;
                }
            }
        }
        return out;
    }
    let c = product(
        MatRef::from_row_major_slice(a, m, k),
        MatRef::from_row_major_slice(b, k, n),
    );
    row_major(c.as_ref(), n)
}

/// A matrix product state over `n` qubits.
///
/// Site `s` holds one qubit (`qubit_at[s]`); gates on several qubits first
/// move them next to each other with SWAPs and leave them there (lazy
/// permutation). Gates that may truncate run one at a time in mixed
/// canonical form around an orthogonality center, so each truncation keeps
/// exactly the largest Schmidt coefficients of its bond. Gates that cannot
/// truncate run in parallel layers, in the form where every site is
/// right-canonical and every bond's Schmidt coefficients are stored.
#[derive(Clone, Debug)]
pub struct Mps {
    n: usize,
    sites: Vec<Site>,
    qubit_at: Vec<usize>,
    site_of: Vec<usize>,
    /// Sites left of the center are left-canonical, sites right of it
    /// right-canonical.
    center: usize,
    /// `spectra[b]`: the Schmidt coefficients across the bond between
    /// sites `b` and `b + 1`, with unit sum of squares.
    spectra: Vec<Vec<f64>>,
    /// Whether `spectra` is up to date (measurements invalidate it).
    spectra_valid: bool,
    /// Per bond: splits left before trying the Gram SVD again after it was
    /// rejected there (see [`truncating_svd`]).
    gram_skip: Vec<u8>,
    truncation: Truncation,
    /// Product over truncations of the kept fraction of the norm.
    fidelity: f64,
    max_bond_seen: usize,
}

/// One step of a gate layer, on sites the planner has already arranged.
#[derive(Clone, Debug)]
enum SiteOp {
    /// A one-qubit gate `[g00, g01, g10, g11]`.
    One { site: usize, matrix: [C64; 4] },
    /// A `2^k x 2^k` matrix (row-major) on sites `start..start + k`,
    /// physical index with the first site most significant. `growth` bounds
    /// the factor by which it can multiply a bond's dimension (its operator
    /// Schmidt rank for two sites).
    Block {
        start: usize,
        k: usize,
        matrix: Arc<[C64]>,
        growth: usize,
    },
}

impl SiteOp {
    fn sites(&self) -> std::ops::Range<usize> {
        match self {
            SiteOp::One { site, .. } => *site..*site + 1,
            SiteOp::Block { start, k, .. } => *start..*start + *k,
        }
    }
}

/// The SWAP gate in block order, shared by every routing step.
static SWAP: LazyLock<Arc<[C64]>> = LazyLock::new(|| {
    let mut m = vec![C64::new(0.0, 0.0); 16];
    for (row, col) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
        m[row * 4 + col] = C64::new(1.0, 0.0);
    }
    m.into()
});

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
            spectra: vec![vec![1.0]; n - 1],
            spectra_valid: true,
            gram_skip: vec![0; n - 1],
            truncation,
            fidelity: 1.0,
            max_bond_seen: 1,
        }
    }

    /// The state `|0...0>` with qubit `order[s]` on site `s` (a
    /// permutation of `0..n`). Placing interacting qubits close together
    /// saves SWAPs and keeps bonds small; see [`super::qubit_order`].
    pub fn with_order(order: &[usize], truncation: Truncation) -> Mps {
        let mut mps = Mps::new(order.len(), truncation);
        let mut seen = vec![false; order.len()];
        for (s, &q) in order.iter().enumerate() {
            assert!(
                q < order.len() && !seen[q],
                "order must be a permutation of the qubits"
            );
            seen[q] = true;
            mps.qubit_at[s] = q;
            mps.site_of[q] = s;
        }
        mps
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
        self.apply_gates(&[(matrix, qubits)]);
    }

    /// Apply gates in order. Their SWAP routing is planned first. Runs of
    /// steps that cannot truncate are scheduled in layers on disjoint sites
    /// and run in parallel; steps that may truncate run one at a time, in
    /// order, so truncating the bonds of a layer simultaneously (measured up
    /// to 3000x lower fidelity than one after another) never happens.
    pub fn apply_gates(&mut self, gates: &[(&Matrix, &[usize])]) {
        let mut ops = Vec::new();
        for &(matrix, qubits) in gates {
            self.plan(matrix, qubits, &mut ops);
        }
        // Steps that cannot truncate commute with each other whenever their
        // sites are disjoint; steps that may truncate keep the circuit's
        // order, so the result is the same as applying the gates one by one.
        let classes = self.classify(&ops);
        let mut i = 0;
        while i < ops.len() {
            if !classes[i].0 {
                self.run_step(&ops[i], ops.get(i + 1));
                i += 1;
                continue;
            }
            let end = (i..ops.len()).find(|&j| !classes[j].0).unwrap_or(ops.len());
            let work: f64 = classes[i..end].iter().map(|c| c.1).sum();
            self.run_exact(&ops[i..end], work);
            i = end;
        }
    }

    /// Run steps that cannot truncate, scheduled in layers of steps on
    /// disjoint sites (each as early as its sites allow), in parallel.
    /// Parallel layers need every site right-canonical and current bond
    /// spectra; restoring them after truncations costs an SVD sweep, so
    /// short runs then go step by step instead.
    fn run_exact(&mut self, ops: &[SiteOp], work: f64) {
        let blocks = ops
            .iter()
            .filter(|op| matches!(op, SiteOp::Block { .. }))
            .count();
        // Tiny steps (blocks of a few hundred numbers) are cheaper one by
        // one than dispatched in parallel layers.
        if blocks < 2 || work < 1e4 * blocks as f64 || (!self.spectra_valid && blocks < self.n) {
            for (i, op) in ops.iter().enumerate() {
                self.run_step(op, ops.get(i + 1));
            }
            return;
        }
        let mut ready = vec![0usize; self.n];
        let mut layers: Vec<Vec<SiteOp>> = Vec::new();
        for op in ops {
            let sites = op.sites();
            let layer = sites.clone().map(|s| ready[s]).max().unwrap_or(0);
            for s in sites {
                ready[s] = layer + 1;
            }
            if layers.len() <= layer {
                layers.resize_with(layer + 1, Vec::new);
            }
            layers[layer].push(op.clone());
        }
        self.canonicalize();
        for mut layer in layers {
            layer.sort_by_key(|op| op.sites().start);
            self.run_layer(&layer);
        }
    }

    /// For each step, whether it certainly cannot truncate (its new bonds,
    /// bounded by the dimensions around the block, which earlier steps may
    /// grow, all fit within χ_max), and a bound on its work: the size of its
    /// block to the power 1.5, as for an SVD.
    fn classify(&self, ops: &[SiteOp]) -> Vec<(bool, f64)> {
        let mut bonds: Vec<usize> = self.bond_dimensions();
        let max = self.truncation.max_bond;
        ops.iter()
            .map(|op| match *op {
                SiteOp::One { .. } => (true, 0.0),
                SiteOp::Block {
                    start, k, growth, ..
                } => {
                    let left = if start == 0 { 1 } else { bonds[start - 1] };
                    let right = if start + k == self.n {
                        1
                    } else {
                        bonds[start + k - 1]
                    };
                    let mut exact = true;
                    for j in 1..k {
                        // A gate multiplies a bond's Schmidt rank by at most
                        // its own operator Schmidt rank across that cut.
                        // A gate multiplies a bond's Schmidt rank by at most
                        // its own operator Schmidt rank across that cut.
                        let growth = growth.min(1 << (2 * j.min(k - j)));
                        let bound = left
                            .saturating_mul(1 << j)
                            .min(right.saturating_mul(1 << (k - j)))
                            .min(bonds[start + j - 1].saturating_mul(growth));
                        exact &= bound <= max;
                        bonds[start + j - 1] = bound.min(max);
                    }
                    let size = (left * right) as f64 * (1u64 << k) as f64;
                    (exact, size.powf(1.5))
                }
            })
            .collect()
    }

    /// Run one step on its own: move the orthogonality center into the
    /// block, so each SVD sees the exact Schmidt coefficients of its bond
    /// and every truncation is optimal.
    fn run_step(&mut self, op: &SiteOp, next: Option<&SiteOp>) {
        match op {
            SiteOp::One { site, matrix } => apply_one(&mut self.sites[*site], matrix),
            SiteOp::Block {
                start, k, matrix, ..
            } => {
                let (start, k) = (*start, *k);
                self.move_center(self.center.clamp(start, start + k - 1));
                let (left, mut theta, right) = contract(&self.sites[start..start + k]);
                if Arc::ptr_eq(matrix, &SWAP) {
                    // theta[l, p1, p2, r] -> theta[l, p2, p1, r]: swap the
                    // middle two of the four physical blocks.
                    for l in 0..left {
                        let base = l * 4 * right;
                        let (head, tail) =
                            theta[base + right..base + 3 * right].split_at_mut(right);
                        head.swap_with_slice(tail);
                    }
                } else {
                    apply_physical(&mut theta, matrix, 1 << k, left, right);
                }
                // Leave the center where the next step will need it.
                let center_right = next.is_none_or(|n| n.sites().start + 1 >= start + k);
                self.split(start, k, left, theta, right, center_right);
            }
        }
    }

    /// Split `theta[l, p, r]` back into sites `start..start + k` by SVDs,
    /// truncating each bond and recording its spectrum. With
    /// `center_right` the SVDs go from the left and the center ends at the
    /// last site; otherwise from the right, ending at the first site (so a
    /// chain of steps moving left needs no QR in between).
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
                let (u, values, vh) = self.cut(s, &theta, rows, cols);
                let m = values.len();
                self.sites[s] = Site {
                    left: l,
                    right: m,
                    data: u,
                };
                theta = vh;
                for (row, &x) in theta.chunks_mut(cols).zip(&values) {
                    row.iter_mut().for_each(|a| *a *= x);
                }
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
                let (u, values, vh) = self.cut(s - 1, &theta, rows, cols);
                let m = values.len();
                self.sites[s] = Site {
                    left: m,
                    right: r,
                    data: vh,
                };
                theta = u;
                for row in theta.chunks_mut(m) {
                    row.iter_mut().zip(&values).for_each(|(a, &x)| *a *= x);
                }
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

    /// The truncated SVD across bond `bond` of a `rows x cols` matrix:
    /// `U` (`rows x m`), the kept singular values divided by their norm (the
    /// bond's new spectrum, also recorded), and `V†` (`m x cols`), all
    /// row-major. Tracks the fidelity, the largest bond and whether the
    /// stored spectra are still current.
    fn cut(
        &mut self,
        bond: usize,
        theta: &[C64],
        rows: usize,
        cols: usize,
    ) -> (Vec<C64>, Vec<f64>, Vec<C64>) {
        let try_gram = self.gram_skip[bond] == 0;
        let (u, s_values, v, rejected) = truncating_svd(
            MatRef::from_row_major_slice(theta, rows, cols),
            self.truncation.max_bond,
            try_gram,
        );
        self.gram_skip[bond] = if rejected {
            32
        } else {
            self.gram_skip[bond].saturating_sub(1)
        };
        let (u, v) = (u.as_ref(), v.as_ref());
        let m = keep(self.truncation, &s_values);
        let total: f64 = s_values.iter().map(|x| x * x).sum();
        let kept: f64 = s_values[..m].iter().map(|x| x * x).sum();
        if kept < total {
            self.fidelity *= kept / total;
            if kept < total * (1.0 - 1e-12) {
                // A real truncation changes the other bonds' spectra.
                self.spectra_valid = false;
            }
        }
        self.max_bond_seen = self.max_bond_seen.max(m);
        let norm = kept.sqrt();
        let values: Vec<f64> = s_values[..m].iter().map(|x| x / norm).collect();
        self.spectra[bond] = values.clone();
        let vh = (0..m)
            .flat_map(|a| (0..cols).map(move |c| v[(c, a)].conj()))
            .collect();
        (row_major(u, m), values, vh)
    }

    /// Turn a gate into site steps: SWAPs that bring its qubits together,
    /// then the gate on their block.
    fn plan(&mut self, matrix: &Matrix, qubits: &[usize], ops: &mut Vec<SiteOp>) {
        assert_eq!(
            matrix.qubits(),
            qubits.len(),
            "matrix size does not match the qubits"
        );
        if qubits.len() == 1 {
            let site = self.site_of[qubits[0]];
            let matrix = [
                matrix[(0, 0)],
                matrix[(0, 1)],
                matrix[(1, 0)],
                matrix[(1, 1)],
            ];
            ops.push(SiteOp::One { site, matrix });
            return;
        }
        let start = self.gather(qubits, ops);
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
        let matrix: Arc<[C64]> = (0..d * d)
            .map(|i| matrix[(to_gate(i / d), to_gate(i % d))])
            .collect();
        let growth = if k == 2 {
            operator_schmidt_rank(&matrix)
        } else {
            usize::MAX
        };
        ops.push(SiteOp::Block {
            start,
            k,
            matrix,
            growth,
        });
    }

    /// Plan adjacent SWAPs that bring `qubits` next to each other, keeping
    /// their relative order, and return the first site of the block. The
    /// qubit at the median site stays where it is.
    fn gather(&mut self, qubits: &[usize], ops: &mut Vec<SiteOp>) -> usize {
        let mut by_site: Vec<usize> = qubits.to_vec();
        by_site.sort_by_key(|&q| self.site_of[q]);
        let k = by_site.len();
        let mid = k / 2;
        let start = self.site_of[by_site[mid]] - mid;
        for j in (0..mid).rev() {
            while self.site_of[by_site[j]] < start + j {
                let s = self.site_of[by_site[j]];
                self.plan_swap(s, ops);
            }
        }
        for (j, &q) in by_site.iter().enumerate().skip(mid + 1) {
            while self.site_of[q] > start + j {
                let s = self.site_of[q] - 1;
                self.plan_swap(s, ops);
            }
        }
        start
    }

    /// Swap the qubits at sites `s` and `s + 1`.
    fn plan_swap(&mut self, s: usize, ops: &mut Vec<SiteOp>) {
        ops.push(SiteOp::Block {
            start: s,
            k: 2,
            matrix: SWAP.clone(),
            growth: 4,
        });
        let (a, b) = (self.qubit_at[s], self.qubit_at[s + 1]);
        self.qubit_at.swap(s, s + 1);
        self.site_of[a] = s + 1;
        self.site_of[b] = s;
    }

    /// Run the steps of one layer (on disjoint sites, sorted, none of which
    /// can truncate) in parallel. Needs every site right-canonical and the
    /// spectra current, which the steps keep so.
    fn run_layer(&mut self, layer: &[SiteOp]) {
        // Each step gets its own sites and the bonds inside its block, and
        // a copy of the spectrum of the bond to its left (which no step of
        // the layer changes).
        let lefts: Vec<Vec<f64>> = layer
            .iter()
            .map(|op| match op.sites().start {
                0 => vec![1.0],
                s => self.spectra[s - 1].clone(),
            })
            .collect();
        let mut jobs = Vec::with_capacity(layer.len());
        let (mut sites, mut bonds) = (&mut self.sites[..], &mut self.spectra[..]);
        let mut offset = 0;
        for (op, left) in layer.iter().zip(&lefts) {
            let range = op.sites();
            let (_, rest) = std::mem::take(&mut sites).split_at_mut(range.start - offset);
            let (block, rest) = rest.split_at_mut(range.len());
            sites = rest;
            let (_, rest) = std::mem::take(&mut bonds).split_at_mut(range.start - offset);
            let (inside, rest) = rest.split_at_mut(range.len() - 1);
            // Skip the bond after the block too, keeping `bonds` aligned
            // with `sites`.
            bonds = if rest.is_empty() {
                rest
            } else {
                &mut rest[1..]
            };
            offset = range.end;
            jobs.push((op, block, inside, left.as_slice()));
        }
        let truncation = self.truncation;
        let run =
            |(op, block, inside, left): (&SiteOp, &mut [Site], &mut [Vec<f64>], &[f64])| match op {
                SiteOp::One { matrix, .. } => {
                    apply_one(&mut block[0], matrix);
                    (1.0, 0)
                }
                SiteOp::Block { matrix, .. } => {
                    update_block(block, matrix, left, inside, truncation)
                }
            };
        let blocks = layer
            .iter()
            .filter(|op| matches!(op, SiteOp::Block { .. }))
            .count();
        let results: Vec<(f64, usize)> = if blocks > 1 {
            use rayon::prelude::*;
            jobs.into_par_iter().map(run).collect()
        } else {
            jobs.into_iter().map(run).collect()
        };
        for (fidelity, bond) in results {
            self.fidelity *= fidelity;
            self.max_bond_seen = self.max_bond_seen.max(bond);
        }
    }

    /// Make every site right-canonical and the bond spectra current: a QR
    /// sweep to the right, then an SVD sweep back.
    fn canonicalize(&mut self) {
        if !self.spectra_valid {
            self.move_center(self.n - 1);
            for c in (1..self.n).rev() {
                let site = &self.sites[c];
                let (rows, cols, right) = (site.left, 2 * site.right, site.right);
                let (u, s, v) = svd(MatRef::from_row_major_slice(&site.data, rows, cols));
                let (u, v) = (u.as_ref(), v.as_ref());
                let m = keep(self.truncation, &s);
                let total: f64 = s.iter().map(|x| x * x).sum();
                let kept: f64 = s[..m].iter().map(|x| x * x).sum();
                if kept < total {
                    self.fidelity *= kept / total;
                }
                self.max_bond_seen = self.max_bond_seen.max(m);
                let norm = kept.sqrt();
                self.spectra[c - 1] = s[..m].iter().map(|x| x / norm).collect();
                let site_data = (0..m)
                    .flat_map(|a| (0..cols).map(move |j| v[(j, a)].conj()))
                    .collect();
                self.sites[c] = Site {
                    left: m,
                    right,
                    data: site_data,
                };
                let us: Vec<C64> = (0..rows)
                    .flat_map(|i| (0..m).map(move |a| (i, a)))
                    .map(|(i, a)| u[(i, a)] * (s[a] / norm))
                    .collect();
                let prev = &self.sites[c - 1];
                let data = matmul(&prev.data, prev.left * 2, rows, &us, m);
                let prev_left = prev.left;
                self.sites[c - 1] = Site {
                    left: prev_left,
                    right: m,
                    data,
                };
                self.center = c - 1;
            }
            self.spectra_valid = true;
        }
        self.move_center(0);
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
        self.spectra_valid = false;
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
            let w = [product(v.as_ref(), slice(0)), product(v.as_ref(), slice(1))];
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

/// The operator Schmidt rank of a two-site gate (row-major 4x4, first site
/// most significant): the rank of its matrix regrouped as
/// `(out1 in1) x (out2 in2)`. 2 for CX and CZ, 4 for SWAP.
fn operator_schmidt_rank(g: &[C64]) -> usize {
    // Gaussian elimination with partial pivoting; for 16 numbers this costs
    // far less than calling an SVD (once per gate of the circuit).
    let mut m = [[C64::new(0.0, 0.0); 4]; 4];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            let (o1, i1, o2, i2) = (r >> 1, r & 1, c >> 1, c & 1);
            *x = g[(o1 * 2 + o2) * 4 + (i1 * 2 + i2)];
        }
    }
    let scale = m.iter().flatten().map(|x| x.norm()).fold(0.0, f64::max);
    let mut rank = 0;
    for col in 0..4 {
        let pivot = (rank..4).max_by(|&a, &b| m[a][col].norm().total_cmp(&m[b][col].norm()));
        let Some(p) = pivot.filter(|&p| m[p][col].norm() > scale * 1e-10) else {
            continue;
        };
        m.swap(rank, p);
        let pivot_row = m[rank];
        for row in &mut m[rank + 1..] {
            let factor = row[col] / pivot_row[col];
            for (x, &y) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= factor * y;
            }
        }
        rank += 1;
    }
    rank.max(1)
}

/// How many singular values (sorted, descending) to keep.
fn keep(truncation: Truncation, s: &[f64]) -> usize {
    let total: f64 = s.iter().map(|x| x * x).sum();
    let mut m = s.len().min(truncation.max_bond).max(1);
    let mut discarded: f64 = s[m..].iter().map(|x| x * x).sum();
    while m > 1 && discarded + s[m - 1] * s[m - 1] <= truncation.threshold * total {
        m -= 1;
        discarded += s[m] * s[m];
    }
    m
}

/// Contract consecutive sites into one tensor `theta[l, p, r]` with `p`
/// of one bit per site (the first site most significant).
fn contract(sites: &[Site]) -> (usize, Vec<C64>, usize) {
    let first = &sites[0];
    let left = first.left;
    let mut theta = first.data.clone();
    let mut rows = left * 2;
    let mut bond = first.right;
    for site in &sites[1..] {
        theta = matmul(&theta, rows, bond, &site.data, 2 * site.right);
        rows *= 2;
        bond = site.right;
    }
    (left, theta, bond)
}

/// Apply a one-qubit gate to a site. A unitary on the physical index keeps
/// the site's canonical form.
fn apply_one(site: &mut Site, g: &[C64; 4]) {
    let r = site.right;
    for l in 0..site.left {
        let (zero, one) = site.data[l * 2 * r..(l + 1) * 2 * r].split_at_mut(r);
        for (a0, a1) in zero.iter_mut().zip(one.iter_mut()) {
            let (x0, x1) = (*a0, *a1);
            *a0 = g[0] * x0 + g[1] * x1;
            *a1 = g[2] * x0 + g[3] * x1;
        }
    }
}

/// Apply a block matrix to right-canonical sites whose left bond has
/// Schmidt coefficients `left`, with Hastings' update (Phys. Rev. B 79,
/// 165102, 2009): SVDs of `diag(left) θ` from the right give the new
/// right-canonical sites and the spectra of the inner bonds, and the first
/// site is `θ` times the adjoint of the new rest, so no coefficient is ever
/// inverted. Returns the kept fraction of the norm and the largest new bond.
fn update_block(
    sites: &mut [Site],
    g: &[C64],
    left: &[f64],
    inside: &mut [Vec<f64>],
    truncation: Truncation,
) -> (f64, usize) {
    let k = sites.len();
    let (l, mut theta, r) = contract(sites);
    apply_physical(&mut theta, g, 1 << k, l, r);
    let row = theta.len() / l;
    let mut m: Vec<C64> = theta
        .iter()
        .enumerate()
        .map(|(i, &x)| x * left[i / row])
        .collect();
    let (mut fidelity, mut max_bond) = (1.0, 0);
    let mut right = r;
    for j in (1..k).rev() {
        let cols = 2 * right;
        let rows = m.len() / cols;
        let (u, s, v) = svd(MatRef::from_row_major_slice(&m, rows, cols));
        let (u, v) = (u.as_ref(), v.as_ref());
        let kept_count = keep(truncation, &s);
        let total: f64 = s.iter().map(|x| x * x).sum();
        let kept: f64 = s[..kept_count].iter().map(|x| x * x).sum();
        if kept < total {
            fidelity *= kept / total;
        }
        max_bond = max_bond.max(kept_count);
        let norm = kept.sqrt();
        inside[j - 1] = s[..kept_count].iter().map(|x| x / norm).collect();
        let data = (0..kept_count)
            .flat_map(|a| (0..cols).map(move |c| v[(c, a)].conj()))
            .collect();
        sites[j] = Site {
            left: kept_count,
            right,
            data,
        };
        m = (0..rows)
            .flat_map(|i| (0..kept_count).map(move |a| (i, a)))
            .map(|(i, a)| u[(i, a)] * s[a])
            .collect();
        right = kept_count;
    }
    let (inner, rest, _) = contract(&sites[1..]);
    let width = rest.len() / inner;
    let rest_adjoint: Vec<C64> = (0..width)
        .flat_map(|c| (0..inner).map(move |a| (a, c)))
        .map(|(a, c)| rest[a * width + c].conj())
        .collect();
    let mut first = matmul(&theta, l * 2, width, &rest_adjoint, inner);
    // Renormalise: the state's norm is |diag(left) first| when the rest is
    // right-canonical.
    let norm_sqr: f64 = first
        .chunks(2 * inner)
        .zip(left)
        .map(|(block, &w)| w * w * block.iter().map(|x| x.norm_sqr()).sum::<f64>())
        .sum();
    if norm_sqr > 0.0 {
        let scale = 1.0 / norm_sqr.sqrt();
        first.iter_mut().for_each(|x| *x *= scale);
    }
    sites[0] = Site {
        left: l,
        right: inner,
        data: first,
    };
    (fidelity, max_bond)
}

/// The singular values of `a` (all of them, descending) and the singular
/// vectors of at least the largest `max_keep`, for a split whose bond is
/// capped at `max_keep`.
///
/// When the cap binds on a large matrix, the eigendecomposition of the
/// smaller Gram matrix is about twice as fast as an SVD (2.3x at 512x512).
/// Its other factor, `a` times the eigenvectors divided by the singular
/// values, is accurate to about `1e-16 (s_max / s)^2`, so it is used only
/// when every kept singular value is at least `1e-3 s_max`; otherwise, and
/// for small matrices, this is the SVD. The flag says whether the Gram
/// method was tried and rejected, so callers can stop trying it on bonds
/// whose kept spectra reach far down.
fn truncating_svd(
    a: MatRef<'_, C64>,
    max_keep: usize,
    try_gram: bool,
) -> (Mat<C64>, Vec<f64>, Mat<C64>, bool) {
    let small = a.nrows().min(a.ncols());
    if try_gram && small > max_keep && small >= 256 {
        match gram_svd(a, max_keep) {
            Some((u, s, v)) => return (u, s, v, false),
            None => {
                let (u, s, v) = svd(a);
                return (u, s, v, true);
            }
        }
    }
    let (u, s, v) = svd(a);
    (u, s, v, false)
}

/// See [`truncating_svd`]: all singular values from the Gram matrix's
/// eigenvalues, singular vectors for the largest `keep`, or `None` if the
/// smallest of those is too small for the other factor to be accurate.
fn gram_svd(a: MatRef<'_, C64>, keep: usize) -> Option<(Mat<C64>, Vec<f64>, Mat<C64>)> {
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
    let s: Vec<f64> = order
        .iter()
        .map(|&i| values[i].re.max(0.0).sqrt())
        .collect();
    if s[keep - 1] < 1e-3 * s[0] || !s.iter().all(|x| x.is_finite()) {
        return None;
    }
    let vectors = eigen.U();
    let x = Mat::<C64>::from_fn(vectors.nrows(), keep, |r, c| vectors[(r, order[c])]);
    let y: Mat<C64> = if wide { a.adjoint() * &x } else { a * &x };
    let y = Mat::<C64>::from_fn(y.nrows(), keep, |r, c| y[(r, c)] / s[c]);
    let (u, v) = if wide { (x, y) } else { (y, x) };
    (finite(u.as_ref()) && finite(v.as_ref())).then_some((u, s, v))
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
    fn gram_svd_matches_the_svd_on_the_kept_part() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(4);
        for (rows, cols) in [(256, 300), (300, 256), (256, 256)] {
            // A decaying spectrum, as on a capped bond.
            let data: Vec<C64> = (0..rows * cols)
                .map(|i| {
                    C64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5)
                        * (-((i % cols) as f64) / 200.0).exp()
                })
                .collect();
            let a = MatRef::from_row_major_slice(&data, rows, cols);
            let keep = 100;
            let (u, s, v) = gram_svd(a, keep).expect("well-conditioned kept part");
            let (_, exact, _) = svd(a);
            for i in 0..keep {
                assert!(
                    (s[i] - exact[i]).abs() < 1e-10 * exact[0],
                    "singular value {i}"
                );
            }
            // The kept rank-`keep` part is the best approximation: compare
            // U_k S_k V_k† with the SVD's.
            let (eu, es, ev) = svd(a);
            let mut error = 0.0f64;
            for i in 0..rows {
                for j in 0..cols {
                    let x: C64 = (0..keep).map(|k| u[(i, k)] * s[k] * v[(j, k)].conj()).sum();
                    let y: C64 = (0..keep)
                        .map(|k| eu[(i, k)] * es[k] * ev[(j, k)].conj())
                        .sum();
                    error = error.max((x - y).norm());
                }
            }
            assert!(error < 1e-9, "{rows}x{cols}: error {error}");
            let gram = u.adjoint() * &u;
            for i in 0..keep {
                assert!((gram[(i, i)].re - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn operator_schmidt_ranks() {
        use crate::circuit::Gate;
        let block = |gate: Gate| -> Vec<C64> {
            // Qubit 0 is the less significant bit of a gate's index; the
            // block's first site is the more significant one.
            let m = gate.matrix();
            let flip = |b: usize| ((b & 1) << 1) | (b >> 1);
            (0..16).map(|i| m[(flip(i / 4), flip(i % 4))]).collect()
        };
        assert_eq!(operator_schmidt_rank(&block(Gate::CX)), 2);
        assert_eq!(operator_schmidt_rank(&block(Gate::CZ)), 2);
        assert_eq!(operator_schmidt_rank(&block(Gate::CP(0.3))), 2);
        assert_eq!(operator_schmidt_rank(&block(Gate::SWAP)), 4);
        assert_eq!(operator_schmidt_rank(&block(Gate::RZZ(0.7))), 2);
        assert_eq!(operator_schmidt_rank(&SWAP), 4);
        let identity: Vec<C64> = (0..16)
            .map(|i| C64::new((i % 5 == 0) as u8 as f64, 0.0))
            .collect();
        assert_eq!(operator_schmidt_rank(&identity), 1);
    }

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

#[cfg(test)]
mod spectra_tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn layer_spectra_are_schmidt_coefficients() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(8);
        let n = 10;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut mps = Mps::new(n, Truncation::exact());
        let h = crate::circuit::Gate::H.matrix();
        let gates: Vec<(Matrix, Vec<usize>)> = (0..80)
            .map(|i| {
                if i % 3 == 0 {
                    (h.clone(), vec![rng.random_range(0..n)])
                } else {
                    let a = rng.random_range(0..n);
                    let b = (a + 1 + rng.random_range(0..n - 1)) % n;
                    (
                        crate::circuit::Gate::CRY(rng.random::<f64>() * 3.0).matrix(),
                        vec![a, b],
                    )
                }
            })
            .collect();
        let refs: Vec<(&Matrix, &[usize])> = gates.iter().map(|(m, q)| (m, q.as_slice())).collect();
        pool.install(|| mps.apply_gates(&refs));
        let stored = mps.spectra.clone();
        mps.spectra_valid = false;
        mps.canonicalize();
        for (b, (a, e)) in stored.iter().zip(&mps.spectra).enumerate() {
            assert_eq!(a.len(), e.len(), "bond {b}: dimensions");
            for (x, y) in a.iter().zip(e) {
                assert!(
                    (x - y).abs() < 1e-9,
                    "bond {b}: stored {a:?} vs recomputed {e:?}"
                );
            }
        }
    }
}
