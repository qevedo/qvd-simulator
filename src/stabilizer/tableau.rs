//! Stabilizer tableau simulation of Clifford circuits.
//!
//! A stabilizer state is `C|0...0>` for a Clifford unitary `C`. Following
//! Stim (Gidney, Quantum 5, 497, 2021), the tableau stores the *inverse*
//! of `C` acting on Paulis by conjugation: for each generator
//! `g` in `X_0..X_{n-1}, Z_0..Z_{n-1}`, the Pauli string `inv(g) = C† g C`.
//!
//! - Applying a gate `G` (so `C -> G C`) replaces `inv(g)` by
//!   `inv(G† g G)`: products of a few rows, i.e. whole-row XORs, `O(n/64)`
//!   word operations per gate.
//! - Measuring `Z_q` looks at `P = inv(Z_q)`. If `P` has no X or Y
//!   component, the outcome is fixed by its sign: `O(n/64)`.
//! - Otherwise the outcome is random. Picking a qubit `k` where `P` has an
//!   X or Y, a Clifford `U` made of `CX(k, j)`, `CZ(k, j)` and `S(k)` gates
//!   (all of which fix `|0...0>`) turns `P` into `±X_k`, and the collapsed
//!   state is `C U H_k X_k^b |0...0>` for a random bit `b`. Every row is
//!   conjugated by `U H_k X_k^b`; with the CX and CZ layers applied in
//!   closed form per row, that is `O(n^2/64)` word operations.
//!
//! Signs follow Aaronson & Gottesman's convention: a row is `(-1)^sign`
//! times a product of `I, X, Y, Z`, with `(x, z) = (1, 1)` meaning `Y`.

use std::sync::atomic::{AtomicU64, Ordering};

use rand::{Rng, RngExt};
use rayon::prelude::*;

use super::Clifford;

/// A Pauli string: per qubit `I`, `X`, `Y` or `Z`, with a sign.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PauliString {
    /// `(x, z)` bits per qubit: `(1, 0)` = X, `(1, 1)` = Y, `(0, 1)` = Z.
    pub xs: Vec<bool>,
    pub zs: Vec<bool>,
    /// `true` for a leading minus sign.
    pub negative: bool,
}

impl PauliString {
    /// Parse a string such as `"+XIZY"` or `"-ZZ"`; character `j` (after the
    /// optional sign) is qubit `j`.
    pub fn parse(text: &str) -> Option<PauliString> {
        let (negative, body) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        let mut xs = Vec::new();
        let mut zs = Vec::new();
        for c in body.chars() {
            let (x, z) = match c {
                'I' | '_' => (false, false),
                'X' => (true, false),
                'Y' => (true, true),
                'Z' => (false, true),
                _ => return None,
            };
            xs.push(x);
            zs.push(z);
        }
        Some(PauliString { xs, zs, negative })
    }
}

/// `Σ g(x1, z1, x2, z2)` over the qubits of two rows (mod 4), where
/// `σ1 σ2 = i^g σ(x1^x2, z1^z2)`: the phase picked up when multiplying the
/// Pauli strings (Aaronson & Gottesman's `g`, bit-sliced over 64 qubits).
#[inline]
fn product_phase(x1: &[u64], z1: &[u64], x2: &[u64], z2: &[u64]) -> u32 {
    let mut plus = 0u32;
    let mut minus = 0u32;
    for w in 0..x1.len() {
        let (a, b, c, d) = (x1[w], z1[w], x2[w], z2[w]);
        // +1: Y·Z, X·Y, Z·X.   -1: Y·X, X·Z, Z·Y.
        let p = (a & b & !c & d) | (a & !b & c & d) | (!a & b & c & !d);
        let m = (a & b & c & !d) | (a & !b & !c & d) | (!a & b & c & d);
        plus += p.count_ones();
        minus += m.count_ones();
    }
    plus.wrapping_sub(minus) & 3
}

/// `(x1, z1) := (x1, z1) · (x2, z2)` over all words, returning the phase
/// exponent (powers of i, mod 4) of the product.
///
/// Stim's bit-sliced formulation: every bit position keeps a 2-bit counter
/// (`cnt1`, `cnt2`) of the `±i` factors from anticommuting single-qubit
/// products, updated with a few bitwise operations per word (which the
/// compiler vectorises); the counters are popcounted once at the end.
#[inline]
fn multiply_in_place(x1: &mut [u64], z1: &mut [u64], x2: &[u64], z2: &[u64]) -> u32 {
    let mut cnt1 = 0u64;
    let mut cnt2 = 0u64;
    for (((x1, z1), &x2), &z2) in x1.iter_mut().zip(z1.iter_mut()).zip(x2).zip(z2) {
        let (old_x1, old_z1) = (*x1, *z1);
        *x1 ^= x2;
        *z1 ^= z2;
        let x1z2 = old_x1 & z2;
        let anticommutes = (x2 & old_z1) ^ x1z2;
        cnt2 ^= (cnt1 ^ *x1 ^ *z1 ^ x1z2) & anticommutes;
        cnt1 ^= anticommutes;
    }
    (cnt1.count_ones() ^ (cnt2.count_ones() << 1)) & 3
}

/// The inverse stabilizer tableau of an `n`-qubit state.
#[derive(Clone, Debug)]
pub struct Tableau {
    n: usize,
    /// u64 words per bit row.
    words: usize,
    /// Row `r` occupies `xs[r * words..(r + 1) * words]`; rows `0..n` are
    /// `inv(X_k)`, rows `n..2n` are `inv(Z_k)`.
    ///
    /// While random measurements are being collapsed the bits are kept
    /// transposed instead (`columns`): qubit `k` occupies
    /// `xs[k * column_words..(k + 1) * column_words]`, bit `r` of it
    /// belonging to row `r`.
    xs: Vec<u64>,
    zs: Vec<u64>,
    /// Bit `r` is the sign of row `r`.
    signs: Vec<u64>,
    /// Row layout only: bit `w` of row `r`'s `occ_words` words is set iff
    /// word `w` of row `r` is nonzero in `xs` or `zs`. Rows of circuits
    /// with local structure (error-correcting codes) are mostly zero, and
    /// row products then touch only the source row's nonzero words.
    occ: Vec<u64>,
    occ_words: usize,
    /// u64 words per qubit column (`2n` bits) in the column layout.
    column_words: usize,
    columns: bool,
    layout: Layout,
    /// How much the other layout would have saved recently, in units of
    /// one gate applied in the column layout (see [`Layout::Adaptive`]).
    balance: usize,
    /// The balance at which switching layouts pays off.
    switch_cost: usize,
    /// Reused destination of transpositions.
    scratch: Vec<u64>,
    /// In the column layout, the x bits of the 64 rows `64 * cache_group..`
    /// in row layout (64 rows of `words` words), so that finding a row's
    /// X/Y qubits does not read every column. `usize::MAX` when empty.
    cache_group: usize,
    cache: Vec<u64>,
}

/// Which bit layout a [`Tableau`] keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Rows (cheap gates) while gates dominate, columns (cheap random
    /// measurements) while random measurements do. Each layout keeps a
    /// running balance of what the other would have saved; once it exceeds
    /// the cost of a transposition the tableau switches.
    Adaptive,
    /// Always rows: every random measurement is a pass over all rows.
    Rows,
    /// Columns from the first random measurement on: every gate is a
    /// strided row operation.
    Columns,
}

/// The cost of a random measurement in the row layout, in units of one gate
/// in the column layout: both are strided passes over the tableau, the
/// former touching about twice as much memory.
const ROW_COLLAPSE_COST: usize = 2;

/// Transpose a 64x64 bit matrix in place: bit `j` of `a[i]` swaps with bit
/// `i` of `a[j]`.
fn transpose64(a: &mut [u64; 64]) {
    let mut j = 32;
    let mut m = 0x0000_0000_FFFF_FFFFu64;
    while j != 0 {
        let mut k = 0;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// Transpose a `rows x cols` bit matrix stored row-major (`cols.div_ceil(64)`
/// words per row) into `dst` (`rows.div_ceil(64)` words per row).
fn transpose_bits(src: &[u64], rows: usize, cols: usize, dst: &mut Vec<u64>) {
    let src_words = cols.div_ceil(64);
    let dst_words = rows.div_ceil(64);
    dst.clear();
    dst.resize(cols * dst_words, 0);
    // Each task produces 512 output rows: 8 source words, one cache line
    // of every source row.
    dst.par_chunks_mut(512 * dst_words)
        .enumerate()
        .for_each(|(g, out)| {
            let w0 = g * 8;
            let width = (src_words - w0).min(8);
            let mut blocks = [[0u64; 64]; 8];
            for rb in 0..dst_words {
                for i in 0..64 {
                    let r = rb * 64 + i;
                    for (b, block) in blocks[..width].iter_mut().enumerate() {
                        block[i] = if r < rows {
                            src[r * src_words + w0 + b]
                        } else {
                            0
                        };
                    }
                }
                for (b, block) in blocks[..width].iter_mut().enumerate() {
                    transpose64(block);
                    for (j, &v) in block.iter().enumerate() {
                        if let Some(word) = out.get_mut((b * 64 + j) * dst_words + rb) {
                            *word = v;
                        }
                    }
                }
            }
        });
}

/// Gates in parallel layers are handed out in chunks of this many.
const LAYER_CHUNK: usize = 32;

/// The row operations gates are made of.
trait RowOps {
    fn num_qubits(&self) -> usize;
    fn multiply_rows(&mut self, dst: usize, src: usize, extra: u32);
    fn swap_rows(&mut self, a: usize, b: usize);
    fn flip_sign(&mut self, row: usize, flip: bool);
}

/// Apply a gate, `C -> G C`: each row `inv(g)` becomes `inv(G† g G)`, a
/// product of rows. Rows `0..n` are `inv(X_k)`, rows `n..2n` `inv(Z_k)`.
fn apply_gate(rows: &mut impl RowOps, op: Clifford) {
    let n = rows.num_qubits();
    match op {
        // H† X H = Z, H† Z H = X.
        Clifford::H(q) => rows.swap_rows(q, n + q),
        // S† X S = -Y = -i X Z.
        Clifford::S(q) => rows.multiply_rows(q, n + q, 3),
        // S X S† = Y = i X Z.
        Clifford::Sdg(q) => rows.multiply_rows(q, n + q, 1),
        // X† Z X = -Z.
        Clifford::X(q) => rows.flip_sign(n + q, true),
        Clifford::Z(q) => rows.flip_sign(q, true),
        Clifford::Y(q) => {
            rows.flip_sign(n + q, true);
            rows.flip_sign(q, true);
        }
        // CX X_c CX = X_c X_t, CX Z_t CX = Z_c Z_t.
        Clifford::CX(c, t) => {
            rows.multiply_rows(c, t, 0);
            rows.multiply_rows(n + t, n + c, 0);
        }
        // CZ X_a CZ = X_a Z_b.
        Clifford::CZ(a, b) => {
            rows.multiply_rows(a, n + b, 0);
            rows.multiply_rows(b, n + a, 0);
        }
        Clifford::Swap(a, b) => {
            rows.swap_rows(a, b);
            rows.swap_rows(n + a, n + b);
        }
    }
}

/// Whether no two gates share a qubit.
fn disjoint(ops: &[Clifford]) -> bool {
    let mut seen = std::collections::HashSet::new();
    ops.iter().all(|op| op.qubits().all(|q| seen.insert(q)))
}

impl RowOps for Tableau {
    fn num_qubits(&self) -> usize {
        self.n
    }

    fn multiply_rows(&mut self, dst: usize, src: usize, extra: u32) {
        Tableau::multiply_rows(self, dst, src, extra);
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        Tableau::swap_rows(self, a, b);
    }

    fn flip_sign(&mut self, row: usize, flip: bool) {
        Tableau::flip_sign(self, row, flip);
    }
}

/// Raw pointers to a row-layout tableau. Operations on distinct rows may
/// run on different threads: each row's words and occupancy bits belong to
/// it alone, and sign bits, which share words, are updated atomically.
#[derive(Clone, Copy)]
struct RowView<'a> {
    n: usize,
    words: usize,
    occ_words: usize,
    xs: *mut u64,
    zs: *mut u64,
    occ: *mut u64,
    signs: *mut u64,
    _tableau: std::marker::PhantomData<&'a mut Tableau>,
}

// SAFETY: see the type's documentation; callers give each thread its own
// rows.
unsafe impl Send for RowView<'_> {}
unsafe impl Sync for RowView<'_> {}

impl RowView<'_> {
    fn sign_word(&self, row: usize) -> &AtomicU64 {
        // SAFETY: `signs` has a word for every row and is 8-byte aligned;
        // all concurrent access to sign words goes through atomics.
        unsafe { AtomicU64::from_ptr(self.signs.add(row / 64)) }
    }

    fn sign(&self, row: usize) -> bool {
        self.sign_word(row).load(Ordering::Relaxed) >> (row % 64) & 1 == 1
    }

    /// Update the occupancy bit of word `w` of row `row`.
    ///
    /// # Safety
    /// The caller must own row `row`.
    unsafe fn update_occ(&self, row: usize, w: usize) {
        unsafe {
            let nonzero =
                *self.xs.add(row * self.words + w) | *self.zs.add(row * self.words + w) != 0;
            let occ = self.occ.add(row * self.occ_words + w / 64);
            let b = 1u64 << (w % 64);
            *occ = (*occ & !b) | if nonzero { b } else { 0 };
        }
    }
}

impl RowOps for RowView<'_> {
    fn num_qubits(&self) -> usize {
        self.n
    }

    fn multiply_rows(&mut self, dst: usize, src: usize, extra: u32) {
        debug_assert_ne!(dst, src);
        let (words, ow) = (self.words, self.occ_words);
        let (d, s) = (dst * words, src * words);
        // SAFETY: the caller owns rows `dst` and `src`, which are distinct,
        // so the mutable and shared slices below do not overlap.
        let phase = unsafe {
            let src_occ = std::slice::from_raw_parts(self.occ.add(src * ow), ow);
            let occupied: u32 = src_occ.iter().map(|w| w.count_ones()).sum();
            if occupied as usize * 4 > words {
                // Dense rows: the vectorised loop over every word.
                let phase = multiply_in_place(
                    std::slice::from_raw_parts_mut(self.xs.add(d), words),
                    std::slice::from_raw_parts_mut(self.zs.add(d), words),
                    std::slice::from_raw_parts(self.xs.add(s), words),
                    std::slice::from_raw_parts(self.zs.add(s), words),
                );
                for (i, &bits) in src_occ.iter().enumerate() {
                    let mut bits = bits;
                    while bits != 0 {
                        self.update_occ(dst, i * 64 + bits.trailing_zeros() as usize);
                        bits &= bits - 1;
                    }
                }
                phase
            } else {
                // Sparse rows: only the words where `src` is nonzero change,
                // and only they contribute to the phase (see
                // `multiply_in_place`).
                let (mut cnt1, mut cnt2) = (0u64, 0u64);
                for (i, &bits) in src_occ.iter().enumerate() {
                    let mut bits = bits;
                    let dst_occ = self.occ.add(dst * ow + i);
                    let mut occ = *dst_occ | bits;
                    while bits != 0 {
                        let w = i * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let (x2, z2) = (*self.xs.add(s + w), *self.zs.add(s + w));
                        let (xd, zd) = (self.xs.add(d + w), self.zs.add(d + w));
                        let (old_x1, old_z1) = (*xd, *zd);
                        let (x1, z1) = (old_x1 ^ x2, old_z1 ^ z2);
                        *xd = x1;
                        *zd = z1;
                        let x1z2 = old_x1 & z2;
                        let anticommutes = (x2 & old_z1) ^ x1z2;
                        cnt2 ^= (cnt1 ^ x1 ^ z1 ^ x1z2) & anticommutes;
                        cnt1 ^= anticommutes;
                        if x1 | z1 == 0 {
                            occ &= !(1 << (w % 64));
                        }
                    }
                    *dst_occ = occ;
                }
                (cnt1.count_ones() ^ (cnt2.count_ones() << 1)) & 3
            }
        };
        let total = 2 * self.sign(dst) as u32 + 2 * self.sign(src) as u32 + phase + extra;
        debug_assert!(total.is_multiple_of(2), "row product is not Hermitian");
        self.flip_sign(dst, (total / 2 + self.sign(dst) as u32) % 2 == 1);
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        let (words, ow) = (self.words, self.occ_words);
        // SAFETY: the caller owns rows `a` and `b`.
        unsafe {
            for i in 0..ow {
                let (oa, ob) = (self.occ.add(a * ow + i), self.occ.add(b * ow + i));
                let mut bits = *oa | *ob;
                while bits != 0 {
                    let w = i * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    std::ptr::swap(self.xs.add(a * words + w), self.xs.add(b * words + w));
                    std::ptr::swap(self.zs.add(a * words + w), self.zs.add(b * words + w));
                }
                std::ptr::swap(oa, ob);
            }
        }
        let (sa, sb) = (self.sign(a), self.sign(b));
        self.flip_sign(a, sa != sb);
        self.flip_sign(b, sa != sb);
    }

    fn flip_sign(&mut self, row: usize, flip: bool) {
        if flip {
            self.sign_word(row)
                .fetch_xor(1 << (row % 64), Ordering::Relaxed);
        }
    }
}

impl Tableau {
    /// The state `|0...0>` on `n` qubits.
    pub fn new(n: usize) -> Tableau {
        let words = n.div_ceil(64).max(1);
        let column_words = (2 * n).div_ceil(64).max(1);
        let occ_words = words.div_ceil(64);
        let mut t = Tableau {
            n,
            words,
            xs: vec![0; 2 * n * words],
            zs: vec![0; 2 * n * words],
            signs: vec![0; column_words],
            occ: vec![0; 2 * n * occ_words],
            occ_words,
            column_words,
            columns: false,
            layout: Layout::Adaptive,
            balance: 0,
            // A transposition streams the whole tableau twice; a column-
            // layout gate reads two rows, `4n` strided words.
            switch_cost: 16 + n / 256,
            scratch: Vec::new(),
            cache_group: usize::MAX,
            cache: Vec::new(),
        };
        for k in 0..n {
            t.xs[k * words + k / 64] |= 1 << (k % 64);
            t.zs[(n + k) * words + k / 64] |= 1 << (k % 64);
        }
        t.recompute_occ();
        t
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Choose the layout policy (the default is [`Layout::Adaptive`]).
    pub fn set_layout(&mut self, layout: Layout) {
        self.layout = layout;
    }

    /// Set how much the other layout must save before [`Layout::Adaptive`]
    /// switches to it, in units of one gate applied in the column layout.
    pub fn set_switch_cost(&mut self, cost: usize) {
        self.switch_cost = cost;
    }

    /// Whether the next gate runs in the column layout, switching to rows
    /// if that has become cheaper.
    fn gate_in_columns(&mut self) -> bool {
        if !self.columns {
            self.balance = self.balance.saturating_sub(1);
            return false;
        }
        self.balance += 1;
        match self.layout {
            Layout::Columns => true,
            Layout::Adaptive if self.balance <= self.switch_cost => true,
            _ => {
                self.use_rows();
                false
            }
        }
    }

    fn sign(&self, row: usize) -> bool {
        self.signs[row / 64] >> (row % 64) & 1 == 1
    }

    fn flip_sign(&mut self, row: usize, flip: bool) {
        self.signs[row / 64] ^= (flip as u64) << (row % 64);
    }

    fn use_rows(&mut self) {
        self.balance = 0;
        if self.columns {
            self.transpose(self.n, 2 * self.n);
            self.columns = false;
            self.recompute_occ();
        }
    }

    fn recompute_occ(&mut self) {
        let (words, xs, zs) = (self.words, &self.xs, &self.zs);
        self.occ
            .par_chunks_mut(self.occ_words)
            .enumerate()
            .for_each(|(row, occ)| {
                occ.fill(0);
                for w in 0..words {
                    if xs[row * words + w] | zs[row * words + w] != 0 {
                        occ[w / 64] |= 1 << (w % 64);
                    }
                }
            });
    }

    /// Update the occupancy bit of word `w` of row `row` (row layout).
    fn update_occ(&mut self, row: usize, w: usize) {
        let nonzero = self.xs[row * self.words + w] | self.zs[row * self.words + w] != 0;
        let (i, b) = (row * self.occ_words + w / 64, 1u64 << (w % 64));
        self.occ[i] = (self.occ[i] & !b) | if nonzero { b } else { 0 };
    }

    fn use_columns(&mut self) {
        self.balance = 0;
        self.cache_group = usize::MAX;
        if !self.columns {
            self.transpose(2 * self.n, self.n);
            self.columns = true;
        }
    }

    fn transpose(&mut self, rows: usize, cols: usize) {
        for table in [&mut self.xs, &mut self.zs] {
            transpose_bits(table, rows, cols, &mut self.scratch);
            std::mem::swap(table, &mut self.scratch);
        }
    }

    /// The `(x, z)` words of row `row`, in either layout.
    fn row(&self, row: usize) -> (Vec<u64>, Vec<u64>) {
        if !self.columns {
            let r = self.range(row);
            return (self.xs[r.clone()].to_vec(), self.zs[r].to_vec());
        }
        let (mut xs, mut zs) = (vec![0u64; self.words], vec![0u64; self.words]);
        let (rw, rb) = (row / 64, row % 64);
        for k in 0..self.n {
            xs[k / 64] |= (self.xs[k * self.column_words + rw] >> rb & 1) << (k % 64);
            zs[k / 64] |= (self.zs[k * self.column_words + rw] >> rb & 1) << (k % 64);
        }
        (xs, zs)
    }

    fn x_row(&self, k: usize) -> usize {
        k
    }

    fn z_row(&self, k: usize) -> usize {
        self.n + k
    }

    fn range(&self, row: usize) -> std::ops::Range<usize> {
        row * self.words..(row + 1) * self.words
    }

    /// `row dst := row dst · row src`, times `i^extra`. The result must be
    /// Hermitian (the total phase a power of -1).
    fn multiply_rows(&mut self, dst: usize, src: usize, extra: u32) {
        debug_assert_ne!(dst, src);
        if self.gate_in_columns() {
            return self.multiply_rows_in_columns(dst, src, extra);
        }
        self.row_view().multiply_rows(dst, src, extra);
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if self.gate_in_columns() {
            return self.swap_rows_in_columns(a, b);
        }
        self.row_view().swap_rows(a, b);
    }

    /// Raw access to the rows (row layout), for gates on distinct rows.
    fn row_view(&mut self) -> RowView<'_> {
        debug_assert!(!self.columns);
        RowView {
            n: self.n,
            words: self.words,
            occ_words: self.occ_words,
            xs: self.xs.as_mut_ptr(),
            zs: self.zs.as_mut_ptr(),
            occ: self.occ.as_mut_ptr(),
            signs: self.signs.as_mut_ptr(),
            _tableau: std::marker::PhantomData,
        }
    }

    /// Apply gates that act on pairwise disjoint qubits. They commute and
    /// touch disjoint rows, so in the row layout they run in parallel; the
    /// result is the same as applying them in order.
    pub fn apply_layer(&mut self, ops: &[Clifford]) {
        let mut rest = ops;
        // The column layout runs gates one by one (and may switch to rows).
        while self.columns && !rest.is_empty() {
            apply_gate(self, rest[0]);
            rest = &rest[1..];
        }
        if rest.len() < 2 * LAYER_CHUNK || rayon::current_num_threads() == 1 {
            for &op in rest {
                apply_gate(self, op);
            }
            return;
        }
        debug_assert!(disjoint(rest), "layer gates must act on disjoint qubits");
        // Each row operation would have cost the column layout one unit.
        self.balance = self.balance.saturating_sub(2 * rest.len());
        let view = self.row_view();
        rest.par_chunks(LAYER_CHUNK).for_each(|chunk| {
            let mut view = view;
            for &op in chunk {
                apply_gate(&mut view, op);
            }
        });
    }

    /// [`Tableau::multiply_rows`] in the column layout: gather both rows,
    /// then flip the bits of `dst` where `src` has them.
    fn multiply_rows_in_columns(&mut self, dst: usize, src: usize, extra: u32) {
        let (sx, sz) = self.row(src);
        let (mut dx, mut dz) = self.row(dst);
        let phase = multiply_in_place(&mut dx, &mut dz, &sx, &sz);
        let total = 2 * self.sign(dst) as u32 + 2 * self.sign(src) as u32 + phase + extra;
        debug_assert!(total.is_multiple_of(2), "row product is not Hermitian");
        self.flip_sign(dst, (total / 2 + self.sign(dst) as u32) % 2 == 1);
        self.flip_row_bits(dst, &sx, &sz);
        self.set_cached_row(dst, &dx);
    }

    fn swap_rows_in_columns(&mut self, a: usize, b: usize) {
        let (ax, az) = self.row(a);
        let (bx, bz) = self.row(b);
        let dx: Vec<u64> = ax.iter().zip(&bx).map(|(p, q)| p ^ q).collect();
        let dz: Vec<u64> = az.iter().zip(&bz).map(|(p, q)| p ^ q).collect();
        self.flip_row_bits(a, &dx, &dz);
        self.flip_row_bits(b, &dx, &dz);
        let (sa, sb) = (self.sign(a), self.sign(b));
        self.flip_sign(a, sa != sb);
        self.flip_sign(b, sa != sb);
        self.set_cached_row(a, &bx);
        self.set_cached_row(b, &ax);
    }

    /// Flip row `row`'s bits at the set bits of `x` and `z` (column layout).
    fn flip_row_bits(&mut self, row: usize, x: &[u64], z: &[u64]) {
        let (rw, rb, cw) = (row / 64, 1u64 << (row % 64), self.column_words);
        for (table, bits) in [(&mut self.xs, x), (&mut self.zs, z)] {
            for (w, &word) in bits.iter().enumerate() {
                let mut rest = word;
                while rest != 0 {
                    let k = w * 64 + rest.trailing_zeros() as usize;
                    table[k * cw + rw] ^= rb;
                    rest &= rest - 1;
                }
            }
        }
    }

    /// Record that row `row` now has x words `x`, if it is cached.
    fn set_cached_row(&mut self, row: usize, x: &[u64]) {
        if row / 64 == self.cache_group {
            self.cache[(row % 64) * self.words..][..self.words].copy_from_slice(x);
        }
    }

    // -- gates: C -> G C, i.e. inv(g) -> inv(G† g G) ----------------------

    pub fn h(&mut self, q: usize) {
        apply_gate(self, Clifford::H(q));
    }

    pub fn s(&mut self, q: usize) {
        apply_gate(self, Clifford::S(q));
    }

    pub fn sdg(&mut self, q: usize) {
        apply_gate(self, Clifford::Sdg(q));
    }

    pub fn x(&mut self, q: usize) {
        apply_gate(self, Clifford::X(q));
    }

    pub fn z(&mut self, q: usize) {
        apply_gate(self, Clifford::Z(q));
    }

    pub fn y(&mut self, q: usize) {
        apply_gate(self, Clifford::Y(q));
    }

    pub fn sx(&mut self, q: usize) {
        // SX = H S H.
        self.h(q);
        self.s(q);
        self.h(q);
    }

    pub fn sxdg(&mut self, q: usize) {
        self.h(q);
        self.sdg(q);
        self.h(q);
    }

    pub fn cx(&mut self, control: usize, target: usize) {
        apply_gate(self, Clifford::CX(control, target));
    }

    pub fn cz(&mut self, a: usize, b: usize) {
        apply_gate(self, Clifford::CZ(a, b));
    }

    pub fn cy(&mut self, control: usize, target: usize) {
        // CY = S_t CX S_t†: apply S†, CX, S in that order.
        self.sdg(target);
        self.cx(control, target);
        self.s(target);
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        apply_gate(self, Clifford::Swap(a, b));
    }

    // -- measurement ------------------------------------------------------

    /// The outcome of measuring qubit `q` in the Z basis if it is
    /// determined, without changing the state.
    pub fn peek_z(&self, q: usize) -> Option<bool> {
        let row = self.z_row(q);
        let random = if self.columns {
            self.column_pivot(row, 0).is_some()
        } else {
            let base = row * self.words;
            self.occ[row * self.occ_words..(row + 1) * self.occ_words]
                .iter()
                .enumerate()
                .any(|(i, &bits)| {
                    let mut bits = bits;
                    while bits != 0 {
                        if self.xs[base + i * 64 + bits.trailing_zeros() as usize] != 0 {
                            return true;
                        }
                        bits &= bits - 1;
                    }
                    false
                })
        };
        (!random).then(|| self.sign(row))
    }

    /// Measure qubit `q` in the Z basis and collapse the state.
    pub fn measure(&mut self, q: usize, rng: &mut impl Rng) -> bool {
        self.measure_with(q, || rng.random::<bool>())
    }

    /// Measure with the random outcome bit supplied by `coin` (called only
    /// when the outcome is not determined).
    pub fn measure_with(&mut self, q: usize, coin: impl FnOnce() -> bool) -> bool {
        if !self.columns {
            if let Some(outcome) = self.peek_z(q) {
                return outcome;
            }
            self.balance += ROW_COLLAPSE_COST;
            let stay = match self.layout {
                Layout::Rows => true,
                Layout::Columns => false,
                Layout::Adaptive => self.balance <= self.switch_cost,
            };
            if stay {
                self.collapse(q, coin());
                return self
                    .peek_z(q)
                    .expect("the measurement is determined after collapse");
            }
            self.use_columns();
        }
        self.collapse_columns(q, coin)
    }

    /// The first qubit `k >= from` at which row `row` has an X or Y, in
    /// the column layout.
    fn column_pivot(&self, row: usize, from: usize) -> Option<usize> {
        let (rw, rb) = (row / 64, 1u64 << (row % 64));
        (from..self.n).find(|&k| self.xs[k * self.column_words + rw] & rb != 0)
    }

    /// The x words of `row` (column layout), through the row cache.
    fn cached_x_row(&mut self, row: usize) -> &[u64] {
        let (group, words, cw) = (row / 64, self.words, self.column_words);
        if self.cache_group != group {
            self.cache.clear();
            self.cache.resize(64 * words, 0);
            let mut block = [0u64; 64];
            for kw in 0..words {
                for (j, word) in block.iter_mut().enumerate() {
                    let k = kw * 64 + j;
                    *word = if k < self.n {
                        self.xs[k * cw + group]
                    } else {
                        0
                    };
                }
                transpose64(&mut block);
                for (b, &word) in block.iter().enumerate() {
                    self.cache[b * words + kw] = word;
                }
            }
            self.cache_group = group;
        }
        &self.cache[(row % 64) * words..][..words]
    }

    /// Update the row cache after the x bits of column `k` changed.
    fn refresh_cache(&mut self, k: usize) {
        if self.cache_group == usize::MAX {
            return;
        }
        let column = self.xs[k * self.column_words + self.cache_group];
        let (kw, kb) = (k / 64, 1u64 << (k % 64));
        for b in 0..64 {
            let word = &mut self.cache[b * self.words + kw];
            *word = (*word & !kb) | if column >> b & 1 == 1 { kb } else { 0 };
        }
    }

    /// Measure `Z_q` in the column layout. As in Stim, gates that fix
    /// `|0...0>` (CX controlled by the pivot) are prepended to the circuit
    /// until `inv(Z_q)` has a single X or Y, at the pivot; a prepended H
    /// (or H_YZ) on the pivot then collapses the state, and an X sets the
    /// outcome. Prepending a gate conjugates every row by it: a column
    /// operation.
    fn collapse_columns(&mut self, q: usize, coin: impl FnOnce() -> bool) -> bool {
        let row = self.z_row(q);
        // CX(pivot, k) only changes column k, so the X/Y qubits can be read
        // once up front.
        let x_row = self.cached_x_row(row).to_vec();
        let mut qubits = x_row.iter().enumerate().flat_map(|(w, &word)| {
            std::iter::successors((word != 0).then_some(word), |&rest| {
                Some(rest & (rest - 1)).filter(|&r| r != 0)
            })
            .map(move |rest| w * 64 + rest.trailing_zeros() as usize)
        });
        let Some(pivot) = qubits.next() else {
            return self.sign(row);
        };
        self.balance = self.balance.saturating_sub(ROW_COLLAPSE_COST);
        for k in qubits {
            self.cx_columns(pivot, k);
            self.refresh_cache(k);
        }
        let (rw, rb) = (row / 64, 1u64 << (row % 64));
        let cw = self.column_words;
        let (p, z) = (pivot * cw, self.zs[pivot * cw + rw] & rb != 0);
        for i in 0..cw {
            let (x, zi) = (self.xs[p + i], self.zs[p + i]);
            if z {
                // H_YZ: X -> -X, Y <-> Z.
                self.signs[i] ^= x & !zi;
                self.xs[p + i] = x ^ zi;
            } else {
                // H: Y -> -Y, X <-> Z.
                self.signs[i] ^= x & zi;
                self.xs[p + i] = zi;
                self.zs[p + i] = x;
            }
        }
        self.refresh_cache(pivot);
        let outcome = coin();
        if self.sign(row) != outcome {
            // X on the pivot flips every row with a Z or Y there.
            for i in 0..cw {
                self.signs[i] ^= self.zs[p + i];
            }
        }
        outcome
    }

    /// Conjugate every row by `CX(c, t)` (column layout).
    fn cx_columns(&mut self, c: usize, t: usize) {
        let cw = self.column_words;
        let (c, t) = (c * cw, t * cw);
        for i in 0..cw {
            let (xc, zc, xt, zt) = (
                self.xs[c + i],
                self.zs[c + i],
                self.xs[t + i],
                self.zs[t + i],
            );
            self.signs[i] ^= xc & zt & !(xt ^ zc);
            self.xs[t + i] = xt ^ xc;
            self.zs[c + i] = zc ^ zt;
        }
    }

    /// Reset qubit `q` to |0>.
    pub fn reset(&mut self, q: usize, rng: &mut impl Rng) {
        if self.measure(q, rng) {
            self.x(q);
        }
    }

    /// Make the Z measurement of `q` deterministic by conjugating every row
    /// by `U H_k X_k^b` (see the module documentation).
    fn collapse(&mut self, q: usize, b: bool) {
        let words = self.words;
        let p = self.z_row(q);
        let pr = self.range(p);
        let pivot = self.xs[pr.clone()]
            .iter()
            .enumerate()
            .find(|&(_, &w)| w != 0)
            .map(|(i, &w)| i * 64 + w.trailing_zeros() as usize)
            .expect("collapse needs a random outcome");
        let (kw, kb) = (pivot / 64, 1u64 << (pivot % 64));
        // CX(k, j) for each X component j != k, CZ(k, j) for each Z component.
        let mut cx_mask = self.xs[pr.clone()].to_vec();
        cx_mask[kw] &= !kb;
        let mut cz_mask = self.zs[pr].to_vec();
        cz_mask[kw] &= !kb;
        // Only the words the masks touch need per-row work.
        let active: Vec<usize> = (0..words)
            .filter(|&w| cx_mask[w] | cz_mask[w] != 0)
            .collect();
        // Apply the CX/CZ layers to P first to learn whether an S is needed
        // (P has Y on the pivot), then to every row.
        self.conjugate_layers(p, kw, kb, &cx_mask, &cz_mask, &active);
        let needs_s = self.zs[p * words + kw] & kb != 0;
        // Rows that are zero on every word involved do not change.
        let ow = self.occ_words;
        let mut touched = vec![0u64; ow];
        for &w in active.iter().chain([&kw]) {
            touched[w / 64] |= 1 << (w % 64);
        }
        for row in 0..2 * self.n {
            let occ = &self.occ[row * ow..(row + 1) * ow];
            if row != p && occ.iter().zip(&touched).all(|(o, t)| o & t == 0) {
                continue;
            }
            if row != p {
                self.conjugate_layers(row, kw, kb, &cx_mask, &cz_mask, &active);
            }
            let (xi, zi) = (row * words + kw, row * words + kw);
            let mut x = self.xs[xi] & kb != 0;
            let mut z = self.zs[zi] & kb != 0;
            let mut sign = self.sign(row);
            if needs_s {
                // S† R S on qubit k: X -> -Y, Y -> X.
                sign ^= x && !z;
                z ^= x;
            }
            // H R H.
            sign ^= x && z;
            std::mem::swap(&mut x, &mut z);
            if b {
                // X R X: flips Z and Y.
                sign ^= z;
            }
            self.flip_sign(row, sign != self.sign(row));
            self.xs[xi] = (self.xs[xi] & !kb) | if x { kb } else { 0 };
            self.zs[zi] = (self.zs[zi] & !kb) | if z { kb } else { 0 };
            for &w in active.iter().chain([&kw]) {
                self.update_occ(row, w);
            }
        }
    }

    /// Conjugate row `row` by `Π CX(k, j)` over `cx_mask`, then by
    /// `Π CZ(k, j)` over `cz_mask`, in closed form: the commuting gates'
    /// combined phase is a function of popcounts.
    fn conjugate_layers(
        &mut self,
        row: usize,
        kw: usize,
        kb: u64,
        cx_mask: &[u64],
        cz_mask: &[u64],
        active: &[usize],
    ) {
        let base = row * self.words;
        let xk = self.xs[base + kw] & kb != 0;
        let z0 = self.zs[base + kw] & kb != 0;
        // CX layer: x_j ^= x_k, z_k ^= z_j, sign per the CHP CX rule applied
        // to each gate in turn.
        let mut a = 0u32; // popcount(z & J)
        let mut c = 0u32; // popcount(z & J & (x if z0 else !x))
        for &w in active {
            let (x, z) = (self.xs[base + w], self.zs[base + w]);
            let zj = z & cx_mask[w];
            a += zj.count_ones();
            c += (zj & if z0 { x } else { !x }).count_ones();
        }
        if xk {
            let pairs = a * a.wrapping_sub(1) / 2;
            self.flip_sign(row, (c + pairs) & 1 == 1);
            for &w in active {
                self.xs[base + w] ^= cx_mask[w];
            }
        }
        if a & 1 == 1 {
            self.zs[base + kw] ^= kb;
        }
        // CZ layer: z_j ^= x_k, z_k ^= x_j, sign per the CZ rule.
        let zk = self.zs[base + kw] & kb != 0;
        let mut px = 0u32; // popcount(x & J')
        let mut xz = 0u32; // popcount(x & z & J')
        for &w in active {
            let (x, z) = (self.xs[base + w], self.zs[base + w]);
            let xj = x & cz_mask[w];
            px += xj.count_ones();
            xz += (xj & z).count_ones();
        }
        if xk {
            let pairs = px * px.wrapping_sub(1) / 2;
            self.flip_sign(row, (xz + if zk { px } else { 0 } + pairs) & 1 == 1);
            for &w in active {
                self.zs[base + w] ^= cz_mask[w];
            }
        }
        if px & 1 == 1 {
            self.zs[base + kw] ^= kb;
        }
    }

    // -- near-Clifford support --------------------------------------------

    /// Prepend a Clifford `V` to the state's circuit (`C -> C V`), for the
    /// near-Clifford backend's virtual frame: every row `inv(g)` becomes
    /// `V† inv(g) V`, a column operation. Keeps the column layout from then
    /// on.
    pub(crate) fn prepend(&mut self, op: Clifford) {
        self.layout = Layout::Columns;
        self.use_columns();
        let cw = self.column_words;
        let column = |q: usize| q * cw..(q + 1) * cw;
        match op {
            Clifford::H(q) => {
                // H X H = Z, H Y H = -Y.
                for i in column(q) {
                    let (x, z) = (self.xs[i], self.zs[i]);
                    self.signs[i - q * cw] ^= x & z;
                    self.xs[i] = z;
                    self.zs[i] = x;
                }
                self.refresh_cache(q);
            }
            Clifford::S(q) => {
                // S† X S = -Y, S† Y S = X.
                for i in column(q) {
                    let (x, z) = (self.xs[i], self.zs[i]);
                    self.signs[i - q * cw] ^= x & !z;
                    self.zs[i] = z ^ x;
                }
            }
            Clifford::Sdg(q) => {
                // S X S† = Y, S Y S† = -X.
                for i in column(q) {
                    let (x, z) = (self.xs[i], self.zs[i]);
                    self.signs[i - q * cw] ^= x & z;
                    self.zs[i] = z ^ x;
                }
            }
            Clifford::X(q) => {
                for i in column(q) {
                    self.signs[i - q * cw] ^= self.zs[i];
                }
            }
            Clifford::Z(q) => {
                for i in column(q) {
                    self.signs[i - q * cw] ^= self.xs[i];
                }
            }
            Clifford::Y(q) => {
                for i in column(q) {
                    self.signs[i - q * cw] ^= self.xs[i] ^ self.zs[i];
                }
            }
            Clifford::CX(c, t) => {
                self.cx_columns(c, t);
                self.refresh_cache(t);
            }
            Clifford::CZ(a, b) => {
                // CZ X_a CZ = X_a Z_b.
                for i in 0..cw {
                    let (xa, za, xb, zb) = (
                        self.xs[a * cw + i],
                        self.zs[a * cw + i],
                        self.xs[b * cw + i],
                        self.zs[b * cw + i],
                    );
                    self.signs[i] ^= xa & xb & (za ^ zb);
                    self.zs[a * cw + i] = za ^ xb;
                    self.zs[b * cw + i] = zb ^ xa;
                }
            }
            Clifford::Swap(a, b) => {
                for i in 0..cw {
                    self.xs.swap(a * cw + i, b * cw + i);
                    self.zs.swap(a * cw + i, b * cw + i);
                }
                self.refresh_cache(a);
                self.refresh_cache(b);
            }
        }
    }

    /// `C† P C` for a physical Pauli `P` given as `(qubit, 'X' | 'Y' | 'Z')`
    /// factors on distinct qubits: a Pauli on the virtual qubits, returned
    /// as its sign (`true` for minus) and x and z bits (64 qubits a word,
    /// `(1, 1)` meaning Y).
    pub(crate) fn pull_back(&self, pauli: &[(usize, char)]) -> (bool, Vec<u64>, Vec<u64>) {
        let words = self.words;
        let (mut xs, mut zs) = (vec![0u64; words], vec![0u64; words]);
        let mut exponent = 0u32;
        let multiply = |row: usize, xs: &mut [u64], zs: &mut [u64]| -> u32 {
            let (rx, rz) = self.row(row);
            let phase = product_phase(xs, zs, &rx, &rz);
            for w in 0..words {
                xs[w] ^= rx[w];
                zs[w] ^= rz[w];
            }
            phase + 2 * self.sign(row) as u32
        };
        for &(q, p) in pauli {
            match p {
                'X' => exponent += multiply(self.x_row(q), &mut xs, &mut zs),
                'Z' => exponent += multiply(self.z_row(q), &mut xs, &mut zs),
                'Y' => {
                    // Y = i X Z.
                    exponent += multiply(self.x_row(q), &mut xs, &mut zs);
                    exponent += multiply(self.z_row(q), &mut xs, &mut zs);
                    exponent += 1;
                }
                other => panic!("not a Pauli: {other}"),
            }
        }
        debug_assert!(
            exponent.is_multiple_of(2),
            "the pulled-back Pauli must be Hermitian"
        );
        ((exponent / 2) % 2 == 1, xs, zs)
    }

    /// `<psi| P |psi>` for a Pauli string: `+1`, `-1` or `0`.
    pub fn expectation(&self, pauli: &PauliString) -> i32 {
        assert_eq!(
            pauli.xs.len(),
            self.n,
            "Pauli string length must equal the number of qubits"
        );
        let words = self.words;
        let mut xs = vec![0u64; words];
        let mut zs = vec![0u64; words];
        // Accumulate the phase exponent (powers of i) of the running product.
        let mut exponent = 2 * pauli.negative as u32;
        let multiply = |row: usize, xs: &mut [u64], zs: &mut [u64]| -> u32 {
            let (rx, rz) = self.row(row);
            let phase = product_phase(xs, zs, &rx, &rz);
            for w in 0..words {
                xs[w] ^= rx[w];
                zs[w] ^= rz[w];
            }
            phase + 2 * self.sign(row) as u32
        };
        for j in 0..self.n {
            match (pauli.xs[j], pauli.zs[j]) {
                (true, false) => exponent += multiply(self.x_row(j), &mut xs, &mut zs),
                (false, true) => exponent += multiply(self.z_row(j), &mut xs, &mut zs),
                (true, true) => {
                    // Y = i X Z.
                    exponent += multiply(self.x_row(j), &mut xs, &mut zs);
                    exponent += multiply(self.z_row(j), &mut xs, &mut zs);
                    exponent += 1;
                }
                (false, false) => {}
            }
        }
        if xs.iter().any(|&w| w != 0) {
            return 0;
        }
        debug_assert!(exponent.is_multiple_of(2));
        if (exponent / 2).is_multiple_of(2) {
            1
        } else {
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    /// Random layers of gates on disjoint qubits.
    fn random_layer(rng: &mut impl Rng, n: usize) -> Vec<Clifford> {
        let mut qubits: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            qubits.swap(i, rng.random_range(0..=i));
        }
        let mut ops = Vec::new();
        let mut rest = &qubits[..];
        while !rest.is_empty() {
            let (a, b) = (rest[0], rest.get(1).copied());
            let pick = rng.random_range(0..9);
            match (pick, b) {
                (6..=8, Some(b)) => {
                    ops.push(
                        [Clifford::CX(a, b), Clifford::CZ(a, b), Clifford::Swap(a, b)][pick - 6],
                    );
                    rest = &rest[2..];
                }
                _ => {
                    ops.push(
                        [
                            Clifford::H(a),
                            Clifford::S(a),
                            Clifford::Sdg(a),
                            Clifford::X(a),
                            Clifford::Y(a),
                            Clifford::Z(a),
                        ][pick % 6],
                    );
                    rest = &rest[1..];
                }
            }
        }
        ops
    }

    #[test]
    fn parallel_layers_match_sequential_gates() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut rng = rand_pcg::Pcg64::seed_from_u64(11);
        for (n, layout) in [
            (300, Layout::Adaptive),
            (700, Layout::Columns),
            (130, Layout::Rows),
        ] {
            let (mut parallel, mut sequential) = (Tableau::new(n), Tableau::new(n));
            for t in [&mut parallel, &mut sequential] {
                t.set_layout(layout);
            }
            for round in 0..12 {
                for _ in 0..3 {
                    let ops = random_layer(&mut rng, n);
                    pool.install(|| parallel.apply_layer(&ops));
                    for &op in &ops {
                        apply_gate(&mut sequential, op);
                    }
                }
                // Measurements make rows sparse again and, with
                // Layout::Columns, leave the tableau in the column layout.
                for _ in 0..n / 4 {
                    let (q, coin) = (rng.random_range(0..n), rng.random_bool(0.5));
                    let a = parallel.measure_with(q, || coin);
                    let b = sequential.measure_with(q, || coin);
                    assert_eq!(a, b, "n {n} round {round}");
                }
                for row in 0..2 * n {
                    assert_eq!(
                        parallel.row(row),
                        sequential.row(row),
                        "n {n} round {round} row {row}"
                    );
                    assert_eq!(
                        parallel.sign(row),
                        sequential.sign(row),
                        "n {n} round {round} row {row}"
                    );
                }
            }
        }
    }

    #[test]
    fn transposes_bit_matrices() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(9);
        for (rows, cols) in [(1usize, 1usize), (64, 64), (130, 70), (7, 600), (1100, 513)] {
            let words = cols.div_ceil(64);
            let mut src: Vec<u64> = (0..rows * words).map(|_| rng.random::<u64>()).collect();
            if cols % 64 != 0 {
                // Padding bits must be zero.
                for r in 0..rows {
                    src[r * words + words - 1] &= (1u64 << (cols % 64)) - 1;
                }
            }
            let (mut t, mut back) = (Vec::new(), Vec::new());
            transpose_bits(&src, rows, cols, &mut t);
            for r in 0..rows {
                for c in 0..cols {
                    let bit = src[r * words + c / 64] >> (c % 64) & 1;
                    assert_eq!(t[c * rows.div_ceil(64) + r / 64] >> (r % 64) & 1, bit);
                }
            }
            transpose_bits(&t, cols, rows, &mut back);
            assert_eq!(back, src, "{rows}x{cols}");
        }
    }

    #[test]
    fn bit_sliced_phase_matches_direct_formula() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(5);
        for words in [1, 2, 3, 7] {
            for _ in 0..2000 {
                let mut v = || -> Vec<u64> { (0..words).map(|_| rng.random::<u64>()).collect() };
                let (x1, z1, x2, z2) = (v(), v(), v(), v());
                let expected = product_phase(&x1, &z1, &x2, &z2);
                let (mut a, mut b) = (x1.clone(), z1.clone());
                let got = multiply_in_place(&mut a, &mut b, &x2, &z2);
                assert_eq!(got, expected);
                let xor: Vec<u64> = x1.iter().zip(&x2).map(|(p, q)| p ^ q).collect();
                assert_eq!(a, xor);
            }
        }
    }
}
