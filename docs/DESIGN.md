# Design and measurements

This document records how the research in
[`reports/Fastest quantum circuit simulator design.md`](../reports/Fastest%20quantum%20circuit%20simulator%20design.md)
turned into code, and what each decision measured on the development
machine: Intel i9-14900K (8 P-cores with hyper-threading + 16 E-cores, 32
threads, AVX2 + FMA, no AVX-512), 62 GB DDR5, Linux.

## 1. The budget is memory bandwidth

`examples/bandwidth.rs` measures sustained bandwidth for an in-place
read-modify-write sweep (the access pattern of a gate) and a STREAM triad:

| Threads | Read-modify-write | Triad |
|---|---|---|
| 8 P-cores | 78.9 GB/s | 76.2 GB/s |
| 16 P-core threads | 78.3 GB/s | 79.1 GB/s |
| 24 cores (P + E) | 77.2 GB/s | 67.2 GB/s |
| 32 threads | 76.6 GB/s | 65.0 GB/s |

Eight P-cores saturate memory; E-cores add nothing to streaming and cost
10–15% on the triad. One pass over a 30-qubit single-precision state (8 GiB
read + 8 GiB written) therefore costs at least **0.218 s**, and every
optimisation below is about doing fewer such passes, or doing more work
per pass for free.

## 2. Layout and kernels (`simd.rs`, `kernels.rs`, `state.rs`)

**Layout.** Amplitudes are stored in 64-byte blocks: 8 real parts then 8
imaginary parts in single precision (4 + 4 in double), as in qsim. A
complex multiply-accumulate on split registers is exactly four FMAs. Bit
`q` of an amplitude index is qubit `q`; the lowest 3 qubits (2 in double)
select a lane inside a register, the rest select blocks.

**Kernel.** For a gate on `k` qubits, targets above the lane bits select
`NH = 2^high` blocks per iteration; targets inside the register are handled
by `NT = 2^low` lane-permuted copies of each input register
(`vpermps`), with per-lane coefficient vectors precomputed per gate. The
loop index is spread over the remaining bits with BMI2 `pdep`; controls on
block bits are fixed bits of the loop index, so each control halves the
memory touched. Every `(NH, NT)` shape is a separate monomorphised kernel.
Diagonal gates get a kernel with one complex multiply per amplitude.

**Measured** (`examples/kernels.rs`, 30 qubits, single precision; time
relative to one 0.218 s memory sweep):

| Gate | Cost (sweeps) |
|---|---|
| dense, 1–4 qubits, any positions | 1.02–1.08 |
| dense, 5 qubits | 2.09 (8 P-cores), 1.60 (32 threads) |
| dense, 6 qubits | 5.17 (8 P-cores), 3.23 (32 threads) |
| diagonal, 4 qubits | 1.02 |
| CX | 0.51 |

Kernels up to four qubits run at 97% of memory bandwidth. Double precision
costs the same per byte (a 29-qubit double state behaves like a 30-qubit
single state).

**Two kernel fixes found by measurement.**

- *FMA latency.* The first kernel summed every term of an output into one
  accumulator: a chain of dependent FMAs at ~4 cycles each. Splitting each
  sum into four accumulators and computing two output rows per pass over the
  inputs (`dense_row_pair`) raised in-cache throughput 1.4–2×.
- *Which in-cache number matters.* A 4-qubit gate on high qubits looked 3×
  slower than on low qubits when the state lived in the shared L3 (8 MiB):
  16 concurrent streams per thread defeat the prefetchers. Target spacing
  made no difference (tested from adjacent qubits to qubits 3, 8, 13, 19).
  On an L2-resident region, the regime cache blocking runs in, every shape
  reaches 8–10 G FMA/s per P-core, about 2 FMAs per cycle, the core's peak
  (`examples/l2.rs`).

## 3. Gate fusion (`fusion.rs`)

Because a 4-qubit gate costs the same memory pass as a 1-qubit gate,
merging gates into 4-qubit unitaries cuts passes directly. The fuser
follows only true dependencies: a gate joins a group if no earlier,
not-yet-fused gate shares one of its qubits, so gates on unrelated qubits in
between do not stop it. Groups first grow along gates connected to them,
then fill spare room with unrelated gates (filling first wasted room on
qubits the chain never needed).

Measured on a 28-qubit, depth-20 random circuit (830 gates), without cache
blocking:

| Max fused qubits | Fused gates | Time |
|---|---|---|
| 1 | 812 | 39.1 s |
| 2 | 270 | 15.3 s |
| 3 | 137 | 8.1 s |
| 4 | 69 | 4.2 s |
| 5 | 50 | 5.0 s |

Four is the optimum: five-qubit kernels are compute-bound. This matches
qsim's documented advice (f = 4 for large circuits) and the research's
roofline argument.

## 4. Cache-blocked stages (`blocking.rs`)

The state is divided into regions of 2^14 blocks (1 MiB). Physical qubits
inside a region are *local*. A *stage* is a set of gates that can run with
the current local qubits: non-diagonal targets must be local, while
diagonal gates and controls may use any qubit, since within a region the
other qubits are fixed. Each thread applies all of a stage's gates to one
region while it stays in L2, so a stage costs one memory pass.

Stages are extracted from the *unfused* gate list by dependency: every gate
whose predecessors are done and whose qubits fit is taken, wherever it sits
in the program. When nothing more fits, a lookahead simulates the next
stage with a growing set of wanted qubits, and one pass swaps the wanted
global qubits with unneeded local ones. The state records the new
logical-to-physical layout instead of moving data back, so every public
method still speaks logical qubits. Only then is each stage fused, never
merging a global control or diagonal qubit into a dense matrix.

Measured on the same 28-qubit circuit with fusion up to 4 qubits:

| Scheduling | Memory passes | Time (32 threads) |
|---|---|---|
| No blocking | 69 | 4.28 s |
| Stages cut from the fused list in program order | 39 | 3.22 s |
| Dependency-aware stages with lookahead | 9 | 1.96 s |

The first, program-order version stopped a stage at the first fused gate
that needed a global qubit. Taking gates by dependency instead brought
passes from 39 to 9. With memory no longer the limit, the work becomes
compute-bound, which is where the E-cores finally help: 32 threads beat
16 P-core threads (1.96 s vs 2.61 s). With blocking, fusion widths 2–4
perform alike (1.70–1.74 s), since fusion now saves compute rather than
memory.

## 5. Measurement (`measure.rs`, `simulator.rs`)

Sampling makes two passes over the state for any number of shots: parallel
per-chunk probability sums, then sorted uniforms walked through the chunks
in parallel. Counts are formed from integer keys sorted in parallel, and
only distinct outcomes are formatted as strings. All sums accumulate in
double precision. Circuits whose measurements all come at the end are
simulated once; mid-circuit measurements and resets collapse the state and
run shot by shot.

| Case | Time |
|---|---|
| 28 qubits, 1M shots | 0.60 s sampling (1.7 s gates) |
| 30 qubits, 100k shots | 0.25 s sampling |
| 24 qubits, 10M shots, 6.25M distinct outcomes | 3.4 s (mostly building the result map) |

## 6. Correctness

- 800 random gates per precision on random qubit sets, controls and
  diagonals, against a plain reference simulator (`tests/kernels.rs`).
- Random circuits over the whole gate set at every fusion width, and with
  tiny regions that force many stages, global controls, global diagonals and
  relabeling passes (`tests/simulator.rs`).
- Sampling statistics, mid-circuit measurement, reset and layout-permuted
  sampling.
- Against Qiskit Aer through OpenQASM: random circuit and QFT amplitudes
  agree to 7×10⁻¹³ and 8×10⁻¹⁴ in double precision, including the global
  phase, so gate conventions match Qiskit's exactly.

## 7. Stabilizer backend (`stabilizer/`)

Circuits of Clifford gates (H, S, CX, CZ, Paulis, SWAP, √X, and rotations
by multiples of π/2), measurements and resets keep the state a stabilizer
state, which `O(n²)` bits describe exactly. `Backend::Auto` sends such
circuits here (`stabilizer::lower` decides, gate by gate); `stabilizer::sample`
returns bit-packed shots, 64 per word.

**Two passes, as in Stim.** One *reference* shot runs on a stabilizer
tableau and records an outcome for every measurement. All other shots are
Pauli *frames* relative to it: `x`/`z` bits per qubit, 64 shots per word,
pushed through the gates with a few XORs. A frame's X component flips a
measurement; its Z component is randomised at the start and after each
measurement and reset, which makes exactly the random measurements come out
random (Gidney, *Quantum* 5, 497, 2021).

**Inverse tableau (`tableau.rs`).** For each generator `g` among
`X_0…X_{n-1}, Z_0…Z_{n-1}` the tableau stores `C† g C` as `x`/`z` bit rows
plus a sign. A gate is a product of two rows, with the phase accumulated
bit-sliced over 64 qubits at a time. A Z measurement is deterministic when
row `Z_q` has no X bits, and its sign is then the outcome. Otherwise one of
two collapse methods runs. In the row layout, every row is conjugated by a
closed-form layer of CX/CZ gates. In the column layout, Stim's method
prepends CX gates, then H, then X: each is a column operation.

**What made the d = 101 surface code fast.** The benchmark is a
rotated surface code of distance 101 (20,401 qubits, 101 rounds:
5,110,600 gates and 1,040,401 measurements), sampled 10,000 times. The
first version of the generator measured each ancilla straight after its
closing H (`h a; measure a; h b; measure b; …`), which is the hard case
for every layout policy:

| Step | Time |
|---|---|
| Row-major tableau, collapse = a pass over all 40,802 rows | 37.3 s (with 64 shots) |
| Transpose to columns for runs of random measurements | 19.0 s |
| Cache the x bits of 64 rows while in columns | 16.6 s |
| Gates in the column layout as sparse row updates; adaptive layout switching | 11.2 s |
| Per-row bitmap of nonzero words: row products touch only those | 4.3 s |
| Frames in parallel blocks on the P-cores | 3.5 s |
| Generator emits the H layer, then the measurement layer | 2.4 s |
| Reference shot: gates on disjoint qubits in parallel | **1.5 s** (under load, see below) |

The steps, in order:

- **Columns for collapses.** Each random measurement in the row layout
  touches a few words of every one of the `2n` rows, which is a strided
  pass over 208 MB. Transposing once (64×64 bit-block transposes, in
  parallel) turns the collapses into column operations of `O(n/64)` words
  each.
- **Row cache.** Locating a row's X bits in the column layout reads one
  word per column. A transposed copy of the x bits of the current 64-row
  group, patched after each column operation, serves a run of consecutive
  measurements instead.
- **Adaptive layout.** Interleaved H gates forced a transposition back to
  rows before every gate. Now a gate in the column layout gathers its two
  rows and flips only the bits that change. A running balance of what the
  other layout would have saved triggers a transposition once it exceeds
  the transposition's cost. Measurement-heavy stretches stay in columns and
  gate-heavy ones go back to rows. `Layout::Rows` and `Layout::Columns`
  force one layout (the tests run all three).
- **Sparse rows.** In a code with local checks, inverse-tableau rows have
  weight `O(d)`: about 2 nonzero words out of 319. Each row keeps a bitmap
  of its nonzero words, so a row product reads and writes only the source
  row's nonzero words, and a row-layout collapse skips rows that are zero
  on the words it changes. Dense rows fall back to the vectorised loop.
- **Parallel frames.** The reference pass records the circuit as steps.
  Frame blocks of up to 2,048 shots then walk the steps in parallel, in
  32k-step segments so that the step list stays in cache. Each block's
  frames fit in L2. The blocks synchronise after each segment, so a pool
  of P-cores is faster than all 32 threads (frames 1.0 s vs 3.8 s at
  d = 101). Each 64-shot word has its own random stream, which makes
  results depend only on the seed and not on the thread count.
- **Parallel reference shot.** The reference pass was then most of the
  time: about 2.2 s, all cache misses in the 208 MB tableau, on one core.
  Consecutive gates on pairwise disjoint qubits commute and touch disjoint
  rows, so the reference pass collects them into layers (a layer ends at a
  gate that reuses a qubit, or at a measurement or reset) and applies each
  layer in parallel. Each row's words and occupancy bits belong to one gate
  of the layer; sign bits, 64 rows to a word, are flipped with atomic XORs.
  The sequential and parallel paths share the same row code, and the
  result is identical to applying the gates in order. The reference shot
  at d = 101 drops to 0.75 s. This step was measured while a virtual
  machine kept about 14 cores busy; under that same load, the previous
  build took 2.57 s and this one 1.52 s (best of five, interleaved).
  Checking runs of determined measurements in parallel as well made no
  measurable difference, so the measurements stay sequential.

**Against Stim and Qiskit Aer.** Same OpenQASM circuits (layered
generator), 10,000 shots, timed by `benchmarks/compare_stim.py`. Stim 1.16
is single-threaded; qvd is shown on 8 P-cores and on one thread. Stim is
timed two ways: `reference_sample()` plus a `FlipSimulator` with
measurement-major output (the fair comparison), and
`compile_sampler().sample(bit_packed=True)`.

| Distance | Qubits | Gates | Measurements | qvd, 8 P-cores (1 thread) | Stim 1.16, reference + `FlipSimulator` | Stim `sample()` | Qiskit Aer 0.17 stabilizer |
|---|---|---|---|---|---|---|---|
| 11 | 241 | 6,160 | 1,441 | **0.001 s** (0.003 s) | 0.004 s | 0.11 s | 10.8 s |
| 21 | 881 | 44,520 | 9,681 | **0.010 s** (0.018 s) | 0.036 s | 0.32 s | 636 s |
| 51 | 5,201 | 652,800 | 135,201 | **0.14 s** (0.29 s) | 0.45 s | 6.9 s | — |
| 101 | 20,401 | 5,110,600 | 1,040,401 | **1.5 s** (4.4 s) | 12.4 s | 131 s | — |

These runs shared the machine with a virtual machine. Stim, Aer and qvd on
one thread ran while it used about six cores; qvd on 8 P-cores was
re-measured after the parallel reference shot while it used about 14, which
can only have slowed qvd down. (One thread takes the same sequential path as
before.) At d = 51 Stim spends 0.17 s on its reference shot and the rest in
frames; at d = 101 its reference shot alone takes 8.0 s, against 0.75 s
for qvd's. `sample()`, which returns shot-major bits, is 9–27× slower than
Stim's flip simulator here. Single-threaded, qvd is 1.3–2.8× faster than
Stim; with 8 P-cores it is 3–8× faster.

**Correctness** (`tests/stabilizer.rs`): expectation values of random Pauli
strings and deterministic measurement outcomes against the dense simulator
on random Clifford circuits. Forced collapse outcomes are checked against
dense projection under every layout policy, including frequent switching.
Sampled distributions are checked against exact branching distributions
(5σ). Further tests cover 1,000-qubit GHZ sampling, the surface code's
stabilizers repeating from round to round (d = 3, 5, 7), reproducibility
across thread counts, parallel gate layers against the same gates in order
(from both layouts), and 64×64 transposes.

## 8. Matrix product state backend (`mps/`)

Past the state vector's memory, circuits with limited entanglement still
fit as a matrix product state: one tensor `A[l, p, r]` per qubit, joined by
bonds of dimension χ, `O(n χ²)` memory. `Backend::Auto` uses it for
non-Clifford circuits whose state vector would not fit in 80% of memory;
`Backend::Mps` forces it.

**Gates.** A one-qubit gate updates one site and keeps its canonical form.
A gate on `k` qubits moves them next to each other with adjacent SWAPs,
around the qubit at the median site, and leaves them there (lazy
permutation, as quimb's `CircuitPermMPS`; swapping them back after each
gate made the fidelity neither better nor worse on grid circuits and costs
twice the SWAPs). Their sites are then contracted into one tensor, the
gate's matrix is applied to its physical index, and the block is split back
by `k - 1` SVDs. The state is kept in mixed canonical form around an
orthogonality center (moved with QR), so each SVD sees the exact Schmidt
coefficients of its bond: keeping the largest χ, and dropping the smallest
while their squares sum to at most the threshold, is the optimal cut. The
product of the kept weights is the fidelity estimate. On a 4×5 grid it
tracks the true overlap with the exact state within a factor of two down to
10⁻⁹, and on 1D circuits it matches quimb's estimate to four digits.

**Parallel layers, only where exact.** Gates on disjoint sites can run in
parallel in a different form: every site right-canonical and every bond's
Schmidt coefficients stored. A step then needs only its own sites and the
coefficients `Λ` of the bond to its left. The block is split by SVDs of
`diag(Λ) θ` from the right, and its first site becomes `θ` times the adjoint
of the new rest, so no coefficient is ever inverted (Hastings, PRB 79,
165102). The planner routes every gate first, as bookkeeping that emits SWAP
steps. Each step is then classified with an upper bound on its new bonds:
neighbouring bonds times `2^j`, and the old bond times the gate's operator
Schmidt rank (2 for CZ and CX, 4 for SWAP). Runs of steps that cannot reach
χ_max are scheduled as early as their sites allow and run in parallel
layers. Steps that may truncate run one at a time, in circuit order, with
the moving center.

Running truncating steps in parallel was tried and rejected. It truncates
all bonds of a layer at once, each from the state before the others, and it
needs a full SVD sweep after every truncating layer to restore the form.
Even then, on the 64-qubit depth-20 circuit at χ = 256 the fidelity fell
from 1.8×10⁻⁵ to 5.7×10⁻⁹, and on small circuits the true overlap fell by
2–5×. As built, results under truncation equal the sequential ones (the same
fidelity estimates as quimb's), apart from the choice among exactly equal
singular values at a cut, which these √X/CZ circuits often have.

**Small products.** faer dispatches every matrix product to the thread pool.
For the χ = 1 or 2 tensors that dominate early layers and QFT-like circuits,
that overhead was most of the time: a 100-qubit QFT took 0.92 s on the
default 32-thread pool, 0.10 s once small products (under 2¹⁸
multiply-adds) ran sequentially. The operator Schmidt rank of each gate,
needed for the bounds above, is a 4×4 Gaussian elimination rather than an
SVD call for the same reason.

Against the previous version, same machine and load, best of three:

| Circuit | Before | After |
|---|---|---|
| Random 1D, 64 qubits, depth 10 (exact) | 0.070 s | **0.018 s** |
| Random 1D, 40 qubits, depth 16 (exact) | 2.31 s | **0.54 s** |
| Random 1D, 64 qubits, depth 20, χ = 256 | 15.2 s | **8.5 s** (same fidelity) |
| Random 1D, 100 qubits, depth 16, χ = 128 | 4.08 s | **1.90 s** (same fidelity) |
| Random 8×8 grid, depth 10, χ = 64 | 6.28 s | 6.28 s |
| QFT, 100 qubits | 0.084 s | **0.040 s** |

**Qubit order.** The bond between two sites is at most `2^(gates crossing
it)`, so the qubits are placed on sites to keep interacting qubits close.
Three orders compete: the circuit's own, reverse Cuthill–McKee, and the
spectral order (the Laplacian's Fiedler vector). The winner is the one with
the fewest gates across the worst cut, then across all cuts. Circuits whose
labels follow their geometry keep their order. Circuits whose labels do not
gain hugely:

| Circuit | Circuit's order | Chosen order |
|---|---|---|
| 1D, 64 qubits, depth 12, labels scrambled, χ = 64 | 5.2 s, fidelity 4×10⁻¹² | **0.028 s, exact** |
| 1D, 40 qubits, depth 16, labels scrambled, χ = 256 | 19.4 s, fidelity 1.2×10⁻⁴ | **0.42 s, exact** |
| 6×10 grid, depth 12, χ = 64 (columns first) | 6.4 s, fidelity 5×10⁻¹⁹ | 5.1 s, fidelity 7×10⁻¹⁷ |
| 8×8 grid, depth 10, labels scrambled, χ = 64 | fidelity 1.5×10⁻¹⁷ | fidelity 1.5×10⁻¹⁶ |

**SVD.** faer 0.24's thin SVD, which runs on the rayon pool (12.6 s vs 22 s
single-threaded for the 64-qubit depth-20 case). On one 128×128 matrix, finite
and well scaled but with nearly degenerate leading singular values, it
returned NaN factors. Every SVD is therefore checked; the SVD of the
adjoint, then a Hermitian eigendecomposition of the Gram matrix, are the
fallbacks, and the unit tests run all three methods.

When χ_max binds on a large bond (at least 256 rows and columns), the
eigendecomposition of the smaller Gram matrix is used instead: 46 ms
against 106 ms for a 512×512 SVD, with singular values within 10⁻¹⁴. The
other factor, `a` times the eigenvectors divided by the singular values, is
accurate to about `10⁻¹⁶ (s_max/s)²`. It is therefore accepted only if every
kept singular value is at least 10⁻³ of the largest. Otherwise the SVD runs,
and that bond skips the Gram attempt for its next 32 splits. On the capped
benchmarks this saves 30–35% (64 qubits, depth 20: 7.1 s → 5.0 s;
100 qubits: 1.47 s → 1.0 s) with identical fidelity.

**Sampling.** With every site but the first right-canonical, a shot is drawn
site by site from the left, keeping only a running left vector. The vectors
of a batch of 256 shots form a matrix, so each site costs two matrix
products: 1000 shots of the 64-qubit χ = 256 state take 0.7 s instead of
2.8 s shot by shot. Each batch has its own random stream, so results depend
only on the seed. Circuits with mid-circuit measurements or resets run the
gates before the first one once, then each shot continues from a copy.

**Against Qiskit Aer and quimb** (same QASM, same truncation, 1000 shots, 8
threads; `benchmarks/compare_mps.py`):

| Circuit | χ cap | qvd | Qiskit Aer 0.17 MPS | quimb 1.15 `CircuitPermMPS` |
|---|---|---|---|---|
| Random 1D, 64 qubits, depth 10 | 256 | **0.018 s** | 0.25 s | 10.3 s (0.23 s gates) |
| Random 1D, 64 qubits, depth 20 | 256 | **8.5 s** | 114 s | 31.9 s (14.4 s gates) |
| Random 1D, 100 qubits, depth 16 | 128 | **1.9 s** | 18.0 s | 25.5 s (4.1 s gates) |
| Random 8×8 grid, depth 10 | 64 | **6.3 s** | 10.2 s | 18.4 s (5.8 s gates) |
| QFT, 100 qubits | 256 | **0.040 s** | 0.11 s | 21.7 s (3.1 s gates) |
| Random 1D, 64 qubits, depth 12, labels scrambled | 64 | **0.049 s** | 21.3 s | 26.6 s (10.5 s gates) |

qvd's gates are 2–2.6× faster than quimb's on the 1D circuits and about
equal on the grid. Aer is 1.6–14× slower in total, and quimb's total is
dominated by sampling shot by shot in Python. On scrambled labels the
qubit order makes the difference: exact at χ = 64 for qvd, 2×10⁻¹³ fidelity
for quimb. qvd's column was measured last, under heavier load than the
others. The research target, a 64-qubit depth-10 random circuit within
quimb's ~10 s, is met by a wide margin for the 1D circuit. The 2D grid is
the harder case, where the SWAP routing multiplies the SVDs.

**Correctness** (`tests/mps.rs`): exact states against the dense reference
for random circuits over the full gate set (non-adjacent and three-qubit
gates), measurement against dense projection, sampled distributions with
mid-circuit measurement and reset against exact branching, truncation
bounds and fidelity tracking, a 300-qubit GHZ state chosen automatically,
long-range routing, and reproducibility across thread counts.

## 9. Near-Clifford backend (`nearclifford/`)

Circuits that are mostly Clifford, with a few non-Clifford gates (T gates,
small rotations, Toffolis), are what error-correction and magic-state
experiments look like. Following Clifft (Chase & Labib, arXiv:2604.27058),
the state is kept as `|ψ> = C (|φ>_A ⊗ |0...0>_D)`: a Clifford frame `C`
(an inverse stabilizer tableau, reusing `stabilizer/`) around a dense
vector `φ` of the few *active* virtual qubits, with the other virtual
qubits *dormant* in `|0>`.

**Gates.** Clifford gates update only the frame. Every other gate is
lowered, exactly up to global phase, to Cliffords and Pauli rotations
`exp(-iθ/2 P)`:
- T, RZ, RX, RY and U become one to three rotations;
- CP, CRZ, CRX, CRY, RZZ and RXX become one to three rotations, some on two
  qubits;
- CH, CCX and CSWAP use their standard Clifford+T circuits;
- one-qubit unitaries use ZYZ Euler angles.

A rotation's Pauli, pulled back through the frame (`C† P C`), acts on
dormant qubits only through its X parts, since Z on `|0>` is `+1`. If it has
none, the rotation is applied to `φ` alone. Otherwise CX gates between the
dormant qubits gather those X parts onto one of them. The CX gates are
prepended to the frame (`C -> C W`), and `W` fixes `|0...0>`, so `φ` does
not change. That qubit then becomes active, doubling `φ`. So each
non-Clifford rotation activates at most one qubit.

**Measurements.** A Z measurement pulls `Z_q` back the same way:
- **X on a dormant qubit `d`:** the outcome is exactly 50/50. The collapsed
  state, `φ⊗|0> ± (P_A φ)⊗|1>`, equals `CP_A · H_d · X_d^b (φ⊗|0>)`, so the
  collapse is three Cliffords prepended to the frame, with no dense work.
- **Active qubits only:** it is measured on `φ`. A Clifford on the active
  qubits (applied to `φ` and prepended to the frame) then turns the measured
  Pauli into Z on one qubit, which is retired to dormant, halving `φ`.

The cost is exponential only in the number of active qubits. A dry run of
the frame alone finds the exact peak, because which qubits are active never
depends on measurement outcomes: `nearclifford::peak_active_qubits`.

**Sampling.** Shots are not simulated one by one:
- **Before the final measurements,** each measurement or reset is done once
  per branch of a measurement tree. Its probability splits the branch's
  shots binomially. Branches share `φ` copy-on-write.
- **The final measurements** use per-shot Pauli frames, as in Clifft. Each
  shot's state is `C P_s (φ⊗0)` for a Pauli `P_s`, kept 64 shots per word,
  conjugated by every Clifford prepended to the frame. A fixed outcome flips
  where `P_s` anticommutes with the measured Pauli. A 50/50 one collapses
  the shared state once and puts each shot's random outcome into its frame
  as `X_d`.
- **Only measurements on `φ` branch,** and only on their outcome in the
  shared state. The measurements that act on `φ` go first, each halving it,
  so the dense work is about `k·2^k` for any number of shots.

On 100 qubits with 23 active, 1000 shots took 214 s shot by shot, 38 s with
the tree alone and 1.3 s with the frames.

**Correctness** (`tests/nearclifford.rs`, plus unit tests of the dense
kernels):
- expectation values of random Pauli strings after random circuits over the
  full gate set, against the dense reference;
- measurement collapse against dense projection;
- sampled distributions with mid-circuit measurement and reset against exact
  branching (checked again with 2 million shots: worst deviation 2.1σ over
  48 outcomes);
- a 200-qubit GHZ state with a T gate, checked against its analytic
  expectation values;
- the dry-run peak against real runs, automatic selection, and
  reproducibility across thread counts.

**Against Qiskit Aer** (random Clifford+T circuits from
`library::random_clifford_t`, 1000 shots, 8 threads;
`benchmarks/compare_nearclifford.py`):

| Circuit | Peak active qubits | qvd near-Clifford | qvd state vector | Qiskit Aer state vector | Aer extended stabilizer (approximate) | Aer MPS |
|---|---|---|---|---|---|---|
| 24 qubits, depth 20, 10 T | 10 | **0.003 s** | 0.55 s | 2.9 s | 508 s | > 600 s |
| 30 qubits, depth 20, 12 T | 12 | **0.003 s** | 51 s | 210 s | > 600 s | > 600 s |
| 60 qubits, depth 20, 16 T | 16 | **0.013 s** | — | — | > 600 s | > 600 s |
| 100 qubits, depth 30, 24 T | 23 | **0.85 s** | — | — | over its 63-qubit limit | > 600 s |
| 200 qubits, depth 40, 20 T | 20 | **0.11 s** | — | — | over its 63-qubit limit | > 600 s |

Runs were stopped at 600 s. The state vector is double precision (qvd's
30-qubit run spends most of its time on the CX gates between random qubit
pairs, which fuse poorly). The CX layers make these circuits highly
entangled, which is why the MPS methods cannot follow. The dense work is exponential in the peak
number of active qubits (at most one per T gate), not in the qubit count,
so 200 qubits with 20 T gates (peak 20) cost less than 100 qubits with 24
(peak 23).

## 10. Not done yet

- **Calibration.** Fusion width (4) and region size (2^14 blocks) are fixed
  defaults chosen from the measurements above. A per-machine calibration
  step and a cost-based fuser (Aer-style dynamic programming) would adapt
  them.
- **Smarter stage planning.** The lookahead is greedy; Atlas finds a
  minimal number of stages with an ILP.
- **Other backends.** Pauli propagation for expectation values, and a
  cost-based selector calibrated on this machine (the current one compares
  rough work estimates), are the roadmap in the README.
- **Near-Clifford.** Gates after mid-circuit measurements multiply with the
  branches of the measurement tree. Pauli frames could carry them too (as
  in Clifft) when the gates are Clifford. Rotations whose angle is a
  multiple of π/2 inside decompositions could be frame updates rather
  than dense passes.
- **Faster capped MPS.** Gates that truncate still run one at a time
  (simultaneous truncation costs too much fidelity, see section 8). Each one
  is a Gram eigendecomposition or SVD; a randomized range finder would cut
  that further when χ_max is far below the bond's rank. For 2D circuits,
  routing that also moves qubits apart again (or a 2D tensor network) would
  reduce the SWAPs.
- **Fewer cache misses in the stabilizer reference shot.** Its gates are
  cache misses in a 208 MB tableau at d = 101. Storing each row's x and z
  words side by side would halve them. Random measurements (the first and
  last rounds of a surface code) still collapse one at a time.
- **Encoded or out-of-core states.** A 2-byte encoding would reach 34
  qubits here at some precision cost; SSD-backed states need a dedicated
  multi-terabyte NVMe array to be practical.
