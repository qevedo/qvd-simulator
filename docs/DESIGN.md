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

## 8. Not done yet

- **Calibration.** Fusion width (4) and region size (2^14 blocks) are fixed
  defaults chosen from the measurements above. A per-machine calibration
  step and a cost-based fuser (Aer-style dynamic programming) would adapt
  them.
- **Smarter stage planning.** The lookahead is greedy; Atlas finds a
  minimal number of stages with an ILP.
- **Other backends.** MPS, near-Clifford and Pauli propagation backends,
  and selection between them beyond the Clifford check, are the roadmap in
  the README. With the stabilizer backend, they are what takes structured
  circuits past the 32-qubit wall.
- **Fewer cache misses in the stabilizer reference shot.** Its gates are
  cache misses in a 208 MB tableau at d = 101. Storing each row's x and z
  words side by side would halve them. Random measurements (the first and
  last rounds of a surface code) still collapse one at a time.
- **Encoded or out-of-core states.** A 2-byte encoding would reach 34
  qubits here at some precision cost; SSD-backed states need a dedicated
  multi-terabyte NVMe array to be practical.
