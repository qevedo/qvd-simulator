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

## 7. Not done yet

- **Calibration.** Fusion width (4) and region size (2^14 blocks) are fixed
  defaults chosen from the measurements above. A per-machine calibration
  step and a cost-based fuser (Aer-style dynamic programming) would adapt
  them.
- **Smarter stage planning.** The lookahead is greedy; Atlas finds a
  minimal number of stages with an ILP.
- **Other backends.** Stabilizer, MPS, near-Clifford and Pauli propagation
  backends, and automatic selection between them, are the roadmap in the
  README. They are what takes structured circuits past the 32-qubit wall.
- **Encoded or out-of-core states.** A 2-byte encoding would reach 34
  qubits here at some precision cost; SSD-backed states need a dedicated
  multi-terabyte NVMe array to be practical.
