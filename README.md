# qevedo-simulator

[![CI](https://github.com/qevedo/qevedo-simulator/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/qevedo/qevedo-simulator/actions/workflows/ci.yml)

`qvd`, the simulator in this repository, is a Rust crate and the Python
package `qevedo-simulator` (imported as `qevedo.simulator`).

A quantum circuit simulator in Rust, built around one fact: on a modern CPU,
simulating a large circuit is limited by **memory bandwidth**, not by
arithmetic. Every gate on a 30-qubit state reads and writes 8–16 GiB, so the
fastest simulator is the one that makes the fewest passes over memory.

On an Intel i9-14900K (24 cores, AVX2, 62 GB RAM) it runs 3–4× faster than
Google's qsim and 7–12× faster than Qiskit Aer on the same circuits (details
below).

Clifford circuits go to a stabilizer backend instead, which has no 2ⁿ
memory wall: a distance-101 surface code (20,401 qubits, 5.1M gates, 1M
measurements) samples 10,000 shots in 1.5 s, 8× faster than Stim and
2.8× faster on a single thread (details below). Other circuits too large
for a state vector run on a matrix product state backend, exact while the
entanglement is low and approximate, with a fidelity estimate, beyond: it is
1.6–14× faster than the faster of Qiskit Aer's and quimb's MPS simulators on
the same circuits. Circuits with few non-Clifford gates run on a
near-Clifford backend (a Clifford frame around a small dense state, after
Clifft): a 200-qubit Clifford+T circuit with 20 T gates samples 1000 shots in
0.11 s, where every Qiskit Aer method either takes over 10 minutes or
does not support it.

## How it works

| Technique | What it does | Effect measured here |
|---|---|---|
| Blocked state layout | 8 real + 8 imaginary floats per 64-byte cache line (qsim's layout); the 3 lowest qubits live inside one AVX2 register | Complex multiply-add is 4 FMAs, no shuffles |
| AVX2 + FMA kernels | Dense kernels for 1–6-qubit gates, a diagonal kernel, and controls that skip the uncontrolled half of the state | 1–4-qubit gates run at **97% of DRAM bandwidth** |
| Gate fusion | Merges dependent gates into up to 4-qubit matrices, respecting only true dependencies | 830 gates → 69 kernels; 9.3× faster |
| Cache-blocked stages | Applies every gate of a stage to 1 MiB regions that stay in L2; relabels qubits instead of moving them back; diagonal gates and controls never break a stage | 69 memory passes → 9; 2.2× faster |
| Huge pages, parallel first touch, P/E-core-aware thread pools | `mmap` + `MADV_HUGEPAGE`, pinned rayon pools | Bandwidth saturated with 8 P-cores; all 32 threads for compute-bound stages |
| Two-pass sampling | Any number of shots costs two passes over the state | 1M shots of 28 qubits: 0.6 s |

The design follows a deep-research survey of the fastest simulators and the
literature, in [`reports/`](reports/) (with the underlying notes in
[`research_notes/`](research_notes/)). [docs/DESIGN.md](docs/DESIGN.md)
explains how each finding maps to the code and what was measured.

## Results

The recorded data (CSV), the scripts that produced it and the charts are in
[`benchmarks/`](benchmarks/), with the full method in
[benchmarks/README.md](benchmarks/README.md).

### State-vector simulation

![State-vector benchmark: qvd vs qsim vs Qiskit Aer](benchmarks/charts/statevector.svg)

Same OpenQASM circuits, same machine, best of 16 or 32 threads for each
simulator. Random circuits are depth-20 Google-style circuits (√X, √Y, √W
on every qubit and CZ on a brick pattern); QFT is the quantum Fourier
transform.

| Circuit | Qubits | Precision | qvd | qsim 0.22 | Qiskit Aer 0.17 | Speed-up vs. best |
|---|---|---|---|---|---|---|
| Random | 26 | single | **0.43 s** | 1.84 s | 5.51 s | 4.3× |
| Random | 28 | single | **2.09 s** | 7.63 s | 23.0 s | 3.7× |
| Random | 30 | single | **9.75 s** | 30.4 s | 89.8 s | 3.1× |
| Random | 26 | double | **0.91 s** | — | 8.52 s | 9.4× |
| Random | 28 | double | **4.26 s** | — | 34.5 s | 8.1× |
| Random | 29 | double | **9.60 s** | — | 68.7 s | 7.2× |
| QFT | 26 | single | **0.18 s** | 9.75 s | 3.18 s | 17.5× |
| QFT | 28 | single | **0.98 s** | 43.2 s | 12.1 s | 12.3× |
| QFT | 30 | single | **4.44 s** | 192 s | 45.7 s | 10.3× |

qvd's times include allocating the state; the others include sampling one
shot, and qsim's include converting the circuit in Python. qsim's CPU
backend is single precision only. Amplitudes were checked against Qiskit Aer
(maximum difference 7×10⁻¹³ in double precision).

### Clifford circuits

![Stabilizer benchmark: qvd vs Stim vs Qiskit Aer](benchmarks/charts/stabilizer.svg)

Rotated surface code memory experiments (d rounds), 10,000 shots, from the
same OpenQASM file. qvd on 8 P-cores (single-threaded in brackets); Stim is
single-threaded. Details are in
[docs/DESIGN.md](docs/DESIGN.md#7-stabilizer-backend-stabilizer).

| Distance | Qubits | Gates | Measurements | qvd, 8 P-cores (1 thread) | Stim 1.16, reference + `FlipSimulator` | Stim `sample()` | Qiskit Aer 0.17 stabilizer |
|---|---|---|---|---|---|---|---|
| 11 | 241 | 6,160 | 1,441 | **0.001 s** (0.003 s) | 0.004 s | 0.11 s | 10.8 s |
| 21 | 881 | 44,520 | 9,681 | **0.010 s** (0.018 s) | 0.036 s | 0.32 s | 636 s |
| 51 | 5,201 | 652,800 | 135,201 | **0.14 s** (0.29 s) | 0.45 s | 6.9 s | — |
| 101 | 20,401 | 5,110,600 | 1,040,401 | **1.5 s** (4.4 s) | 12.4 s | 131 s | — |

These runs shared the machine with a virtual machine using 6–14 cores, so
absolute times are a little high; the qvd 8-core column was measured under
the heavier load ([details](benchmarks/README.md#clifford-circuits)). Stim's `sample()` returns shot-major arrays and is much slower
here; its flip simulator returns measurement-major results, as qvd does,
and is the fairer comparison.

The stabilizer backend runs a Stim-style reference shot on an inverse
tableau and propagates bit-packed Pauli frames for all other shots. Four
additions matter at this scale:
- the tableau switches between row and column layouts depending on whether
  gates or random measurements dominate;
- rows keep a bitmap of their nonzero words, so products of the sparse rows
  of a local code touch a few words instead of 319;
- gates on disjoint qubits run in parallel in the reference shot;
- frames run in parallel, cache-sized blocks.

### Low-entanglement circuits (MPS)

![MPS benchmark: qvd vs Qiskit Aer vs quimb](benchmarks/charts/mps.svg)

1000 shots each, the same truncation in all three (at most χ singular
values per bond, discarded weight below 10⁻¹⁶), 8 threads. Random 1D
circuits are brick-pattern versions of the random circuits above; the grid
circuit applies CZ on all four coupler orientations of an 8×8 grid in turn.

| Circuit | χ cap | qvd | Qiskit Aer 0.17 MPS | quimb 1.15 `CircuitPermMPS` | Fidelity estimate |
|---|---|---|---|---|---|
| Random 1D, 64 qubits, depth 10 | 256 | **0.018 s** | 0.25 s | 10.3 s | 1 (exact, χ = 32) |
| Random 1D, 64 qubits, depth 20 | 256 | **8.5 s** | 114 s | 31.9 s | 1.8×10⁻⁵ |
| Random 1D, 100 qubits, depth 16 | 128 | **1.9 s** | 18.0 s | 25.5 s | 7.6×10⁻⁷ |
| Random 8×8 grid, depth 10 | 64 | **6.3 s** | 10.2 s | 18.4 s | 10⁻¹⁶ |
| QFT, 100 qubits | 256 | **0.040 s** | 0.11 s | 21.7 s | 1 (χ = 1) |
| Random 1D, 64 qubits, depth 12, labels scrambled | 64 | **0.049 s** | 21.3 s | 26.6 s | 1 (qvd), 2×10⁻¹³ (quimb) |

qvd's fidelity estimates agree with quimb's to four digits on the 1D
circuits. Low numbers are the circuits' entanglement exceeding χ, not
numerical trouble: on a 4×5 grid small enough to check against the exact
state, the estimate tracks the true overlap within a factor of two.
quimb's time is mostly sampling, which it does shot by shot in Python.
On the last row the circuit's qubit labels are permuted at random, as for a
circuit written for other hardware. qvd places the qubits on the chain by
their interactions first, so the circuit stays exact at χ = 64; the others
route every gate as long-range. The runs shared the machine with a busy
virtual machine (qvd's column was measured last, under the heavier load).

The MPS is SVD-bound. Gates that cannot truncate run in parallel layers.
Gates that may truncate run one at a time, so that every truncation sees
the exact state; truncating a whole layer at once lost up to 3000× in
fidelity. Capped bonds use a Gram-matrix eigendecomposition, about twice as
fast as an SVD. Details are in
[docs/DESIGN.md](docs/DESIGN.md#8-matrix-product-state-backend-mps).

### Clifford+T circuits (near-Clifford)

Random Clifford+T circuits (`library::random_clifford_t`: layers of random
one-qubit Cliffords and CX on random pairs, with T gates on random qubits),
1000 shots, 8 threads.

![Clifford+T benchmark: qvd vs Qiskit Aer](benchmarks/charts/nearclifford.svg)

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
entangled, which is why the MPS methods cannot follow.

The backend keeps `|ψ> = C (|φ> ⊗ |0...0>)`. `C` is a Clifford frame (an
inverse stabilizer tableau), and `φ` is a dense state of the few *active*
qubits:
- Clifford gates only update `C`.
- A non-Clifford rotation activates at most one qubit.
- Measurements retire active qubits again.

The cost is exponential only in the peak number of active qubits. A dry run
of the frame computes that peak exactly before simulating, and `Backend::Auto`
uses it to choose this backend when it beats the state vector by a wide
margin. Final measurements are sampled with per-shot Pauli frames, so 1000
shots cost little more than one.

## How many qubits?

A full state of *n* qubits holds 2ⁿ amplitudes: 8 bytes each in single
precision, 16 in double.

| Qubits | Single | Double |
|---|---|---|
| 30 | 8 GiB | 16 GiB |
| 32 | 32 GiB | 64 GiB |
| 40 | 8 TiB | 16 TiB |

With 62 GB, a general circuit tops out at **32 qubits in single precision
or 31 in double**, and that ceiling holds for every simulator. No credible
source shows a general 40-qubit simulation on one workstation: the records
(45 qubits in 2017, 50 in 2025) used supercomputers. Larger qubit counts on
one machine are possible only for circuits with structure: Clifford
circuits (qvd's stabilizer backend handles 20,000 qubits in seconds),
Clifford circuits with a few non-Clifford gates (qvd's near-Clifford
backend: 200 qubits with 20 T gates in a tenth of a second),
low-entanglement circuits (qvd's MPS backend: 100 qubits and more), or a few
amplitudes rather than the whole state.

## Usage

### From Python

```bash
pip install qevedo-simulator        # imports as qevedo.simulator
```

```python
from qevedo.simulator import Circuit, run, statevector

result = run(Circuit(2).h(0).cx(0, 1).measure_all(), shots=1000)
print(result.counts, result.backend)   # backend chosen automatically

run(open("circuit.qasm").read(), shots=1000)   # OpenQASM 2 or 3 text
state = statevector(Circuit(3).h(0).cx(0, 1))  # NumPy array if installed
```

See [python/README.md](python/README.md). The bindings are a PyO3 module
built with maturin; simulations release the GIL.

### From Rust

```rust
use qvd::{run, statevector, Circuit, Options};

fn main() -> std::io::Result<()> {
    let mut circuit = Circuit::new(3);
    circuit.h(0).cx(0, 1).cx(1, 2).measure_all();

    // Sample shots (single precision).
    let result = run::<f32>(&circuit, 1000, &Options::default())?;
    println!("{:?}", result.counts); // e.g. {"000": 480, "111": 520}

    // Or get the final state (double precision) of a unitary circuit.
    let mut ghz = Circuit::new(3);
    ghz.h(0).cx(0, 1).cx(1, 2);
    let (state, _stats) = statevector::<f64>(&ghz, &Options::default())?;
    println!("{:.4} {:.4}", state.get(0), state.get(7)); // 0.7071+0.0000i 0.7071+0.0000i
    Ok(())
}
```

This is `examples/quickstart.rs`. Gates follow Qiskit's definitions and qubit order (qubit 0 is the least
significant bit). `Circuit::to_qasm` exports OpenQASM 2, and `qvd::qasm::parse`
reads OpenQASM 2 and 3:
- **Gates:** the gates of `qelib1.inc` and `stdgates.inc`, gate definitions
  and parameter expressions.
- **Operations:** register broadcasting and ranges, measure, reset and
  barrier.
- **Modifiers:** OpenQASM 3's `inv @`, `ctrl @`, `negctrl @` and integer
  `pow @`.

Programs that need classical state (`if`, loops, variables) are rejected
with the line and column of the statement. The Python tests check it
against Qiskit on 40 random circuits exported both ways.

`run` picks the backend: with the default `Backend::Auto`, circuits made only
of Clifford gates (including rotations by multiples of π/2), measurements and
resets run on the stabilizer backend. Circuits with few non-Clifford gates
run on the near-Clifford backend when at most 28 qubits are ever active in
it and it is much cheaper than the state vector (or the state vector does
not fit). Everything else runs on the state vector if it fits in 80% of
memory, and on a matrix product state if not.
`Backend::Mps` forces the MPS; `Options::max_bond_dimension` (default 256)
and `Options::truncation_threshold` (default 10⁻¹⁶) set its truncation, and
`result.stats.fidelity` reports the estimated fidelity. For many shots or
many classical bits, `stabilizer::sample` returns bit-packed results without
building strings:

```rust
let circuit = qvd::library::surface_code(25, 25);
let samples = qvd::stabilizer::sample(&circuit, 100_000, Some(1)).unwrap();
let first_shot_bit_0 = samples.get(0, 0);
```

Threads run on rayon's global pool by default. To choose cores explicitly
(the stabilizer backend is fastest on P-cores only):

```rust
use qvd::threads::{pool, Placement};
pool(&Placement::AllThreads).install(|| { /* simulate */ });
```

## Building and benchmarking

The Python package lives in `python/`:

```sh
cd python
pip install "maturin==1.15.0"
maturin develop --release          # into the active virtualenv
pip install pytest numpy && pytest
```

Pushing a tag `vX.Y.Z` that matches `python/Cargo.toml` builds wheels for
Linux, macOS and Windows and publishes them to PyPI
(`.github/workflows/release.yml`). x86-64 wheels target `x86-64-v3` (AVX2 +
FMA) for the AVX2 kernels.

Requires Rust 1.87 or newer. `.cargo/config.toml` builds with
`-C target-cpu=native` so the AVX2 kernels are used; without AVX2 + FMA a
portable fallback with the same layout is compiled.

```sh
cargo test --release
cargo run --release --example bandwidth            # memory bandwidth per thread placement
cargo run --release --example kernels -- 30 f32    # gate cost vs. one memory sweep
cargo run --release --example bench -- random 28 20 f32 4 14 --qasm /tmp/c.qasm
cargo run --release --example shots -- 28 1000000
cargo run --release --example stabilizer -- surface 101 101 10000 --qasm /tmp/s.qasm
cargo run --release --example mps -- random 64 20 256 1000 --qasm /tmp/m.qasm
cargo run --release --example nearclifford -- 100 30 24 1000 --qasm /tmp/n.qasm

# Compare with Qiskit Aer and qsim on the same circuit:
python3 -m venv .bench-venv
.bench-venv/bin/pip install qiskit qiskit-aer qsimcirq cirq-core ply stim matplotlib quimb
.bench-venv/bin/python benchmarks/compare.py /tmp/c.qasm --precision single --threads 32

# Clifford circuits against Stim and Aer's stabilizer method:
.bench-venv/bin/python benchmarks/compare_stim.py /tmp/s.qasm --shots 10000

# MPS circuits against Qiskit Aer and quimb (pip install quimb):
.bench-venv/bin/python benchmarks/compare_mps.py /tmp/m.qasm --max-bond 256 --shots 1000

# Clifford+T circuits against Qiskit Aer (extended stabilizer, MPS, state vector):
.bench-venv/bin/python benchmarks/compare_nearclifford.py /tmp/n.qasm --shots 1000

# Redraw benchmarks/charts/ from benchmarks/results/*.csv:
.bench-venv/bin/python benchmarks/plot.py
```

`bench` arguments: circuit (`random`, `qft`, `ghz`), qubits, depth,
precision, maximum fused qubits, region bits for cache blocking (0 turns it
off). `QVD_PLACEMENT=pcores|pthreads|allcores|all` selects the threads.

## Roadmap

1. ~~Stabilizer (Clifford) backend~~ (done, with automatic selection of
   Clifford circuits).
2. ~~Matrix product state backend~~ (done; chosen automatically when the
   state vector does not fit).
3. ~~Near-Clifford backend~~ (done; chosen automatically when it is much
   cheaper than the state vector).
4. Pauli propagation for expectation values, and a backend selector
   calibrated by benchmarks on this machine.
5. ~~OpenQASM input and Python bindings~~ (done: `qvd::qasm`, and the
   `qevedo-simulator` package in `python/`, imported as `qevedo.simulator`).
6. Noise: Pauli noise channels fit the stabilizer and near-Clifford frame
   samplers directly.

## License

MIT
