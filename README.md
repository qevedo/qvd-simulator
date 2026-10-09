# qvd-simulator

A quantum circuit simulator in Rust, built around one fact: on a modern CPU,
simulating a large circuit is limited by **memory bandwidth**, not by
arithmetic. Every gate on a 30-qubit state reads and writes 8–16 GiB, so the
fastest simulator is the one that makes the fewest passes over memory.

On an Intel i9-14900K (24 cores, AVX2, 62 GB RAM) it runs 3–4× faster than
Google's qsim and 7–12× faster than Qiskit Aer on the same circuits (details
below).

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
circuits (stabilizer methods reach ~20,000 qubits), low-entanglement
circuits (matrix product states), or a few amplitudes rather than the whole
state. Those backends are the next steps on the roadmap.

## Usage

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
significant bit), and `Circuit::to_qasm` exports OpenQASM 2.

Threads run on rayon's global pool by default. To choose cores explicitly:

```rust
use qvd::threads::{pool, Placement};
pool(&Placement::AllThreads).install(|| { /* simulate */ });
```

## Building and benchmarking

Requires Rust 1.87 or newer. `.cargo/config.toml` builds with
`-C target-cpu=native` so the AVX2 kernels are used; without AVX2 + FMA a
portable fallback with the same layout is compiled.

```sh
cargo test --release
cargo run --release --example bandwidth            # memory bandwidth per thread placement
cargo run --release --example kernels -- 30 f32    # gate cost vs. one memory sweep
cargo run --release --example bench -- random 28 20 f32 4 14 --qasm /tmp/c.qasm
cargo run --release --example shots -- 28 1000000

# Compare with Qiskit Aer and qsim on the same circuit:
python3 -m venv .bench-venv
.bench-venv/bin/pip install qiskit qiskit-aer qsimcirq cirq-core ply
.bench-venv/bin/python benchmarks/compare.py /tmp/c.qasm --precision single --threads 32
```

`bench` arguments: circuit (`random`, `qft`, `ghz`), qubits, depth,
precision, maximum fused qubits, region bits for cache blocking (0 turns it
off). `QVD_PLACEMENT=pcores|pthreads|allcores|all` selects the threads.

## Roadmap

1. Stabilizer (Clifford) backend in the style of Stim: bit-packed AVX2
   tableaux, for thousands of qubits.
2. Automatic backend selection.
3. Matrix product state backend for low-entanglement circuits past 32 qubits.
4. Near-Clifford backend (Clifford frame + small dense state, reusing these
   kernels) and Pauli propagation for expectation values.
5. OpenQASM input, Python bindings and noise.

## License

MIT
