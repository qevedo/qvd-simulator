# Benchmarks

Recorded results, the scripts that produced them, and the charts drawn from
them. Every comparison runs the same OpenQASM file, written by qvd, through
each simulator.

```
benchmarks/
├── results/
│   ├── machine.json            hardware, versions, measurement notes
│   ├── statevector.csv         dense simulation: qvd vs qsim vs Qiskit Aer
│   ├── stabilizer.csv          Clifford circuits: qvd vs Stim vs Qiskit Aer
│   └── stabilizer_steps.csv    how each optimisation changed the d=101 time
├── charts/                     SVGs drawn from results/ by plot.py
├── compare.py                  times Qiskit Aer and qsim on a QASM file
├── compare_stim.py             times Stim and Qiskit Aer's stabilizer method
└── plot.py                     results/*.csv -> charts/*.svg
```

The CSV files are in long format (one row per simulator and case), with
times in seconds, so they can be loaded directly by other tools or pages.

## Machine

Intel Core i9-14900K (8 P-cores with hyper-threading + 16 E-cores, 32
threads, AVX2 + FMA, no AVX-512), 62 GB DDR5, Linux 7.0, Rust 1.96. Python
packages: qiskit 2.5.2, qiskit-aer 0.17.2, qsimcirq 0.22.1, cirq-core 1.7.0,
stim 1.16.0. Measured on 2026-10-09.

## State-vector simulation

![State-vector benchmark](charts/statevector.svg)

Random circuits are depth-20 Google-style circuits (√X, √Y, √W on every
qubit and CZ on a brick pattern); QFT is the quantum Fourier transform.
Each simulator gets its best of 16 or 32 threads. qvd's times include
allocating the state; the others include sampling one shot, and qsim's
include converting the circuit in Python. qsim's CPU backend is single
precision only. Amplitudes agree with Qiskit Aer to 7×10⁻¹³ in double
precision.

## Clifford circuits

![Stabilizer benchmark](charts/stabilizer.svg)

Rotated surface-code memory experiments, d rounds, 10,000 shots. qvd ran
on a pool of 8 P-cores and on one thread; Stim is single-threaded. Stim is
timed two ways:
- `reference_sample()` plus a `FlipSimulator`, whose results are
  measurement-major like qvd's (the fair comparison);
- `compile_sampler().sample(bit_packed=True)`, which returns shot-major
  arrays and is much slower at these sizes.

Qiskit Aer's stabilizer method was not run past d = 21 (it took 636 s
there). A virtual machine used about six cores during these runs, for every
simulator alike, so absolute times are a little high.

![Stabilizer optimisation steps](charts/stabilizer_steps.svg)

The steps that took the distance-101 code (20,401 qubits, 101 rounds) from
37 s to 2.4 s, measured on a quiet machine. [docs/DESIGN.md](../docs/DESIGN.md)
explains each one.

## Reproducing

```sh
python3 -m venv .bench-venv
.bench-venv/bin/pip install qiskit qiskit-aer qsimcirq cirq-core ply stim matplotlib

# State vector: qvd's time is printed; the QASM file feeds the others.
cargo run --release --example bench -- random 28 20 f32 4 14 --qasm /tmp/c.qasm
.bench-venv/bin/python benchmarks/compare.py /tmp/c.qasm --precision single --threads 32

# Clifford: surface code of distance 51, 51 rounds, 10,000 shots.
cargo run --release --example stabilizer -- surface 51 51 10000 --qasm /tmp/s.qasm
.bench-venv/bin/python benchmarks/compare_stim.py /tmp/s.qasm --shots 10000

# After editing results/*.csv:
.bench-venv/bin/python benchmarks/plot.py
```

`bench` takes the circuit (`random`, `qft`, `ghz`), qubits, depth, precision
(`f32`, `f64`), maximum fused qubits and region bits (0 turns cache blocking
off); `QVD_PLACEMENT=pcores|pthreads|allcores|all` selects the threads. The
`stabilizer` example takes `surface <distance> <rounds> <shots>` or
`random <qubits> <depth> <shots>`; `QVD_THREADS=pcores|all|<n>` selects the
threads (P-cores by default).
