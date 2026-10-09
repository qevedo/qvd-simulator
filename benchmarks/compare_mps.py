"""Run an OpenQASM 2 circuit on matrix product state simulators, for timing:
Qiskit Aer's `matrix_product_state` method and quimb's `CircuitPermMPS`.

    python benchmarks/compare_mps.py circuit.qasm [--max-bond 256] [--shots 1000]
        [--threshold 1e-16] [--threads 8] [--skip aer,quimb]

Both use qvd's truncation: at most `max_bond` singular values per bond, and
the smallest ones are dropped while the sum of their squares stays below
`threshold` (quimb's `cutoff_mode="rsum2"`). quimb's `CircuitPermMPS`, like
qvd, leaves qubits where SWAPs moved them.
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm")
    parser.add_argument("--max-bond", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=1e-16)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip", default="")
    args = parser.parse_args()
    if "aer" not in args.skip:
        run_aer(args)
    if "quimb" not in args.skip:
        run_quimb(args)


def run_aer(args):

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit.from_qasm_file(args.qasm)
    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=args.max_bond,
        matrix_product_state_truncation_threshold=args.threshold,
        max_parallel_threads=args.threads,
    )
    start = time.perf_counter()
    result = simulator.run(circuit, shots=args.shots).result()
    wall = time.perf_counter() - start
    print(f"aer mps: wall {wall:.3f} s, simulation {result.results[0].time_taken:.3f} s")


def run_quimb(args):
    import re

    import quimb.tensor as qtn

    # Warm up (numba compiles quimb's kernels on first use).
    warm = qtn.CircuitPermMPS(4, max_bond=4)
    warm.h(0)
    warm.cx(0, 3)
    for _ in warm.sample(2):
        pass
    text = open(args.qasm).read()
    # quimb applies gates only; measurements are sampled at the end.
    text = "\n".join(line for line in text.splitlines() if not re.match(r"\s*(measure|creg|barrier)", line))
    start = time.perf_counter()
    circuit = qtn.CircuitPermMPS.from_openqasm2_str(
        text,
        max_bond=args.max_bond,
        cutoff=args.threshold,
        gate_opts={"cutoff_mode": "rsum2"},
    )
    circuit.psi  # make sure every gate has been applied
    gates = time.perf_counter() - start
    start = time.perf_counter()
    for _ in circuit.sample(args.shots):
        pass
    sampling = time.perf_counter() - start
    print(
        f"quimb mps: gates {gates:.3f} s, {args.shots} shots {sampling:.3f} s, total {gates + sampling:.3f} s, "
        f"fidelity {circuit.fidelity_estimate():.3e}"
    )


if __name__ == "__main__":
    main()
