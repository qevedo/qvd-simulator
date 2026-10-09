"""Run the same OpenQASM circuit on Qiskit Aer and qsim, for timing and checks.

    python benchmarks/compare.py circuit.qasm [--precision single|double]
        [--threads N] [--check amplitudes.txt] [--shots 1]
"""

import argparse
import time

import numpy as np


def load_qiskit(path):
    from qiskit import QuantumCircuit

    return QuantumCircuit.from_qasm_file(path)


def run_aer(circuit, precision, threads, shots, statevector):
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector", precision=precision, max_parallel_threads=threads)
    circuit = circuit.copy()
    if statevector:
        circuit.save_statevector()
    else:
        circuit.measure_all()
    start = time.perf_counter()
    result = sim.run(circuit, shots=shots).result()
    wall = time.perf_counter() - start
    state = np.asarray(result.get_statevector()) if statevector else None
    return wall, result.results[0].time_taken, state


def run_qsim(path, precision, threads, statevector):
    import cirq
    import qsimcirq
    from cirq.contrib.qasm_import import circuit_from_qasm

    circuit = circuit_from_qasm(open(path).read())
    qubits = sorted(circuit.all_qubits(), key=lambda q: int(str(q).split("_")[-1]))
    options = qsimcirq.QSimOptions(cpu_threads=threads, max_fused_gate_size=4, use_gpu=False)
    sim = qsimcirq.QSimSimulator(qsim_options=options)
    start = time.perf_counter()
    if statevector:
        # Reverse the order so qubit 0 is the least significant bit, as in qvd and Qiskit.
        result = sim.simulate(circuit, qubit_order=list(reversed(qubits)))
        state = result.final_state_vector
    else:
        circuit = circuit + cirq.measure(*qubits, key="m")
        sim.run(circuit, repetitions=1)
        state = None
    return time.perf_counter() - start, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm")
    parser.add_argument("--precision", default="single")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--check", help="qvd amplitudes file to compare with")
    parser.add_argument("--skip", default="", help="comma-separated: aer,qsim")
    args = parser.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    statevector = args.check is not None
    reference = None
    if "aer" not in skip:
        wall, inner, reference = run_aer(load_qiskit(args.qasm), args.precision, args.threads, args.shots, statevector)
        print(f"aer  {args.precision}: wall {wall:.3f} s, simulation {inner:.3f} s")
    if "qsim" not in skip:
        if args.precision != "single":
            print("qsim: single precision only on CPU")
        wall, qsim_state = run_qsim(args.qasm, args.precision, args.threads, statevector)
        print(f"qsim single: wall {wall:.3f} s")
        if statevector and reference is not None:
            # qsim and Aer may differ by a global phase convention; compare up to phase.
            overlap = abs(np.vdot(reference, qsim_state))
            print(f"qsim vs aer fidelity: {overlap**2:.8f}")
    if statevector and reference is not None:
        ours = np.loadtxt(args.check)
        ours = ours[:, 0] + 1j * ours[:, 1]
        error = np.max(np.abs(ours - reference[: len(ours)]))
        print(f"qvd vs aer: max amplitude difference {error:.2e} over {len(ours)} amplitudes")


if __name__ == "__main__":
    main()
