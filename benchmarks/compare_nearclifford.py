"""Run a Clifford+T OpenQASM 2 circuit on Qiskit Aer's methods, for timing.

    python benchmarks/compare_nearclifford.py circuit.qasm [--shots 1000]
        [--methods extended_stabilizer,statevector,matrix_product_state] [--threads 8]

`extended_stabilizer` is Aer's Clifford+T method (Bravyi et al. 2019): an
approximation, with its default error target of 0.05, limited to 63 qubits.
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--methods", default="extended_stabilizer,matrix_product_state")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit.from_qasm_file(args.qasm)
    for method in args.methods.split(","):
        simulator = AerSimulator(method=method, max_parallel_threads=args.threads)
        start = time.perf_counter()
        try:
            result = simulator.run(circuit, shots=args.shots).result()
            ok = result.success
        except Exception as error:  # noqa: BLE001 - report and move on
            print(f"aer {method}: failed ({error})")
            continue
        wall = time.perf_counter() - start
        status = "" if ok else f" (failed: {result.status})"
        print(f"aer {method}: wall {wall:.3f} s{status}")


if __name__ == "__main__":
    main()
