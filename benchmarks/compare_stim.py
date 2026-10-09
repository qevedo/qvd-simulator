"""Run a Clifford OpenQASM 2 circuit (as written by qvd) on Stim and Qiskit Aer's
stabilizer method, for timing.

    python benchmarks/compare_stim.py circuit.qasm [--shots N] [--skip aer]

Stim is timed two ways: `compile_sampler().sample()` (shot-major output) and
`reference_sample()` plus a `FlipSimulator` (measurement-major, like qvd).
"""

import argparse
import re
import time

GATES = {"h": "H", "s": "S", "sdg": "S_DAG", "x": "X", "y": "Y", "z": "Z", "sx": "SQRT_X",
         "sxdg": "SQRT_X_DAG", "cx": "CX", "cz": "CZ", "cy": "CY", "swap": "SWAP", "id": "I"}


def qasm_to_stim(text):
    import stim

    lines = []
    for line in text.splitlines():
        line = line.strip().rstrip(";")
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "barrier")):
            continue
        if line.startswith("measure"):
            q = re.findall(r"q\[(\d+)\]", line)[0]
            lines.append(f"M {q}")
        elif line.startswith("reset"):
            q = re.findall(r"q\[(\d+)\]", line)[0]
            lines.append(f"R {q}")
        else:
            name = line.split()[0]
            qubits = " ".join(re.findall(r"q\[(\d+)\]", line))
            lines.append(f"{GATES[name]} {qubits}")
    return stim.Circuit("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--skip", default="")
    args = parser.parse_args()
    text = open(args.qasm).read()
    circuit = qasm_to_stim(text)
    start = time.perf_counter()
    sampler = circuit.compile_sampler()
    compiled = time.perf_counter() - start
    samples = sampler.sample(args.shots, bit_packed=True)
    total = time.perf_counter() - start
    print(f"stim sample(): compile (reference sample) {compiled:.3f} s, total with {args.shots} shots {total:.3f} s")
    # sample() returns shot-major bits, which costs a transposition; the
    # flip simulator leaves them measurement-major, as qvd does.
    import stim

    start = time.perf_counter()
    reference = circuit.reference_sample()
    simulator = stim.FlipSimulator(batch_size=args.shots)
    simulator.do(circuit)
    flips = simulator.get_measurement_flips(bit_packed=True)
    total = time.perf_counter() - start
    print(f"stim reference_sample + FlipSimulator: {total:.3f} s ({len(reference)} x {flips.shape[1] * 8} bits)")
    if "aer" not in args.skip:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit.from_qasm_str(text)
        start = time.perf_counter()
        result = AerSimulator(method="stabilizer").run(qc, shots=args.shots).result()
        print(f"aer stabilizer: wall {time.perf_counter() - start:.3f} s, simulation {result.results[0].time_taken:.3f} s")


if __name__ == "__main__":
    main()
