import cmath
import math
import random

import pytest

from qevedo.simulator import Circuit, choose_backend, run, statevector


def probabilities(circuit):
    """Exact outcome probabilities of a circuit measured at the end."""
    unitary = Circuit(circuit.num_qubits)
    measures = []
    for name, qubits, params in circuit.instructions():
        if name == "measure":
            measures.append((qubits[0], int(params[0])))
        elif name not in ("barrier",):
            unitary.gate(name, qubits, params)
    state = statevector(unitary)
    out = {}
    width = circuit.num_clbits
    for index, amplitude in enumerate(state):
        p = abs(amplitude) ** 2
        if p < 1e-15:
            continue
        bits = ["0"] * width
        for qubit, clbit in measures:
            bits[width - 1 - clbit] = str((index >> qubit) & 1)
        key = "".join(bits)
        out[key] = out.get(key, 0.0) + p
    return out


def assert_matches(counts, exact, shots):
    for key in counts:
        assert key in exact, f"impossible outcome {key}"
    for key, p in exact.items():
        observed = counts.get(key, 0) / shots
        sigma = math.sqrt(p * (1 - p) / shots)
        assert abs(observed - p) <= 5 * sigma + 3 / shots, (key, observed, p)


def test_bell_state_on_the_stabilizer_backend():
    bell = Circuit(2).h(0).cx(0, 1).measure_all()
    result = run(bell, shots=4000, seed=1)
    assert set(result.counts) == {"00", "11"}
    assert sum(result.counts.values()) == 4000
    assert result.backend == "stabilizer"
    assert result.gates == 2


def test_qasm_text_runs_directly():
    program = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[3] c;
        h q[0];
        cx q[0], q[1];
        ccx q[0], q[1], q[2];
        c = measure q;
    """
    result = run(program, shots=2000, seed=3)
    assert set(result.counts) == {"000", "111"}
    assert result.backend in ("near_clifford", "statevector")


def test_statevector_matches_known_states():
    ghz = statevector(Circuit(3).h(0).cx(0, 1).cx(1, 2))
    r = 1 / math.sqrt(2)
    assert abs(ghz[0] - r) < 1e-12 and abs(ghz[7] - r) < 1e-12
    assert sum(abs(a) ** 2 for a in ghz) == pytest.approx(1.0)
    phased = statevector(Circuit(1).h(0).t(0))
    assert abs(phased[1] - r * cmath.exp(1j * math.pi / 4)) < 1e-12


@pytest.mark.parametrize("backend", ["statevector", "mps", "near_clifford"])
def test_every_backend_samples_the_exact_distribution(backend):
    rng = random.Random(7)
    circuit = Circuit(5)
    for _ in range(40):
        a, b = rng.sample(range(5), 2)
        choice = rng.randrange(6)
        if choice == 0:
            circuit.h(a)
        elif choice == 1:
            circuit.t(a)
        elif choice == 2:
            circuit.rx(rng.uniform(0, 6.28), a)
        elif choice == 3:
            circuit.cx(a, b)
        elif choice == 4:
            circuit.cp(rng.uniform(0, 6.28), a, b)
        else:
            circuit.ccx(a, b, rng.choice([q for q in range(5) if q not in (a, b)]))
    circuit.measure_all()
    shots = 20000
    result = run(circuit, shots=shots, backend=backend, seed=5, max_bond_dimension=64)
    assert result.backend == backend
    assert_matches(result.counts, probabilities(circuit), shots)
    if backend == "mps":
        assert result.fidelity == pytest.approx(1.0)
    if backend == "near_clifford":
        assert result.peak_active_qubits is not None


def test_stabilizer_backend_rejects_non_clifford_gates():
    with pytest.raises(RuntimeError, match="not a Clifford"):
        run(Circuit(1).t(0).measure_all(), shots=10, backend="stabilizer")


def test_automatic_choice():
    big_clifford = Circuit(500).h(0)
    for q in range(1, 500):
        big_clifford.cx(q - 1, q)
    assert choose_backend(big_clifford) == "stabilizer"
    assert choose_backend(Circuit(4).h(0).t(0).cx(0, 1)) in ("statevector", "near_clifford")
    many_t = Circuit(60)
    for q in range(60):
        many_t.h(q).t(q).h(q)
    assert choose_backend(many_t) == "mps"


def test_qasm_round_trip_and_instructions():
    circuit = Circuit(3, 3)
    circuit.h(0).cx(0, 1).rz(0.25, 2).gate("cu3", [2, 0], [0.1, 0.2, 0.3]).measure(1, 2)
    with pytest.raises(ValueError, match="unitaries"):
        circuit.to_qasm()  # cu3 is not native, so it is stored as a unitary
    plain = Circuit(3, 3).h(0).cx(0, 1).rz(0.25, 2).measure(1, 2)
    again = Circuit.from_qasm(plain.to_qasm())
    assert again.instructions() == plain.instructions()
    assert plain.instructions()[2] == ("rz", [2], [0.25])
    assert plain.instructions()[3] == ("measure", [1], [2.0])


def test_unitary_from_nested_lists():
    sx = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
    a = statevector(Circuit(1).unitary(sx, [0]))
    b = statevector(Circuit(1).sx(0))
    assert all(abs(x - y) < 1e-12 for x, y in zip(a, b))
    with pytest.raises(ValueError, match="not unitary"):
        Circuit(1).unitary([[1, 1], [0, 1]], [0])


def test_errors():
    with pytest.raises(ValueError, match="out of range"):
        Circuit(2).h(2)
    with pytest.raises(ValueError, match="same qubit"):
        Circuit(2).cx(1, 1)
    with pytest.raises(ValueError, match="line 3"):
        Circuit.from_qasm('OPENQASM 2.0;\nqreg q[1];\nfoo q[0];')
    with pytest.raises(ValueError, match="unknown backend"):
        run(Circuit(1).measure_all(), backend="quantum")
    with pytest.raises(ValueError, match="unknown gate"):
        Circuit(1).gate("nope", [0])


def test_threads_and_reproducibility():
    circuit = Circuit(6)
    for q in range(6):
        circuit.h(q).t(q)
    for q in range(5):
        circuit.cx(q, q + 1)
    circuit.measure_all()
    a = run(circuit, shots=3000, seed=9, backend="statevector", threads=1)
    b = run(circuit, shots=3000, seed=9, backend="statevector", threads=4)
    assert a.counts == b.counts


def test_against_qiskit():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import qasm2, qasm3
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import Statevector

    for seed in range(40):
        qc = random_circuit(5, 6, max_operands=3, seed=seed)
        expected = Statevector(qc).data
        for text in (qasm2.dumps(qc), qasm3.dumps(qc)):
            got = statevector(text)
            # Qiskit's exported definitions of some gates (ecr, for one) drop
            # their global phase, so compare up to a global phase.
            overlap = abs(sum(x.conjugate() * y for x, y in zip(got, expected)))
            assert overlap == pytest.approx(1.0, abs=1e-9), (seed, text[:200])
