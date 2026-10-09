"""qvd: a fast quantum circuit simulator, as ``qevedo.simulator``.

Build circuits with :class:`Circuit` (or read OpenQASM 2/3 with
:meth:`Circuit.from_qasm`), then sample them with :func:`run` or get the
final state with :func:`statevector`. ``run`` chooses the method
automatically: the stabilizer backend for Clifford circuits, the
near-Clifford backend when few non-Clifford gates are ever active, the
state vector when it fits in memory, and a matrix product state otherwise.

    >>> from qevedo.simulator import Circuit, run
    >>> bell = Circuit(2).h(0).cx(0, 1).measure_all()
    >>> result = run(bell, shots=1000, seed=1)
    >>> sorted(result.counts)
    ['00', '11']
    >>> result.backend
    'stabilizer'
"""

from __future__ import annotations

import platform as _platform

def _check_cpu() -> None:
    # x86-64 wheels use AVX2 + FMA kernels; fail clearly instead of crashing
    # on an older processor.
    if _platform.machine().lower() not in ("x86_64", "amd64"):
        return
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line for line in f if line.startswith("flags")), "")
    except OSError:
        return
    if flags and not {"avx2", "fma"} <= set(flags.split()):
        raise ImportError(
            "qevedo.simulator's x86-64 wheels need a processor with AVX2 and FMA "
            "(Intel Haswell or AMD Excavator, 2013-2015, or newer); "
            "build from source with `pip install --no-binary qevedo-simulator qevedo-simulator`"
        )


_check_cpu()

from qevedo.simulator._native import (  # noqa: E402
    Circuit,
    RunResult,
    __version__,
    choose_backend as _choose_backend,
    run as _run,
    statevector_bytes as _statevector_bytes,
    uses_avx2,
)

__all__ = ["Circuit", "RunResult", "run", "statevector", "choose_backend", "uses_avx2", "__version__"]

BACKENDS = ("auto", "statevector", "stabilizer", "mps", "near_clifford")


def _circuit(circuit: Circuit | str) -> Circuit:
    if isinstance(circuit, str):
        return Circuit.from_qasm(circuit)
    if not isinstance(circuit, Circuit):
        raise TypeError("expected a qevedo.simulator.Circuit or OpenQASM text")
    return circuit


def run(
    circuit: Circuit | str,
    shots: int = 1024,
    *,
    backend: str = "auto",
    seed: int | None = None,
    precision: str = "single",
    max_bond_dimension: int = 256,
    truncation_threshold: float = 1e-16,
    threads: int | str | None = None,
) -> RunResult:
    """Sample ``shots`` shots of a circuit (or OpenQASM 2/3 text).

    ``backend`` is one of ``"auto"``, ``"statevector"``, ``"stabilizer"``,
    ``"mps"`` and ``"near_clifford"``. ``precision`` (``"single"`` or
    ``"double"``) applies to the state vector. ``max_bond_dimension`` and
    ``truncation_threshold`` control the MPS: at most that many singular
    values per bond, and the smallest are dropped while the sum of their
    squares stays below the threshold. ``threads`` is a thread count or a
    placement (``"pcores"``, ``"pthreads"``, ``"cores"``, ``"all"``);
    ``None`` uses all cores. The simulation runs without holding the GIL.

    The result has ``counts`` (bitstrings with classical bit 0 on the
    right, as in Qiskit), ``backend`` (the method that ran), timings, and
    method-specific statistics: ``fidelity`` and ``max_bond_dimension`` for
    the MPS, ``peak_active_qubits`` for the near-Clifford backend.
    """
    return _run(
        _circuit(circuit),
        shots,
        backend=backend,
        seed=seed,
        precision=precision,
        max_bond_dimension=max_bond_dimension,
        truncation_threshold=truncation_threshold,
        threads=threads,
    )


def statevector(circuit: Circuit | str, *, threads: int | str | None = None):
    """The final state of a circuit without measurements, in double
    precision: a NumPy array of complex128 if NumPy is installed, a list of
    complex numbers otherwise. Index bit ``q`` is qubit ``q`` (Qiskit's
    order)."""
    raw = _statevector_bytes(_circuit(circuit), threads=threads)
    try:
        import numpy as np
    except ImportError:
        import struct

        values = struct.unpack(f"<{len(raw) // 8}d", raw)
        return [complex(values[i], values[i + 1]) for i in range(0, len(values), 2)]
    return np.frombuffer(raw, dtype="<c16").copy()


def choose_backend(
    circuit: Circuit | str,
    *,
    backend: str = "auto",
    precision: str = "single",
    max_bond_dimension: int = 256,
    truncation_threshold: float = 1e-16,
) -> str:
    """The method :func:`run` would use for this circuit."""
    return _choose_backend(
        _circuit(circuit),
        backend=backend,
        precision=precision,
        max_bond_dimension=max_bond_dimension,
        truncation_threshold=truncation_threshold,
    )
