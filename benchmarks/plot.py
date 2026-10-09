"""Draw the charts in benchmarks/charts/ from the CSV files in benchmarks/results/.

    python benchmarks/plot.py

Needs matplotlib. The SVGs are deterministic, so regenerating them without
changing the data leaves git clean.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHARTS = HERE / "charts"

plt.rcParams.update({
    "svg.hashsalt": "qvd",
    "svg.fonttype": "path",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLORS = {
    "qvd": "#d1495b",
    "qvd, 1 thread": "#f4a4ae",
    "qsim": "#00798c",
    "qiskit-aer": "#edae49",
    "stim, flip simulator": "#30638e",
    "stim, sample()": "#8fb8de",
    "qiskit-aer, stabilizer": "#edae49",
    "quimb": "#66a182",
    "qvd, near-Clifford": "#d1495b",
    "qvd, state vector": "#f4a4ae",
    "qiskit-aer, statevector": "#edae49",
    "qiskit-aer, extended_stabilizer": "#00798c",
    "qiskit-aer, matrix_product_state": "#8fb8de",
}


def read(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    CHARTS.mkdir(exist_ok=True)
    fig.savefig(CHARTS / name, metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


def statevector():
    rows = read("statevector.csv")
    cases = []
    times = defaultdict(dict)
    for r in rows:
        case = (r["circuit"], int(r["qubits"]), r["precision"])
        if case not in cases:
            cases.append(case)
        times[r["simulator"]][case] = float(r["seconds"])
    simulators = ["qvd", "qsim", "qiskit-aer"]
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, sim in enumerate(simulators):
        xs = [c + (i - 1) * width for c, case in enumerate(cases) if case in times[sim]]
        ys = [times[sim][case] for case in cases if case in times[sim]]
        bars = ax.bar(xs, ys, width, label=sim, color=COLORS[sim])
        if sim == "qvd":
            ax.bar_label(bars, labels=[f"{y:g}" for y in ys], fontsize=8, padding=2)
    ax.set_yscale("log")
    ax.set_ylabel("seconds (log scale, lower is better)")
    ax.set_xticks(range(len(cases)))
    names = {"random": "Random", "qft": "QFT"}
    ax.set_xticklabels([f"{names[c]}\n{q} qubits\n{p}" for c, q, p in cases], fontsize=9)
    ax.set_title("State-vector simulation, i9-14900K (qsim's CPU backend is single precision only)")
    ax.legend(frameon=False, ncols=3, loc="upper left")
    save(fig, "statevector.svg")


def stabilizer():
    rows = read("stabilizer.csv")
    series = defaultdict(list)
    for r in rows:
        if r["simulator"] == "qvd":
            name = "qvd" if r["threads"] != "1" else "qvd, 1 thread"
        elif r["simulator"] == "stim":
            name = "stim, flip simulator" if "Flip" in r["method"] else "stim, sample()"
        else:
            name = "qiskit-aer, stabilizer"
        series[name].append((int(r["qubits"]), float(r["seconds"]), int(r["distance"])))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = {
        "qvd": "qvd, 8 P-cores",
        "qvd, 1 thread": "qvd, 1 thread",
        "stim, flip simulator": "Stim, reference + FlipSimulator",
        "stim, sample()": "Stim, compile_sampler().sample()",
        "qiskit-aer, stabilizer": "Qiskit Aer, stabilizer",
    }
    for name, label in labels.items():
        points = sorted(series[name])
        style = "-" if name.startswith("qvd") else "--"
        ax.plot([p[0] for p in points], [p[1] for p in points], style, marker="o",
                color=COLORS[name], label=label, linewidth=2 if name == "qvd" else 1.5)
    distances = sorted({(q, d) for pts in series.values() for q, _, d in pts})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([q for q, _ in distances])
    ax.set_xticklabels([f"{q:,}\nd={d}" for q, d in distances])
    ax.minorticks_off()
    ax.set_xlabel("qubits (rotated surface code, d rounds)")
    ax.set_ylabel("seconds for 10,000 shots (log scale)")
    ax.set_title("Clifford circuits: surface code sampling")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "stabilizer.svg")


def stabilizer_steps():
    rows = read("stabilizer_steps.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [
        r["description"]
        + (f" ({r['shots']} shots)" if r["shots"] != "10000" else "")
        + (" (under load)" if r["conditions"] != "quiet" else "")
        for r in rows
    ]
    values = [float(r["seconds"]) for r in rows]
    bars = ax.barh(range(len(rows)), values, color=COLORS["qvd"])
    ax.bar_label(bars, labels=[f"{v:g} s" for v in values], padding=3, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("seconds, distance-101 surface code (20,401 qubits, 101 rounds, 10k shots)")
    ax.set_title("What made the stabilizer backend fast (each step adds to the previous)")
    save(fig, "stabilizer_steps.svg")


def mps():
    rows = read("mps.csv")
    cases = []
    times = defaultdict(dict)
    for r in rows:
        case = (r["circuit"], int(r["qubits"]), r["depth"], int(r["max_bond"]))
        if case not in cases:
            cases.append(case)
        times[r["simulator"]][case] = float(r["seconds"])
    simulators = ["qvd", "qiskit-aer", "quimb"]
    width = 0.27
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    for i, sim in enumerate(simulators):
        xs = [c + (i - 1) * width for c in range(len(cases))]
        ys = [times[sim][case] for case in cases]
        bars = ax.bar(xs, ys, width, label=sim, color=COLORS[sim])
        if sim == "qvd":
            ax.bar_label(bars, labels=[f"{y:g}" for y in ys], fontsize=8, padding=2)
    ax.set_yscale("log")
    ax.set_ylabel("seconds, gates + 1000 shots (log scale)")
    ax.set_xticks(range(len(cases)))
    names = {
        "random-1d": "Random 1D",
        "random-1d-scrambled": "Random 1D,\nscrambled labels",
        "random-grid-8x8": "Random 8x8 grid",
        "qft": "QFT",
    }
    ax.set_xticklabels(
        [f"{names[c]}\n{q} qubits" + (f", depth {d}" if d else "") + f"\nχ ≤ {b}" for c, q, d, b in cases],
        fontsize=9,
    )
    ax.set_ylim(top=max(max(t.values()) for t in times.values()) * 8)
    ax.set_title("Matrix product states, same truncation (i9-14900K, 8 threads)")
    ax.legend(frameon=False, ncols=3, loc="upper left")
    save(fig, "mps.svg")


def nearclifford():
    rows = read("nearclifford.csv")
    cases = []
    times = defaultdict(dict)
    for r in rows:
        case = (int(r["qubits"]), int(r["t_count"]))
        if case not in cases:
            cases.append(case)
        name = f"{r['simulator']}, {r['method']}"
        times[name][case] = (float(r["seconds"]) if r["seconds"] else None, r["status"])
    series = [
        "qvd, near-Clifford",
        "qvd, state vector",
        "qiskit-aer, statevector",
        "qiskit-aer, extended_stabilizer",
        "qiskit-aer, matrix_product_state",
    ]
    labels = {
        "qvd, near-Clifford": "qvd near-Clifford",
        "qvd, state vector": "qvd state vector",
        "qiskit-aer, statevector": "Aer state vector",
        "qiskit-aer, extended_stabilizer": "Aer extended stabilizer (approximate)",
        "qiskit-aer, matrix_product_state": "Aer MPS",
    }
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 4.6))
    limit = max(t for s in times.values() for t, status in s.values() if t is not None)
    for i, name in enumerate(series):
        for c, case in enumerate(cases):
            if case not in times[name]:
                continue
            t, status = times[name][case]
            if status == "failed":
                # Not supported (Aer's extended stabilizer stops at 63 qubits).
                continue
            x = c + (i - 2) * width
            if status == "timeout":
                # Stopped at the time limit: an open bar to the limit.
                ax.bar(x, t, width, color="white", edgecolor=COLORS[name], hatch="///", linewidth=1)
                ax.text(x, t * 1.15, ">", ha="center", fontsize=8, color=COLORS[name])
            else:
                first = all(times[name].get(cs, (None, "failed"))[1] != "ok" for cs in cases[:c])
                bar = ax.bar(x, t, width, color=COLORS[name], label=labels[name] if first else None)
                if name == "qvd, near-Clifford":
                    ax.bar_label(bar, labels=[f"{t:g}"], fontsize=8, padding=2)
    ax.set_yscale("log")
    ax.set_ylim(top=limit * 20)
    ax.set_ylabel("seconds for 1000 shots (log scale)")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"{q} qubits\n{t} T gates" for q, t in cases])
    ax.set_title("Clifford+T circuits, 1000 shots (hatched: stopped at 600 s; no bar: not supported or too large)")
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=COLORS[name], label=labels[name]) for name in series]
    ax.legend(handles=handles, frameon=False, ncols=3, loc="upper left", fontsize=9)
    save(fig, "nearclifford.svg")


if __name__ == "__main__":
    statevector()
    nearclifford()
    mps()
    stabilizer()
    stabilizer_steps()
    print(f"wrote {', '.join(p.name for p in sorted(CHARTS.glob('*.svg')))}")
