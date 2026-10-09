//! Time the MPS backend.
//!
//!     cargo run --release --example mps -- <random|grid|qft|ghz> <qubits | ROWSxCOLS> <depth> <max bond> [shots] [--qasm FILE]
//!
//! `random` is a 1D brick circuit, `grid` a 2D Google-style circuit (qubits
//! `ROWSxCOLS`). Prints the time to apply the gates and to sample, the
//! largest bond dimension and the fidelity estimate. `QVD_THREADS=all|pcores|<n>`
//! picks the thread pool (default: P-cores).

use std::time::Instant;

use qvd::library::{ghz, qft, random_circuit, random_grid_circuit};
use qvd::threads::{Placement, pool};
use qvd::{Backend, Options};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kind = args.first().map_or("random", |s| s.as_str());
    let size = args.get(1).map_or("64", |s| s.as_str());
    let depth: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    let max_bond: usize = args.get(3).map_or(256, |s| s.parse().unwrap());
    let shots: usize = args
        .get(4)
        .filter(|s| !s.starts_with("--"))
        .map_or(1000, |s| s.parse().unwrap());
    let mut circuit = match kind {
        "grid" => {
            let (rows, cols) = size.split_once('x').expect("grid size as ROWSxCOLS");
            random_grid_circuit(rows.parse().unwrap(), cols.parse().unwrap(), depth, 1)
        }
        "qft" => qft(size.parse().unwrap()),
        "ghz" => ghz(size.parse().unwrap()),
        _ => random_circuit(size.parse().unwrap(), depth, 1),
    };
    circuit.measure_all();
    if let Some(i) = args.iter().position(|s| s == "--qasm") {
        std::fs::write(&args[i + 1], circuit.to_qasm().unwrap()).unwrap();
    }
    let placement = match std::env::var("QVD_THREADS").as_deref() {
        Ok("all") => Placement::AllThreads,
        Ok("pcores") | Err(_) => Placement::PerformanceCores,
        Ok(n) => Placement::Unpinned {
            threads: n.parse().expect("QVD_THREADS: all, pcores or a number"),
        },
    };
    let options = Options {
        backend: Backend::Mps,
        seed: Some(1),
        max_bond_dimension: max_bond,
        ..Options::default()
    };
    let start = Instant::now();
    let result = pool(&placement)
        .install(|| qvd::run::<f64>(&circuit, shots, &options))
        .unwrap();
    let stats = &result.stats;
    println!(
        "qvd mps: {kind} {size}, depth {depth}, {} gates, χ ≤ {max_bond}: gates {:.3} s, {shots} shots {:.3} s, total {:.3} s; max bond {}, fidelity {:.3e}",
        stats.gates,
        stats.gate_seconds,
        stats.measure_seconds,
        start.elapsed().as_secs_f64(),
        stats.max_bond_dimension.unwrap(),
        stats.fidelity.unwrap(),
    );
}
