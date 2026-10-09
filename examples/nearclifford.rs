//! Time the near-Clifford backend on random Clifford+T circuits.
//!
//!     cargo run --release --example nearclifford -- <qubits> <depth> <T count> [shots] [--backend nc|sv|mps] [--qasm FILE]
//!
//! The circuit is `library::random_clifford_t`, every qubit measured at the
//! end. `QVD_THREADS=all|pcores|<n>` picks the thread pool (default:
//! P-cores).

use std::time::Instant;

use qvd::library::random_clifford_t;
use qvd::threads::{Placement, pool};
use qvd::{Backend, Options};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: usize| {
        args.get(i)
            .filter(|s| !s.starts_with("--"))
            .map_or(default, |s| s.parse().unwrap())
    };
    let (n, depth, t_count, shots) = (arg(0, 60), arg(1, 20), arg(2, 16), arg(3, 1000));
    let mut circuit = random_clifford_t(n, depth, t_count, 1);
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
    let backend = match args
        .iter()
        .position(|s| s == "--backend")
        .map(|i| args[i + 1].as_str())
    {
        Some("sv") => Backend::StateVector,
        Some("mps") => Backend::Mps,
        _ => Backend::NearClifford,
    };
    let options = Options {
        backend,
        seed: Some(1),
        ..Options::default()
    };
    let start = Instant::now();
    let result = pool(&placement)
        .install(|| qvd::run::<f64>(&circuit, shots, &options))
        .unwrap();
    println!(
        "qvd {backend:?}: {n} qubits, depth {depth}, {t_count} T, {} gates: gates {:.3} s, {shots} shots {:.3} s, total {:.3} s; peak active {:?}",
        result.stats.gates,
        result.stats.gate_seconds,
        result.stats.measure_seconds,
        start.elapsed().as_secs_f64(),
        result.stats.peak_active_qubits,
    );
}
