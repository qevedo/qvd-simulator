//! Time the stabilizer backend.
//!
//!     cargo run --release --example stabilizer -- surface <distance> <rounds> <shots> [--qasm FILE]
//!     cargo run --release --example stabilizer -- random <qubits> <depth> <shots> [--qasm FILE]
//!
//! `QVD_THREADS=all|pcores|<n>` picks the thread pool (default: P-cores).

use std::time::Instant;

use qvd::library::{random_clifford, surface_code};
use qvd::stabilizer;
use qvd::threads::{Placement, pool};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kind = args.first().map_or("surface", |s| s.as_str());
    let a: usize = args.get(1).map_or(15, |s| s.parse().unwrap());
    let b: usize = args.get(2).map_or(15, |s| s.parse().unwrap());
    let shots: usize = args.get(3).map_or(10_000, |s| s.parse().unwrap());
    let mut circuit = match kind {
        "random" => random_clifford(a, b, 1),
        _ => surface_code(a, b),
    };
    if kind == "random" {
        circuit.measure_all();
    }
    if let Some(i) = args.iter().position(|s| s == "--qasm") {
        std::fs::write(&args[i + 1], circuit.to_qasm().unwrap()).unwrap();
    }
    let measurements = circuit
        .instructions
        .iter()
        .filter(|i| matches!(i, qvd::Instruction::Measure { .. }))
        .count();
    let placement = match std::env::var("QVD_THREADS").as_deref() {
        Ok("all") => Placement::AllThreads,
        Ok("pcores") | Err(_) => Placement::PerformanceCores,
        Ok(n) => Placement::Unpinned {
            threads: n.parse().expect("QVD_THREADS: all, pcores or a number"),
        },
    };
    let pool = pool(&placement);
    let start = Instant::now();
    let samples = pool
        .install(|| stabilizer::sample(&circuit, shots, Some(1)))
        .unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let ones: usize = (0..samples.num_clbits.min(64))
        .filter(|&c| samples.get(0, c))
        .count();
    println!(
        "qvd stabilizer: {} qubits, {} gates, {} measurements, {} shots: {:.3} s ({ones} ones in shot 0's first 64 bits)",
        circuit.num_qubits,
        circuit.gate_count(),
        measurements,
        shots,
        elapsed,
    );
}
