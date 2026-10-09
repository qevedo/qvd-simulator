//! End-to-end run with sampling: `run()` on a measured circuit.
//!
//!     cargo run --release --example shots -- [qubits] [shots]
use qvd::library::random_circuit;
use qvd::threads::{Placement, pool};
use qvd::{Options, run};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(28, |s| s.parse().unwrap());
    let shots: usize = args.next().map_or(1_000_000, |s| s.parse().unwrap());
    let mut circuit = random_circuit(n, 20, 42);
    circuit.measure_all();
    pool(&Placement::AllThreads).install(|| {
        let start = Instant::now();
        let result = run::<f32>(&circuit, shots, &Options { seed: Some(1), ..Options::default() }).unwrap();
        println!(
            "{n} qubits, {shots} shots: total {:.3} s (gates {:.3} s, sampling {:.3} s), {} distinct outcomes",
            start.elapsed().as_secs_f64(),
            result.stats.gate_seconds,
            result.stats.measure_seconds,
            result.counts.len()
        );
    });
}
