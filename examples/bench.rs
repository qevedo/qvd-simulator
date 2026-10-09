//! Time a benchmark circuit end to end.
//!
//!     cargo run --release --example bench -- <random|qft|ghz> <qubits> [depth] [f32|f64] [fusion] [region bits, 0 = off]
//!         [--qasm FILE] [--amplitudes FILE]

use std::time::Instant;

use qvd::library::{ghz, qft, random_circuit};
use qvd::simd::Real;
use qvd::threads::{Placement, pool};
use qvd::{Circuit, Options, statevector};

fn run<T: Real>(circuit: &Circuit, options: &Options, amplitudes: Option<&str>) {
    let start = Instant::now();
    let (state, stats) = statevector::<T>(circuit, options).unwrap();
    let total = start.elapsed().as_secs_f64();
    println!(
        "qvd {}: {} qubits, {} gates -> {} fused -> {} passes, gates {:.3} s, total {:.3} s (incl. allocation), norm {:.6}",
        std::any::type_name::<T>(),
        circuit.num_qubits,
        stats.gates,
        stats.kernels,
        stats.passes,
        stats.gate_seconds,
        total,
        state.norm_sqr()
    );
    if let Some(path) = amplitudes {
        let text: String = state
            .to_vec()
            .iter()
            .take(1 << 12)
            .map(|a| format!("{:.10e} {:.10e}\n", a.re, a.im))
            .collect();
        std::fs::write(path, text).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let positional: Vec<&String> = args.iter().take_while(|a| !a.starts_with("--")).collect();
    let kind = positional.first().map_or("random", |s| s.as_str());
    let n: usize = positional.get(1).map_or(24, |s| s.parse().unwrap());
    let depth: usize = positional.get(2).map_or(20, |s| s.parse().unwrap());
    let precision = positional.get(3).map_or("f32", |s| s.as_str());
    let fusion: usize = positional.get(4).map_or(4, |s| s.parse().unwrap());
    // Region bits for cache blocking; 0 disables it.
    let region: u32 = positional
        .get(5)
        .map_or(qvd::blocking::DEFAULT_REGION_BITS, |s| s.parse().unwrap());
    let circuit = match kind {
        "qft" => qft(n),
        "ghz" => ghz(n),
        _ => random_circuit(n, depth, 42),
    };
    if let Some(path) = flag("--qasm") {
        std::fs::write(path, circuit.to_qasm().unwrap()).unwrap();
    }
    let options = Options {
        backend: qvd::Backend::StateVector,
        max_fused_qubits: fusion,
        seed: None,
        region_bits: (region > 0).then_some(region),
    };
    let amplitudes = flag("--amplitudes");
    let placement = match std::env::var("QVD_PLACEMENT").as_deref() {
        Ok("pcores") => Placement::PerformanceCores,
        Ok("allcores") => Placement::AllCores,
        Ok("all") => Placement::AllThreads,
        Ok("pthreads") => Placement::PerformanceThreads,
        _ => Placement::AllThreads,
    };
    pool(&placement).install(|| match precision {
        "f64" => run::<f64>(&circuit, &options, amplitudes.as_deref()),
        _ => run::<f32>(&circuit, &options, amplitudes.as_deref()),
    });
}
