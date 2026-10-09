//! Time single gates of each width on a large state and compare with the
//! time one memory sweep takes at the measured DRAM bandwidth.
//!
//!     cargo run --release --example kernels -- [qubits] [f32|f64] [placement]

use std::time::Instant;

use qvd::simd::Real;
use qvd::threads::{Placement, pool};
use qvd::{C64, Matrix, StateVector};

fn random_unitary(k: usize, seed: u64) -> Matrix {
    // Deterministic dense unitary: a DFT-like matrix with phases.
    let dim = 1 << k;
    let mut entries = Vec::with_capacity(dim * dim);
    for r in 0..dim {
        for c in 0..dim {
            let angle = 2.0 * std::f64::consts::PI * (r * c) as f64 / dim as f64
                + seed as f64 * 0.1 * r as f64;
            entries.push(C64::from_polar(1.0 / (dim as f64).sqrt(), angle));
        }
    }
    Matrix::new(entries)
}

fn time<T: Real>(state: &mut StateVector<T>, matrix: &Matrix, qubits: &[usize]) -> f64 {
    state.apply(matrix, qubits, &[]); // warm-up
    (0..3)
        .map(|_| {
            let start = Instant::now();
            state.apply(matrix, qubits, &[]);
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn run<T: Real>(n: usize, bandwidth: f64) {
    let mut state = StateVector::<T>::zero(n).unwrap();
    let bytes = state.memory_bytes() as f64;
    let sweep = 2.0 * bytes / bandwidth;
    println!(
        "{n} qubits, {}, {:.1} GiB, threads {}: one sweep at {:.0} GB/s = {:.3} s\n",
        std::any::type_name::<T>(),
        bytes / (1u64 << 30) as f64,
        rayon::current_num_threads(),
        bandwidth / 1e9,
        sweep
    );
    println!(
        "{:<34} {:>9} {:>10} {:>9}",
        "gate", "time (s)", "GB/s", "x sweep"
    );
    let report = |name: String, t: f64| {
        println!(
            "{name:<34} {t:>9.3} {:>10.1} {:>9.2}",
            2.0 * bytes / t / 1e9,
            t / sweep
        );
    };
    let high = |k: usize| -> Vec<usize> { (0..k).map(|i| n - 1 - 3 * i).collect() };
    for k in 1..=6 {
        let m = random_unitary(k, k as u64);
        report(
            format!("dense k={k} high qubits"),
            time(&mut state, &m, &high(k)),
        );
    }
    for k in 1..=5 {
        let m = random_unitary(k, 7);
        let qubits: Vec<usize> = (0..k).collect();
        report(
            format!("dense k={k} lowest qubits"),
            time(&mut state, &m, &qubits),
        );
    }
    let m = random_unitary(4, 3);
    report(
        "dense k=4 mixed (0, 5, 17, n-1)".into(),
        time(&mut state, &m, &[0, 5, 17, n - 1]),
    );
    let diag = Matrix::diagonal(
        &(0..16)
            .map(|i| C64::from_polar(1.0, i as f64))
            .collect::<Vec<_>>(),
    );
    report(
        "diagonal k=4 mixed".into(),
        time(&mut state, &diag, &[0, 5, 17, n - 1]),
    );
    let x = qvd::Gate::X.matrix();
    let start = Instant::now();
    state.apply(&x, &[n - 1], &[(n - 2, true)]);
    report(
        "CX (control halves traffic)".into(),
        start.elapsed().as_secs_f64(),
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(28, |s| s.parse().unwrap());
    let precision = args.next().unwrap_or_else(|| "f32".into());
    let placement = match args.next().as_deref() {
        Some("all") => Placement::AllThreads,
        Some("pthreads") => Placement::PerformanceThreads,
        Some("allcores") => Placement::AllCores,
        _ => Placement::PerformanceCores,
    };
    let bandwidth = 78.9e9;
    pool(&placement).install(|| match precision.as_str() {
        "f64" => run::<f64>(n, bandwidth),
        _ => run::<f32>(n, bandwidth),
    });
}
