//! Measure sustainable memory bandwidth for the access patterns of a
//! state-vector simulator, under several thread placements.
//!
//!     cargo run --release --example bandwidth -- [GiB per array]

use std::time::Instant;

use qvd::memory::Buffer;
use qvd::threads::{Placement, Topology, pool};
use rayon::prelude::*;

const CHUNK: usize = 1 << 16;

fn best_of<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let gib: f64 = std::env::args()
        .nth(1)
        .map_or(2.0, |s| s.parse().expect("GiB"));
    let len = (gib * (1u64 << 30) as f64) as usize / 4;
    let topology = Topology::detect();
    println!(
        "P-cores {:?}\nE-cores {:?}",
        topology.performance, topology.efficiency
    );
    println!(
        "array: {:.2} GiB of f32 ({} elements)\n",
        len as f64 * 4.0 / (1u64 << 30) as f64,
        len
    );
    println!(
        "{:<22} {:>8} {:>16} {:>16}",
        "placement", "threads", "rmw GB/s", "triad GB/s"
    );
    for placement in [
        Placement::PerformanceCores,
        Placement::PerformanceThreads,
        Placement::AllCores,
        Placement::AllThreads,
    ] {
        let pool = pool(&placement);
        pool.install(|| {
            let mut a = Buffer::<f32>::zeroed(len).unwrap();
            let mut b = Buffer::<f32>::zeroed(len).unwrap();
            let c = Buffer::<f32>::zeroed(len).unwrap();
            b.as_mut_slice()
                .par_chunks_mut(CHUNK)
                .for_each(|x| x.fill(1.0));
            // In-place read-modify-write: the pattern of applying a gate.
            let rmw = best_of(5, || {
                a.as_mut_slice().par_chunks_mut(CHUNK).for_each(|x| {
                    for v in x {
                        *v = *v * 0.999 + 0.5;
                    }
                })
            });
            let triad = best_of(5, || {
                a.as_mut_slice()
                    .par_chunks_mut(CHUNK)
                    .zip(b.as_slice().par_chunks(CHUNK))
                    .zip(c.as_slice().par_chunks(CHUNK))
                    .for_each(|((a, b), c)| {
                        for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
                            *a = *b + 0.5 * *c;
                        }
                    })
            });
            let bytes = len as f64 * 4.0;
            println!(
                "{:<22} {:>8} {:>16.1} {:>16.1}",
                format!("{placement:?}"),
                rayon::current_num_threads(),
                2.0 * bytes / rmw / 1e9,
                3.0 * bytes / triad / 1e9
            );
        });
    }
}
