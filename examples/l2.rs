//! Single-core kernel throughput on an L2-resident state (the regime of
//! cache-blocked stages): FMA instructions per cycle-equivalent.
use qvd::threads::{Placement, pool};
use qvd::{C64, Matrix, StateVector};
use std::time::Instant;

fn main() {
    let n = 17; // 1 MiB of f32 amplitudes
    pool(&Placement::Cpus(vec![0])).install(|| {
        let mut state = StateVector::<f32>::zero(n).unwrap();
        let cases: Vec<(&str, Vec<usize>)> = vec![
            ("k=1 high", vec![10]),
            ("k=2 high", vec![9, 13]),
            ("k=3 high", vec![5, 9, 13]),
            ("k=4 high", vec![4, 8, 11, 14]),
            ("k=4 low+high", vec![0, 1, 2, 9]),
            ("k=4 contiguous", vec![3, 4, 5, 6]),
            ("k=5 high", vec![4, 7, 10, 13, 16]),
        ];
        for (name, targets) in cases {
            let k = targets.len();
            let dim = 1 << k;
            let m = Matrix::new(
                (0..dim * dim)
                    .map(|i| C64::from_polar(0.2, i as f64 * 0.37))
                    .collect(),
            );
            state.apply(&m, &targets, &[]);
            let reps = 300;
            let start = Instant::now();
            for _ in 0..reps {
                state.apply(&m, &targets, &[]);
            }
            let per = start.elapsed().as_secs_f64() / reps as f64;
            let fmas = (1u64 << n) as f64 / 8.0 * (4 * dim) as f64; // per block: 4*2^k FMAs
            println!(
                "{name:<16} {:>8.1} us  {:>6.2} G FMA/s  {:>6.1} GB/s",
                per * 1e6,
                fmas / per / 1e9,
                2.0 * state.memory_bytes() as f64 / per / 1e9
            );
        }
    });
}
