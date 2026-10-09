//! qvd: a quantum circuit simulator built for memory bandwidth.
//!
//! The state vector is stored in cache-line blocks of split real and
//! imaginary parts, gates are applied by AVX2 kernels that run at the
//! machine's memory bandwidth, and gate fusion merges up to four-qubit
//! groups of gates into one pass over memory.

pub mod blocking;
pub mod circuit;
pub mod fusion;
pub mod kernels;
pub mod library;
pub mod matrix;
mod measure;
pub mod memory;
pub mod mps;
pub mod nearclifford;
pub mod reference;
pub mod simd;
pub mod simulator;
pub mod stabilizer;
pub mod state;
pub mod threads;

pub use circuit::{Circuit, Gate, Instruction};
pub use matrix::{C64, Matrix};
pub use simulator::{Backend, Options, RunResult, Stats, run, statevector};
pub use state::StateVector;
