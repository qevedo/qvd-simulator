//! The example from the README.
use qvd::{Circuit, Options, run, statevector};

fn main() -> std::io::Result<()> {
    let mut circuit = Circuit::new(3);
    circuit.h(0).cx(0, 1).cx(1, 2).measure_all();

    // Sample shots (single precision).
    let result = run::<f32>(&circuit, 1000, &Options::default())?;
    println!("{:?}", result.counts);

    // Or get the final state (double precision) of a unitary circuit.
    let mut ghz = Circuit::new(3);
    ghz.h(0).cx(0, 1).cx(1, 2);
    let (state, _stats) = statevector::<f64>(&ghz, &Options::default())?;
    println!("{:.4} {:.4}", state.get(0), state.get(7));
    Ok(())
}
