//! Choosing which qubit sits on which MPS site.
//!
//! A gate between qubits far apart along the chain costs SWAPs, and every
//! gate across a cut can double that bond's dimension. A good order keeps
//! interacting qubits close: the bond between sites `s` and `s + 1` is at
//! most `2^(number of two-qubit gates crossing it)`.

use std::collections::BTreeMap;

use faer::Mat;

use crate::circuit::{Circuit, Instruction};

/// Interacting pairs `(a, b)` with `a < b`, and how many gates couple them.
fn interactions(circuit: &Circuit) -> BTreeMap<(usize, usize), usize> {
    let mut edges = BTreeMap::new();
    for instruction in &circuit.instructions {
        if let Instruction::Gate { qubits, .. } = instruction {
            for (i, &a) in qubits.iter().enumerate() {
                for &b in &qubits[i + 1..] {
                    *edges.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
        }
    }
    edges
}

/// The cost of placing `order[s]` on site `s`: the largest number of gates
/// crossing any cut, then the total over all cuts.
pub fn cut_cost(order: &[usize], edges: &BTreeMap<(usize, usize), usize>) -> (usize, usize) {
    let n = order.len();
    let mut position = vec![0; n];
    for (s, &q) in order.iter().enumerate() {
        position[q] = s;
    }
    // Difference array over cuts 0..n-1 (cut s is between sites s, s+1).
    let mut delta = vec![0isize; n + 1];
    for (&(a, b), &w) in edges {
        let (l, r) = (position[a].min(position[b]), position[a].max(position[b]));
        delta[l] += w as isize;
        delta[r] -= w as isize;
    }
    let (mut running, mut max, mut total) = (0isize, 0, 0);
    for d in &delta[..n.saturating_sub(1)] {
        running += d;
        max = max.max(running as usize);
        total += running as usize;
    }
    (max, total)
}

/// Reverse Cuthill-McKee: breadth-first from a low-degree qubit of each
/// connected group, neighbours in increasing degree, reversed. Keeps the
/// bandwidth of the interaction graph small.
fn cuthill_mckee(n: usize, edges: &BTreeMap<(usize, usize), usize>) -> Vec<usize> {
    let mut neighbours = vec![Vec::new(); n];
    for &(a, b) in edges.keys() {
        neighbours[a].push(b);
        neighbours[b].push(a);
    }
    let degree: Vec<usize> = neighbours.iter().map(Vec::len).collect();
    for list in &mut neighbours {
        list.sort_by_key(|&q| (degree[q], q));
    }
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut starts: Vec<usize> = (0..n).collect();
    starts.sort_by_key(|&q| (degree[q], q));
    for start in starts {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(q) = queue.pop_front() {
            order.push(q);
            for &next in &neighbours[q] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
    }
    order.reverse();
    order
}

/// Spectral order: qubits sorted by their entry in the Fiedler vector (the
/// eigenvector of the graph Laplacian's second-smallest eigenvalue), which
/// minimises the weighted squared distance between interacting qubits.
fn spectral(n: usize, edges: &BTreeMap<(usize, usize), usize>) -> Option<Vec<usize>> {
    let mut laplacian = Mat::<f64>::zeros(n, n);
    for (&(a, b), &w) in edges {
        let w = w as f64;
        laplacian[(a, b)] -= w;
        laplacian[(b, a)] -= w;
        laplacian[(a, a)] += w;
        laplacian[(b, b)] += w;
    }
    let eigen = laplacian.self_adjoint_eigen(faer::Side::Lower).ok()?;
    let values = eigen.S().column_vector();
    let mut index: Vec<usize> = (0..n).collect();
    index.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
    let fiedler = index.get(1)?;
    let vectors = eigen.U();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        vectors[(a, *fiedler)]
            .total_cmp(&vectors[(b, *fiedler)])
            .then(a.cmp(&b))
    });
    Some(order)
}

/// The best qubit order among the circuit's own (`0..n`), reverse
/// Cuthill-McKee and the spectral order (whose dense eigendecomposition is
/// limited to 256 qubits), by [`cut_cost`]. The circuit's order wins ties,
/// and is kept outright when more than half of all pairs interact (as in a
/// QFT), since then no order is much better.
pub fn qubit_order(circuit: &Circuit) -> Vec<usize> {
    let n = circuit.num_qubits;
    let edges = interactions(circuit);
    let mut best: Vec<usize> = (0..n).collect();
    if edges.len() * 4 > n * n.saturating_sub(1) {
        return best;
    }
    let mut best_cost = cut_cost(&best, &edges);
    let mut candidates = vec![cuthill_mckee(n, &edges)];
    if (3..=256).contains(&n) {
        candidates.extend(spectral(n, &edges));
    }
    for order in candidates {
        let cost = cut_cost(&order, &edges);
        if cost < best_cost {
            best = order;
            best_cost = cost;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::random_grid_circuit;

    #[test]
    fn recovers_a_good_order_for_scrambled_labels() {
        let grid = random_grid_circuit(6, 6, 8, 1);
        let natural = cut_cost(&(0..36).collect::<Vec<_>>(), &interactions(&grid));
        // Relabel the qubits at random.
        let mut labels: Vec<usize> = (0..36).collect();
        let mut x = 7u64;
        for i in (1..36).rev() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            labels.swap(i, (x >> 33) as usize % (i + 1));
        }
        let mut scrambled = Circuit::new(36);
        for instruction in &grid.instructions {
            if let Instruction::Gate { gate, qubits } = instruction {
                let mapped: Vec<usize> = qubits.iter().map(|&q| labels[q]).collect();
                scrambled.gate(gate.clone(), &mapped);
            }
        }
        let edges = interactions(&scrambled);
        let identity = cut_cost(&(0..36).collect::<Vec<_>>(), &edges);
        let chosen = cut_cost(&qubit_order(&scrambled), &edges);
        assert!(
            chosen.0 <= natural.0 + natural.0 / 2,
            "chosen {chosen:?}, grid order {natural:?}"
        );
        assert!(
            chosen.0 * 2 < identity.0,
            "chosen {chosen:?}, scrambled order {identity:?}"
        );
        let mut sorted = qubit_order(&scrambled);
        sorted.sort();
        assert_eq!(sorted, (0..36).collect::<Vec<_>>());
    }
}
