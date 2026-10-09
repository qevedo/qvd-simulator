//! Thread pools pinned to the right cores.
//!
//! Intel hybrid CPUs (like the i9-14900K) mix performance and efficiency
//! cores. Linux lists them in `/sys/devices/cpu_core/cpus` and
//! `/sys/devices/cpu_atom/cpus`, and its scheduler does not reliably keep
//! bandwidth-bound work on P-cores, so placement is explicit here.

use std::fs;

/// Which logical CPUs the simulator's threads run on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Let the OS schedule `threads` threads anywhere.
    Unpinned { threads: usize },
    /// One thread per physical P-core (no hyper-threads).
    PerformanceCores,
    /// Every hardware thread of the P-cores.
    PerformanceThreads,
    /// One thread per physical core, P and E.
    AllCores,
    /// Every hardware thread.
    AllThreads,
    /// An explicit list of logical CPU ids.
    Cpus(Vec<usize>),
}

/// The CPU topology as Linux reports it.
#[derive(Clone, Debug)]
pub struct Topology {
    /// Logical CPUs of performance cores (empty on non-hybrid CPUs).
    pub performance: Vec<usize>,
    /// Logical CPUs of efficiency cores (empty on non-hybrid CPUs).
    pub efficiency: Vec<usize>,
    /// All online logical CPUs.
    pub all: Vec<usize>,
    /// For each logical CPU, the id of its physical core.
    core_of: Vec<Option<usize>>,
}

fn parse_cpu_list(text: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    cpus.extend(a..=b);
                }
            }
            None => cpus.extend(part.parse::<usize>().ok()),
        }
    }
    cpus
}

impl Topology {
    pub fn detect() -> Topology {
        let read = |path: &str| {
            fs::read_to_string(path)
                .map(|t| parse_cpu_list(&t))
                .unwrap_or_default()
        };
        let mut all = read("/sys/devices/system/cpu/online");
        if all.is_empty() {
            all = (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect();
        }
        let max = all.iter().copied().max().unwrap_or(0);
        let mut core_of = vec![None; max + 1];
        for &cpu in &all {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id");
            let package = format!("/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id");
            let core = fs::read_to_string(path)
                .ok()
                .and_then(|t| t.trim().parse::<usize>().ok());
            let pkg = fs::read_to_string(package)
                .ok()
                .and_then(|t| t.trim().parse::<usize>().ok())
                .unwrap_or(0);
            core_of[cpu] = core.map(|c| pkg << 16 | c);
        }
        Topology {
            performance: read("/sys/devices/cpu_core/cpus"),
            efficiency: read("/sys/devices/cpu_atom/cpus"),
            all,
            core_of,
        }
    }

    pub fn is_hybrid(&self) -> bool {
        !self.performance.is_empty() && !self.efficiency.is_empty()
    }

    /// The first logical CPU of each distinct physical core in `cpus`.
    fn one_per_core(&self, cpus: &[usize]) -> Vec<usize> {
        let mut seen = std::collections::HashSet::new();
        cpus.iter()
            .copied()
            .filter(|&cpu| match self.core_of.get(cpu).copied().flatten() {
                Some(core) => seen.insert(core),
                None => true,
            })
            .collect()
    }

    /// The logical CPUs for `placement`, or `None` for unpinned placement.
    pub fn cpus(&self, placement: &Placement) -> Option<Vec<usize>> {
        let performance = if self.is_hybrid() {
            &self.performance
        } else {
            &self.all
        };
        match placement {
            Placement::Unpinned { .. } => None,
            Placement::PerformanceCores => Some(self.one_per_core(performance)),
            Placement::PerformanceThreads => Some(performance.clone()),
            Placement::AllCores => Some(self.one_per_core(&self.all)),
            Placement::AllThreads => Some(self.all.clone()),
            Placement::Cpus(cpus) => Some(cpus.clone()),
        }
    }
}

/// Build a rayon pool for `placement`, pinning each worker to one CPU.
pub fn pool(placement: &Placement) -> rayon::ThreadPool {
    let topology = Topology::detect();
    let builder = rayon::ThreadPoolBuilder::new();
    let builder = match topology.cpus(placement) {
        None => {
            let threads = match placement {
                Placement::Unpinned { threads } => *threads,
                _ => unreachable!(),
            };
            builder.num_threads(threads)
        }
        Some(cpus) => {
            let ids: Vec<core_affinity::CoreId> = cpus
                .iter()
                .map(|&id| core_affinity::CoreId { id })
                .collect();
            builder
                .num_threads(ids.len().max(1))
                .start_handler(move |index| {
                    if let Some(id) = ids.get(index) {
                        core_affinity::set_for_current(*id);
                    }
                })
        }
    };
    builder.build().expect("failed to build the thread pool")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu_lists() {
        assert_eq!(parse_cpu_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
    }
}
