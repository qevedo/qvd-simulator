# Simulating Past the RAM Limit on One Machine, and Verified Large-Simulation Records

Scope: (a) out-of-core (SSD), compression and hybrid Schrödinger-Feynman methods that push generic circuit simulation past the RAM limit on one workstation; (b) verified records for single-node and supercomputer simulations. Target machine: i9-14900K (24 cores, AVX2), 62 GB RAM, one NVMe SSD with about 49 GB free (typical NVMe about 3-7 GB/s), no GPU. Research date: 2026-10-09.

Terminology used below:
- **Generic full state vector (SV)**: stores all 2^n amplitudes exactly (FP64 or FP32) and works for any circuit.
- **Special-purpose**: lossy or structure-dependent compression, tensor networks, hybrid Schrödinger-Feynman (SF) for a subset of amplitudes, decision diagrams, MPS, stabilizer methods. These only reach large n for some circuit classes or outputs.

---

## Q1. Memory arithmetic: bytes per amplitude and maximum dense qubits per RAM size

### Takeaway
A dense state vector needs 2^(n+4) bytes in complex128 and 2^(n+3) bytes in complex64. With 62 GB the target machine holds **31 qubits (complex128) or 32 qubits (complex64)**. A 2-byte "byte-encoded" amplitude, as in JUQCS, would allow 34. Exact 40 qubits needs 16 TiB (complex128) or 8 TiB (complex64), roughly 130-260x this machine's RAM.

### Cited Findings
- A standard SV simulation needs 2^(n+4) bytes for n qubits (16 B per complex128 amplitude). — [BMQSim, arXiv:2410.14088](https://arxiv.org/html/2410.14088v1)
- 40 qubits in double precision needs 16 × 2^40 B = 16 TiB (1.76e13 B, "17.6 TB" in decimal units). The same lecture notes say a laptop handles about 20-30 qubits in SV mode. — [Lecture notes arXiv:2601.03035](https://arxiv.org/pdf/2601.03035); [PennyLane forum](https://discuss.pennylane.ai/t/hardware-requirement/4829)
- Google's qsim docs give the rule of thumb "8 × 2^N bytes" (complex64), so memory doubles per qubit. They recommend laptops (8 GB+) for "noiseless simulations that use fewer than 29 qubits" and noisy simulations under 18 qubits. Multi-threaded SV performance benefits from memory bandwidth "above 100 GB/s". — [qsim: choose hardware](https://quantumai.google/qsim/choose_hw)
- JUQCS-50 paper: a 50-qubit FP64 state needs 2^14 TiB (16 PiB). Byte encoding, at 2 bytes per complex amplitude, cuts this 8x to 2,048 TiB (about 2 PiB). — [De Raedt et al., arXiv:2511.03359 v3](https://arxiv.org/html/2511.03359v3)
- Simulator overheads matter: ProjectQ is reported to have about 1.5x memory overhead during allocation, and QuEST clones the state vector in distributed mode. — via search summary of [arXiv:1801.01037](https://arxiv.org/pdf/1801.01037) (secondary; not verified in full text)
- Supercomputer memory ceilings per Wu et al. (2019), Table 1, with the maximum qubits for arbitrary circuits: Summit 2.8 PB → 47; Sierra 1.38 PB → 46; Sunway TaihuLight 1.31 PB → 46; Theta 0.8 PB → 45. — [Wu et al., SC19, arXiv:1911.04034](https://arxiv.org/pdf/1911.04034)

### Inferences
Arithmetic from the formulas above, with GiB = 2^30 B. The state vector alone must fit, with headroom for the OS and buffers, so a state exactly equal to RAM does not count:

| n qubits | complex128 (16 B) | complex64 (8 B) | 2-byte encoded (JUQCS-style) |
|---|---|---|---|
| 30 | 16 GiB | 8 GiB | 2 GiB |
| 31 | 32 GiB | 16 GiB | 4 GiB |
| 32 | 64 GiB | 32 GiB | 8 GiB |
| 33 | 128 GiB | 64 GiB | 16 GiB |
| 34 | 256 GiB | 128 GiB | 32 GiB |
| 35 | 512 GiB | 256 GiB | 64 GiB |
| 36 | 1 TiB | 512 GiB | 128 GiB |
| 37 | 2 TiB | 1 TiB | 256 GiB |
| 38 | 4 TiB | 2 TiB | 512 GiB |
| 40 | 16 TiB | 8 TiB | 2 TiB |
| 45 | 512 TiB | 256 TiB | 64 TiB |
| 48 | 4 PiB | 2 PiB | 512 TiB |
| 50 | 16 PiB | 8 PiB | 2 PiB |

Maximum dense qubits that fit with headroom:

| Machine RAM | complex128 | complex64 | 2-byte encoded |
|---|---|---|---|
| 62 GB (target) / 64 GB | 31 | 32 | 34 |
| 128 GB | 32 | 33 | 35 |
| 256 GB | 33 | 34 | 36 |
| 1 TB | 35 | 36 | 38 |
| 2 TB | 36 | 37 | 39 |

- A 64 GiB machine cannot run 32 qubits in complex128: the state alone is 64 GiB.
- Each extra qubit doubles both the memory and the work per gate.
- "40 qubits exact, generic" needs ≥8 TiB even in FP32. That is a multi-socket big-memory server (or a cluster), or an SSD array of that size.

### Gaps
- No primary source found for the exact simulator overhead factors (scratch buffers) of qsim, Qiskit Aer or QuEST on a single node.

---

## Q2. Out-of-core / SSD-backed state-vector simulation

### Takeaway
Storage-backed SV simulation is real and gives exact results. Demonstrated systems reach **42 qubits (SnuQS, workstation with many NVMe SSDs and HDDs)**, **up to 43 qubits (Keio/Tsukuba FPGA board with 32 SATA disks)** and **47 qubits (BMQSim, SSD plus lossy compression, structured circuits)**. Every one of them needed storage arrays far larger than one consumer NVMe. Their speed is set by storage bandwidth and the number of passes over the state. Circuits are partitioned so that each pass applies many gates to an in-RAM chunk, and only "global-qubit" gates force a data-permutation pass.

### Cited Findings
- **SnuQS (Seoul National Univ., Thunder lab):** a full-state simulator that uses HDDs and NVMe SSDs "to enlarge the available main memory capacity at a small cost."
  - Techniques: automatic circuit partitioning based on qubit permutation and amplitude layout; prefetching amplitudes so computation and I/O overlap; a "three-step qubit permutation" that keeps I/O in large contiguous blocks for high bandwidth.
  - Claims: a **42-qubit quantum-supremacy circuit on a workstation-level system with many NVMe SSDs and HDDs**, where a workstation's in-memory limit is about **34 qubits**. Also "256 times larger" circuits than conventional full-state simulators, at about 300x lower cost than a DDR4-only system.
  - Self-reported; no runtimes, bandwidth or SSD counts on the page.
  - [SNU Thunder lab page](https://thunder.snu.ac.kr/?p=875)
- **Related patent (EP4418162):** a host connected to HDDs, SSDs or NVMe devices simulates sub-circuit by sub-circuit, uses a three-step data permutation to maximize bandwidth, and prefetches. It argues supercomputer-based simulation struggles above about 50 qubits. — [EPO EP4418162](https://data.epo.org/publication-server/rest/v1.2/patents/EP4418162NWA1/document.html) (via search summary)
- **NTU (National Taiwan Univ.) NVM-based simulator (thesis):** treats non-volatile memory as slower I/O-accessed storage and merges gates into k-qubit unitaries to cut the number of fetches. It is benchmarked against QuEST and can simulate beyond DRAM capacity "at a reasonable speed". A related NTU thesis says circuit size is "bounded by the volume sizes of the adopted SSDs" and performance "limited by the maximum bandwidth of the storage system". — [NTU thesis 85098](https://tdr.lib.ntu.edu.tw/handle/123456789/85098?mode=full); [NTU thesis 87736](https://tdr.lib.ntu.edu.tw/handle/123456789/87736) (via search summaries)
- **Keio Univ. + Univ. of Tsukuba (press release, 11 Dec 2023):** an FPGA (reconfigurable LSI) board connected to 32 SATA disks performs SV simulation of "40 or more qubits", up to **43 qubits**. It is described as relatively inexpensive and installable in a research lab. — [Keio press release](https://www.keio.ac.jp/en/press-release/20231211-1/)
  - Secondary coverage adds 32 SATA SSDs totalling 8 TB, a desktop system cost of about 4 million yen, and **about 3 hours for a 40-qubit simulation**. These figures are not in the primary page. — [Archyde](https://www.archyde.com/groundbreaking-fpga-board-for-quantum-computer-simulations-developed-by-keio-university-and-university-of-tsukuba/)
- **BMQSim (arXiv:2410.14088, Oct 2024):**
  - Design: a two-level memory system that spills compressed state-vector blocks to SSD when CPU and GPU memory run out.
  - Main test box: 28-core Xeon Gold 6238R, 2x 16 GB GPUs, 128 GB DDR4, 4 TB Samsung 870 EVO **SATA** SSD.
  - Reach: **42 qubits in memory and up to 47 with SSD fallback** (structured circuits).
  - SSD test: RAM capped at 8 GB. The SSD was only touched above about 32 qubits for some circuits; Ising kept 39% of its blocks on SSD at 32 qubits and 70% at 33.
  - Overhead: two-level management cost about 0.7% performance on average.
  - No SSD pass counts or throughput figures are reported.
  - [BMQSim HTML](https://arxiv.org/html/2410.14088v1)
- **Pednault et al. (IBM, 2017):** said secondary storage would let 7×7 circuits be simulated to arbitrary depth, with an estimate of under a day for depth 83. This was **estimated, not run**; the reported experiments used main memory only. — [arXiv:1710.05867](https://ar5iv.labs.arxiv.org/html/1710.05867)
- **QuEST with SSD swap (Edinburgh MSc project):** OS-level SSD swap extends memory but performs worse than RAM; the project suggests CXL memory as an alternative. — [Edinburgh MSc project](https://project-archive.inf.ed.ac.uk/msc/20247458/msc_proj.pdf) (via search summary)
- **Q-GPU (GPU offloading):**
  - Idea: keep the state in host memory and stream it to the GPU, using "dynamic zero state amplitude pruning and lossless compression of non-zero amplitudes to reduce the data transfer time."
  - Source: secondary only (a Scribd upload); the primary HPCA paper was not retrieved. — [Scribd doc](https://www.scribd.com/document/785129302/Accelerating-Quantum-Computer-Simulations-Using-GPUs)
  - BMQSim also notes that CPU↔GPU data movement "incurs significant overhead". — [BMQSim](https://arxiv.org/html/2410.14088v1)
- **DSLSQS (2025):** a "Distributed Shared Layered Storage Quantum Simulator" in which multiple nodes share a common disk pool. — [arXiv:2508.15542](https://arxiv.org/pdf/2508.15542) (title and summary only)

### Inferences
Arithmetic for the target machine, assuming about 7 GB/s read and 5 GB/s write sequential:
- **Capacity:**
  - 49 GB free cannot even hold a 33-qubit complex64 state (64 GiB), so SSD spilling is impossible on the current disk.
  - 38 qubits in complex64 (2 TiB) would need at least a dedicated 2-4 TB NVMe.
  - 40 qubits in complex64 needs about 8.8 TB of storage; complex128 needs about 17.6 TB.
- **Time per pass:** one full pass (read + write) over a 40-qubit complex64 state is about 8.8e12/7e9 + 8.8e12/5e9 ≈ 1,260 s + 1,760 s ≈ **50 minutes**, if the drive sustains these rates.
  - The same pass from DRAM at about 100 GB/s would take about 3 minutes.
  - So SSD-backed simulation is about 15-30x slower per pass than in-RAM, before thermal throttling or write-cache exhaustion. Consumer drives often drop well below their spec once the SLC cache fills (an assumption; not measured here).
- **Passes per circuit:**
  - With about 32 GiB of RAM chunks in complex64, about 32 qubits are "local" and 8 qubits "global" for n = 40.
  - Each partition (pass) applies every gate that acts on local qubits. A qubit permutation (a full read/write pass) is needed whenever a gate touches global qubits.
  - The number of passes is therefore the number of partitions, which SnuQS-style partitioners minimize. Deep random circuits might need tens of passes, giving many hours to days.
  - This is consistent with the about 3 h for 40 qubits reported for the Keio board, which used 32 parallel disks.
- **SSD endurance:** each pass writes about 8.8 TB at 40 qubits in complex64. A consumer 1-2 TB NVMe rated around 600-1,200 TBW (typical vendor ratings, not sourced here) would wear out after roughly 70-140 passes. A single consumer NVMe is a poor fit for routine 38-40-qubit exact simulation.

### Gaps
- Could not retrieve the primary SnuQS paper (venue, runtimes, I/O bandwidth, SSD count). Figures are from the lab's own page.
- No primary source for the Keio 40-qubit runtime (about 3 h) or the 8 TB capacity; only secondary coverage.
- No published per-pass counts for out-of-core simulators on standard benchmark circuits were found.
- Intel-QS and QuEST have no documented built-in SSD out-of-core mode in the sources found. QuEST + SSD is only via OS swap in a student project.

---

## Q3. Compression-based simulation (lossless and lossy)

### Takeaway
Compression helps hugely for **structured, low-entanglement states**: Grover, GHZ and Bernstein-Vazirani compress about 400-80,000x. It gives only about **5-21x for random, QAOA and QFT circuits**, which is worth about 2-4 extra qubits, at a large time cost and some fidelity loss. The 61-qubit Wu et al. result is a Grover circuit, not a generic circuit. On a workstation, compression credibly gives about +3-4 qubits for "hard" circuits (about 34-35 qubits at 62 GB) and 40+ only for highly compressible circuits.

### Cited Findings
**Wu et al., "Full-State Quantum Circuit Simulation by Using Data Compression", SC19** ([arXiv:1911.04034](https://arxiv.org/pdf/1911.04034); [SC19 page](https://www.sc19.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap180.html)):
- **Hybrid scheme:**
  - Zstd lossless compression early on, while the ratio is high enough.
  - It switches to a tailored SZ-based lossy, point-wise-relative-error compressor with adaptive error bounds (1E-5 to 1E-1) when memory runs short.
  - The state is kept in compressed blocks; at most two blocks per rank are decompressed per gate step, with a 64-line compressed-block cache.
  - Fidelity lower bound: F ≥ ∏(1 − δ_i).
- **Compressor assessment (36-qubit QAOA and random-circuit datasets):**
  - SZ reaches up to about 100:1 on qaoa_36, while ZFP is always under 10:1.
  - On the supremacy random circuit sup_36: SZ about 28-126, ZFP 4.25-12.6, depending on error bound.
  - SC19 Table 2 shows the much lower minimum ratios actually reached during full runs (see the next item).
- **Results (SC19 Table 2):**

| Run | Original memory | Hardware | Ratio | Fidelity | Time |
|---|---|---|---|---|---|
| Grover 61 qubits (314 gates) | 32 EB | 4,096 Theta nodes (768 TB) | 7.39×10^4 | 0.996 | 8.14 h (93 s/gate) |
| Grover 59 qubits | 8 EB | 4,096 Theta nodes | 8.26×10^4 | 0.996 | 3.48 h |
| Grover 47 qubits | 2 PB | 128 nodes (24 TB) | 1.06×10^4 | 1.0 | 0.49 h |
| Random circuit, 5×9 = 45 qubits, depth 11 | 512 TB | 1,024 nodes (192 TB) | 6.03 | 0.987 | 4.87 h |
| Random circuit, 6×7 = 42 qubits, depth 11 | 64 TB | 128 nodes | 9.40 | 0.993 | 8.64 h |
| Random circuit, 6×6 = 36 qubits | 1 TB | **one node, 192 GB** | 8.16 | **0.933** | **7.96 h** (173.65 s/gate, 165 gates) |
| Random circuit, 7×5 = 35 qubits | 512 GB | **one node** | 10.05 | 0.985 | 6.23 h |
| QAOA 45 qubits | – | 1,024 nodes | 5.38 | 0.895 | 13.34 h |
| QAOA 43 qubits | – | 256 nodes | 4.85 | 0.999 | – |
| QAOA 42 qubits | – | 128 nodes | 9.25 | 0.999 | – |
| **QFT 36 qubits (3,258 gates)** | 1 TB | **one node, 192 GB** | 21.34 | 0.962 | **78.98 h** (87 s/gate) |

- **Time breakdown:** for the non-Grover runs, compression plus decompression take about **63-96% of runtime**; for example, QFT-36 is 57.9% compression plus 37.7% decompression, and only 1.9% computation.
- **Authors' caveats:**
  - "It does not work as well on random circuits"; depth was limited to 11.
  - "More entanglement leads to less compressible vectors."
  - For general circuits the approach adds "2 to 16 qubits".
  - Projected 63 qubits on Summit and 64 on Aurora are estimates, not runs.
- **Hardware:** each Theta node is a 64-core Xeon Phi 7230 with 16 GB MCDRAM and 192 GB DDR4. — [UChicago news](https://www.computerscience.uchicago.edu/news/quantum-compression/)
- The University of Chicago reports the 61-qubit Grover run had **0.4% error**. — [UChicago news](https://www.computerscience.uchicago.edu/news/quantum-compression/)

**Wu et al., "Memory-Efficient Quantum Circuit Simulation by Using Lossy Data Compression"** (PMES workshop, 2018; 2 pages):
- Memory reduced to 16.5% of the original for QFT and to 2.24E-06 for Grover.
- It "suggests" deep circuits up to 63 qubits with 0.8 PB; this is a projection.
- [arXiv:1811.05630](https://arxiv.org/abs/1811.05630)

**BMQSim (Oct 2024)** ([arXiv:2410.14088](https://arxiv.org/html/2410.14088v1)):
- **Design:**
  - GPU lossy compressor with point-wise error control; relative bounds are converted to absolute ones via log scaling.
  - Circuit partitioning cuts compression events; for a 33-qubit QFT, from 2,673 (one per gate) to 28 (one per stage).
  - Compression, transfer and compute are pipelined on CUDA streams.
- **Single workstation results** (28-core Xeon, 128 GB RAM, 2x 16 GB GPUs):
  - 42 qubits for cat_state, bv and ghz; 37 for cc; **36 for QFT**; **35 for Ising, QSVM and QAOA**.
  - Up to 47 with SSD. Other simulators reach about 30 on average (Qiskit 33, cuQuantum 31, HyQuas 29, SV-Sim 26).
  - Claims "14 more qubits at best, 10 on average".
- **Memory-reduction ratios:** cat/GHZ about 679x; BV 425x; cc 15.5x; QFT 10.5x.
- **Fidelity:** "above 0.99" in the abstract and §5.3, but "above 0.999 in almost all cases" in the conclusion, so the paper is internally inconsistent.
- **Speed:**
  - About 1,385x faster than the SC19 CPU simulator and 539x faster than its GPU port.
  - About 75x faster than SV-Sim; roughly on par with Qiskit-Aer (time ratio 0.99-1.05).
  - Slower than HyQuas (about 12x) and cuQuantum (about 9x) where those fit in memory.
  - In SC19-Sim, compression was about 61% of simulation time.

**MEMQSim (SC23 workshop, arXiv:2309.16979):** memory-efficient, modularized SV simulation using compression across CPUs and GPUs; no SSD use found. — [arXiv:2309.16979](https://arxiv.org/pdf/2309.16979) (title/summary only)

**JUQCS adaptive byte encoding** (a form of lossy compression used for records):
- 2018/2019: adaptive encoding of the wave function cuts memory about 8x, enabling 48 qubits on the K computer and Sunway TaihuLight. — [De Raedt et al., arXiv:1805.04708](https://arxiv.org/pdf/1805.04708)
- 2025: 2 bytes per complex amplitude in a polar (magnitude/phase) representation, tuned on the fly.
  - Costs about 2-3x extra compute for gates such as Hadamard.
  - The Hadamard benchmark output is "FP64-accurate by design", but the 50-qubit adder showed deviations in x/y expectation values, e.g. 0.48 or 0.52 instead of 0.5.
  - [arXiv:2511.03359 v3](https://arxiv.org/html/2511.03359v3)

### Inferences
- **For a workstation (62 GB)**, using the measured ratios above:
  - Random, QAOA and QFT circuits at about 5-20x buy about 2-4 qubits, i.e. about 34-35 qubits in complex128, at 10-100x slowdown and fidelity of about 0.93-0.99. This is consistent with Wu's one-node runs: 36-qubit random circuit in 8 h; 36-qubit QFT in 79 h on a 192 GB node.
  - 40 qubits in complex128 needs a ratio of about 270x (16 TiB / 60 GiB). Only GHZ/cat, BV, Grover-like and other low-entropy states have shown such ratios.
- **A 2-byte encoding (JUQCS-style) is the most predictable trick:** a fixed 8x saving relative to complex128, or 4x relative to complex64. That is about +3 qubits over complex128 (34 qubits in 62 GB) for any circuit, with bounded precision loss and about 2-3x compute cost.
- **Compression time dominates on CPUs.** A fast AVX2 SV kernel is memory-bandwidth bound. Adding SZ-style compression makes it compressor-bound; SC19 shows compression plus decompression at about 60-95% of runtime.

### Gaps
- No source found giving compression ratios for deep (depth >11) random circuits. Wu et al. explicitly avoided them, which strongly suggests ratios approach 1 as entanglement saturates.
- BMQSim's fidelity statement is internally inconsistent (0.99 vs 0.999).
- No CPU-only (no-GPU) BMQSim results.

---

## Q4. Hybrid Schrödinger-Feynman (SF) and amplitude-subset methods

### Takeaway
Hybrid SF cuts the qubits into two halves, each simulated as a dense SV. The cost is about 2^d × (2^|A| + 2^|B|), where d is the number of cut (cross-partition) gates. Memory therefore stays at about 2^(n/2), so a 50-60-qubit circuit fits in workstation RAM, but runtime explodes with circuit depth. It is practical for **shallow circuits or for a few amplitudes or low-fidelity sampling**. It is not a way to obtain the full 2^n state.

### Cited Findings
- **Cost model:** gates crossing the cut determine the number of independent simulation runs. For CZ-type cross gates (Schmidt rank 2), d cross gates give a 2^d path space. The qsim hybrid (qsimh) does dense SV simulation of each half, so total cost is O(2^d · (2^|A| + 2^|B|)). — [Survey, arXiv:2311.16505](https://arxiv.org/pdf/2311.16505) (via search summary)
- **Markov, Fatima, Isakov, Boixo, "Quantum supremacy is both closer and farther than it appears" (arXiv:1807.10749, 2018):**
  - The "Rollright" massively-parallel simulator needs no inter-process communication.
  - It simulated approximate sampling for a **7×8 (56-qubit) circuit at depth 1+40+1**, estimating **$35,184** on Google Cloud for 1M bitstring probabilities at **0.5% fidelity**, and about $1M at depth 1+48+1.
  - "Simulation costs scale linearly with fidelity", because only a fraction of the paths is summed.
  - [arXiv:1807.10749](https://arxiv.org/abs/1807.10749)
- **Google 2019:** it estimated that the 53-qubit, 20-cycle Sycamore circuit at 0.1% fidelity via hybrid SF would take about **50 trillion core-hours** on Google Cloud. A later 60-qubit paper's SF estimate lists the 53-qubit 20-cycle case at 1,332 years on 7,630,848 CPU cores (fidelity 0.224%). — via search summary of [arXiv:2109.03494](https://arxiv.org/pdf/2109.03494) and [arXiv:2311.16505](https://arxiv.org/pdf/2311.16505)
- **Burgholzer, Bauer, Wille, "Hybrid Schrödinger-Feynman Simulation of Quantum Circuits With Decision Diagrams" (IEEE QCE 2021):**
  - Combines SF with decision diagrams (DDs) to use multiple cores.
  - Some hard circuits finished in minutes instead of not finishing within a day.
  - Implemented in MQT DDSIM with a "dd" mode (sample from the final DD) and an "amplitude" mode (each path via DDs, summed in arrays; more memory but often faster).
  - [arXiv:2105.07045](https://www.arxiv.org/pdf/2105.07045); [DDSIM docs](https://ddsim.readthedocs.io/en/latest/simulators/HybridSchrodingerFeynman.html)
- **Pednault et al. (IBM, Oct 2017):** "deferring tensor contractions" (Schrödinger-style slicing with tensor contractions, not a dense full SV).
  - Computed **all amplitudes of a 49-qubit (7×7), depth-27** circuit using just over 4.5 TB of main memory.
  - Computed an arbitrary slice of **2^37 amplitudes of a 56-qubit (8×7), depth-23** circuit with 3.0 TB.
  - Previous methods would have needed 8 PB and 1 EB respectively.
  - Run on LLNL's Vulcan Blue Gene/Q, September 2017.
  - [arXiv:1710.05867](https://ar5iv.labs.arxiv.org/html/1710.05867)
- **Tensor networks on few GPUs (amplitude subsets):**
  - Pan & Zhang (PRL 128, 030501, Jan 2022) computed exact amplitudes of 2×10^6 correlated bitstrings for Sycamore 53-qubit, 20-cycle using **60 GPUs**.
  - They also obtained the full state vector of a 43-qubit, 14-cycle "simplifiable" Sycamore circuit on **one GPU**.
  - [arXiv:2103.03074](https://www.arxiv.org/abs/2103.03074); [PRL](https://www.doi.org/10.1103/physrevlett.128.030501)

### Inferences
- **For the target machine:** hybrid SF with two halves of about 25 qubits (2^25 × 16 B = 512 MiB each) can handle 50 qubits in RAM easily. The cost is 2^d path-pairs, so with d ≈ 20-30 cut CZs the work is about 10^6-10^9 half-simulations.
  - This is feasible for **shallow circuits, or for single amplitudes / small batches / low-fidelity sampling** (cost scales linearly with target fidelity).
  - It is infeasible for deep random circuits at full fidelity.
- **SF and tensor methods yield amplitudes, not the full state.** Producing all 2^40 amplitudes would still need 8-16 TiB of output storage.

### Gaps
- Could not retrieve qsimh documentation pages directly (the fetched tutorial was a plain qsim 32-qubit example). Exact qsimh API and benchmark numbers are not cited here.
- No primary measured single-workstation runtimes for qsimh found.

---

## Q5. Verified records: supercomputer vs single node (chronological)

### Takeaway
- **The record for a universal, exact-ish full state vector is 50 qubits (JUQCS-50 on JUPITER, Nov 2025).** It used 16,384 GH200 superchips and 2-byte amplitude encoding (about 2 PiB); FP64 tops out at 47 qubits and FP32 at 48.
- Larger "qubit counts" (49/56 IBM 2017; 61 Grover SC19; 53-qubit Sycamore) used slicing, compression of structured states, or tensor networks, not generic dense SV.
- On a single node, practical exact SV is about **30-34 qubits**: qsim 34 qubits on a 60-vCPU cloud VM; "workstation limit about 34 qubits" per SnuQS; 33 qubits FP64 on one GH200. Up to **36-39 qubits** is possible on extreme nodes or with byte encoding.

### Cited Findings
| Date | Who / system | Qubits | Method | Key details | Source |
|---|---|---|---|---|---|
| 2007 | JUMPIQCS (Jülich) | 36 | Dense SV | Up to 4,096 processors, up to 1 TB | [arXiv:1805.04708](https://arxiv.org/pdf/1805.04708) |
| Apr 2017 (SC17) | Häner & Steiger (ETH), Cori II (NERSC) | 45 | Dense SV, supremacy-style circuits | 8,192 nodes, about 0.5 PB; about 0.428 PFLOPS; communication about 75% of time (per HPCwire) | [arXiv:1704.01127](https://arxiv.org/pdf/1704.01127); [HPCwire](https://www.hpcwire.com/2017/04/13/doe-supercomputer-45-qubit-quantum-computer/) |
| Oct 2017 | Pednault et al. (IBM), Vulcan BG/Q (LLNL) | 49 (all amplitudes), 56 (2^37-amplitude slice) | Tensor slicing / deferred contraction, **not** dense SV | 4.5 TB / 3.0 TB | [arXiv:1710.05867](https://ar5iv.labs.arxiv.org/html/1710.05867) |
| 2018-19 (CPC 2019) | JUQCS-E (De Raedt et al.), K computer & Sunway TaihuLight | 48 | SV with adaptive encoding (about 8x compression) | 48-qubit Hadamard benchmark: 3,102 s on K, 8,548 s on TaihuLight; Shor runs up to 48 qubits (factoring 65,531) | [arXiv:1805.04708](https://arxiv.org/pdf/1805.04708); [arXiv:2511.03359](https://arxiv.org/html/2511.03359v3) |
| Nov 2019 (SC19) | Wu et al., Argonne Theta | 61 (Grover only) | SV + lossless/lossy compression | 4,096 nodes, 768 TB, ratio 7.39×10^4, fidelity 0.996, 8.14 h. Random circuits only to 45 qubits at depth 11 | [arXiv:1911.04034](https://arxiv.org/pdf/1911.04034) |
| Nov 2021 (SC21 Gordon Bell) | Liu et al., "Closing the Quantum Supremacy Gap", new Sunway (SWQSIM) | 53 (Sycamore-class) | Tensor network | 304 s; 41.9M cores; 1.2 EFLOPS single / 4.4 EFLOPS mixed precision. Disputed: ORNL's Liakh says complexity was "dialed back" and only "sampling from the space of 21 qubits" | [HPCwire](https://www.hpcwire.com/2021/11/18/2021-gordon-bell-prize-goes-to-exascale-powered-quantum-supremacy-challenge/); [Next Platform](https://www.nextplatform.com/hpc/2021/11/18/chinas-exascale-quantum-simulation-not-all-it-appears/1659235) |
| Jan 2022 | Pan & Zhang (PRL 128, 030501) | 53 (2M correlated amplitudes); 43 (full SV, simplifiable circuit) | Big-batch tensor network | 60 GPUs; 43-qubit/14-cycle full state on 1 GPU | [arXiv:2103.03074](https://www.arxiv.org/abs/2103.03074) |
| Aug 2022 | Pan, Chen & Zhang (PRL 129, 090502) | 53, 20 cycles | Tensor network sampling | 1M uncorrelated samples, fidelity about 0.0037, about 15 h on 512 GPUs | via search summary of [Quantum Insider](https://thequantuminsider.com/2021/03/05/scientists-say-they-used-classical-computers-to-outperform-googles-sycamore-qc/) / [Cambridge talk](https://webapp.prod.talks.gcp.uis.cam.ac.uk/talk/index/183923/) |
| Mar 2022 | Fujitsu mpiQulacs, "Todoroki" 64-node A64FX cluster | 36 | Distributed dense SV | Fujitsu's "world's fastest 36-qubit simulator"; a 40-qubit target was set for Sep 2022 | [arXiv:2203.16044](https://arxiv.org/pdf/2203.16044); [Fujitsu PR](https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2022/0330-01.html) |
| 2024-26 | Fujitsu 40-qubit system, 1,024 FX700 (A64FX) nodes | 40 | Distributed dense SV (Qulacs + MPI) | Offered for Fujitsu's simulator challenges (2024; 2025-26). 1,024 × 32 GiB = 32 TiB fits 40 qubits in complex128 (16 TiB) | [Fujitsu challenge 2024](https://global.fujitsu/en-global/technology/research/article/topics/202405-quantum-simulator-challenge); [arXiv:2402.11878](https://arxiv.org/pdf/2402.11878) |
| Dec 2023 | Keio + Tsukuba FPGA board, 32 SATA disks | 40+ (up to 43) | Storage-backed dense SV | Lab-affordable; about 3 h for 40 qubits (secondary) | [Keio PR](https://www.keio.ac.jp/en/press-release/20231211-1/) |
| Oct 2024 | BMQSim, one workstation (128 GB, 2 GPUs, SATA SSD) | 42 (GHZ/BV/cat), 35-36 (QFT/QAOA/Ising); 47 with SSD | Lossy compression + SSD | Fidelity >0.99 | [arXiv:2410.14088](https://arxiv.org/html/2410.14088v1) |
| **Nov 2025** (FGCS 2026) | **JUQCS-50, JUPITER (Jülich + NVIDIA)** | **50** | Universal SV with 2-byte adaptive encoding | 4,096 nodes / 16,384 GH200; about 2 PiB; FP64 max 47, FP32 max 48; per-gate 36-qubit time 2.83 s vs 47.03 s on K (16.6x) | [arXiv:2511.03359 v3](https://arxiv.org/html/2511.03359v3); [Quantum Computing Report](https://quantumcomputingreport.com/julich-supercomputing-centre-utilizes-jupiter-exascale-hardware-to-set-50-qubit-quantum-simulation-benchmark/) |
| Dec 2025 | Quantum Rings (Wold & Kasirajan), ASU Sol | 53 (Sycamore-class) | Proprietary approximate method (GPU builds state, CPU jobs sample) | 20-cycle, 2.5M shots in 4,536 s over 100 CPU jobs; 14-cycle XEB 0.549. Prior 2024 CPU-only single-node run about 2.5 days | [arXiv:2512.07311](https://arxiv.org/html/2512.07311v1) |
| Apr 2026 | Osaka Univ. QIQB + Fixstars, ABCI-Q | 41 (Fe₂S₂ IQPE circuit) | GPU-cluster SV (chemqulacs-gpu) | Up to 1,024 H100 GPUs; "exceeded the previous limit of 40 qubits" for SV-based quantum-chemistry simulation | [Quantum Insider](https://thequantuminsider.com/2026/04/02/quantum-circuit-simulation-beyond-40-qubits/) |
| Apr 2026 | Montanez-Barrera & Michielsen, JUPITER | 48 (noiseless, up to 3,384 two-qubit gates) | Large-scale SV (JUQCS family implied) | 16,384 GH200; used to benchmark Quantinuum Helios-1 (98 qubits) | [arXiv:2604.26423](https://arxiv.org/abs/2604.26423) |

**Single-node reference points:**
- JUPITER, single node:
  - One GH200 superchip ran a **33-qubit FP64** benchmark (128 GB) and a **36-qubit byte-encoded** run.
  - One node (4 superchips) ran **39 qubits** using 58 Wh. — [arXiv:2511.03359 v3](https://arxiv.org/html/2511.03359v3)
- qsim CPU benchmark on a c2-standard-60 (60 vCPU) cloud VM, depth 20: **30 qubits ≈ 13.5 s; 34 qubits ≈ 292 s**. GPU: 1x A100-80GB reaches 33 qubits; 8x A100-80GB reaches 36 qubits (about 17.6 s for 100k samples). — [qsim docs](https://quantumai.google/qsim/choose_hw)
- SnuQS: a conventional workstation's full-state limit is "about 34 qubits". — [SNU](https://thunder.snu.ac.kr/?p=875)
- Wu SC19: a single Theta node (192 GB) ran compressed 35-36-qubit random circuits (depth 11) in about 6-8 h and a 36-qubit QFT in about 79 h. — [arXiv:1911.04034](https://arxiv.org/pdf/1911.04034)
- A 2018 cluster study reported 30 qubits on a single node and 33 qubits over 64 nodes. — via search summary of [arXiv:1801.01037](https://arxiv.org/pdf/1801.01037)
- NVIDIA cuStateVec material shows a 40-qubit SV distributed over 32 A100 nodes (multi-node, not single-node). — via search summary of [Morino slides](https://qsw.phys.s.u-tokyo.ac.jp/assets/files/20231225_morino.pdf)

### Inferences
- **Generic exact dense SV:**
  - About 45 qubits (2017) → 47 FP64 / 48 FP32 / 50 byte-encoded (2025) at the supercomputer scale.
  - The 2025 50-qubit record needed about 2 PiB, i.e. about 35,000x the target machine's RAM.
- **"40 qubits generic, exact" is a multi-node achievement:** 1,024 A64FX nodes (Fujitsu), 32+ A100 nodes (NVIDIA), up to 1,024 H100s (Osaka), or an FPGA plus 32-disk storage array (Keio).

### Gaps
- Could not retrieve Tom's Hardware's "46 qubits on world's top supercomputers" article body, so the details (systems, date) are unverified.
- The JUQCS-50 speedup figure conflicts across versions and sources: search snippets of v1/v2 and press coverage say 11.4x; the v3 abstract and HTML say 16.6x. Use v3 (16.6x) as the latest.
- The press-release date for the earlier 48-qubit record conflicts: SciTechDaily says 2019, HPCwire says 2022 (both per search summaries). The CPC paper (arXiv:1805.04708) dates the work to 2018-2019.
- No measured total wall time for the 50-qubit run was found in the paper; only per-gate and estimated network figures.
- No documented single-node CPU-only run at 38-39 qubits (e.g. on a 4-8 TB RAM server) was found.
- Not verified: whether Fujitsu formally announced achieving the 40-qubit system in 2022. Only later challenge pages describe it as existing.

---

## Q6. What "40 qubits on one normal machine" credibly means

### Takeaway
Credible sources do **not** support exact, generic 40-qubit state-vector simulation on a normal workstation. It needs 8-16 TiB of state, and the 2023-2026 demonstrations of 40-43 qubits used clusters, dedicated FPGA-plus-disk-array hardware, or circuit-specific compression. "40 qubits on a laptop or workstation" is only true for **special circuit classes or outputs**:
- low-entanglement circuits (MPS)
- sparse or factorized states
- Clifford-dominated circuits (stabilizer)
- highly compressible structured states (GHZ, BV, Grover)
- single amplitudes or small batches (hybrid SF / tensor networks)
- low-fidelity approximate sampling

### Cited Findings
- A 40-qubit SV is about 1,000x larger than a 16 GB laptop can hold, so a plain laptop simulation is impossible. Laptops do about 20-30 qubits in SV mode. — [arXiv:2601.03035](https://arxiv.org/pdf/2601.03035); [PennyLane forum](https://discuss.pennylane.ai/t/hardware-requirement/4829)
- If states are sparse or factorized, SV simulators such as the Qrisp simulator can handle larger systems. MPS simulators (e.g. Qiskit Aer) are efficient only for circuits with restricted entanglement. — [arXiv:2602.21803](https://arxiv.org/pdf/2602.21803)
- Compressed SV reaches 42 qubits on one 128 GB workstation **only for cat/GHZ/BV circuits** (ratios about 425-680x). QFT tops out at 36 and QAOA/Ising at 35 on the same box. — [BMQSim](https://arxiv.org/html/2410.14088v1)
- Wu et al.: compression "exploits non-uniformity and structure… it does not work as well on random circuits". The 61-qubit result is Grover-specific. — [arXiv:1911.04034](https://arxiv.org/pdf/1911.04034)
- Quantum Rings' 53-qubit Sycamore-class results are judged by XEB (0.549 at 14 cycles), not exact amplitudes. The 2024 CPU-only single-node run took about 2.5 days. This is an approximate/proprietary method, not exact dense SV. — [arXiv:2512.07311](https://arxiv.org/html/2512.07311v1)
- The Pan & Zhang single-GPU 43-qubit full-SV result applies to Google's "simplifiable" 14-cycle circuit, which has structure that tensor contraction exploits. — [arXiv:2103.03074](https://www.arxiv.org/abs/2103.03074)
- The Keio/Tsukuba "40+ qubits, lab-affordable" claim needed a dedicated FPGA board and 32 disks. — [Keio PR](https://www.keio.ac.jp/en/press-release/20231211-1/)
- Osaka/Fixstars (2026) describe 40 qubits as "the previous limit" for SV quantum-chemistry simulation, which they passed with 1,024 H100s. — [Quantum Insider](https://thequantuminsider.com/2026/04/02/quantum-circuit-simulation-beyond-40-qubits/)

### Inferences
Realistic tiers for the target machine (62 GB RAM, about 49 GB free NVMe, AVX2, no GPU):

| Tier | Qubits | Conditions | Expected cost |
|---|---|---|---|
| Exact generic SV, complex128 | 31 | – | Seconds to minutes per circuit (memory-bandwidth bound) |
| Exact generic SV, complex64 | 32 | – | Same as above |
| 2-byte adaptive encoding (JUQCS-style) | 34 | Any circuit, small precision loss | About 2-3x compute |
| Generic lossy compression (SZ-like) | about 34-35 | Random/QAOA/QFT, ratios about 5-20x; fidelity about 0.93-0.99 | Hours (cf. Wu one-node 36-qubit runs: 8-79 h) |
| Out-of-core SSD, 36-38 qubits | 36-38 | Requires ~1-4 TB of free NVMe (unavailable now); 40 qubits needs ≥8.8 TB | About 50 min per full pass at 40 qubits on one NVMe, many passes per circuit, plus heavy SSD wear |
| 40+ qubits | 40+ | Only structured/compressible circuits (GHZ, BV, Grover, sparse), low-entanglement MPS, Clifford(+few T), or single amplitudes / small batches via hybrid SF or tensor networks (shallow circuits) | – |

- **Honest framing for "40 qubits on a normal machine":**
  - Exact generic full-state simulation at 40 qubits on this workstation is out of reach: it would need about 130-260x more RAM, or a dedicated multi-TB SSD array plus days of runtime.
  - 40 qubits is achievable for restricted circuit classes, or by computing selected amplitudes instead of the full state.

### Gaps
- No peer-reviewed source documents a generic, exact 40-qubit SV run on a single commodity workstation with ≤128 GB RAM.
- Vendor claims of "40+ qubits on a laptop" (e.g. from approximate-simulation startups) were not systematically surveyed. Where found (Quantum Rings), they rely on approximate methods evaluated by XEB rather than exact fidelity.
