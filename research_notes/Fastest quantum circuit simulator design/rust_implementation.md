# Rust implementation of a maximum-performance state-vector simulator (i9-14900K, AVX2+FMA, Rust 1.96 stable), plus a survey of existing Rust simulators

Scope note: The research ran 2026-10-09. Each finding has a source. Items under "Inferences" are engineering reasoning or the researcher's prior knowledge. They are labeled as such and have not been verified against a source in this session.

---

## 1. SIMD in Rust: intrinsics, target_feature, runtime dispatch, portable SIMD, helper crates, checking asm

### Takeaway
On stable Rust in 2026, the practical path is `std::arch::x86_64` AVX2/FMA intrinsics inside `#[target_feature(enable = "avx2,fma")]` functions. Since 1.86/1.87 this code is largely safe. Pair it with runtime dispatch (`is_x86_feature_detected!`, the `multiversion` crate, or `pulp`'s `Arch`) or with `-C target-cpu=native` for a single-machine build. `std::simd` is still nightly-only. Spinoza (Rust, Wells Fargo) argues that hand-written SIMD gives little benefit for memory-bound gate kernels if the loops are simple and use a split real/imag layout. PRISM-Q (2026) uses explicit AVX2/FMA/BMI2 kernels.

### Cited Findings
- **Rust 1.86 (Apr 2025), target_feature 1.1:** Safe functions can now carry `#[target_feature]`. They "can only be safely called from other functions marked with the target feature attribute". Calling one from unmarked code needs an `unsafe` block, and "it is the caller's responsibility to ensure that the target feature is available". Such functions "cannot be passed to functions accepting generics bounded by the `Fn*` traits" and can be coerced to function pointers only inside `target_feature` functions. — [Rust 1.86.0 blog](https://blog.rust-lang.org/2025/04/03/Rust-1.86.0/)
- **Rust 1.87 (May 2025):** "Most `std::arch` intrinsics that are unsafe only due to requiring target features to be enabled ... are now callable in safe code that has those features enabled." The official example checks AVX2 at runtime, calls `sum_avx2` through `unsafe`, and the `#[target_feature(enable = "avx2")]` body's "core loop is now fully safe code". — [Rust 1.87.0 blog](https://blog.rust-lang.org/2025/05/15/Rust-1.87.0/)
- Pointer-based load/store intrinsics (e.g. `_mm256_loadu_ps(ptr)`) remain `unsafe`. — [archmage crate README (third party)](https://github.com/imazen/archmage); see also [InfoWorld on 1.87](https://www.infoworld.com/article/3989643/rust-1-87-shines-on-anonymous-pipes-architecture-intrinsics.html)
- Rust 1.89 (Aug 2025) stabilized "many intrinsics for x86", including AVX-512 intrinsics and target features such as `avx512bw`. This does not apply to the i9-14900K, which has no AVX-512, but it means a future AVX-512 path needs no nightly. — [Rust 1.89.0 blog](https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/)
- `std::simd` (portable SIMD) is still a "nightly-only experimental API" under tracking issue #86656. The project group's original goal of stabilizing for the 2021 edition was missed. — [std::simd docs](https://doc.rust-lang.org/std/simd); [tracking issue #86656](https://github.com/rust-lang/rust/issues/86656); [RFC 2977](https://www.ncameron.org/rfcs/2977.html)
- The `wide` crate is recommended as a stable-Rust portable SIMD alternative to `std::simd`. — [pythonspeed: Using portable SIMD in stable Rust](https://pythonspeed.com/articles/simd-stable-rust)
- **`multiversion` crate (v0.9.0, docs dated Sep 2026):**
  - The `#[multiversion]` attribute uses "CPU feature detection at runtime to dispatch the appropriate function" when the `std` feature is on. Without `std` it "will only allow compile-time function dispatch".
  - Targets are written "arch+feature1+feature2", e.g. `"x86_64+avx+avx2"`.
  - Limitations: no `self`/`Self` and no `impl Trait` return types.
  - The `nightly` feature adds nightly-only targets.
  - [multiversion docs](https://docs.rs/multiversion/latest/multiversion/)
- **`pulp` (v0.22.3):**
  - A "safe abstraction over SIMD instructions" that dispatches at runtime to vectorized versions based on detected CPU features.
  - Two usage modes: wrap scalar code in `Arch::new().dispatch(closure)` and let it autovectorize under the chosen feature set, or implement `WithSimd` for manual SIMD with generic `Simd` ops (`splat_f64s`, `mul_f64s`, `as_mut_simd_f64s` + scalar tail). A `#[with_simd]` macro is also available.
  - Has complex SIMD types (`c64x1/x2/x4`, `c32x2/x4/x8`), a `num_complex` re-export, and an `Interleave` trait for (de)interleaving.
  - [pulp docs](https://docs.rs/pulp/latest/pulp/)
- **`faer` (v0.24.4, 2026-06-24)** depends on `pulp` ^0.22.2 (required), `num-complex` ^0.4.6, and optionally `rayon` ^1.11. Its SIMD layer is therefore pulp. The project moved to Codeberg, and the GitHub repo is a mirror. — [faer on docs.rs](https://docs.rs/crate/faer/latest); [faer-rs GitHub](https://github.com/sarah-quinones/faer-rs)
- **Spinoza's position on manual SIMD:**
  - It is compiled with `target-cpu=native` so that "all instruction subsets supported by the local machine" are enabled.
  - Gate loops are written as "fast paths" that contain "only ... low-latency instructions such as add, mul". It "precludes the need for manual SIMD".
  - Rationale: "quantum state simulation is highly memory bound. As a result, the potential savings from handwritten SIMD would not provide any tangible performance benefits."
  - It uses FMA for the P/U-gate matrix-vector products.
  - [Spinoza paper, arXiv:2303.01493 §5.2](https://arxiv.org/pdf/2303.01493)
- Spinoza's build flags: `-C opt-level=3 -C target-cpu=native`, `codegen-units = 1`, `lto = true`, `panic = "abort"`. — [arXiv:2303.01493 §6](https://arxiv.org/pdf/2303.01493)
- PRISM-Q (Rust, 2026) uses explicit AVX2, FMA and BMI2 kernels with a scalar fallback on non-x86_64. A switch (`PRISM_NO_AVX2_2Q`) forces a 128-bit FMA two-qubit kernel for A/B comparison. — [PRISM-Q performance guide](https://docs.rs/crate/prism-q/0.28.0/source/docs/guides/performance.md)
- **`cargo-show-asm`** (`cargo install cargo-show-asm`):
  - Show a function's assembly with `cargo asm -p crate --lib path::to::fn`.
  - Other outputs: `--llvm` (IR), `--mir`, `--mca` (llvm-mca throughput analysis), `--rust` (interleave source).
  - `--native` or `--target-cpu=CPU` sets the target CPU.
  - Generic or inlined functions do not appear as symbols. Write a monomorphic `pub` wrapper or add `#[inline(never)]`.
  - [cargo-show-asm](https://github.com/pacak/cargo-show-asm)

### Inferences
- **Recommended structure (researcher's design reasoning):**
  - Write each gate kernel generically over a "backend" with an AVX2+FMA monomorphization (`#[target_feature(enable = "avx2,fma,bmi2")]`) and a scalar fallback.
  - Dispatch once per gate or circuit, not per element, using a cached `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")` result.
  - Since 1.86/1.87, the kernel body can be safe Rust apart from raw-pointer loads and stores.
- **`-C target-cpu=native` vs dispatch:**
  - `native` on a 14900K enables AVX2/FMA/BMI2 everywhere, including autovectorized glue code. The binary then won't run on older CPUs.
  - Dispatch keeps the binary portable, and only the hot kernels get AVX2.
  - For a personal maximum-performance build, `native` plus `lto="fat"`, `codegen-units=1`, `panic="abort"` (Spinoza's recipe) is the simplest option.
  - Because Raptor Lake E-cores (Gracemont) also implement AVX2/FMA, `native` code runs correctly on all 32 threads. This is the researcher's prior knowledge, not verified in this session.
- **AVX2 complex multiply idioms (researcher's prior knowledge):**
  - Interleaved layout: one `__m256d` holds two complex numbers `[re0, im0, re1, im1]`. Multiply by a scalar complex `(a + ib)` with `t = _mm256_permute_pd(v, 0b0101)` (swap re/im), then `r = _mm256_fmaddsub_pd(v, splat(a), _mm256_mul_pd(t, splat(b)))`.
  - Split layout: plain `_mm256_fmadd_pd`/`_mm256_fnmadd_pd` on four reals plus four imags at a time, with no shuffles.
  - For low target qubits (stride < 4 doubles), pairs fall within one register. These need `_mm256_permute2f128_pd`, `_mm256_unpacklo_pd`/`_mm256_unpackhi_pd` or `_mm256_shuffle_pd` lane shuffles, or a "reorder" trick.
- Spinoza's "autovectorization is enough" claim was measured single-threaded on 2023 hardware. Under 32 threads with the bus saturated, arithmetic efficiency matters even less for 1-qubit gates. It matters more for fused 2–5-qubit dense gates, which are compute-bound and where qsim-style explicit SIMD pays off. PRISM-Q's choice of explicit kernels plus fusion fits this reasoning.
- Autovectorization of complex loops is fragile: `num_complex::Complex<f64>` multiplication through operator overloading may or may not vectorize. Check with `cargo asm --native --rust` and look for `vfmadd*pd ymm` and `vfmaddsub*pd`.

### Gaps
- No primary source found on the exact 2026 status of `std::simd` stabilization beyond "nightly-only". The tracking issue's latest state was not read directly.
- `pulp`'s exact x86 levels (V3 = AVX2/FMA, V4 = AVX-512) are not listed on the docs page fetched. This is prior knowledge only.
- `simdeez` and `wide` internals and maintenance status were not researched in detail.
- `multiversion` dispatch overhead (a function-pointer cache) was not quantified by any source.

---

## 2. Complex-number layout, bounds checks, unsafe practice

### Takeaway
`num_complex::Complex<T>` is `#[repr(C)]` and layout-compatible with `[T; 2]`, so an interleaved `Vec<Complex64>` can be reinterpreted as `&[f64]` for SIMD. Spinoza instead uses split real/imag vectors. It claims this layout enables optimizations (such as cheap swaps for Y and X gates) that are "not possible with a single vector". Avoid bounds checks with iterators, `chunks_exact`, pre-slicing, or asserts, and keep `get_unchecked` as a last resort.

### Cited Findings
- **num-complex 0.4.6:** "`Complex<T>` is memory layout compatible with an array `[T; 2]`", declared `#[repr(C)]` with `re` before `im`. For floats it is "only memory layout compatible with C's complex types, not necessarily calling convention compatible". Pass it by pointer across FFI. — [num-complex docs](https://docs.rs/num-complex/latest/num_complex/struct.Complex.html)
- **Spinoza's state layout:**
  - The state is "a structure consisting of two vectors of single or double-precision floating-point types—one for real components and one for the imaginary components", plus a `u8` qubit count.
  - "separating the real and imaginary components into two separate vectors creates opportunities for optimizations that are not possible with a single vector."
  - Gates are stored as 1-D arrays rather than 2×2 matrices, for "better memory locality, less memory consumption, and less allocations".
  - [arXiv:2303.01493 §4.2.2](https://arxiv.org/pdf/2303.01493)
- **Spinoza precision:**
  - A `single` feature flag switches to f32 complex numbers. "In our testing, single-precision complex numbers offered a 30% performance improvement."
  - 32 qubits needs about 68.72 GB in complex128 versus 34.36 GB in complex64. On a 64 GB machine, double precision "would lead to page faults".
  - [arXiv:2303.01493 §4.2.1](https://arxiv.org/pdf/2303.01493)
- LogosQ's dense state uses `Complex64` from num-complex in one contiguous `Vec`, allocated once, and applies gates by bitwise index manipulation rather than building 2ⁿ×2ⁿ matrices. — [LogosQ, arXiv:2512.23183](https://arxiv.org/html/2512.23183v2)
- **Rust Performance Book on bounds checks:** "Replace direct element accesses in a loop by using iteration". "Make a slice of the `Vec` before the loop and then index into the slice within the loop". "Add assertions on the ranges of index variables". "As a last resort, there are the unsafe methods `get_unchecked` and `get_unchecked_mut`." — [Rust Performance Book: Bounds Checks](https://nnethercote.github.io/perf-book/bounds-checks.html)
- RustQIP (`qip`) shows a badge saying it forbids `unsafe` code. — [RustQIP GitHub](https://github.com/Renmusxd/RustQIP)

### Inferences
- **Memory budget for this machine (researcher's arithmetic):**
  - complex128: 30 qubits = 16 GiB, 31 = 32 GiB, 32 = 64 GiB, which exceeds the 62 GB RAM.
  - complex64: 32 qubits = 32 GiB, 33 = 64 GiB, which also does not fit.
  - So the in-RAM ceiling is 31 qubits in f64 and 32 qubits in f32, unless out-of-core or compression is used.
- **Layout choice:**
  - For a bandwidth-bound simulator, interleaved and split layouts move the same bytes. Split (SoA) needs no shuffles for complex arithmetic and makes the 4-wide f64 lanes independent amplitudes.
  - Interleaved keeps each amplitude's re/im in one cache line. That helps when gathering scattered amplitudes, e.g. high-qubit gates with two streams, and it matches the NumPy/Qiskit ABI for zero-copy Python export.
  - A compromise used by qsim-style C++ code is blocked SoA: chunks of re[8] and im[8]. In Rust this is a `#[repr(C, align(64))] struct Block { re: [f64; 4], im: [f64; 4] }` or similar. This is design reasoning, not sourced in this session.
- **Iteration pattern:**
  - The canonical 1-qubit kernel for target qubit t iterates `chunks_exact_mut(2 << t)` and `split_at_mut(1 << t)`, then zips the lo and hi halves. This is bounds-check-free and safe, and it parallelizes directly with `par_chunks_mut`.
  - For t below about 2, the chunks are smaller than one SIMD register and need in-register shuffles.
- Unsafe best practice: confine `unsafe` to small, documented kernels. Use `debug_assert!` for index invariants. Validate with `cargo miri test` on small sizes (Miri cannot run AVX intrinsics, so test the scalar path) and against a scalar reference implementation.

### Gaps
- No source found that benchmarks interleaved vs split layout specifically in Rust state-vector kernels. Spinoza asserts the benefit but publishes no ablation.

---

## 3. Parallelism: rayon, thread pools, pinning on hybrid Raptor Lake, work splitting, false sharing

### Takeaway
Rayon (`par_chunks_mut` with `with_min_len` or explicit chunk sizing) is the standard choice. PRISM-Q, a 2026 Rust simulator, enables Rayon at ≥14 qubits with at least 4096 elements per task. Hybrid P/E-core topology is exposed on Linux via `/sys/devices/cpu_core/cpus` and `/sys/devices/cpu_atom/cpus`. Pinning is done with `core_affinity` or `sched_setaffinity`. No source was found that benchmarks P-only vs all-cores for a memory-bound Rust kernel, so this must be measured.

### Cited Findings
- Rayon 1.12.0 `IndexedParallelIterator`:
  - `with_min_len` "sets the minimum length of iterators desired to process in each rayon job".
  - `with_max_len` sets the maximum.
  - `by_uniform_blocks` and `by_exponential_blocks` divide work into sequential blocks.
  - [rayon docs](https://docs.rs/rayon/latest/rayon/iter/trait.IndexedParallelIterator.html)
- **PRISM-Q threading:**
  - Rayon kernels engage at ≥14 qubits with `MIN_PAR_ELEMS = 4096` elements per task. The pool defaults to all logical cores (`RAYON_NUM_THREADS` overrides).
  - "Hyperthreading reportedly helps at 24+ qubits by hiding memory latency."
  - Fixed, deterministic work partitioning gives identical results regardless of thread count.
  - Cache-resident tiling runs batched gates on L2/L3-sized tiles.
  - [PRISM-Q performance guide](https://docs.rs/crate/prism-q/0.28.0/source/docs/guides/performance.md); [prism-q 0.27.0 crate page](https://docs.rs/crate/prism-q/0.27.0)
- Spinoza's paper benchmarks were single-threaded ("Spinoza will be parallelized" in future work). The current README shows a `-t <num-threads>` flag, so multithreading was added later. — [arXiv:2303.01493](https://arxiv.org/pdf/2303.01493); [Spinoza GitHub](https://github.com/QuState/spinoza)
- **Hybrid topology on Linux:**
  - The kernel exports `/sys/devices/cpu_core` and `/sys/devices/cpu_atom` PMUs, each with a `cpus` file. The Alder Lake example shows `cpu_core/cpus = 0-15` and `cpu_atom/cpus = 16-23`.
  - Layouts differ by chip, so read the files at runtime.
  - The fallback is per-CPU `topology/core_type`.
  - [Linux perf intel-hybrid docs](https://gbmc.googlesource.com/linux/+/refs/heads/linux-6.4.y/tools/perf/Documentation/intel-hybrid.txt); [nova-shell commit (detection order)](https://git.berlin.ccc.de/vinzenz/nova-shell/commit/446edaab9c193de82d7da53fd774455cbe96c420)
- A kernel bug report (218195) describes the "Intel hybrid CPU scheduler always prefers E cores" in some situations. This shows that the scheduler alone is not a reliable guarantee of P-core placement. — [kernel bugzilla 218195](https://bugzilla.kernel.org/show_bug.cgi?id=218195)
- **`core_affinity` crate:**
  - Pattern: `get_core_ids()` then `set_for_current(id)` in each thread. Supports Linux, macOS and Windows.
  - Core IDs are plain indices with no P/E classification.
  - A Rust forum user reported core_affinity "nearly doubled the throughput with the 64 and 96 thread use cases" for a large-buffer workload (not hybrid-specific).
  - [core_affinity](https://docs.rs/crate/core_affinity/^0.8); [Rust users forum: Rayon and work locality](https://users.rust-lang.org/t/rayon-and-work-locality-over-large-buffers-with-large-thread-pools/114770)
- `gdt-cpus` advertises P/E-core detection and affinity, but version 25.5.0 was yanked. — [gdt-cpus](https://docs.rs/crate/gdt-cpus/25.5.0)

### Inferences
- **Pinning recipe (researcher's prior knowledge):**
  - Build a dedicated pool with `rayon::ThreadPoolBuilder::new().num_threads(k).start_handler(|i| core_affinity::set_for_current(ids[i]))`.
  - Take `ids` from parsing `/sys/devices/cpu_core/cpus` (P-cores; on the 14900K expect 16 logical CPUs, 8 cores × 2 HT, likely 0–15) and/or `cpu_atom/cpus` (16 E-cores, likely 16–31).
  - Run kernels via `pool.install(|| ...)`.
  - The `start_handler` API is prior knowledge and was not verified in this session.
- **P-only vs all cores:**
  - A streaming kernel saturates dual-channel DDR5 with fewer than 32 threads. P-cores alone (8 cores, maybe 16 threads) may hit the bandwidth roofline.
  - Adding E-cores mainly helps compute-bound fused-gate kernels. With static equal partitioning, it can hurt, because slow E-core chunks become stragglers.
  - Rayon's work stealing mitigates stragglers only if there are many tasks, e.g. 4–8× the thread count.
  - Benchmark three configurations: 8P / 16P-threads / 32 all.
- **Task granularity:** For a 2^30-element state, `par_chunks_mut` with chunks of about 2^14–2^16 amplitudes (256 KiB–1 MiB, sized to L2: 2 MB per P-core, 4 MB shared per 4-E-core cluster) gives thousands of tasks. Rayon's per-task overhead (sub-µs) is then negligible.
- False sharing only occurs when threads write within the same 64-byte line. Chunk boundaries at powers of two ≥ 8 complex128 elements (128 B) avoid it automatically. Per-thread reduction accumulators (norms, expectation values) should be padded (`crossbeam_utils::CachePadded`) or reduced via rayon's `reduce`.
- Scoped threads (`std::thread::scope`) with a persistent spin-barrier pool can beat rayon's fork-join latency for circuits with many tiny gates at 20–24 qubits. This is unbenchmarked; consider it only after profiling.

### Gaps
- No published measurement found of rayon fork-join overhead per `par_chunks_mut` call on Raptor Lake.
- No source found benchmarking P-core-only vs all-core state-vector simulation on Intel hybrid CPUs.
- hwloc Rust bindings (`hwlocality`) were not researched.

---

## 4. Memory: aligned allocations, first-touch, huge pages, mmap/out-of-core

### Takeaway
Allocate the state with a 64-byte (or 2 MiB) aligned `Layout` or an anonymous `mmap`. Request THP with `madvise(MADV_HUGEPAGE)` (`memmap2::Advice::HugePage`). Pre-fault in parallel, either by a rayon first-touch write or with `MADV_POPULATE_WRITE` on Linux ≥ 5.14, so page faults don't pollute the first gate's timing. NUMA is moot on a single-socket 14900K.

### Cited Findings
- **memmap2 0.9.11 `Advice`:**
  - `HugePage` maps to `MADV_HUGEPAGE` and `NoHugePage` to `MADV_NOHUGEPAGE` (Linux ≥ 2.6.38, requires `CONFIG_TRANSPARENT_HUGEPAGE`).
  - `PopulateRead` and `PopulateWrite` (Linux ≥ 5.14) "prefault page tables" without accessing memory. Failures return an error rather than SIGBUS.
  - `WillNeed` maps to `MADV_WILLNEED`. `Advice::is_supported()` checks at runtime.
  - Applied via `MmapMut::advise`.
  - [memmap2 Advice](https://docs.rs/memmap2/latest/memmap2/enum.Advice.html)
- **Linux THP:**
  - `/sys/kernel/mm/transparent_hugepage/enabled` takes `always`, `madvise` or `never`. Per-size controls are at `hugepages-<size>kB/enabled`.
  - With `defrag = madvise`, the kernel does direct reclaim on fault "only for regions that have used madvise(MADV_HUGEPAGE)".
  - `khugepaged` collapses small pages in the background.
  - Usage is visible in `AnonHugePages` in `/proc/meminfo` or `/proc/PID/smaps`.
  - [kernel THP docs](https://docs.kernel.org/admin-guide/mm/transhuge.html)
- Spinoza notes that exceeding RAM (32 qubits in f64 on 64 GB) "would lead to page faults, which would dramatically reduce the performance". — [arXiv:2303.01493](https://arxiv.org/pdf/2303.01493)
- PRISM-Q derives its max state-vector qubits from available RAM (override: `PRISM_MAX_SV_QUBITS`). It auto-routes oversized circuits to MPS with bond dimension 256. — [prism-q crate page](https://docs.rs/crate/prism-q/0.27.0)

### Inferences
- **Allocation recipe (researcher's prior knowledge):**
  - For a 16–32 GiB state, `std::alloc::alloc(Layout::from_size_align(bytes, 2 << 20))` goes to glibc, which uses `mmap` for large sizes. Alternatively, `memmap2::MmapMut::map_anon(bytes)` or `libc::mmap` directly gives page-aligned memory.
  - Then call `madvise(MADV_HUGEPAGE)` and run a parallel first-touch loop: `par_chunks_mut(1 << 18).for_each(|c| c.fill(0.0))`. This spreads page-fault cost across threads and gets 2 MiB pages faulted in as huge pages.
  - With 4 KiB pages, a 16 GiB state is about 4M pages, whose sequential first touch costs seconds. With THP it is about 8k faults, and the TLB covers much more memory.
  - Avoid `alloc_zeroed`/`vec![0.0; n]` when timing matters. It returns lazily-zeroed `calloc`/mmap pages, so the first gate pays the faults single-threaded or arbitrarily.
- `MADV_POPULATE_WRITE` is a one-syscall alternative, but it runs on the calling thread. A parallel touch is likely faster on 32 threads; measure both.
- **Out-of-core:** A file-backed `memmap2::MmapMut` on NVMe would allow 32–33 qubits. However, NVMe bandwidth (single-digit GB/s) is about 10× below DRAM, so f32 precision (32 qubits in 32 GiB) is a better first lever than paging, as Spinoza suggests.
- **NUMA:** The 14900K is single-socket and single-node, so no NUMA policy is needed. First-touch matters only for page-fault cost, not placement.

### Gaps
- No Rust-specific benchmark found quantifying THP vs 4 KiB pages for state-vector kernels.
- Whether this machine's kernel has THP set to `always` or `madvise` is unknown. Check `/sys/kernel/mm/transparent_hugepage/enabled`.

---

## 5. Benchmarking and roofline in Rust

### Takeaway
Use `criterion` or `divan` for micro-kernels, plus a STREAM-like Rust triad to get achievable DRAM bandwidth. Report each gate kernel as bytes moved ÷ time ÷ measured STREAM bandwidth. PRISM-Q keeps baseline and regression benchmark suites and warns that contended hosts add noise.

### Cited Findings
- PRISM-Q ships circuit, gate-microbenchmark and GPU benchmark suites. Its baselines are taken with Rayon enabled, and it warns that "contended hosts add benchmark noise". — [PRISM-Q performance guide](https://docs.rs/crate/prism-q/0.28.0/source/docs/guides/performance.md)
- `cargo-show-asm --mca` runs llvm-mca on a function for static throughput analysis. — [cargo-show-asm](https://github.com/pacak/cargo-show-asm)

### Inferences
- **Roofline accounting (researcher's reasoning):**
  - A 1-qubit gate on n qubits reads and writes the whole state once: 2 × 16 × 2^n bytes for complex128. At 30 qubits that is 32 GiB of traffic per gate.
  - At a realistic dual-channel DDR5 sustained bandwidth (STREAM-triad likely in the 70–90 GB/s range for DDR5-5600/6000, unverified for this specific machine), that is roughly 0.35–0.45 s per non-fused gate at 30 qubits. This is the floor that fusion and cache-blocking must beat.
  - Measure the real number with a Rust triad `a[i] = b[i] + s*c[i]` over a ≥ 4 GiB array with rayon, pinned threads and THP.
- Use `criterion` (statistical, HTML reports) for kernel A/B tests and `divan` for quick, low-boilerplate benches with thread-count parameters. Pin CPU frequency/governor (`cpupower frequency-set -g performance`) for stability. Hybrid boost behavior on the 14900K adds variance. These are tool recommendations; the crates' docs were not fetched in this session.

### Gaps
- No measured STREAM numbers for the i9-14900K were found in this session. Measure locally.
- criterion and divan docs were not fetched. Their feature comparison here comes from the researcher's prior knowledge.

---

## 6. Survey of existing Rust quantum simulators and performance vs C++ (qsim, Qulacs, QuEST, Aer)

### Takeaway
No Rust simulator found has a published head-to-head benchmark against qsim, QuEST or Qiskit Aer with current versions:
- **Spinoza (2023)** is the only one with a published comparison against a C++ simulator (Qulacs 0.6.0). That comparison was single-threaded, on a GCP c2-standard-16, against an old Qiskit 0.16, and reports Spinoza as "one of the fastest".
- **LogosQ (Dec 2025)** compares only against Python/Julia/Q#.
- **PRISM-Q (2026)** is the most performance-engineered Rust project found: AVX2/FMA/BMI2 kernels, Rayon, fusion, cache tiling and multiple backends. Its docs give no cross-simulator numbers.
- **roqoqo/qoqo-quest** simply wraps QuEST (C) via FFI.

### Cited Findings
**Spinoza (QuState / Wells Fargo), Apache-2.0**
- **Design:** The core idea is that a 1-qubit gate acts on amplitude pairs differing only in the target bit. It uses several "pair selection strategies" per gate type:
  - concatenation for Y, Z, P, Ry, U, X, H, Rx, with a special case for target 0;
  - group-and-traverse for Rz;
  - a double-shift/insertion strategy for controlled gates.
  - Other features: split re/im vectors, FMA, autovectorization (no manual SIMD), and Python bindings (`spynoza`).
  - [arXiv:2303.01493](https://arxiv.org/pdf/2303.01493)
- **Benchmarks:**
  - Setup: GCP c2-standard-16 (1 vCPU per core), single thread, rustc 1.67.1. Circuits were parameterized random circuits from a shared public benchmark repo, run through Spinoza's Python bindings.
  - Compared against Qulacs 0.6.0, ProjectQ 0.8.0, Qiskit 0.16.0 and PennyLane 0.30.0.
  - Figure 6 plots time and time relative to Spinoza for 5–25 qubits. The abstract says Spinoza is "one of the fastest" simulators.
  - Exact numeric ratios are only in plots, so no reliable digits could be extracted. A Rust-vs-Qulacs-C++ comparison is referenced as Appendix D but could not be read from the extracted text.
  - [arXiv:2303.01493](https://arxiv.org/pdf/2303.01493)
- **Not compared to qsim:** No Spinoza vs qsim benchmark was found. — [search summary; Quantum Zeitgeist coverage](https://quantumzeitgeist.com/researchers-from-wells-fargo-introduce-spinoza-a-high-speed-quantum-simulator/)
- **Current repo:** 133 commits, 91 stars, a `-t <num-threads>` flag, Python bindings via `pip install git+...#subdirectory=spynoza`, and a `qasm` folder. No benchmarks in the README. — [Spinoza GitHub](https://github.com/QuState/spinoza)

**LogosQ (An, Wang, Slavakis; arXiv:2512.23183, Dec 2025)**
- **Design:**
  - Modules: Circuit, State, MPS and Gate.
  - Gates are `Arc<dyn Gate>` with `apply_dense`/`apply_mps`. The dense state is a `Vec<Complex64>`.
  - MPS uses `Array3<Complex64>` with bond dimension 64 and threshold 1e-8, SWAP networks and TEBD.
  - Adaptive backend selection uses state vector up to about 12–15 qubits and MPS above that. Dense QFT uses RustFFT.
  - A "parallel feature" exists. The paper text does not discuss SIMD or Rayon.
  - [arXiv:2512.23183](https://arxiv.org/html/2512.23183v2)
- **Benchmarks:**
  - Hardware: i9-10980XE (18C/36T), 125 GB RAM.
  - QFT: "up to 900×" over PennyLane/Qiskit (Python), 6–22× over Yao.jl, and "competitive" with Q#. LogosQ QFT took 0.72 µs at 1 qubit and 169 µs at 24 qubits.
  - VQE H₂: 2–5× faster than Qiskit/PennyLane.
  - There is no comparison to qsim, Qulacs, QuEST or Aer.
  - Note: 169 µs for a 24-qubit QFT state (256 MiB) is below what a DRAM streaming pass would allow. The FFT shortcut or a product-state input likely explains it, so the number is not comparable to gate-by-gate simulators. This is a researcher inference.
  - [arXiv:2512.23183](https://arxiv.org/html/2512.23183v2)

**PRISM-Q ("Performance Rust Interoperable Simulator for Quantum"), MIT/Apache-2.0**
- Very active: 0.2.0 on 2026-04-10, 0.27.0 on 2026-07-24, 0.34.0 on 2026-10-06. — [prism-q crate page](https://docs.rs/crate/prism-q/0.27.0)
- **Backends:** Statevector, Stabilizer, Sparse, MPS, Product State, Tensor Network and Factored. Auto-dispatch routes circuits by structure (Clifford → stabilizer, oversized → MPS χ=256). An optional CUDA `gpu` feature exists.
- **Performance features:**
  - AVX2/FMA/BMI2 kernels.
  - Rayon at ≥14 qubits, 4096 elements minimum per task.
  - Multi-pass gate fusion and cache-resident tiling of fused batches.
  - Disjoint two-qubit gate reordering into tiers.
- **Interop:** OpenQASM 3.0 parser (with 2.0 compatibility, `inv@`/`ctrl@`/`pow@`, user gates) and PyO3/maturin Python bindings with NumPy output (LSB-first bit order, reversed vs Qiskit).
- **Benchmarks:** No cross-simulator numbers on the pages read. A separate Benchmarks page exists but was not retrieved.
- [prism-q 0.27.0](https://docs.rs/crate/prism-q/0.27.0); [performance guide](https://docs.rs/crate/prism-q/0.28.0/source/docs/guides/performance.md)

**roqoqo / qoqo / qoqo-quest (HQS Quantum Simulations)**
- The repo contains three components:
  - `qoqo_quest`, a Python backend;
  - `roqoqo-quest`, a Rust backend with `Backend::call_circuit`;
  - `quest-sys`, low-level Rust FFI bindings to the QuEST C library.
- It supports noiseless and noisy simulation. It is "not tested against distributed builds" and has "preliminary support for GPU". Performance is therefore QuEST's C performance plus FFI overhead.
- [qoqo-quest PyPI](https://pypi.org/project/qoqo-quest/); [roqoqo-quest docs](https://docs.rs/roqoqo-quest/); [quest-sys](https://docs.rs/crate/quest-sys/0.19.0)

**Other Rust simulators**
- **RustQIP (`qip`):** A graph-building circuit construction API. It uses the borrow checker as an analogy for no-cloning and forbids `unsafe`. It has 315 stars. The README does not describe its state layout, SIMD or benchmarks. — [RustQIP GitHub](https://github.com/Renmusxd/RustQIP)
- **q1tsim:** A CPU simulator. Features include arbitrary gates, X/Y/Z-basis measurement, classical conditioning, OpenQASM/c-QASM export, LaTeX export and efficient stabilizer simulation. Releases run 0.1.0 (Jan 2019) to 0.5.0, so it is effectively dormant. — [lib.rs q1tsim](https://lib.rs/crates/q1tsim); [docs.rs q1tsim](https://docs.rs/crate/q1tsim/latest)
- **qcgpu:**
  - An OpenCL-based simulator from 2018 that requires nightly Rust. Its arXiv paper benchmarks against other libraries.
  - At 5 qubits, a single gate took about 215 µs on the GPU vs about 160 µs on the CPU, so the GPU does not win at small sizes.
  - Dormant.
  - [qcgpu docs.rs](https://docs.rs/crate/qcgpu/latest); [Simulating Quantum Computers Using OpenCL](https://www.paperswithcode.com/paper/simulating-quantum-computers-using-opencl)
- **quantrs2:** A facade crate over the QuantRS2 framework with state-vector and stabilizer sims. The docs.rs build of 0.1.3 failed. — [quantrs2 docs](https://docs.rs/quantrs2)
- **Correctness warning:** quantrs2-core defines Rz(λ) with the opposite exponent sign from the IBM/Qiskit convention. This gave fidelity 0.0 against roqoqo/q1tsim/native on a QAOA test. — [dev.to: We benchmarked 4 Rust quantum simulators](https://dev.to/cleiton_augusto_/we-benchmarked-4-rust-quantum-simulators-three-agreed-one-didnt-4dd2)
- The same dev.to article (CleitonForge) reports only sub-ms timings for 2–3-qubit circuits, not attributed to a backend. These are not useful for performance comparison. — [dev.to](https://dev.to/cleiton_augusto_/we-benchmarked-4-rust-quantum-simulators-three-agreed-one-didnt-4dd2)
- `qvass` appeared in crate search results but was not investigated. — [qvass docs.rs](https://docs.rs/crate/qvass/0.1.1)

**Qiskit's Rust code**
- The `crates/` directory contains:
  - `accelerate` (one-off accelerators);
  - `circuit`, `circuit_library`, `transpiler`, `synthesis` and `quantum_info`;
  - `qasm2` and `qasm3`;
  - `qpy`, `providers` and `cext` (C API);
  - `pyext`, the only crate building the `qiskit._accelerate` extension.
- None is described as a state-vector simulator. Qiskit's high-performance simulation remains Qiskit Aer (C++).
- [Qiskit crates/](https://github.com/Qiskit/qiskit/tree/main/crates)

### Inferences
- No Rust simulator currently has a credible, published, multi-threaded benchmark against qsim, QuEST or Aer on modern hardware. A new Rust simulator claiming "fastest" status would have to produce these numbers itself, using the same circuits, thread counts and precision.
- **Design lineage to borrow:**
  - Spinoza: per-gate-type specialized pair iteration, split layout, f32 option.
  - PRISM-Q: explicit AVX2 kernels, fusion, L2/L3 tiling, deterministic partitioning, auto-backend dispatch.
  - qsim (C++, from the broader literature): explicit-SIMD fused k-qubit dense gates.
- Wrapping QuEST via `quest-sys` gives a cheap C-performance baseline accessible from Rust for A/B comparisons on the same machine.

### Gaps
- Spinoza's exact speedup numbers vs Qulacs (Fig. 6 and Appendix D) could not be extracted as text from the PDF. Read the figure directly if precise ratios are needed.
- PRISM-Q's Benchmarks page (wall-clock results) was not retrieved. It may hold comparisons to Aer or qsim.
- The "quantr" crate and Stim-like Rust ports were not found in the searches performed. Their status is unknown.
- No information found on whether Spinoza's current multithreading uses rayon.

---

## 7. Interop: Python bindings (PyO3/maturin) and OpenQASM parsing crates

### Takeaway
PyO3 + maturin is the de-facto standard. Spinoza, PRISM-Q, qoqo-quest, Qiskit's own `_accelerate` and the CleitonForge benchmark layer all use it. For OpenQASM 3, use Qiskit's `openqasm3_parser` crate family (`oq3_lexer` → `oq3_parser` → `oq3_syntax` → `oq3_semantics`). Its authors report about an 80× faster parse than the reference parser. Qiskit has a separate in-tree Rust OpenQASM 2 parser.

### Cited Findings
- **Qiskit `openqasm3_parser`:**
  - A front end "built to be faster and to give better diagnostics than the reference parser". "A crude test with large source files showed parse time reduced by a factor of 80."
  - Crates: `oq3_lexer` (a lightly modified rustc lexer), `oq3_parser` (CST), `oq3_syntax` (AST) and `oq3_semantics` (semantic analysis).
  - It is independent of Qiskit. Qiskit's `crates/qasm3` adapts it to build QuantumCircuits, and it backs the experimental `qasm3.loads_experimental`.
  - [openqasm3_parser GitHub](https://github.com/Qiskit/openqasm3_parser); [oq3_parser docs](https://docs.rs/oq3_parser); [Qiskit crates/](https://github.com/Qiskit/qiskit/tree/main/crates)
- `oq3_syntax` and `oq3_semantics` are listed at version 0.7.0 (Apache-2.0) by a third-party index. — [depscope oq3_semantics](https://mcp.depscope.dev/pkg/cargo/oq3_semantics)
- Qiskit has a Rust `qasm2` crate (an OpenQASM 2 parser depending on `circuit`). — [Qiskit crates/](https://github.com/Qiskit/qiskit/tree/main/crates)
- PRISM-Q has its own OpenQASM 3.0/2.0 parser (`src/circuit/openqasm.rs`) covering stdgates, Qiskit exporter gates and IonQ/Cirq native names. It also has PyO3/maturin bindings with NumPy output. — [prism-q](https://docs.rs/crate/prism-q/0.27.0)
- qoqo-quest ships prebuilt PyPI wheels for common x86_64 targets and macOS arm64. Other platforms need Rust + maturin to build from sdist. — [qoqo-quest PyPI](https://pypi.org/project/qoqo-quest/)
- CleitonForge exposes a `SimulationBackend` trait over a parsed QASM 2 circuit, with PyO3 + maturin Python bindings. — [dev.to](https://dev.to/cleiton_augusto_/we-benchmarked-4-rust-quantum-simulators-three-agreed-one-didnt-4dd2)

### Inferences
- **Zero-copy export to NumPy (researcher's prior knowledge):**
  - With an interleaved `Vec<Complex64>`/aligned buffer, expose the state through `numpy` (rust-numpy) as a complex128 array without copying, e.g. `PyArray1::borrow_from_array` or by owning the buffer in a PyClass.
  - A split re/im layout forces a copy or two float arrays. This is an argument for interleaved or blocked layouts when Python interop is a priority.
- Release the GIL (`py.allow_threads`) around long simulations so rayon threads aren't blocked by Python. Build wheels with `maturin build --release`, with `RUSTFLAGS="-C target-cpu=native"` only for local use.
- **Bit order:** PRISM-Q's LSB-first counts are reversed relative to Qiskit. A new simulator must document and test its bit order against Qiskit Aer. The quantrs2 Rz sign bug shows the need for cross-simulator fidelity tests.

### Gaps
- Exact current versions of PyO3, maturin and rust-numpy, and their APIs for 2026 (e.g. Bound API changes), were not fetched.
- The older `openqasm` crate (OpenQASM 2 parser by another author) was not researched.
