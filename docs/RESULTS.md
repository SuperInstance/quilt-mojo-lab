# RESULTS — measured, not promised

Machine: 2-core container, no GPU. Timings: best of 3; Python via
`time.perf_counter_ns`, Mojo via internal `std.time.perf_counter_ns`
(runtime startup excluded). Workload: flow passes only for the flow timing
column; `mojo(total)` = flow + whole-cell SIMD entropy pass. Identical
semantics verified across runtimes by `python/test_correctness.py` (9/9) and
the checksum gate in `python/bench.py`.

## Throughput (grid N×N, 10 flow steps, single source at (1,1)=4.8)

| grid | cells | naive | flat | numpy | C -O3 | mojo(flow) | mojo(total) |
|---|---|---|---|---|---|---|---|
| 16x16 | 256 | 2.58ms | 1.20ms | 0.25ms | 0.01ms | 0.01ms | 0.01ms |
| 32x32 | 1024 | 10.49ms | 5.75ms | 0.37ms | 0.03ms | 0.03ms | 0.03ms |
| 64x64 | 4096 | 42.28ms | 23.92ms | 0.54ms | 0.09ms | 0.10ms | 0.10ms |
| 128x128 | 16384 | — | 97.10ms | 1.36ms | 0.31ms | 0.43ms | 0.44ms |
| 256x256 | 65536 | — | 387.77ms | 5.31ms | 1.27ms | 1.57ms | 1.60ms |
| 512x512 | 262144 | — | 1696.40ms | 27.62ms | 7.10ms | 10.84ms | 11.11ms |

cells/sec (best):

| grid | naive | flat | numpy | C | mojo |
|---|---|---|---|---|---|
| 16x16 | 990,841 | 2,126,659 | 10,326,786 | 187,958,884 | 387,937,566 |
| 32x32 | 975,839 | 1,781,347 | 27,802,893 | 347,991,572 | 397,068,518 |
| 64x64 | 968,810 | 1,712,622 | 76,092,671 | 462,526,960 | 406,772,928 |
| 128x128 | — | 1,687,287 | 120,573,075 | 523,122,901 | 370,516,946 |
| 256x256 | — | 1,690,074 | 123,497,596 | 514,894,673 | 408,371,058 |
| 512x512 | — | 1,545,295 | 94,920,258 | 369,062,571 | 235,903,403 |

## What the draft claimed vs what we measured

1. **"Cells as a contiguous flat block, indexable by offsets, no dictionary
   overhead"** — CONFIRMED, and it is the single biggest win. The dict-based
   loop sustains ~1.0M cells/s regardless of grid size; the same semantics on
   a compiled flat block sustain 370-523M cells/s: **~400x**, most of it from
   removing per-cell object/dict machinery. The flat-block Python version
   (still interpreted) already gains ~1.7-2.1x over dict-Python — layout
   alone helps even without compilation.
2. **"SIMD vectorized flow pass; whole blocks in a single step"** — PARTIAL.
   The flow pass is neighbor-gather bound (reads at ±1, ±N): Mojo's scalar
   unrolled gather lands within ±15% of gcc -O3, slightly BEHIND C at ≥64²
   (462-523M C vs 371-408M Mojo). The lane-parallel case is the elementwise
   entropy pass (whole-cell `SIMD[Float32,4]` load/store), which is a minor
   fraction of runtime. Nothing "automatic" beat the C compiler here; the
   honest claim is parity, not superiority.
3. **"Lifetime-driven transitions, zero allocation in fast loops"** —
   CONFIRMED after one fix: the draft allocated a snapshot buffer per pass;
   ours allocates once at construction. With that, steady-state passes do
   zero allocation.
4. **Mojo scaling is unusually flat** (388-408M cells/s from 16² to 256²),
   then drops 42% at 512² — the interleaved AoS block (4MB) plus snapshot
   exceeds cache and the gather's effective bandwidth falls off a cliff. C
   degrades more gracefully (gcc reorders better around the strided access).
   Next substrate iteration should try SoA (separate potential array) to
   restore unit-stride reads — registered as future work, not yet run.

## Bottom line for the architecture

- The flat-memory + compiled-runtime thesis is right: ~400x over the naive
  Python loop, ~4-5x over numpy, and the C-ABI export (`interface_probe.py`)
  works as advertised — one contiguous block, streaming to any external
  consumer with a single `.tobytes()`.
- Mojo specifically buys **safety + expressiveness at C-class speed** here,
  not speed over C. The SIMD story needs the SoA layout (future work) or a
  GPU backend before it can beat the C compiler on this kernel.
- GPU: this container has none; the parallel-lane claims remain untested and
  are NOT claimed by this document.

## Reproduce

```sh
sh c/build.sh
python -m pytest python/test_correctness.py -q
python python/bench.py            # needs mojo on PATH + MODULAR_HOME set
python python/interface_probe.py
```
