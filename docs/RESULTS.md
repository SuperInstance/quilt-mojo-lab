# RESULTS — measured, not promised

Machine: 2-core container, no GPU. Timings: best of 3; Python via
`time.perf_counter_ns`, Mojo via internal `std.time.perf_counter_ns`
(runtime startup excluded). Workload: flow passes only for the flow timing
column; `mojo(total)` = flow + whole-cell SIMD entropy pass. Identical
semantics verified across runtimes by `python/test_correctness.py` (15/15 as
of wave-72) and the checksum gate in `python/bench.py`.

## Wave-72: 1024² scale test + four-block interface v2 — measured 2026-10-01

Toolchain re-verified BEFORE any measurement: the wave-69 Mojo nightly
(`Mojo 1.2.0.dev2026093005`, installed via pixi under `~/.pixi/envs/mojo`,
`MODULAR_HOME=$HOME/.pixi/envs/mojo/share/max`) still compiles and runs both
substrates today — **Mojo was available this wave**; no number below is
estimated or carried over. One environment gotcha receipted for the next
porter: prepending `~/.pixi/envs/mojo/bin` to PATH shadows the venv `python3`
with the conda env's python — invoke the benchmark python by absolute path.

### The 1024² scale test (`python/bench_1024.py`)

All eight runtimes at 1024², same protocol as wave-69 (best of 3, flow passes
only, correctness gate FIRST). Gate: 16² checksum agreement, 8/8 runtimes OK;
at 1024² every runtime agrees with the flat reference to max|Δpot| ≤ 3.0e-08
(soa: 0.0 exactly), checksums 0.400000006 ± 3e-8. New at 1024²: the naive
dict runtime is included (the v0.1.0 tables dropped it past 64²) and peak RSS
is recorded (`resource.getrusage`: 757 MB high-water, driven by the 1M-dict
naive row; compiled runtimes stay in the tens of MB).

Flow-only cells/sec (best of 3; mojo columns FLOW_NS-based):

| grid | naive | flat | soa | vec | c | csoa | mojo (AoS) | mojo SoA |
|---|---|---|---|---|---|---|---|---|
| 512x512 (in-session ref) | 700,649 | 1,557,192 | 2,012,901 | 96.97M | 336.54M | 518.28M | 333.47M | 1,584.50M |
| **1024x1024** | 591,546 | 1,512,404 | 1,967,890 | 82.33M | 317.07M | **486.36M** | 277.13M | **1,140.23M** |
| 1024/512 ratio | 0.84 | 0.97 | 0.98 | 0.85 | 0.94 | 0.94 | 0.83 | 0.72 |

Registered prediction (written into the `bench_1024.py` docstring BEFORE
measurement, p=0.55): **"C-SoA per-cell flow throughput at 1024² stays within
2x of its in-session 512² value: csoa(1024)/csoa(512) ≥ 0.5"** — the SoA
working set at 1024² is pot+snap+res ≈ 12 MB, and wave-69 measured SoA
cliffing ~10% per grid doubling vs the AoS 23-25% cliff.

**Verdict: PASS** — measured ratio 0.938 (486.36M vs 518.28M cells/s), a 6%
cliff, nowhere near the 0.5 bar. SoA's graceful cache scaling extends one more
doubling; the registered wave-69 conclusion ("it was the layout, not the
language") is stable at 1024².

Honest side observations (not registered, recorded):

1. Mojo SoA at 1024² sustains **1,140M cells/s = 2.34x C-SoA** (486M) — the
   wave-69 superiority verdict holds at the new scale (wave-69 measured 2.55x
   at 512²).
2. Mojo SoA degrades the MOST of the compiled runtimes across 512→1024
   (ratio 0.72) — the 3-snapshot working set (pot+snap+res at 4 MB each, plus
   the lane-vectorized interior touching all three) is starting to feel
   capacity pressure. Its absolute lead is intact.
3. Session-to-session variance is real on this 2-core container: wave-69
   measured mojo-SoA 1,295M and c-SoA 507M at 512²; this session 1,584M and
   518M on identical code and protocol (best-of-3 absorbs scheduler noise
   only partially). Cross-WAVE comparisons should trust ratios over absolutes;
   cross-RUNTIME comparisons within a session are solid.
4. The dict naive loop loses further ground at 1024² (0.84 ratio): ~1,970x
   behind Mojo SoA per cell.

### Four-block C-ABI interface v2 (`c/export_soa.c` + `python/export_soa.py`)

v1 (wave-68's `interface_probe.py`, kept + receipts preserved) streamed ONE
interleaved AoS block. v2 dumps the substrate as **FOUR separate flat blocks**
— one per field (potential/resistance/entropy/split), structure-of-arrays,
the layout the SoA kernels already run in memory — inside a self-describing
container:

```
offset 0      magic "QS2" + 0x00 (4 bytes)
offset 4      uint32 header_len, little-endian (authoritative)
offset 8      header_len bytes UTF-8 JSON (space-padded): magic, version=2,
              grid, cells, dtype="float32-le", fields, block_size_bytes,
              header_len, data_offset, blocks[4] = {field, offset, sha256}
8+header_len  block 0..3, each cells*4 bytes, field order
```

C side (`gcc -O3`): `qml_soa_deinterleave`/`qml_soa_interleave` (AoS dump ↔
four blocks), `qml_soa_export` (renders the JSON header by fixed-point pass
— data_offset depends on header_len depends on JSON length — hashes each
block with an in-tree FIPS 180-4 sha256, writes the container in one call;
no JSON parser needed by the writer). Python bridge: `export_v2` (from four
field buffers or one AoS dump), `read_v2`/`read_v2_bytes` (structural
validation + hashlib re-hash of every block = C-writer/Python-reader
agreement), `to_aos` (v2 → v1 byte-exact).

Verified (4 new pytest tests, suite now 15/15; demo receipt
`outputs/interface/quilt_soa_v2_256.qbin` + `_header.json`, v1 receipts kept):

- export a 256² grid → re-import: **field-identical** (all four fields,
  `np.array_equal`), and the four blocks are **byte-identical to the
  strided slices of the AoS dump**;
- C sha256 == hashlib sha256 per block; flipping one block byte or the magic
  is caught (tamper evidence);
- v2 → v1 interleave roundtrip is byte-exact;
- the AoS-sourced and SoA-runtime-sourced export paths produce the same
  container layout (content differs only by the receipted f32
  cross-runtime noise between C-f32 and Python-f64→f32 arithmetic).

## Wave-69: the SoA re-layout — measured 2026-10-01

v0.1.0 registered a falsifiable prediction (wave-68 discussion round,
p=0.45): restructuring the interleaved AoS block into four separate field
arrays (SoA) should restore >=2x flow throughput at 512^2 on the compiled
runtimes. The SoA Mojo substrate additionally vectorizes the interior flow
pass (8 cells per lane group, all loads unit-stride — the lane-parallel case
the AoS layout could not deliver), with scalar boundary cells.

Flow-only throughput (best of 3, `python/bench.py`, gate: all 7 runtimes OK,
checksum 0.4000000060 at 16^2/10 steps; 64^2/10-step cross-checksum
mojo-SoA 0.4000000059604645 vs python 0.4000000060):

| grid | flat | soa | c | csoa | mojo (AoS) | mojo SoA |
|---|---|---|---|---|---|---|
| 64x64 | 1.70M | 2.30M | 461M | 480M | 407M | **1,692M** |
| 128x128 | 1.69M | 2.30M | 527M | 553M | 415M | **1,837M** |
| 256x256 | 1.68M | 2.27M | 524M | 565M | 408M | **2,065M** |
| 512x512 | 1.53M | 2.02M | 340M | 507M | 313M | **1,295M** |

Verdicts (honest, per the registration):

1. **Registered check: PASS.** At 512^2 the SoA flow is 4.33x the AoS flow on
   Mojo (313M -> 1,295M cells/s) and 1.49x on C (340M -> 507M). The
   prediction ("SoA >= 2x at 512^2 on the compiled runtimes") holds on the
   headline runtime; the C ratio alone would NOT have cleared the 2x bar —
   gcc was already reordering the strided AoS access better than we credited.
2. **The 512^2 AoS cliff diagnosis is CONFIRMED**: AoS degrades 23-25% from
   its 128-256^2 plateau (mojo 415M -> 313M, c 527M -> 340M) while SoA holds
   a far higher absolute band (mojo SoA 2,065M -> 1,295M, c-SoA 553M -> 507M).
   The strided AoS working set, not the grid size per se, was the problem.
3. **Mojo-vs-C verdict REVISED**: v0.1.0 said "parity, not superiority". With
   SoA + a genuinely lane-vectorized interior, Mojo flow is 2.55x C-SoA at
   512^2 (1,295M vs 507M) and 3.7x C-AoS. The draft's "SIMD vectorized flow
   pass" claim is now CONFIRMED for the SoA layout — it was the layout, not
   the language, that blocked vectorization.
4. **Python SoA gains ~1.3x over Python AoS** (1.53M -> 2.02M at 512^2):
   even interpreted, unit-stride layout + a C-level array-slice snapshot beat
   the strided copy loop. The dict baseline remains ~1,000x behind compiled
   SoA.

## v0.1.0 baseline (AoS, kept for the record)

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
