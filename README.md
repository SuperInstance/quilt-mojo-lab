# quilt-mojo-lab

High-performance flat-memory quilt substrate: the draft architecture
(**struct compilation → memory register specs → bare-metal execution**)
built, tested, and benchmarked across **eight runtimes** with identical
semantics — naive dict Python, flat-array Python (AoS), **SoA Python**,
numpy, C `-O3` (AoS + SoA), and Mojo 1.2.0-dev (AoS whole-cell SIMD +
**SoA lane-vectorized flow pass**).

The governing discipline of this repo: **measure first, document second.**
Every performance claim in `docs/RESULTS.md` comes from the harness in this
repo, and every divergence from the original draft (API archaeology included)
is receipted in `docs/MOJO-NOTES.md`.

Headline (wave-69, measured): the SoA re-layout restores unit-stride access
and enables a genuinely lane-vectorized flow pass — **1.29B cells/s at 512²
on Mojo SoA, 4.33x its own AoS build and 2.55x the C-SoA kernel**, passing
the pre-registered "SoA ≥ 2x at 512²" prediction.

## Layout

```
mojo/quilt_high_perf.mojo    the compiling Mojo substrate (1.2.0-dev, AoS)
mojo/quilt_soa.mojo          the SoA substrate: unit-stride + lane-vectorized interior
mojo/original_draft.mojo     the original draft, preserved verbatim (reference)
python/naive_quilt.py        dict-of-dicts baseline ("standard Python loop")
python/flat_quilt.py         the draft's layout in Python (array('f'), AoS 4-slot)
python/soa_quilt.py          the wave-69 SoA re-layout in Python (4 field arrays)
python/vec_quilt.py          numpy whole-grid vectorization
python/cflat.py + c/         compiled C kernel (gcc -O3) + ctypes bridge (AoS)
python/csoa.py               ctypes bridge to the C SoA kernel (c/soa_quilt.c)
python/bench.py              correctness gate + best-of-3 benchmark (8 runtimes)
python/test_correctness.py   cross-runtime agreement + substrate properties
python/interface_probe.py    C-ABI export: raw block -> .bin/.npy/header
docs/RESULTS.md              measured numbers and honest verdicts
docs/MOJO-NOTES.md           24.x -> 1.2.0-dev API archaeology
outputs/                     bench.json, exported memory blocks (receipts)
```

## Semantics (identical across all runtimes)

- Grid of `N x N` cells; each cell is 4 float32 registers:
  `[potential, resistance, entropy, split]`.
- Flow pass (synchronous, snapshot semantics): for each in-bounds N/S/E/W
  neighbor, if `pot - neighbor_pot > resistance`, leakage
  `(diff - resistance) * 0.22` flows out of the higher cell. Outbound-only,
  threshold-gated (not charge-conserving — that is the specified model).
- Entropy pass: `entropy = min(1.0, pot / 5 * 0.95)`; `split = entropy >
  threshold`.

## Quickstart

```sh
sh c/build.sh
python -m pytest python/test_correctness.py -q          # 9 tests
python python/bench.py                                   # gate + benchmark
python python/interface_probe.py                         # C-ABI export
# Mojo (nightly): see docs/MOJO-NOTES.md for the toolchain env
mojo run mojo/quilt_high_perf.mojo 64 10 demo
```

## Headline (from docs/RESULTS.md; 2-core container, no GPU)

- Flat-memory compiled runtimes sustain **370-523M cell-updates/s** vs
  ~1.0M/s for the dict loop: **~400x**, the bulk from layout + compilation.
- Mojo lands at **C-class speed** (within ±15% of gcc -O3; slightly behind
  at larger grids) with memory-safe lifetimes and one-instruction whole-cell
  register I/O — parity, not superiority; see the honest SIMD analysis.
- numpy sits between at ~120M cells/s and is the effort/sweet spot for pure
  Python shops.
- The contiguous block is the interface: one `.tobytes()` streams the whole
  fabric to any C-compatible consumer (verified round-trip).

## Status

- v0.1.0: five runtimes, 9/9 correctness, full scaling curve to 512², C-ABI
  export, CI (Python + Mojo jobs + key-scan gate).
- Registered future work: SoA layout for unit-stride flow reads (the 512²
  cache cliff), GPU backend claim testing (needs hardware), multicomponent
  quilts (routing/filter/projection kernels beyond the leaky flow).
