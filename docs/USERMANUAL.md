# USERMANUAL — quilt-mojo-lab

On-box verified 2026-09-30 on the RTX 4050 / WSL2 box (Ubuntu, 24 logical
cores, CPU-only — no GPU is used anywhere in this repo). Toolchain receipt:
Mojo `1.2.0.dev2026100105` via pixi, gcc via system toolchain, Python 3.12
(pixi) / 3.14 (host) both fine for the pure-Python parts.

Everything below was executed verbatim on this box and passed.

---

## Quickstart (fresh clone → all receipts, ~4 minutes)

```sh
git clone git@github.com:SuperInstance/quilt-mojo-lab.git
cd quilt-mojo-lab

# 1. One-time environment bootstrap (installs Mojo nightly + python 3.12 + numpy + pytest)
pixi run mojo --version          # first run solves the env; subsequent runs are fast

# 2. Build the C runtimes
sh c/build.sh                    # -> built c/libflatquilt.so c/libsoaquilt.so c/libexportsoa.so

# 3. Correctness suite (15 tests, ~1s)
pixi run python -m pytest python/test_correctness.py -q

# 4. Benchmark, 8 runtimes, grids 16²..512² (Mojo first-run JIT compile dominates; total ~2-3 min)
pixi run python python/bench.py

# 5. Scale test at 512² + 1024² (~2 min)
pixi run python python/bench_1024.py
```

Key rule: **run Python entry points through `pixi run`**. The bench harness
needs `mojo` on `PATH` to time the Mojo substrates; the pixi env provides it.
Host-PATH runs of `python3 python/bench.py` fail with
`FileNotFoundError: 'mojo'` on a box that uses the project-local pixi env
(see Troubleshooting).

## The eight runtimes

Identical semantics everywhere (grid of N×N cells, 4 float32 registers
`[potential, resistance, entropy, split]`; outbound-only threshold-gated
flow pass with snapshot semantics; entropy pass `min(1, pot/5*0.95)`;
`split = entropy > 0.6`).

| # | Runtime | File | Layout | What it is |
|---|---------|------|--------|------------|
| 1 | naive | `python/naive_quilt.py` | dict-of-dicts | The "standard Python execution loop" baseline. Correctness anchor, speed floor. |
| 2 | flat (AoS) | `python/flat_quilt.py` | `array('f')`, 4 slots/cell | The draft's flat-memory layout in pure Python — hardware-style offset indexing instead of dicts. |
| 3 | soa | `python/soa_quilt.py` | 4 separate field arrays | Wave-69 SoA re-layout in Python; unit-stride access fixes the 512² cache cliff. |
| 4 | vec | `python/vec_quilt.py` | numpy whole-grid | Neighbor pass as four shifted-array subtractions; SIMD-by-library. |
| 5 | c (AoS) | `python/cflat.py` + `c/flat_quilt.c` → `c/libflatquilt.so` | interleaved AoS | Hot loops in C, `-O3`, via ctypes. Built by `sh c/build.sh`. |
| 6 | csoa | `python/csoa.py` + `c/soa_quilt.c` → `c/libsoaquilt.so` | SoA | C SoA kernel over four contiguous arrays, same ctypes bridge pattern. |
| 7 | mojo (AoS) | `mojo/quilt_high_perf.mojo` | interleaved AoS | Whole-cell SIMD Mojo substrate; timings from its internal `std.time.perf_counter_ns`. |
| 8 | mojo_soa | `mojo/quilt_soa.mojo` | SoA | Lane-vectorized flow pass over unit-stride field arrays — the fastest runtime on this box. |

Supporting pieces (not runtimes): `python/interface_probe.py` (v1 C-ABI dump:
raw interleaved block → `.bin`/`.npy` + header), `python/export_soa.py` +
`c/export_soa.c` (v2 four-block container with sha256-per-block,
byte-exact roundtrip — covered by the test suite).

Pixi tasks (`pixi.toml`):

- `pixi run test` — the pytest suite
- `pixi run bench` — `python python/bench.py`
- `pixi run mojo-demo` — boots the AoS Mojo substrate on a 64² grid

## Expected test output

```
$ pixi run python -m pytest python/test_correctness.py -q
...............                                                          [100%]
15 passed in ~0.5s
```

The 15 tests cover cross-runtime agreement (all 8 runtimes must match the
naive reference within tolerance), substrate properties, and the v2 export
container roundtrip. Bench harnesses run their own correctness gate first
and refuse to time anything that disagrees.

## This-box baseline (2026-09-30, RTX 4050 laptop, WSL2, CPU-only)

Flow-only throughput, cells/sec, best of 3, 10 steps per run.
Measured on this box with Mojo `1.2.0.dev2026100105`.

512² grid (`python/bench.py`, full table):

| grid | naive | flat | soa | vec | c | csoa | mojo | mojo_soa |
|------|-------|------|-----|-----|---|------|------|----------|
| 16² | 2.13M | 4.15M | 4.12M | 12.5M | 162.8M | 151.4M | 467.1M | 762.8M |
| 32² | 1.56M | 2.80M | 3.66M | 27.1M | 308.2M | 333.8M | 496.6M | 1.43G |
| 64² | 1.95M | 3.54M | 4.96M | 98.7M | 613.5M | 640.7M | 501.9M | 2.13G |
| 128² | — | 3.65M | 5.19M | 112.3M | 385.1M | 636.6M | 501.3M | 2.54G |
| 256² | — | 2.89M | 4.40M | 163.2M | 495.9M | 721.2M | 412.4M | 2.21G |
| 512² | — | 2.83M | 4.03M | 139.4M | 631.6M | 667.3M | 456.7M | 2.18G |

Wall-clock per 10 flow steps at the two big grids (`python/bench_1024.py`):

| grid | naive | flat | soa | vec | c | csoa | mojo | mojo_soa |
|------|-------|------|-----|-----|---|------|------|----------|
| 512² | 3280.7ms | 925.4ms | 612.6ms | 15.7ms | 4.4ms | 3.7ms | 5.6ms | 1.05ms |
| 1024² | 15546.3ms | 3685.4ms | 2747.4ms | 103.5ms | 27.1ms | 19.9ms | 32.4ms | 6.13ms |

1024² flow throughput: csoa 527.6M cells/s, mojo_soa 1.86G cells/s
(mojo_soa ≈ 2.5x C-SoA at 1024² on this box).

Pre-registered checks, both PASS on this box:

- wave-69 "SoA ≥ 2× AoS flow at 512²": mojo ratio **5.12×**, c ratio 1.06×.
- wave-72 "C-SoA keeps ≥ 0.5× per-cell throughput at 1024²":
  ratio 1024²/512² = **0.749** (cache-scaling held).
- 1024² checksum agreement: all 8 runtimes within 1e-4 of ref 0.4000000060.

Machine meta recorded in `outputs/bench.json`: 24 logical cores, best-of-3,
flow-passes-only timing. Numbers are reproducible to within normal jitter;
expect ±10-15% run-to-run on a laptop.

## Mojo 1.2.0 compatibility verdict

**Compatible as-is — both substrates compile and run on
1.2.0.dev2026100105 with zero code changes.** The only noise is deprecation
*warnings* (not errors):

```
warning: positional `__getitem__` is deprecated, use `unsafe_offset=` instead
```

These come from raw-pointer indexing in the Mojo substrates. Harmless today;
a future Mojo may remove positional `__getitem__`, at which point the
fix is mechanical (switch to `unsafe_offset=`). `mojo/original_draft.mojo`
targets the 24.x-era API and is preserved verbatim as a reference — it is
not expected to compile on 1.2.0; see `docs/MOJO-NOTES.md` for the API
archaeology.

## Troubleshooting

**`FileNotFoundError: [Errno 2] No such file or directory: 'mojo'`** when
running `python3 python/bench.py` outside pixi.
`bench.py` falls back to `~/.pixi/envs/mojo/bin` (the *global* pixi install
layout documented in MOJO-NOTES). On this box Mojo lives in the
*project-local* env at `.pixi/envs/default/bin/mojo`. Fix: run through pixi
(`pixi run python python/bench.py`), or point it at any mojo explicitly:
`MOJO_BIN=/path/to/mojo python3 python/bench.py`.

**First `pixi run` is slow.** It solves/installs the environment (Mojo
nightly is ~1-2 GB). Every subsequent `pixi run` is fast. `pixi.lock`
pins the resolution — keep it committed for reproducibility.

**pixi warning: "The `project` field is deprecated. Use `workspace`".**
Cosmetic only (pixi ≥ 0.81). Harmless; the manifest works unchanged. If it
annoys you, rename the `[project]` header in `pixi.toml` to `[workspace]`.

**Deprecation warnings on `mojo run`** (`positional __getitem__`): expected
on 1.2.0-dev, see compatibility verdict above. Exit code is 0; results are
correct.

**`sh c/build.sh` fails with missing `gcc`.** Install a C toolchain
(`sudo apt install build-essential`). Only `-lm` is linked; no other deps.

**Benchmarks look 2-3× slower than the table.** Laptop on battery power,
thermals, or another process owning cores (this box also runs GPU jobs on
the side — the benches are CPU-only but share cores). Re-run best-of-3 on
AC power.

**Committing rebuilt artifacts.** `sh c/build.sh` rewrites the `.so` files
in place and they are tracked in git, so a rebuild dirties the tree. That's
intentional (build receipts). `outputs/bench*.json` are likewise rewritten
by every bench run.
