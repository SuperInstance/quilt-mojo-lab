# Runtime #9 — GPU-native CuPy substrate (PRE-REGISTRATION)

Pre-registered 2026-10-01 ~05:55 AKDT, BEFORE any GPU correctness run or
measurement of the cupy runtime. Pushed before fire per fleet discipline.
This file is frozen; the measured verdict lands in docs/RESULTS.md (Wave-73).

## What

Runtime #9 in the runtime grid: `python/cupy_quilt.py` — the SoA quilt
substrate as native CUDA kernels via CuPy RawKernel (one thread per cell,
synchronous snapshot-based flow pass, exact same op order as
`python/soa_quilt.py`: South, North, East, West).

## Parity design (the whole game)

The Python oracle promotes float32 storage to float64 on every read and does
ALL per-cell intermediate math in float64, with a single float32 rounding at
the final store (`pot[idx] = adj`). The kernel therefore does the same:
`double` intermediates, identical op order, one `(float)` store per cell per
pass. Compile options: `-fmad=false` (FMA contraction would skip the explicit
intermediate rounding the oracle performs). Checksum is accumulated HOST-side
in the same sequential float64 order as `SoaQuilt.checksum()`.

## Falsifiable predictions (frozen now)

- **P1 (correctness, primary):** with `-fmad=false`, the cupy runtime's
  checksum after `run(10)` on the bench injection set is **bit-identical**
  (difference 0.0) to the FlatQuilt reference at 16², and within the standing
  bench gate `max(|ref|*1e-4, 1e-4)` at 512².
- **P2 (performance, conservative freeze):** cupy flow cells/s at 1024²
  **≥ 2×** the same-session C-SoA cells/s at 1024². (Memory-bound estimate
  says ~10× is available on the 4050; the gate freezes at 2×. Per-cell math
  is double, but 1M cells × ~10 double-flops is nowhere near the card's
  FP64 wall — the kernel stays memory-bound.)
- **P3 (trap clause, pre-diagnosed):** if P1 fails WITH `-fmad=false` set,
  the run is booked KILL-P1 and the divergence diagnosis (which op, which
  cell) is the finding. No tolerance loosening, no re-rolls, no "fix the
  gate". If P1 passes only when `-fmad=false` is REMOVED, P1 as stated still
  FAILS and the pass-without is recorded as a trap entry, not a KEEP.

## Known non-claims

- 16² GPU timings will lose to CPU on kernel-launch overhead. Expected and
  irrelevant: the performance claim lives at 1024² only.
- Cross-GPU-model bit identity is not claimed (same-op-order per thread is;
  summation order is host-side fixed).

## Verdicts

- **KEEP** = P1 PASS and P2 PASS.
- **PARTIAL** = P1 PASS, P2 FAIL (correct runtime, slow card).
- **KILL** = P1 FAIL (either P3 clause).
