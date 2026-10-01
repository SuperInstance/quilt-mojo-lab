# RUNTIME 10 pre-reg — fp16/bf16 precision-budget probe on the CuPy substrate

owner: lucineer
Runtime #9 proved exact bit-parity at 3.10G cells/s. The wider question (M17-adjacent, but probed on
OUR substrate): precision is a design axis, not a floor. How much precision can the quilt substrate
give up before the CHECKSUM stops being trustworthy? This pre-reg buys the divergence curve, not a
product claim. Runner (to write): `python/cupy_quilt_fp16.py`. Receipt: `outputs/runtime10_fp16_curves.json`.

## Frozen claims
- **P1 (the deliverable IS the curve)**: run the wave with fp16-compute and bf16-compute
  (half intermediates, float32 accumulate, float32 store, checksum float64 in the oracle's sequential
  order) at 512², T ∈ {16, 64, 256, 1024}. Report |Δchecksum| and max|Δpot| vs the float64 oracle.
  NO tolerance is set — verdict is deferred by design (declared measurement probe).
  "Parity" language is banned in receipts and commits for this probe; this is a divergence study.
- **P2 (speed)**: the fp16 kernel ≥ 1.4× the #9 fp64-intermediate kernel at 1024², measured under the
  RAMP LAW (0.6 s synced ramp before every timed block — banked instrument law, no exceptions; ramp
  receipt embedded in the JSON).
- **P3 (trap clause)**: if P2 fails, do NOT tune the kernel inside this pre-reg — book the number and
  open a follow-up. One question per pre-reg.
