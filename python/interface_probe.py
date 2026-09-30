"""Interface probe: stream the flat pointer block out to external consumers.

This is the minimal "interface layer" the high-performance architecture
promises: the substrate's memory is already one contiguous C-compatible
block, so export is a single copy-free view (numpy) or a byte dump.

Outputs (outputs/):
  quilt_state_16.bin   raw 4-float-per-cell block (C ABI: float32[cells*4])
  quilt_state_16.npy   the same block as a (cells, 4) float32 array
  quilt_layout.h       the C header declaring the ABI for external contexts
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
ROOT = os.path.dirname(_HERE)
OUT = os.path.join(ROOT, "outputs", "interface")
os.makedirs(OUT, exist_ok=True)

from flat_quilt import FlatQuilt  # noqa: E402

N, STEPS = 16, 3

q = FlatQuilt(N)
q.inject_force(1, 1, 4.8)
for _ in range(STEPS):
    q.step_flow()
q.step_entropy()

blob = q.raw_bytes()
arr = np.frombuffer(blob, dtype=np.float32).reshape(N * N, 4)

bin_path = os.path.join(OUT, f"quilt_state_{N}.bin")
npy_path = os.path.join(OUT, f"quilt_state_{N}.npy")
with open(bin_path, "wb") as f:
    f.write(blob)
np.save(npy_path, arr)

hdr = f"""/* quilt_layout.h — C ABI of the quilt flat-memory block (auto-generated) */
#ifndef QUILT_LAYOUT_H
#define QUILT_LAYOUT_H
#include <stdint.h>
#define QUILT_SIZE {N}
#define QUILT_CELLS {N * N}
#define QUILT_SLOTS_PER_CELL 4   /* [potential, resistance, entropy, split] */
typedef struct {{
    float potential;
    float resistance;
    float entropy;
    float split;      /* 1.0 = ACTIVE, 0.0 = STATIC */
}} quilt_cell_t;
/* The whole fabric is quilt_cell_t QUILT_CELLS, contiguous, row-major. */
typedef quilt_cell_t quilt_fabric_t[QUILT_CELLS];
#endif
"""
with open(os.path.join(OUT, "quilt_layout.h"), "w") as f:
    f.write(hdr)

# verification round-trip: read the .bin back and re-derive checksum
back = np.fromfile(bin_path, dtype=np.float32).reshape(N * N, 4)
cs_back = float(back[:, 0].sum(dtype=np.float64))
cs_ref = q.checksum()
ok = abs(cs_back - cs_ref) <= 1e-4
print(f"exported {len(blob)} bytes -> {bin_path}")
print(f"npy -> {npy_path}; header -> quilt_layout.h")
print(f"round-trip checksum {cs_back:.10f} vs {cs_ref:.10f}: {'OK' if ok else 'FAIL'}")
assert ok
