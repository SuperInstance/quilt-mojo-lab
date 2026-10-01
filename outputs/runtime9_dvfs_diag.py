import time, numpy as np, cupy as cp
import sys
sys.path.insert(0, "python")
from cupy_quilt import CupySoaQuilt
INJECT = [(1, 1, 4.8)]

for size in (512, 1024, 512, 1024):
    q = CupySoaQuilt(size)
    for r, c, p in INJECT:
        q.inject_force(r, c, p)
    for _ in range(3):
        q.step_flow()          # warm clocks + JIT
    q.flow_sync()
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    ev0.record()
    for _ in range(20):
        q.step_flow()
    ev1.record()
    ev1.synchronize()
    ms = cp.cuda.get_elapsed_time(ev0, ev1)
    cells = size * size * 20
    print(f"{size}x{size}: {ms:.2f} ms / 20 passes -> {cells/(ms/1e3):,.0f} cells/s", flush=True)
    del q
    cp.get_default_memory_pool().free_all_blocks()
