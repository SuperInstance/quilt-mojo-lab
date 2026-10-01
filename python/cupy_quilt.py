"""Runtime #9 — GPU-native CuPy SoA quilt (docs/RUNTIME9-PREREG.md).

Same semantics as soa_quilt.SoaQuilt, as two CUDA kernels via CuPy RawKernel
(one thread per cell, synchronous snapshot-based flow pass, op order South /
North / East / West exactly as the oracle).

Parity design (the whole game, see the frozen pre-reg):

- The Python oracle promotes float32 storage to float64 on every read and
  does ALL per-cell intermediate math in float64, with ONE float32 rounding
  at the final store (`pot[idx] = adj`). The kernels therefore compute in
  `double` and store `(float)` once per cell per pass.
- `-fmad=false` at compile: FMA contraction would skip the explicit
  intermediate rounding the oracle performs (the RUNTIME9 trap clause).
- Checksum is accumulated HOST-side in the same sequential float64 order as
  SoaQuilt.checksum() — the GPU only ever hands back the float32 fields.
"""
import numpy as np
import cupy as cp

_FLOW_SRC = r"""
extern "C" __global__ void flow_kernel(const float* snap, const float* res,
                                       float* pot, int n, long long total) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x
                  + (long long)threadIdx.x;
    if (idx >= total) return;
    int r = (int)(idx / (long long)n);
    int c = (int)(idx - (long long)r * (long long)n);
    double p = (double)snap[idx];
    double rr = (double)res[idx];
    double adj = p;
    double d;
    if (r + 1 < n) { d = p - (double)snap[idx + n]; if (d > rr) adj -= (d - rr) * 0.22; }
    if (r - 1 >= 0) { d = p - (double)snap[idx - n]; if (d > rr) adj -= (d - rr) * 0.22; }
    if (c + 1 < n) { d = p - (double)snap[idx + 1]; if (d > rr) adj -= (d - rr) * 0.22; }
    if (c - 1 >= 0) { d = p - (double)snap[idx - 1]; if (d > rr) adj -= (d - rr) * 0.22; }
    pot[idx] = (float)adj;
}
"""

_ENTROPY_SRC = r"""
extern "C" __global__ void entropy_kernel(const float* pot, float* ent, float* spl,
                                          double thr, long long total) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x
                  + (long long)threadIdx.x;
    if (idx >= total) return;
    double e = (double)pot[idx] / 5.0 * 0.95;
    if (e > 1.0) e = 1.0;
    ent[idx] = (float)e;
    spl[idx] = (e > thr) ? 1.0f : 0.0f;
}
"""

# -fmad=false is the frozen pre-reg clause: nvrtc honors --fmad=<bool>.
_flow_kernel = cp.RawKernel(_FLOW_SRC, "flow_kernel", options=("--fmad=false",))
_entropy_kernel = cp.RawKernel(_ENTROPY_SRC, "entropy_kernel",
                               options=("--fmad=false",))
_THREADS = 256


def _launch(kernel, args, total):
    blocks = (total + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,), args)


class CupySoaQuilt:
    def __init__(self, size: int, split_threshold: float = 0.6):
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("cupy runtime: no CUDA device visible")
        free, _total = cp.cuda.runtime.memGetInfo()
        need = 4 * 4 * size * size + (1 << 20)
        if free < need:
            raise RuntimeError(
                f"cupy runtime: {free} B VRAM free, need {need} B (fail loud)")
        self.size = size
        self.split_threshold = split_threshold
        self.total_cells = size * size
        self.pot = cp.zeros(self.total_cells, dtype=cp.float32)
        self.res = cp.full(self.total_cells, 0.4, dtype=cp.float32)
        self.ent = cp.zeros(self.total_cells, dtype=cp.float32)
        self.spl = cp.zeros(self.total_cells, dtype=cp.float32)

    def inject_force(self, r: int, c: int, potential: float):
        self.pot[r * self.size + c] = np.float32(potential)

    def step_flow(self):
        n = np.int32(self.size)
        total = np.int64(self.total_cells)
        snap = self.pot.copy()  # the oracle's contiguous snapshot
        _launch(_flow_kernel, (snap, self.res, self.pot, n, total),
                self.total_cells)

    def step_entropy(self):
        total = np.int64(self.total_cells)
        thr = np.float64(self.split_threshold)
        _launch(_entropy_kernel, (self.pot, self.ent, self.spl, thr, total),
                self.total_cells)

    def flow_sync(self):
        """Bench hook: kernels are async; the harness syncs before t1."""
        cp.cuda.get_current_stream().synchronize()

    def potential(self, r: int, c: int) -> float:
        return float(self.pot[r * self.size + c].get())

    def entropy(self, r: int, c: int) -> float:
        return float(self.ent[r * self.size + c].get())

    def split(self, r: int, c: int) -> bool:
        return bool(self.spl[r * self.size + c].get() == 1.0)

    def checksum(self) -> float:
        # Host-side sequential float64 accumulation, bit-order-identical to
        # SoaQuilt.checksum() — never a device tree reduction here.
        pot_host = cp.asnumpy(self.pot)
        acc = 0.0
        for i in range(self.total_cells):
            acc += pot_host[i]
        return acc

    def run(self, steps: int):
        for _ in range(steps):
            self.step_flow()
        self.step_entropy()
        self.flow_sync()

    def raw_bytes(self) -> bytes:
        """The C-compatible SoA view: four contiguous field blocks."""
        return (cp.asnumpy(self.pot).tobytes()
                + cp.asnumpy(self.res).tobytes()
                + cp.asnumpy(self.ent).tobytes()
                + cp.asnumpy(self.spl).tobytes())
