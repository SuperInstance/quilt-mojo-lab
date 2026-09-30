# quilt_soa.mojo — wave-69 SoA re-layout of the flat quilt substrate.
#
# v0.1.0 (quilt_high_perf.mojo, interleaved AoS) measured a 42% throughput
# cliff at 512^2 and its flow pass stayed a scalar gather: with 4 floats per
# cell interleaved, neighbor reads at +/-1 and +/-N are strided over the block.
# SoA splits the block into four separate field arrays (pot / res / ent /
# split) so that:
#
#   - the snapshot copy is unit-stride over ONE contiguous potential field
#   - every neighbor read is unit-stride on that snapshot
#   - the interior flow pass finally vectorizes: W=8 consecutive cells per
#     lane group, all loads unit-stride (the lane-parallel case the draft
#     claimed and the AoS layout could not deliver)
#   - boundary cells (rows 0 / N-1, cols 0 / N-1) run the scalar path
#
# Registered falsifiable prediction (wave-68 discussion round, p=0.45):
# SoA flow at 512^2 >= 2x AoS flow. docs/RESULTS.md wave-69 section records
# the measured verdict either way.
#
# Toolchain: Mojo 1.2.0-dev — see docs/MOJO-NOTES.md for the API archaeology
# (Pointer[Scalar[F32], MutUntrackedOrigin], free-function alloc, out/mut/deinit
# self conventions, std.time.perf_counter_ns).

from std import sys
from std import time


struct SoaQuiltSubstrate:
    """Five contiguous field arrays on the heap, single-owner discipline.

    pot / res / ent / split / snap each hold total_cells float32 values.
    The substrate is never copied; main() owns it and mutates it through
    the passes. `__deinit__` frees each block exactly once.
    """
    var size: Int
    var split_threshold: Scalar[DType.float32]
    var total_cells: Int
    var pot_ptr: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var res_ptr: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var ent_ptr: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var split_ptr: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var snap_ptr: Pointer[Scalar[DType.float32], MutUntrackedOrigin]

    def __init__(out self, size: Int, split_threshold: Scalar[DType.float32]):
        self.size = size
        self.split_threshold = split_threshold
        self.total_cells = size * size
        self.pot_ptr = alloc[Scalar[DType.float32]](self.total_cells)
        self.res_ptr = alloc[Scalar[DType.float32]](self.total_cells)
        self.ent_ptr = alloc[Scalar[DType.float32]](self.total_cells)
        self.split_ptr = alloc[Scalar[DType.float32]](self.total_cells)
        self.snap_ptr = alloc[Scalar[DType.float32]](self.total_cells)
        for i in range(self.total_cells):
            self.pot_ptr[i] = 0.0
            self.res_ptr[i] = 0.4  # default resistance barriers
            self.ent_ptr[i] = 0.0
            self.split_ptr[i] = 0.0
            self.snap_ptr[i] = 0.0

    def __deinit__(deinit self):
        self.pot_ptr.unsafe_free()
        self.res_ptr.unsafe_free()
        self.ent_ptr.unsafe_free()
        self.split_ptr.unsafe_free()
        self.snap_ptr.unsafe_free()

    def inject_force(mut self, row: Int, col: Int, potential: Scalar[DType.float32]):
        self.pot_ptr[row * self.size + col] = potential

    @always_inline
    def _flow_cell(
        self,
        snap: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
        pot: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
        res: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
        r: Int,
        c: Int,
    ):
        """Scalar relaxation for one cell with in-bounds neighbor checks."""
        var n = self.size
        var idx = r * n + c
        var p = snap[idx]
        var rr = res[idx]
        var adj = p
        if r + 1 < n:
            var d = p - snap[idx + n]
            if d > rr:
                adj -= (d - rr) * 0.22
        if r - 1 >= 0:
            var d = p - snap[idx - n]
            if d > rr:
                adj -= (d - rr) * 0.22
        if c + 1 < n:
            var d = p - snap[idx + 1]
            if d > rr:
                adj -= (d - rr) * 0.22
        if c - 1 >= 0:
            var d = p - snap[idx - 1]
            if d > rr:
                adj -= (d - rr) * 0.22
        pot[idx] = adj

    def execute_flow_pass(mut self):
        """Bare-metal relaxation, SoA layout: unit-stride snapshot copy,
        lane-vectorized interior (W=8 cells per lane group), scalar boundary.

        The vectorized interior computes, for 8 consecutive cells at once:
          adj = p - sum_k max(diff_k - res, 0) * 0.22   over N/S/E/W
        which is the branchless form of the scalar rule (leak iff diff > res).
        Float-association differs from the sequential scalar form by <= 1 ulp;
        the cross-runtime conformance tolerance (1.2e-5) absorbs it.
        NOTE: `alias` is removed in 1.2.0-dev even at function scope (new
        archaeology vs the wave-68 notes) — lane width is the literal 8.
        """
        var n = self.size
        var pot = self.pot_ptr
        var res = self.res_ptr
        var snap = self.snap_ptr
        # Contiguous unit-stride snapshot (memcpy-class; auto-vectorized)
        for i in range(self.total_cells):
            snap[i] = pot[i]

        # Vectorized interior: rows 1..n-2, cols 1..n-2
        for r in range(1, n - 1):
            var row = r * n
            var c = 1
            while c + 8 <= n - 1:
                var off = row + c
                var p = snap.load[width=8](off)
                var rr = res.load[width=8](off)
                var dS = p - snap.load[width=8](off + n)
                var dN = p - snap.load[width=8](off - n)
                var dE = p - snap.load[width=8](off + 1)
                var dW = p - snap.load[width=8](off - 1)
                var leak = max(dS - rr, 0.0)
                leak += max(dN - rr, 0.0)
                leak += max(dE - rr, 0.0)
                leak += max(dW - rr, 0.0)
                pot.store[width=8](off, p - leak * 0.22)
                c += 8
            while c < n - 1:
                self._flow_cell(snap, pot, res, r, c)
                c += 1

        # Boundary rows 0 and n-1 (all cols), then edge cols of interior rows
        for c in range(n):
            self._flow_cell(snap, pot, res, 0, c)
            self._flow_cell(snap, pot, res, n - 1, c)
        for r in range(1, n - 1):
            self._flow_cell(snap, pot, res, r, 0)
            self._flow_cell(snap, pot, res, r, n - 1)

    def execute_entropy_pass(mut self):
        """Scalar elementwise pass (kept scalar honestly: entropy is <3% of
        runtime; the flow pass is the vectorization target in this wave)."""
        var thr = self.split_threshold
        var pot = self.pot_ptr
        var ent = self.ent_ptr
        var sp = self.split_ptr
        for i in range(self.total_cells):
            var e = pot[i] * 0.19
            if e > 1.0:
                e = 1.0
            ent[i] = e
            if e > thr:
                sp[i] = 1.0
            else:
                sp[i] = 0.0

    def checksum(self) -> Float64:
        var acc: Float64 = 0.0
        for i in range(self.total_cells):
            acc += Float64(self.pot_ptr[i])
        return acc

    def potential(self, row: Int, col: Int) -> Scalar[DType.float32]:
        return self.pot_ptr[row * self.size + col]

    def entropy(self, row: Int, col: Int) -> Scalar[DType.float32]:
        return self.ent_ptr[row * self.size + col]

    def split(self, row: Int, col: Int) -> Bool:
        return self.split_ptr[row * self.size + col] == 1.0


def main():
    print("--- Booting Mojo SoA Quilt Substrate (wave-69) ---")
    var size = 64
    var steps = 10
    var args = sys.argv()
    if len(args) > 1:
        try:
            size = atol(args[1])
        except e:
            size = 64
    if len(args) > 2:
        try:
            steps = atol(args[2])
        except e:
            steps = 10
    print("grid", size, "x", size, "| steps", steps)

    var fabric = SoaQuiltSubstrate(size, 0.6)

    fabric.inject_force(1, 1, 4.8)

    var t0 = time.perf_counter_ns()
    for _ in range(steps):
        fabric.execute_flow_pass()
    var t1 = time.perf_counter_ns()

    fabric.execute_entropy_pass()
    var t2 = time.perf_counter_ns()

    print("CHECKSUM", fabric.checksum())
    print("FLOW_NS", t1 - t0)
    print("ENTROPY_NS", t2 - t1)
    print("SPLIT_AT_1_1", fabric.split(1, 1))
