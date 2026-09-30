# quilt_high_perf.mojo — the flat-memory quilt substrate, modernized.
#
# Original architecture (user draft, preserved verbatim as mojo/original_draft.mojo):
#   [ STRUCT COMPILATION ] -> [ MEMORY REGISTER SPECS ] -> [ BARE-METAL EXECUTION ]
#
# This build targets Mojo 1.2.0-dev (conda.modular.com/max-nightly, 2026-09-30).
# API archaeology vs the draft (full notes in docs/MOJO-NOTES.md):
#   - `fn` removed -> `def`;  `let` removed -> `var`;  file-scope `alias` removed
#   - DTypePointer -> UnsafePointer -> Pointer (deprecated twice)
#   - typed `.alloc(n)` -> free-function `unsafe_alloc[T](n)`, `.free()` -> `.unsafe_free()`
#   - `inout self` ctor -> `out self`; mutating methods -> `mut self`
#   - `time` module -> `std.time.perf_counter_ns()`
#   - the 4-float-per-cell layout is loaded/stored as a whole-cell SIMD[F32, 4]
#     register (one instruction per cell read/write)
#   - snapshot buffer allocated ONCE in __init__ — zero allocation in hot loops
#   - neighbor pass fully unrolled (S, N, E, W) — no VariadicList overhead
#
# Semantics are pinned by python/test_correctness.py across 5 runtimes.

from std import sys
from std import time


struct HighPerfQuiltSubstrate:
    """The contiguous flat block: total_cells * 4 float32 slots on the heap.

    Single-owner discipline: the substrate is never copied; main() owns it and
    mutates it through the passes. `__del__` frees the block exactly once.
    """
    var size: Int
    var split_threshold: Scalar[DType.float32]
    var total_cells: Int
    var memory_ptr: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var snap_ptr: Pointer[Scalar[DType.float32], MutUntrackedOrigin]

    def __init__(out self, size: Int, split_threshold: Scalar[DType.float32]):
        self.size = size
        self.split_threshold = split_threshold
        self.total_cells = size * size
        self.memory_ptr = alloc[Scalar[DType.float32]](
            self.total_cells * 4)
        self.snap_ptr = alloc[Scalar[DType.float32]](self.total_cells)
        # Zero the flat block and the snapshot buffer
        for i in range(self.total_cells * 4):
            self.memory_ptr[i] = 0.0
        for i in range(self.total_cells):
            self.snap_ptr[i] = 0.0
            self.memory_ptr[i * 4 + 1] = 0.4  # default resistance barriers

    def __deinit__(deinit self):
        self.memory_ptr.unsafe_free()
        self.snap_ptr.unsafe_free()

    @always_inline
    def _offset(self, row: Int, col: Int) -> Int:
        return (row * self.size + col) * 4

    def inject_force(mut self, row: Int, col: Int, potential: Scalar[DType.float32]):
        self.memory_ptr[self._offset(row, col)] = potential

    def execute_flow_pass(mut self):
        """Bare-metal relaxation: unrolled neighbor gather, snapshot semantics.

        Honesty note (docs/RESULTS.md): the neighbor reads are data-dependent
        (gather at +/-1 and +/-size), so the win here comes from zero per-cell
        overhead — no objects, no dict lookups, no refcounting — not from lane
        parallelism. The lane-parallel case is the entropy pass below.
        """
        var n = self.size
        var mem = self.memory_ptr
        var snap = self.snap_ptr
        for i in range(self.total_cells):
            snap[i] = mem[i * 4]
        for r in range(n):
            var row = r * n
            for c in range(n):
                var idx = row + c
                var pot = snap[idx]
                var res = mem[idx * 4 + 1]
                var adj = pot
                if r + 1 < n:
                    var d = pot - snap[idx + n]
                    if d > res:
                        adj -= (d - res) * 0.22
                if r - 1 >= 0:
                    var d = pot - snap[idx - n]
                    if d > res:
                        adj -= (d - res) * 0.22
                if c + 1 < n:
                    var d = pot - snap[idx + 1]
                    if d > res:
                        adj -= (d - res) * 0.22
                if c - 1 >= 0:
                    var d = pot - snap[idx - 1]
                    if d > res:
                        adj -= (d - res) * 0.22
                mem[idx * 4] = adj

    def execute_entropy_pass_simd(mut self):
        """Whole-cell SIMD register pass: load the 4 slots as one SIMD[F32, 4],
        update entropy and split lanes, store back. Unit-stride, aligned —
        this is the lane-parallel case, unlike the neighbor gather."""
        var mem = self.memory_ptr
        var thr = self.split_threshold
        for i in range(self.total_cells):
            var off = i * 4
            var reg = mem.load[width=4](off)
            var ent = reg[0] * 0.19
            if ent > 1.0:
                ent = 1.0
            reg[2] = ent
            if ent > thr:
                reg[3] = 1.0
            else:
                reg[3] = 0.0
            mem.store[width=4](off, reg)

    def checksum(self) -> Float64:
        var acc: Float64 = 0.0
        for i in range(self.total_cells):
            acc += Float64(self.memory_ptr[i * 4])
        return acc

    def potential(self, row: Int, col: Int) -> Scalar[DType.float32]:
        return self.memory_ptr[self._offset(row, col)]

    def entropy(self, row: Int, col: Int) -> Scalar[DType.float32]:
        return self.memory_ptr[self._offset(row, col) + 2]

    def split(self, row: Int, col: Int) -> Bool:
        return self.memory_ptr[self._offset(row, col) + 3] == 1.0

    def compile_and_dump_state(self, limit: Int):
        """Register dump (first limit rows/cols) — draft parity, demo mode."""
        print("--- Low-Level Fabric Register Dump ---")
        for r in range(min(limit, self.size)):
            for c in range(min(limit, self.size)):
                var off = self._offset(r, c)
                var status = "STATIC"
                if self.memory_ptr[off + 3] == 1.0:
                    status = "ACTIVE"
                print(
                    "Cell [", r, ",", c, "] -> Potential:",
                    self.memory_ptr[off], " | Entropy:", self.memory_ptr[off + 2],
                    " | Split Status:", status,
                )


def main():
    print("--- Booting Mojo High-Performance Quilt Substrate ---")
    var size = 64
    var steps = 10
    var demo = False
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
    if len(args) > 3 and args[3] == "demo":
        demo = True
    print("grid", size, "x", size, "| steps", steps)

    var fabric = HighPerfQuiltSubstrate(size, 0.6)

    # Phase 1: direct register injection (draft-faithful source)
    fabric.inject_force(1, 1, 4.8)

    # Phase 2: bare-metal flow relaxation passes
    var t0 = time.perf_counter_ns()
    for _ in range(steps):
        fabric.execute_flow_pass()
    var t1 = time.perf_counter_ns()

    # Phase 3: whole-cell SIMD entropy/split pass
    fabric.execute_entropy_pass_simd()
    var t2 = time.perf_counter_ns()

    # Phase 4: export pipeline (checksum = cross-runtime conformance value)
    print("CHECKSUM", fabric.checksum())
    print("FLOW_NS", t1 - t0)
    print("ENTROPY_NS", t2 - t1)
    print("SPLIT_AT_1_1", fabric.split(1, 1))

    if demo:
        fabric.compile_and_dump_state(4)
