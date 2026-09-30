"""SoA (structure-of-arrays) quilt: the wave-69 re-layout.

v0.1.0 measured a 42% throughput cliff at 512^2 on the interleaved AoS block
(4 float32 slots per cell): the snapshot copy and resistance reads are
strided, and the 4MB block + 1MB snapshot exceed cache. SoA splits the block
into four separate arrays (pot / res / ent / split) so that:

  - the flow-pass snapshot becomes a contiguous copy (memcpy-class, C-level
    array slicing instead of a strided Python loop),
  - every neighbor read is unit-stride on one contiguous snapshot,
  - the hot working set is pot+snap+res (3 x 1MB at 512^2), not the full
    strided 5MB AoS working set.

Registered falsifiable prediction (worklog wave-68 discussion round,
p registered 0.45): SoA flow throughput at 512^2 >= 2x the AoS flow on the
compiled runtimes. docs/RESULTS.md wave-69 section records the measured
verdict either way.

Same semantics as naive_quilt.NaiveQuilt / flat_quilt.FlatQuilt.
"""
from array import array


class SoaQuilt:
    def __init__(self, size: int, split_threshold: float = 0.6):
        self.size = size
        self.split_threshold = split_threshold
        self.total_cells = size * size
        self.pot = array("f", bytes(4 * self.total_cells))
        self.res = array("f", [0.4]) * self.total_cells
        self.ent = array("f", bytes(4 * self.total_cells))
        self.spl = array("f", bytes(4 * self.total_cells))  # .split() method stays callable

    def inject_force(self, r: int, c: int, potential: float):
        self.pot[r * self.size + c] = float(potential)

    def step_flow(self):
        n = self.size
        snap = self.pot[:]  # contiguous C-level copy of the potential field
        pot = self.pot
        res = self.res
        for r in range(n):
            row = r * n
            for c in range(n):
                idx = row + c
                p = snap[idx]
                rr = res[idx]
                adj = p
                # South
                if r + 1 < n:
                    d = p - snap[idx + n]
                    if d > rr:
                        adj -= (d - rr) * 0.22
                # North
                if r - 1 >= 0:
                    d = p - snap[idx - n]
                    if d > rr:
                        adj -= (d - rr) * 0.22
                # East
                if c + 1 < n:
                    d = p - snap[idx + 1]
                    if d > rr:
                        adj -= (d - rr) * 0.22
                # West
                if c - 1 >= 0:
                    d = p - snap[idx - 1]
                    if d > rr:
                        adj -= (d - rr) * 0.22
                pot[idx] = adj

    def step_entropy(self):
        thr = self.split_threshold
        pot = self.pot
        ent = self.ent
        spl = self.spl
        for i in range(self.total_cells):
            e = pot[i] / 5.0 * 0.95
            if e > 1.0:
                e = 1.0
            ent[i] = e
            spl[i] = 1.0 if e > thr else 0.0

    def potential(self, r: int, c: int) -> float:
        return self.pot[r * self.size + c]

    def entropy(self, r: int, c: int) -> float:
        return self.ent[r * self.size + c]

    def split(self, r: int, c: int) -> bool:
        return self.spl[r * self.size + c] == 1.0

    def checksum(self) -> float:
        acc = 0.0
        for i in range(self.total_cells):
            acc += self.pot[i]
        return acc

    def run(self, steps: int):
        for _ in range(steps):
            self.step_flow()
        self.step_entropy()

    def raw_bytes(self) -> bytes:
        """The C-compatible SoA view: four contiguous field blocks."""
        return self.pot.tobytes() + self.res.tobytes() + \
            self.ent.tobytes() + self.spl.tobytes()
