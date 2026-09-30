"""Flat-array quilt: mirrors the Mojo/C layout exactly (AoS, 4 float32 slots per
cell: [potential, resistance, entropy, split]) in a single Python array('f') block,
indexed by hardware-style offsets instead of dict lookups.

Same semantics as naive_quilt.NaiveQuilt (see that module's header).
"""
from array import array

SLOT_POT = 0
SLOT_RES = 1
SLOT_ENT = 2
SLOT_SPLIT = 3


class FlatQuilt:
    def __init__(self, size: int, split_threshold: float = 0.6):
        self.size = size
        self.split_threshold = split_threshold
        self.total_cells = size * size
        # Flat heap block: 4 float slots per cell
        self.mem = array("f", [0.0]) * (self.total_cells * 4)
        for i in range(self.total_cells):
            self.mem[i * 4 + SLOT_RES] = 0.4

    def _offset(self, r: int, c: int) -> int:
        return (r * self.size + c) * 4

    def inject_force(self, r: int, c: int, potential: float):
        self.mem[self._offset(r, c) + SLOT_POT] = float(potential)

    def step_flow(self):
        n = self.size
        mem = self.mem
        # Snapshot of potentials (contiguous copy of slot-0 values)
        snap = array("f", bytes(4 * self.total_cells))
        for i in range(self.total_cells):
            snap[i] = mem[i * 4]
        for r in range(n):
            row = r * n
            for c in range(n):
                idx = row + c
                pot = snap[idx]
                res = mem[idx * 4 + SLOT_RES]
                adj = pot
                # South
                if r + 1 < n:
                    diff = pot - snap[idx + n]
                    if diff > res:
                        adj -= (diff - res) * 0.22
                # North
                if r - 1 >= 0:
                    diff = pot - snap[idx - n]
                    if diff > res:
                        adj -= (diff - res) * 0.22
                # East
                if c + 1 < n:
                    diff = pot - snap[idx + 1]
                    if diff > res:
                        adj -= (diff - res) * 0.22
                # West
                if c - 1 >= 0:
                    diff = pot - snap[idx - 1]
                    if diff > res:
                        adj -= (diff - res) * 0.22
                mem[idx * 4 + SLOT_POT] = adj

    def step_entropy(self):
        thr = self.split_threshold
        mem = self.mem
        for i in range(self.total_cells):
            ent = mem[i * 4] / 5.0 * 0.95
            if ent > 1.0:
                ent = 1.0
            mem[i * 4 + SLOT_ENT] = ent
            mem[i * 4 + SLOT_SPLIT] = 1.0 if ent > thr else 0.0

    def potential(self, r: int, c: int) -> float:
        return self.mem[self._offset(r, c) + SLOT_POT]

    def entropy(self, r: int, c: int) -> float:
        return self.mem[self._offset(r, c) + SLOT_ENT]

    def split(self, r: int, c: int) -> bool:
        return self.mem[self._offset(r, c) + SLOT_SPLIT] == 1.0

    def checksum(self) -> float:
        acc = 0.0
        for i in range(self.total_cells):
            acc += self.mem[i * 4]
        return acc

    def run(self, steps: int):
        for _ in range(steps):
            self.step_flow()
        self.step_entropy()

    def raw_bytes(self) -> bytes:
        """The C-compatible view: the flat block, ready to stream to any
        external consumer (the 'interface layer' the design promises)."""
        return self.mem.tobytes()
