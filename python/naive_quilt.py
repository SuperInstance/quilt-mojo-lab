"""Naive dict-of-dicts quilt: the "standard Python execution loop" baseline.

Faithful port of the original high-perf draft semantics (kept identical across
all five runtimes in this lab):

  - Grid of size N x N; each cell carries potential, resistance, entropy, split.
  - inject_force(r, c, p) sets a cell's potential.
  - Flow pass (synchronous): snapshot all potentials; for each cell and each of
    its in-bounds N/S/E/W neighbors, if (pot - neighbor_pot) > resistance then
    leakage = (diff - resistance) * 0.22 flows OUT of this cell. Outbound-only,
    threshold-gated, NOT charge-conserving (that is the specified semantics).
  - Entropy pass: entropy = min(1.0, pot / 5.0 * 0.95); split = entropy > threshold.

This version uses per-cell dict entries and tuple-key lookups: the slow path
the flat-memory substrate is meant to replace.
"""


class NaiveQuilt:
    def __init__(self, size: int, split_threshold: float = 0.6):
        self.size = size
        self.split_threshold = split_threshold
        self.cells = {}
        for r in range(size):
            for c in range(size):
                self.cells[(r, c)] = {
                    "potential": 0.0,
                    "resistance": 0.4,
                    "entropy": 0.0,
                    "split": False,
                }

    def inject_force(self, r: int, c: int, potential: float):
        self.cells[(r, c)]["potential"] = float(potential)

    def step_flow(self):
        snap = {k: v["potential"] for k, v in self.cells.items()}
        n = self.size
        for (r, c), cell in self.cells.items():
            pot = snap[(r, c)]
            res = cell["resistance"]
            adj = pot
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < n and 0 <= nc < n:
                    diff = pot - snap[(nr, nc)]
                    if diff > res:
                        adj -= (diff - res) * 0.22
            cell["potential"] = adj

    def step_entropy(self):
        for cell in self.cells.values():
            ent = min(1.0, cell["potential"] / 5.0 * 0.95)
            cell["entropy"] = ent
            cell["split"] = ent > self.split_threshold

    def potential(self, r: int, c: int) -> float:
        return self.cells[(r, c)]["potential"]

    def entropy(self, r: int, c: int) -> float:
        return self.cells[(r, c)]["entropy"]

    def split(self, r: int, c: int) -> bool:
        return self.cells[(r, c)]["split"]

    def checksum(self) -> float:
        acc = 0.0
        for r in range(self.size):
            for c in range(self.size):
                acc += self.cells[(r, c)]["potential"]
        return acc

    def run(self, steps: int):
        for _ in range(steps):
            self.step_flow()
        self.step_entropy()
