"""Numpy-vectorized quilt: same semantics, whole-grid operations.

The neighbor pass becomes four shifted-array subtractions; the entropy pass is
pure elementwise math. This is what "SIMD vectorization" looks like when the
runtime can see whole rows at once — the fair middle baseline between the
naive dict loop and the compiled flat-memory runtimes.
"""
import numpy as np

LEAK = 0.22


class VecQuilt:
    def __init__(self, size: int, split_threshold: float = 0.6):
        self.size = size
        self.split_threshold = split_threshold
        self.pot = np.zeros((size, size), dtype=np.float32)
        self.res = np.full((size, size), 0.4, dtype=np.float32)
        self.ent = np.zeros((size, size), dtype=np.float32)
        self.split_mask = np.zeros((size, size), dtype=bool)

    def inject_force(self, r: int, c: int, potential: float):
        self.pot[r, c] = np.float32(potential)

    def step_flow(self):
        n = self.size
        pot = self.pot
        res = self.res
        snap = pot.copy()
        adj = snap.copy()
        # South neighbor (row+1): rows 0..n-2 see rows 1..n-1
        diff = snap[: n - 1, :] - snap[1:, :]
        adj[: n - 1, :] -= np.where(diff > res[: n - 1, :], (diff - res[: n - 1, :]) * LEAK, 0.0)
        # North neighbor (row-1): rows 1..n-1 see rows 0..n-2
        diff = snap[1:, :] - snap[: n - 1, :]
        adj[1:, :] -= np.where(diff > res[1:, :], (diff - res[1:, :]) * LEAK, 0.0)
        # East neighbor (col+1)
        diff = snap[:, : n - 1] - snap[:, 1:]
        adj[:, : n - 1] -= np.where(diff > res[:, : n - 1], (diff - res[:, : n - 1]) * LEAK, 0.0)
        # West neighbor (col-1)
        diff = snap[:, 1:] - snap[:, : n - 1]
        adj[:, 1:] -= np.where(diff > res[:, 1:], (diff - res[:, 1:]) * LEAK, 0.0)
        self.pot = adj

    def step_entropy(self):
        self.ent = np.minimum(1.0, self.pot / np.float32(5.0) * np.float32(0.95))
        self.split_mask = self.ent > np.float32(self.split_threshold)

    def potential(self, r: int, c: int) -> float:
        return float(self.pot[r, c])

    def entropy(self, r: int, c: int) -> float:
        return float(self.ent[r, c])

    def split(self, r: int, c: int) -> bool:
        return bool(self.split_mask[r, c])

    def checksum(self) -> float:
        return float(self.pot.sum(dtype=np.float64))

    def run(self, steps: int):
        for _ in range(steps):
            self.step_flow()
        self.step_entropy()
