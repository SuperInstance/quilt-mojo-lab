"""ctypes bridge to the compiled C flat-memory quilt kernel (c/libflatquilt.so).

Same Python API as the other runtimes; the hot loops live in C compiled -O3.
"""
import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "..", "c", "libflatquilt.so")


class CFlatQuilt:
    def __init__(self, size: int, split_threshold: float = 0.6):
        if not os.path.exists(_LIB):
            raise RuntimeError("c/libflatquilt.so missing — run c/build.sh first")
        self.lib = ctypes.CDLL(_LIB)
        self.size = size
        self.split_threshold = split_threshold
        self.total_cells = size * size
        self.mem = (ctypes.c_float * (self.total_cells * 4))()
        for i in range(self.total_cells):
            self.mem[i * 4 + 1] = 0.4
        self._snap = (ctypes.c_float * self.total_cells)()
        self.lib.qml_flow.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.qml_entropy.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
        self.lib.qml_checksum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.lib.qml_checksum.restype = ctypes.c_float
        self.lib.qml_inject.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_float]

    def inject_force(self, r: int, c: int, potential: float):
        self.lib.qml_inject(self.mem, self.size, r, c, float(potential))

    def step_flow(self):
        self.lib.qml_flow(self.mem, self.size, self._snap)

    def step_entropy(self):
        self.lib.qml_entropy(self.mem, self.size, float(self.split_threshold))

    def potential(self, r: int, c: int) -> float:
        return self.mem[(r * self.size + c) * 4]

    def entropy(self, r: int, c: int) -> float:
        return self.mem[(r * self.size + c) * 4 + 2]

    def split(self, r: int, c: int) -> bool:
        return self.mem[(r * self.size + c) * 4 + 3] == 1.0

    def checksum(self) -> float:
        return float(self.lib.qml_checksum(self.mem, self.size))

    def run(self, steps: int):
        for _ in range(steps):
            self.step_flow()
        self.step_entropy()

    def raw_bytes(self) -> bytes:
        return bytes(self.mem)
